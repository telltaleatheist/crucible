# The Crucible API

**GENERATED — do not edit.** `python scripts/gen-api-docs.py` writes this file
from the FastAPI app the server actually runs, and `scripts/release.sh` refuses a
cut when it is stale. Change a request or answer model and regenerate; never edit here.

Every running server serves this same reference, for itself: `GET /docs` (a page),
`GET /v1/docs.md` (this markdown), `GET /v1/docs` (the index as JSON, with each job
type's params schema) and `GET /v1/openapi.json`. None of them needs a token.

Every route is under `/v1` unless it says otherwise. Protected routes need
`Authorization: Bearer <token>` **and** `X-Crucible-Api: 1`, checked in that order.
An error is always a JSON body under an `error` key holding `code`, `message` and
sometimes `details`. Crucible refuses by name and with numbers: branch on `code`,
show a person the `message`.

The prose for WHY a field exists lives in the internals docs (`docs/internals/*.md`); what
is here is what you may send and what comes back.

## Every command

Every route this server answers and every job type it runs, one line each. The sections below give each one in full.

| route | what it does |
| --- | --- |
| `POST /openai/v1/chat/completions` | An OpenAI chat completion, proxied to the resident engine or, for a `<upstream>/<id>` model, forwarded to that upstream. |
| `GET /openai/v1/models` | The resident model in OpenAI's list shape, plus every upstream model a route names. |
| `GET /v1/accelerator` | What is on the card right now and which holders are Crucible's own processes. |
| `GET /v1/activity` | What this server is doing and how far along, in one read with no job id. |
| `GET /v1/capability` | What this server can hold, per capability class, and why not; `enabled: false` is an answer, not an error. |
| `GET /v1/capability/plan` | What an install (`?job_type=`) or a pull (`?subject=`) would give this card, decided live and writing nothing. |
| `GET /v1/catalog` | Every subject this backend can hold, installed or not. |
| `DELETE /v1/catalog/{kind}/{subject_id}` | Delete an installed subject's files. |
| `POST /v1/decide` | One answer distribution per question, read off the resident model's next-token logprobs; with `items`, one choice answer per item in one request. |
| `GET /v1/docs` | Every command this server answers, as data: each route with its door and one line, and each job type with its model, params schema, inputs, returns, notes, an example body and whether it is enabled here. |
| `GET /v1/docs.md` | The whole API reference as markdown: every route, every job type, every model. |
| `GET /v1/events` | Every change on this server as one SSE stream, so an app need not poll: a `snapshot` first, then one event per change (jobs, the queue, queue sessions, the card, chats in flight, tasks, settings, the server stopping). |
| `GET /v1/health` | Is this process alive, in one cheap read: `status` (`ok`, `busy` running a job, `warming` loading a model), the queue depth and what is resident. |
| `GET /v1/info` | Who this server is, what it runs on, what it serves, and the few fixed tables (terminal states, voice sources, service commands) a console shows. |
| `POST /v1/jobs` | Admit one job, or refuse by name. |
| `GET /v1/jobs/{job_id}` | One job's state: `status`, `progress`, `message`, its artifacts and, when it ended, its result or its error. |
| `DELETE /v1/jobs/{job_id}` | Cancel a job; a job still waiting in the queue is removed (reason `client`) and answers `status: removed`. |
| `GET /v1/jobs/{job_id}/artifacts/{name}` | One artifact's bytes, by the name the `done` event lists, with its media type. |
| `GET /v1/jobs/{job_id}/events` | The job's events as SSE, from the first: `queued`, `started`, `warming`, `progress`, `note`, then one of `done` (with its artifacts and the job type's result fields), `failed` (with `error`), `cancelled` or `removed`, after which the stream ends. |
| `POST /v1/jobs/{job_id}/hold` | Keep this job's artifacts for a later job's `{"artifact": {job_id, name}}` inputs until the hold is released or `retention_days` collects it. |
| `DELETE /v1/jobs/{job_id}/hold` | The chain is complete: release the hold and remove the job now. |
| `GET /v1/models` | Every model this build has a manifest for, and where it stands here. |
| `POST /v1/openai/chat/completions` | An OpenAI chat completion, proxied to the resident engine or, for a `<upstream>/<id>` model, forwarded to that upstream. |
| `GET /v1/openai/models` | The resident model in OpenAI's list shape, plus every upstream model a route names. |
| `POST /v1/pairing/decision` | Approve (`allow: true`) or deny a pairing request, naming its `id` and the `user_code` the asking client shows. |
| `POST /v1/pairing/poll` | How a pairing request stands: `pending`, `approved` (the answer then carries the server's `name` and its `token`), `denied` or `expired`. |
| `GET /v1/pairing/requests` | The pairing requests waiting for an answer: who asked, from where, and the `user_code` to compare. |
| `POST /v1/pairing/start` | Ask this server for a token. |
| `GET /v1/peer` | Who manages this engine, and this process's uptime. |
| `POST /v1/peer/claim` | An orchestrator states that it manages this engine; `force` takes it from another orchestrator. |
| `DELETE /v1/peer/claim` | The orchestrator releases its claim; releasing when nothing is claimed succeeds. |
| `GET /v1/ping` | Unauthenticated. |
| `GET /v1/playground` | One page per image, video and audio model this build declares: the params its form shows, with defaults and limits from its manifest, and its `standing`: `ready`; `download`, which its first job fetches by itself (409 `installing` names the task; `download_bytes` when the size is known); or `unavailable`, with `reason` saying why. |
| `GET /v1/playground/presets/{model}` | The presets saved for `model` on this server, by name: each a form's params (no seed) and when it was saved. |
| `PUT /v1/playground/presets/{model}/{name}` | Save (or replace) the preset `name` for `model`: `{"params": {...}}` with the form's own fields - text, numbers and true/false; a seed is refused by name. |
| `DELETE /v1/playground/presets/{model}/{name}` | Remove the preset `name` of `model`; 404 when there is none. |
| `GET /v1/queue` | What waits for the lane, in the order it will be offered it: the open queue session's items first, then first come, first served. |
| `GET /v1/queue/events` | Every change to the queue, for a dashboard: a `snapshot` first, then `added`, `moved`, `started` and `removed` as they happen. |
| `POST /v1/queue/sessions` | Ask for the server for a run of requests you cannot know in advance. |
| `GET /v1/queue/sessions/{session_id}` | Where the session stands: queued (and where), open (what it has run and has in flight, when it would go idle), or closed and why. |
| `DELETE /v1/queue/sessions/{session_id}` | End your session (reason `client`): an open one closes and the card is settled before this answers; a queued one leaves the line. |
| `GET /v1/queue/sessions/{session_id}/events` | The session's own stream: `queued {position, of}` and `moved`, then `opened`, then `closed {reason}`; `removed {reason}` in place of `closed` when it never opened. |
| `POST /v1/queue/sessions/{session_id}/touch` | "Still here", for a long gap on the client's side with nothing in flight. |
| `DELETE /v1/queue/{job_id}` | Take a waiting job, call or queue session out of the queue (reason `operator`), or end the open queue session; a job that has started is cancelled with DELETE /v1/jobs/{id}. |
| `POST /v1/queue/{job_id}/heartbeat` | Say the client that queued this job is still there. |
| `GET /v1/resumable` | Every resume journal this server keeps, newest first, with progress, inputs and expiry. |
| `GET /v1/resumable/{resume_id}` | One journal, as `GET /v1/resumable` lists it. |
| `DELETE /v1/resumable/{resume_id}` | Discard a journal now; refused `resume_in_use` while a job writes it. |
| `POST /v1/server/updating` | Stop admitting work so a deploy can restart this server, if nothing is working. |
| `DELETE /v1/server/updating` | Let go of an update hold (a deploy whose install failed): work is admitted again. |
| `GET /v1/settings` | Where each class's work runs and which upstreams are configured. |
| `PUT /v1/settings` | Apply a partial settings patch, whole or not at all, live without a restart. |
| `PUT /v1/settings/audio/low-vram` | Set `[audio] low_vram` with `{"state": "on" \| "off" \| "auto"}`: `on` and `off` are the operator's and Crucible never changes them; `auto` lets Crucible turn it on exactly where this card cannot hold a splittable audio model whole. |
| `POST /v1/settings/upstreams/{name}/test` | List what an upstream serves, using the body's `key` or `url` when given, else the stored record. |
| `GET /v1/setup` | Everything an app needs to be pointed at this server in one read, including its token and pairing lines, and `network`: whether other devices can reach it (`reachable`, `urls`), said as a `sentence`, and when they cannot, `how` to open it, the one `command` that does (when one exists) and what that `changes`. |
| `GET /v1/tasks` | The last few tasks, newest first. |
| `POST /v1/tasks` | Admit one operator task (pull, install, module, engine or engine-restart), or refuse by name. |
| `GET /v1/tasks/{task_id}` | One task's state: type, status, progress, and its error when it failed. |
| `DELETE /v1/tasks/{task_id}` | Cancel a task. |
| `GET /v1/tasks/{task_id}/events` | The task's events as SSE, ending with `done`, `failed` or `cancelled`; the same shape and resume rules as a job's. |
| `POST /v1/tts/stream` | Open the one streaming session this server will hold at a time, inside a queue session: the client's own (the session header, or the open one it holds), else one opened for the stream, which waits in the line like any session and closes with the stream. |
| `POST /v1/tts/stream/{session_id}` | One op: `say`, `cancel`, `cancel_all` or `close`. |
| `DELETE /v1/tts/stream/{session_id}` | The same as `{"op": "close"}`, for a client that only has verbs. |
| `GET /v1/tts/stream/{session_id}/events` | The session's SSE stream, audio included. |
| `POST /v1/uploads` | Store one file (multipart field `file`) for a later job. |
| `GET /v1/voices` | Every voice this build has a manifest for, and where it stands here. |
| `POST /v1/voices/updates` | Look up, on the Hub, the tag every voice here follows, and say which ones a pull would move. |
| `PUT /v1/voices/{voice_id}` | Pin a voice to a repo revision (`{"pin": ...}`) or write a local manifest override (`{"voice": ...}`), and return its `/v1/voices` row. |
| `DELETE /v1/voices/{voice_id}` | Remove this machine's pin or override for a voice; a shipped voice reverts to its packaged manifest. |
| `GET /v1/voices/{voice_id}/manifest` | One voice's settings as a whole local manifest document, ready to edit and send back with `PUT`. |

Job types are submitted with `POST /v1/jobs` (`{"type", "model", "params", "inputs"}`); each is in full under **Job types** below.

| job type | what it does |
| --- | --- |
| `echo` | Copies every input to an artifact of the same name after a delay. |
| `load-model` | Starts an LLM engine for a model and leaves it resident, so the chat door (POST /v1/openai/chat/completions) and POST /v1/decide can use it. |
| `unload-model` | Takes the resident model off the card now, whichever job put it there (load-model). |
| `load-voice` | Starts narrator with a TTS voice and leaves it resident, for `tts` jobs and serialized TTS streams. |
| `unload-voice` | Takes the resident voice off the card now, whichever job put it there (load-voice or tts). |
| `tts` | Renders a batch of text chunks with a TTS voice, one FLAC per chunk. |
| `asr` | Transcribes one audio file to timed text with Whisper (faster-whisper on cuda-linux, mlx-whisper on a Mac) or Qwen3-ASR. |
| `align` | Forced alignment: places known text in time inside short audio windows with Qwen3-ForcedAligner, one window per chunk, all in one job. |
| `unload-aligner` | Takes the resident aligner off the card now, whichever job put it there (align). |
| `align-longform` | Aligns a whole audiobook to its book text: a rough faster-whisper transcript places each sentence, then Qwen3-ForcedAligner places the words, and out comes a WebVTT with one cue per sentence. |
| `rvc` | Voice conversion: re-voices every input through an RVC model, keeping each input's container, sample format and exact duration. |
| `denoise` | Separates audio into stems with a source-separation model: `vocals-roformer` splits vocals from the instrumental, `denoise-roformer` splits dry speech from noise. |
| `unload-denoiser` | Takes the resident separator off the card now, whichever job put it there (denoise). |
| `image` | Makes one picture from a prompt (text-to-image), redraws an input picture (image-to-image), or regenerates the masked region of one (inpainting and outpainting). |
| `unload-image` | Takes the resident generator off the card now, whichever job put it there (image or load-image). |
| `load-image` | Puts an image model on the card and leaves it there, so the first job that uses it starts at once. |
| `audio` | Makes sound from words: sound effects and instrumental music with Stable Audio 3, songs with sung vocals (or instrumentals) with YuE2. |
| `unload-audio` | Takes the resident audio generator off the card now, whichever job put it there (audio or load-audio). |
| `load-audio` | Puts an audio model on the card and leaves it there, so the first job that uses it starts at once. |
| `segment` | Makes a mask from a picture: the main subject by itself (`birefnet`, background removal), or the object under your points or inside your box (`sam2.1-hiera-large`). |
| `unload-segment` | Takes the resident segmenter off the card now, whichever job put it there (segment or load-segment). |
| `load-segment` | Puts a segment model on the card and leaves it there, so the first job that uses it starts at once. |
| `video` | Makes a video clip with its own sound from a prompt (text-to-video), or brings a start picture to life (image-to-video), with LTX-2.5. |
| `unload-video` | Takes the resident video generator off the card now, whichever job put it there (video or load-video). |
| `load-video` | Puts a video model on the card and leaves it there, so the first job that uses it starts at once. |

## Throughput

What the server already runs together, and what a client sends to get it.

| work | what to send |
| --- | --- |
| Chat (`/v1/chat/completions`) | Each resident LLM takes several chats at once and its engine batches them: `GET /v1/activity` gives `chat.max_in_flight` (16 for the shipped vLLM and mlx-lm models). Send independent chats concurrently, up to that many, not one after another. Past it a chat waits for a slot, or with `"queue": false` is refused 503 `chat_queue_full` at no cost. |
| `asr` (Qwen3-ASR) | One job is one file, cut into pieces server-side; send a long recording whole rather than pre-cut. On cuda-linux (vLLM) the pieces decode `max_batch` (8) at a time; on a Mac one at a time. Every job starts its own ASR engine and aligner (about 30 s on the PC before the first piece), so many short files each pay it. On a Mac `qwen3-asr-1.7b-mlx` decodes about 2.5x faster than `qwen3-asr-1.7b` and hears slightly fewer fillers. |
| `rvc` | One job takes many inputs: send a batch of files as one job, not one job per file. The server runs up to 96 pieces through each conversion process, so the voice loads once per batch rather than once per file. |
| `denoise` (separators) | One input per job, and the separator stays loaded between jobs, so separate jobs cost no reload. There is nothing to gain from cutting a file into chunks: send it whole. The separator runs the model on one window at a time (11 s for `vocals-roformer`, 8 s for `denoise-roformer`), starting one every `hop_s` seconds (8 for both); time is about inversely proportional to the hop. It does not batch windows. |

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

Approve (`allow: true`) or deny a pairing request, naming its `id` and the `user_code` the asking client shows.

*Door:* token + `X-Crucible-Api: 1`

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `id` | string | yes | — |  |
| `user_code` | string | yes | — |  |
| `allow` | boolean | yes | — |  |

*Answers:* `200`, `422` HTTPValidationError

### `POST /v1/pairing/poll`

How a pairing request stands: `pending`, `approved` (the answer then carries the server's `name` and its `token`), `denied` or `expired`. Poll with the `id` and `device_code` from `POST /v1/pairing/start`.

*Door:* `X-Crucible-Api: 1` only

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `id` | string | yes | — |  |
| `device_code` | string | yes | — |  |

*Answers:* `200`, `422` HTTPValidationError

### `GET /v1/pairing/requests`

The pairing requests waiting for an answer: who asked, from where, and the `user_code` to compare.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `POST /v1/pairing/start`

Ask this server for a token. Answers an `id`, a `device_code` to poll with and a short `user_code` the operator compares before approving it (on the operator page, or `POST /v1/pairing/decision`). No token is needed to ask; one is needed to approve.

*Door:* `X-Crucible-Api: 1` only

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `client_name` | string | yes | — |  |

*Answers:* `200`, `422` HTTPValidationError

## Setup

The operator page's own door: it hands out a token and the pairing line.

### `GET /v1/setup`

Everything an app needs to be pointed at this server in one read, including its token and pairing lines, and `network`: whether other devices can reach it (`reachable`, `urls`), said as a `sentence`, and when they cannot, `how` to open it, the one `command` that does (when one exists) and what that `changes`.

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
| `features` | array of string | yes | — | What this server's API offers, by name (crucible/features.py; the list is in docs/API.md under Features). Check for a name rather than comparing versions. It says the routes exist in this build, not that a job type is enabled here: `job_types` says that. |
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

What this server can hold, per capability class, and why not; `enabled: false` is an answer, not an error. A row's `goal` is the size its automatic pick aims at and never exceeds (`params_b`, and the ruling it comes from), or null for a class that has none. A client-sized class may be sized with `?class=&context_tokens=&concurrency=`.

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

### `PUT /v1/settings/audio/low-vram`

Set `[audio] low_vram` with `{"state": "on" \| "off" \| "auto"}`: `on` and `off` are the operator's and Crucible never changes them; `auto` lets Crucible turn it on exactly where this card cannot hold a splittable audio model whole. Decides the audio capability and `[jobs] enable_audio` again, as `crucible audio low-vram` does. Answers the full settings document after the write.

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

## Jobs

The work. Every job type is created, polled and cancelled through the same routes. The `image` job's params, its result and how to prompt it are in docs/IMAGE.md; the `audio` job's (sound effects, music and songs) are in docs/AUDIO.md; the `segment` job's (subject cutouts and point-and-box selections, as masks) are in docs/SEGMENT.md; the `video` job's (clips with sound from words or a start picture) are in docs/VIDEO.md.

### `POST /v1/jobs`

Admit one job, or refuse by name. A busy lane queues the job: it waits with status `queued` (up to an hour, or `queue.max_wait_s`) and its events say where it stands. With `"queue": false` a busy lane is refused `409 server_busy` instead. A missing environment or model is installed while the job is refused `409 installing`. `params.resume` set to a `resume_id` continues a journaled job; without it the job starts fresh. An item of the open queue session (named in the session header, or any submit from the client holding it) goes ahead of everything waiting, waits only behind the session's own jobs, and waits up to a day unless its `queue` says otherwise.

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
| `queue` | QueueRequest or `False` | no | — | Left out, a busy lane queues the job: it waits (status `queued`) up to an hour, or up to a day as an item of the open queue session. `{"max_wait_s": N}` changes the wait; `false` refuses at once with `409 server_busy` instead of waiting. |

*Answers:* `202`, `409` ServerBusy, `422` HTTPValidationError

### `GET /v1/jobs/{job_id}`

One job's state: `status`, `progress`, `message`, its artifacts and, when it ended, its result or its error. Following `/events` is better than polling this.

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

One artifact's bytes, by the name the `done` event lists, with its media type. Every artifact has a `<name>.provenance.json` beside it saying what made it.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |
| `name` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

### `GET /v1/jobs/{job_id}/events`

The job's events as SSE, from the first: `queued`, `started`, `warming`, `progress`, `note`, then one of `done` (with its artifacts and the job type's result fields), `failed` (with `error`), `cancelled` or `removed`, after which the stream ends. Send `Last-Event-ID` to resume after the last event you saw (docs/EVENTS.md).

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

Jobs, calls and queue sessions that find the server busy wait here (waiting is the default; `"queue": false` refuses instead), in order: list them, remove one, keep one alive, or follow every change. How an app should use it is docs/QUEUE.md.

### `GET /v1/queue`

What waits for the lane, in the order it will be offered it: the open queue session's items first, then first come, first served. A queue session waiting to open is a row of kind `session`.

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

Take a waiting job, call or queue session out of the queue (reason `operator`), or end the open queue session; a job that has started is cancelled with DELETE /v1/jobs/{id}.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | path | yes | string |  |

*Answers:* `200` QueueRemoved, `404` ErrorEnvelope, `422` HTTPValidationError

**Answer `200`** (`application/json`), the body:

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `job_id` | string | yes | — |  |
| `status` | `'removed'` or `'closed'` | yes | — |  |
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

One task's state: type, status, progress, and its error when it failed.

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

The task's events as SSE, ending with `done`, `failed` or `cancelled`; the same shape and resume rules as a job's.

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
| `session` | SessionState or null | yes | — |  |
| `slots` | ActivitySlots | yes | — | Every lane this server admits work through. |
| `updating` | object or null | no | — |  |
| `running` | array of ActivityJob | yes | — |  |
| `queued` | array of ActivityJob | yes | — |  |
| `accelerator` | object or null | no | — |  |

## Events

Every change on the server as one SSE stream, so an app follows it instead of polling /v1/activity, /v1/tasks and /v1/health. The event names and payloads, resuming, and what a slow reader is told are in docs/EVENTS.md.

### `GET /v1/events`

Every change on this server as one SSE stream, so an app need not poll: a `snapshot` first, then one event per change (jobs, the queue, queue sessions, the card, chats in flight, tasks, settings, the server stopping). Reconnect with Last-Event-ID to resume; the event names and payloads are in docs/EVENTS.md.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `topics` | query | no | string or null | Comma-separated topics to receive (job, queue, session, card, chat, task, settings, server). Leave it out for all; `server` is always sent. |

*Answers:* `200`, `422` HTTPValidationError

## Health

Is this process alive. Cheaper than /v1/activity and says less.

### `GET /v1/health`

Is this process alive, in one cheap read: `status` (`ok`, `busy` running a job, `warming` loading a model), the queue depth and what is resident.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## Playground

The pages the operator page's playground draws: one per image, video and audio model, with the params its form shows and whether this server can run it now.

### `GET /v1/playground`

One page per image, video and audio model this build declares: the params its form shows, with defaults and limits from its manifest, and its `standing`: `ready`; `download`, which its first job fetches by itself (409 `installing` names the task; `download_bytes` when the size is known); or `unavailable`, with `reason` saying why.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `GET /v1/playground/presets/{model}`

The presets saved for `model` on this server, by name: each a form's params (no seed) and when it was saved.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `model` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

### `PUT /v1/playground/presets/{model}/{name}`

Save (or replace) the preset `name` for `model`: `{"params": {...}}` with the form's own fields - text, numbers and true/false; a seed is refused by name.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `model` | path | yes | string |  |
| `name` | path | yes | string |  |

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `params` | object | yes | — |  |

*Answers:* `200`, `422` HTTPValidationError

### `DELETE /v1/playground/presets/{model}/{name}`

Remove the preset `name` of `model`; 404 when there is none.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `model` | path | yes | string |  |
| `name` | path | yes | string |  |

*Answers:* `200`, `422` HTTPValidationError

## Catalog

Everything this build can serve, of every kind, and what is on disk. Removing a subject here is how weights are reclaimed.

### `GET /v1/catalog`

Every subject this backend can hold, installed or not.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `DELETE /v1/catalog/{kind}/{subject_id}`

Delete an installed subject's files. Refused while the subject is resident or named by a running task.

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
| `serving` | object or null | yes | — | The server narrator starts for this voice (`[voice.serving]`, Higgs v3 only): `max_num_seqs`, `mem_fraction`, `context_length` with their notes, and `stall_guard`, the EFFECTIVE runaway-silence guard: `{enabled, frames, rate, max, window, env, basis, note}`, where `env` is the exact `HIGGS_STALL_GUARD` value narrator is started with (`off` when disabled) and `basis` is `default` or `manifest`. |
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

Open the one streaming session this server will hold at a time, inside a queue session: the client's own (the session header, or the open one it holds), else one opened for the stream, which waits in the line like any session and closes with the stream. Answers once that session is open and the voice is resident (loaded in the session when it is not). Sent with the queue-ticket header set to `1`, an open whose session has to wait answers `202 {queue_session_id, status, position}` at once instead: follow `GET /v1/queue/sessions/{id}/events` and, after `opened`, open again with the session header naming it, which claims that session for the stream (it closes with it). Unclaimed 60 s after opening, it closes. docs/QUEUE.md has the wire.

*Door:* token + `X-Crucible-Api: 1`

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `voice` | string | yes | — |  |
| `language` | string | yes | — |  |
| `idle_s` | integer | no | `900` | When the client holds no queue session, the stream opens one for itself with this idle_s: no row being said, no op and no touch for this long closes the session and the stream with it. Ignored inside the client's own session. |
| `queue` | QueueRequest or `False` | no | — | Left out, the stream's queue session waits in the line to open, up to an hour; `{"max_wait_s": N}` changes the wait. `false`: refuse (`session_open`, `server_busy`) rather than wait when the server is not free now. |

*Answers:* `201`, `422` HTTPValidationError

### `POST /v1/tts/stream/{session_id}`

One op: `say`, `cancel`, `cancel_all` or `close`. `say` answers with the row id; the audio arrives on the stream. Every op is activity of the queue session the stream runs in.

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

The same as `{"op": "close"}`, for a client that only has verbs. A queue session opened for the stream closes with it; one the client opened itself stays open.

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

Store one file (multipart field `file`) for a later job. Answers a `blob_id`, its `bytes` and `sha256`; a job input then names it as `{"blob_id": ...}`. The way to send anything too big for a JSON body.

*Door:* token + `X-Crucible-Api: 1`

**Body** (`multipart/form-data`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `file` | string | yes | — |  |

*Answers:* `201`, `422` HTTPValidationError

## Queue sessions

One client holding the server for a run of requests it cannot know in advance: it waits in the line, opens, runs its items back to back with nothing from anyone else in between, and closes. Not a TTS stream session. docs/QUEUE.md says how an app uses one.

### `POST /v1/queue/sessions`

Ask for the server for a run of requests you cannot know in advance. It waits in the line like any queued item and answers at once with a ticket; follow `GET /v1/queue/sessions/{id}/events` for `opened`. While it is open, nothing from any other client runs.

*Door:* token + `X-Crucible-Api: 1`

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `act` | string | yes | — | The capability class the run is for, as the act header names it. |
| `model` | string or null | no | — | A model to have resident when the session opens; it is loaded for the session (a load-model job attributed to it) when it is not. |
| `idle_s` | integer | no | `300` | Close the session after this long with nothing in flight, no item and no touch. A running job or an answer in flight always counts as activity. |
| `max_wait_s` | integer | no | `3600` | How long it may wait in the line to open before it is removed `expired`. |

*Answers:* `202` SessionTicket, `422` HTTPValidationError

**Answer `202`** (`application/json`), the body:

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | string | yes | — |  |
| `status` | `'queued'` or `'open'` or `'closed'` | yes | — |  |
| `position` | integer or null | yes | — |  |

### `GET /v1/queue/sessions/{session_id}`

Where the session stands: queued (and where), open (what it has run and has in flight, when it would go idle), or closed and why.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | path | yes | string |  |

*Answers:* `200` SessionState, `404` ErrorEnvelope, `422` HTTPValidationError

**Answer `200`** (`application/json`), the body:

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | string | yes | — |  |
| `status` | `'queued'` or `'open'` or `'closed'` | yes | — |  |
| `act` | string | yes | — |  |
| `client` | string or null | yes | — |  |
| `model` | string or null | yes | — |  |
| `position` | integer or null | yes | — |  |
| `idle_s` | integer | yes | — |  |
| `max_wait_s` | integer | yes | — |  |
| `created` | string | yes | — |  |
| `opened_at` | string or null | yes | — |  |
| `idle_deadline` | string or null | yes | — |  |
| `max_hold_deadline` | string or null | yes | — |  |
| `items_run` | integer | yes | — |  |
| `in_flight` | array of object | yes | — |  |
| `stream_session` | object or null | yes | — |  |
| `load_job` | string or null | yes | — |  |
| `closed_at` | string or null | yes | — |  |
| `reason` | string or null | yes | — |  |
| `message` | string or null | yes | — |  |
| `error` | object or null | yes | — |  |

### `DELETE /v1/queue/sessions/{session_id}`

End your session (reason `client`): an open one closes and the card is settled before this answers; a queued one leaves the line. A session already closed answers as it is.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | path | yes | string |  |

*Answers:* `200` SessionState, `404` ErrorEnvelope, `422` HTTPValidationError

**Answer `200`** (`application/json`), the body:

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | string | yes | — |  |
| `status` | `'queued'` or `'open'` or `'closed'` | yes | — |  |
| `act` | string | yes | — |  |
| `client` | string or null | yes | — |  |
| `model` | string or null | yes | — |  |
| `position` | integer or null | yes | — |  |
| `idle_s` | integer | yes | — |  |
| `max_wait_s` | integer | yes | — |  |
| `created` | string | yes | — |  |
| `opened_at` | string or null | yes | — |  |
| `idle_deadline` | string or null | yes | — |  |
| `max_hold_deadline` | string or null | yes | — |  |
| `items_run` | integer | yes | — |  |
| `in_flight` | array of object | yes | — |  |
| `stream_session` | object or null | yes | — |  |
| `load_job` | string or null | yes | — |  |
| `closed_at` | string or null | yes | — |  |
| `reason` | string or null | yes | — |  |
| `message` | string or null | yes | — |  |
| `error` | object or null | yes | — |  |

### `GET /v1/queue/sessions/{session_id}/events`

The session's own stream: `queued {position, of}` and `moved`, then `opened`, then `closed {reason}`; `removed {reason}` in place of `closed` when it never opened.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | path | yes | string |  |

*Answers:* `200`, `404` ErrorEnvelope, `422` HTTPValidationError

### `POST /v1/queue/sessions/{session_id}/touch`

"Still here", for a long gap on the client's side with nothing in flight. Cheap: a timestamp in memory.

*Door:* token + `X-Crucible-Api: 1`

| parameter | in | required | type | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | path | yes | string |  |

*Answers:* `200`, `404` ErrorEnvelope, `422` HTTPValidationError

## OpenAI-compatible

A chat surface shaped like OpenAI's, for clients that already speak it.

### `POST /openai/v1/chat/completions`

An OpenAI chat completion, proxied to the resident engine or, for a `<upstream>/<id>` model, forwarded to that upstream. A chat whose model is not resident, or whose engine has every slot taken, waits in the server's line (up to an hour, or `queue.max_wait_s`) and its model is loaded for it; with `"queue": false` it is refused at once instead. An upstream chat never waits.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `GET /openai/v1/models`

The resident model in OpenAI's list shape, plus every upstream model a route names.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

### `POST /v1/openai/chat/completions`

An OpenAI chat completion, proxied to the resident engine or, for a `<upstream>/<id>` model, forwarded to that upstream. A chat whose model is not resident, or whose engine has every slot taken, waits in the server's line (up to an hour, or `queue.max_wait_s`) and its model is loaded for it; with `"queue": false` it is refused at once instead. An upstream chat never waits.

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

One answer distribution per question, read off the resident model's next-token logprobs; with `items`, one choice answer per item in one request. Every refusal a caller can cause is made before anything is decided. A decision whose model is not resident, or whose engine has every slot taken, waits in the server's line and its model is loaded for it; with `"queue": false` it is refused at once instead.

*Door:* token + `X-Crucible-Api: 1`

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `model` | string | yes | — | The Crucible model id. One that is not resident is loaded for the decision while it waits in the line (`409 model_not_resident` with `"queue": false`). An upstream id (`<upstream>/<id>`) is refused `400 decide_needs_logprobs`: no upstream returns a distribution. |
| `state` | State | yes | — | What the questions are about: a string, used verbatim, or any other JSON value, serialised as compact JSON. Required and never null; may be `""` only when `images` carry the state. |
| `questions` | object of ChoiceQuestion or ScoreQuestion or YesNoQuestion or null | no | — | Question name to question. Names are single path members (no `/`, `\`, leading dot) and key the answers. Answers come back in this order. Exactly one of `questions` and `items` is sent. |
| `instructions` | string or null | no | — | The items form's ask, written under every item's text: "Which of the categories listed above does the speaker do in this passage?". Each item's question is `text`, a newline, then this; absent, `text` alone. Refused with `questions`. |
| `options` | object of string or null | no | — | The items form's shared options (name to one-line description, letters A, B, C… in the order given, 2 to 26). An item without its own `options` uses these. |
| `items` | array of DecideItem or null | no | — | The items form: an ordered list of choice questions about ONE state, each answered exactly as a lone choice question would be (it sees the state and its own question, never another item), in one request: on the Mac the shared state runs once and every item continues from its cache. Answers come back as a list in this order. At most 512 (`too_many_items`); token caps in docs/internals/api.md. |
| `images` | array of string or null | no | — | Base64 image files (PNG, JPEG, GIF or WebP; standard alphabet, padded, no whitespace, no `data:` prefix), read as part of the state, after its text. At most 8 (`too_many_images`), and only on a model whose manifest declares `image` (`400 model_text_only` otherwise). `[]` is the same as none. |
| `missing` | `'refuse'` or `'report'` | no | `'refuse'` | What to do when a label is not among the top tokens the engine returned. `refuse` (the default): the decision is `502 label_not_in_probs` naming the question and the letter. `report`: the door never invents a number — that option's probability and log-probability are null, it is named in the answer's `missing_labels`, and the renormalisation, `confidence`, `score` and `label_mass` run over the letters actually returned. A question whose EVERY label is missing is refused in both modes: there is no answer to report. |
| `queue` | QueueRequest or `False` | no | — | Absent: while the model is not resident, or every slot on its engine is taken, the request is held open in the server's queue (docs/QUEUE.md) up to an hour, and the model is loaded for it when its turn comes. `{"max_wait_s": N}` changes the wait. `false`: refused at once instead (`409 model_not_resident`, `503 chat_queue_full`, `409 session_open`). |

*Answers:* `200`, `422` HTTPValidationError

### `GET /v1/docs`

Every command this server answers, as data: each route with its door and one line, and each job type with its model, params schema, inputs, returns, notes, an example body and whether it is enabled here. The page is `GET /docs`.

*Door:* open

*Answers:* `200`

### `GET /v1/docs.md`

The whole API reference as markdown: every route, every job type, every model.

*Door:* open

*Answers:* `200`

### `POST /v1/server/updating`

Stop admitting work so a deploy can restart this server, if nothing is working. The hold is taken first and the server read second, so nothing slips in between: idle, it answers `holding: true` and every door that creates work refuses `503 server_updating` until the restart, a `DELETE`, or `seconds` pass; working, it lets the hold go at once and refuses `409 server_working`, naming the work.

*Door:* token + `X-Crucible-Api: 1`

**Body** (`application/json`)

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `seconds` | integer | no | `600` | How long the hold stands if nothing restarts the server or lets it go. |
| `release` | string or null | no | — | The release about to be installed, for the refusals to name. |

*Answers:* `200`, `422` HTTPValidationError

### `DELETE /v1/server/updating`

Let go of an update hold (a deploy whose install failed): work is admitted again.

*Door:* token + `X-Crucible-Api: 1`

*Answers:* `200`

## Job types

Each is a `POST /v1/jobs` body: `type`, `model` when it takes one, `params` (validated against the table below; unknown keys are refused), and `inputs`, a map of file name to `{"blob_id"}` (from `POST /v1/uploads`), `{"inline_base64"}` or `{"artifact": {"job_id", "name"}}`. Follow `GET /v1/jobs/{id}/events` to the `done` event and download each artifact it names from `GET /v1/jobs/{id}/artifacts/{name}`.

### `echo`

Copies every input to an artifact of the same name after a delay. The smoke test: it proves submit, events and artifact download without a GPU.

*Model:* none

**Params**

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `delay_ms` | integer | no | `25` | Milliseconds to wait before copying the inputs to artifacts, 0 to 60000; a cancel is honoured during the wait. |

*Inputs:* Any number of files, any names and formats.

*Returns:* One artifact per input, byte for byte the same, under the input's name.

```json
{
  "type": "echo",
  "params": {
    "delay_ms": 25
  },
  "inputs": {
    "hello.txt": {
      "inline_base64": "aGVsbG8="
    }
  }
}
```

### `load-model`

Starts an LLM engine for a model and leaves it resident, so the chat door (POST /v1/openai/chat/completions) and POST /v1/decide can use it. Loading the resident model again at a new `context` is a reload.

*Model:* An LLM id (GET /v1/catalog, or GET /v1/models for the ones this server holds), e.g. `qwen3.5-9b`.

**Params**

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `timeout_s` | number | no | `900.0` | Seconds to wait for the engine to come up and answer before the load fails, 30 to 7200 (default 900). |
| `context` | integer or null | no | `null` | The context length in tokens to start the engine at, at least 2048; null uses the model's own default. Above this host's ceiling it is refused `context_over_limit`; loading the resident model at a new context is a reload. |

*Inputs:* None.

*Returns:* No artifacts. The `done` event gives `resident`, the model now on the card.

- There is no `llm` job type: chat and decide never load a model (`model_not_resident`), so load it here first.
- `context` (at least 2048) is refused above this host's ceiling (`context_over_limit`; GET /v1/capability's `generate` row lists it per model); without it the model's own default is used. A plan that leaves too little KV cache is refused 409 `insufficient_kv_cache`.
- A load that ends `done` leaves the subject resident and held by nothing: the next job to end, a queue session closing or the last chat returning takes it off unless something holds it. Load inside a queue session to keep it for the session.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.

```json
{
  "type": "load-model",
  "model": "qwen3.5-9b",
  "params": {
    "context": 65536
  }
}
```

### `unload-model`

Takes the resident model off the card now, whichever job put it there (load-model).

*Model:* The id of the model that is resident (GET /v1/activity's `resident`, or the resident rows of GET /v1/catalog).

*Params:* none (`{}`).

*Inputs:* None.

*Returns:* No artifacts. The `done` event gives `resident`: what is on the card afterwards, normally null.

- Refused 409 `model_not_resident` when that id is not the resident model; the refusal says what is resident instead.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.
- An unload of the subject the server is already clearing is admitted and ends `done`.

```json
{
  "type": "unload-model",
  "model": "qwen3.5-9b",
  "params": {}
}
```

### `load-voice`

Starts narrator with a TTS voice and leaves it resident, for `tts` jobs and serialized TTS streams. A `zeroshot` voice can only be loaded this way, because only this job carries its reference clip.

*Model:* A voice id (GET /v1/voices; each row gives its `kind` and `takes`), e.g. `deathstalker`.

**Params**

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `timeout_s` | number | no | `900.0` | Seconds to wait for narrator to come up with the voice before the load fails, 30 to 7200 (default 900). |
| `reference` | ReferenceInput or null | no | `null` | The clip a `zeroshot` voice is conditioned on, with its transcript; required for a zeroshot voice (`reference_required`) and refused for every other kind (`reference_not_allowed`). |

*Inputs:* None. A zeroshot voice's reference clip travels in `params.reference`, not as an input.

*Returns:* No artifacts. The `done` event gives `resident` (the voice id), `fingerprint`, and `reference` (the clip's name, sha256 and seconds, or null).

- A `zeroshot` voice needs `reference`: `{"data": "<base64 WAV, no data: prefix>", "transcript": "<the exact words spoken in it>"}`, at most 30 s and 32 MiB (`reference_required`, `reference_malformed`). Any other kind of voice refuses one (`reference_not_allowed`).
- A load that ends `done` leaves the subject resident and held by nothing: the next job to end, a queue session closing or the last chat returning takes it off unless something holds it. Load inside a queue session to keep it for the session.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.

```json
{
  "type": "load-voice",
  "model": "deathstalker",
  "params": {}
}
```

### `unload-voice`

Takes the resident voice off the card now, whichever job put it there (load-voice or tts).

*Model:* The id of the voice that is resident (GET /v1/activity's `resident`, or the resident rows of GET /v1/catalog).

*Params:* none (`{}`).

*Inputs:* None.

*Returns:* No artifacts. The `done` event gives `resident`: what is on the card afterwards, normally null.

- Refused 409 `voice_not_resident` when that id is not the resident voice; the refusal says what is resident instead.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.
- An unload of the subject the server is already clearing is admitted and ends `done`.

```json
{
  "type": "unload-voice",
  "model": "deathstalker",
  "params": {}
}
```

### `tts`

Renders a batch of text chunks with a TTS voice, one FLAC per chunk. It loads the voice itself when it is not resident.

*Model:* A voice id (GET /v1/voices), e.g. `deathstalker`. For Higgs a voice is the merged checkpoint, so the voice id is the model.

**Params**

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `language` | string | yes | — | The language narrator renders in, e.g. `en`; not blank. |
| `take` | integer | yes | — | The rung of the voice's retake ladder to render at (`takes` on GET /v1/voices); no default. A take past the ladder's end renders at the voice's own sampling in that take's seed lane. |
| `chunks` | array of TtsChunk | yes | — | The text to render, one FLAC per chunk; at least one, indexes unique. |
| `retake` | boolean | no | `False` | true renders through the guarded arm, which re-rolls a chunk whose pace falls outside `band`; it then needs `band` (`retake_without_band`). false renders bare. |
| `band` | object or null | no | `null` | The pace band in characters per second: `pace_chars_per_sec`, `min_chars_per_sec`, `max_chars_per_sec`, all three, positive, min < pace < max (`band_malformed`). Never read off the voice: the caller states it. |
| `width` | integer or null | no | `null` | How many chunks are in flight at once. Null sends none, and the engine renders at the width it was started at; on cuda-linux a width above the voice's serving width is refused `width_over_serving`. |

*Inputs:* None; the text is in `params.chunks`.

*Returns:* `<index>.flac` per chunk that rendered: mono FLAC at the voice's own rate. A `chunk` event per row gives `seconds`, `chars`, `chars_per_sec`, `tokens`, `capped`, `take`, `guard` and `pause_cuts`. The `done` event gives `rendered` (a count), `failed` (a list of `{index, message}`), `take`, `sample_rate`, `sampling` (the full triple applied), `voice` (`id`, `identity`, `identity_basis`) and `width` (the width you sent, or null).

- `take` has no default: it is a rung of the voice's retake ladder (`takes` on GET /v1/voices). Chunk indexes must be unique and text not blank.
- A chunk that fails is listed in `failed` and does not fail the job; re-submit only those indexes, at the next take if you like.
- `retake: true` needs `band` (`pace_chars_per_sec`, `min_chars_per_sec`, `max_chars_per_sec`, min < pace < max): `retake_without_band`, `band_malformed`.
- `width` (chunks in flight) above the voice's serving width is refused `width_over_serving` on cuda-linux; left out, the engine runs at the width it was started with.
- A `zeroshot` voice must be loaded with `load-voice` first (`voice_kind_unsupported`).
- Needs ffmpeg on the server (`ffmpeg_missing`).
- The model comes off the card when the job ends unless something holds it. To run several jobs without a reload between them, open a queue session first (POST /v1/queue/sessions, docs/QUEUE.md) and close it at the end.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.

```json
{
  "type": "tts",
  "model": "deathstalker",
  "params": {
    "language": "en",
    "take": 0,
    "chunks": [
      {
        "index": 41,
        "text": "He had been walking for some time."
      },
      {
        "index": 42,
        "text": "The road did not appear to end."
      }
    ]
  }
}
```

### `asr`

Transcribes one audio file to timed text with Whisper (faster-whisper on cuda-linux, mlx-whisper on a Mac) or Qwen3-ASR.

*Model:* An ASR id (GET /v1/catalog), e.g. `qwen3-asr-1.7b`, `whisper-large-v3-turbo`, or `qwen3-asr-1.7b-mlx` on a Mac.

**Params**

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `language` | string | yes | — | The spoken language as an ISO code, e.g. `en`. `auto` asks Whisper to detect it; Qwen3-ASR takes only en, de, fr, es, it, pt, ru, ja, ko, zh or yue. |
| `vad_filter` | boolean | yes | — | true runs faster-whisper's own voice-activity filter; refused on engines without one (mlx-whisper, Qwen3-ASR) and together with `speech_only: true`. |
| `word_timestamps` | boolean | yes | — | true adds `words` with their own start and end seconds to each segment; on Qwen3-ASR it runs the forced aligner after transcription. |
| `initial_prompt` | string or null | no | `null` | Whisper only: text the model is primed with as if it were the transcript so far (a title, the names in it); not blank. |
| `context` | string or null | no | `null` | Qwen3-ASR only: the instruction and vocabulary it reads in its system turn before every piece; at most 8192 characters, not blank. |
| `piece_s` | number or null | no | `null` | Qwen3-ASR only: the longest piece in seconds the audio is cut into, 5 to 180; null is 30. |
| `overlap_s` | number or null | no | `null` | Qwen3-ASR only: seconds of audio each piece also hears on each side, 0 to 5 and under half a piece; null is 0.4 with word timestamps, else 0. Above 0 needs `word_timestamps`. |
| `speech_only` | boolean or null | no | `null` | true takes stretches without speech out before transcribing and lists them in the transcript's `removed`, keeping the original timeline. Null follows `not vad_filter`, so it is on unless `vad_filter` is true. |
| `speech_threshold` | number or null | no | `null` | With `speech_only`: the detector score at which a frame counts as speech, 0.1 to 0.7 (lower keeps more); null is 0.3. |
| `speech_pad_s` | number or null | no | `null` | With `speech_only`: seconds of audio kept either side of speech, 0.1 to 2; null is 0.3. |
| `speech_min_gap_s` | number or null | no | `null` | With `speech_only`: the shortest stretch without speech that is taken out, in seconds, 1 to 60; null is 2. |
| `resume` | string or null | no | `null` | The `resume_id` of an earlier run of this same job, to read its finished pieces back instead of redoing them. Qwen3-ASR only (`resume_unsupported` on Whisper). |

*Inputs:* Exactly one audio file in any format ffmpeg decodes (m4b, mp3, FLAC, WAV, …); the server decodes it.

*Returns:* `transcript.json`: the model and revision, the language, `duration_s`, and `segments` (`start`, `end` in seconds, `text`, and `words` with their own times when `word_timestamps` is true); with `speech_only`, the stretches taken out are listed in `removed`. Qwen3-ASR with word timestamps first publishes `transcript.text.json` (each piece's text before alignment). A Qwen3-ASR job's `done` event gives `context_echo_pieces` and `decode_loop_pieces`.

- `language`, `vad_filter` and `word_timestamps` are required. Qwen3-ASR takes one of en, de, fr, es, it, pt, ru, ja, ko, zh, yue (`language_unsupported_by_engine`); `"auto"` is Whisper's only.
- Engine-specific knobs are refused on the other engine by name: `initial_prompt` is Whisper's, `context`, `piece_s` and `overlap_s` Qwen3-ASR's; `vad_filter: true` only on faster-whisper; `overlap_s` > 0 needs `word_timestamps`.
- `speech_only` (Crucible's speech detector) defaults to the opposite of `vad_filter`, so it is ON unless you send `vad_filter: true`; send `speech_only: false` to transcribe every stretch. Its knobs (`speech_threshold`, `speech_pad_s`, `speech_min_gap_s`) are refused without it.
- Qwen3-ASR keeps a resume journal: the submit answers `resume_id` beside `job_id`; after a failure or cancel, submit the same job with `params.resume` set to it (docs/RESUMABLE-JOBS.md). Whisper refuses `resume` (`resume_unsupported`).
- More than one input is `invalid_inputs`.
- Pieces decode 8 at a time on cuda-linux and one at a time on a Mac; each job loads its own engine. See Throughput.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.

```json
{
  "type": "asr",
  "model": "qwen3-asr-1.7b",
  "params": {
    "language": "en",
    "vad_filter": false,
    "word_timestamps": true
  },
  "inputs": {
    "chapter.m4b": {
      "blob_id": "<from POST /v1/uploads>"
    }
  }
}
```

### `align`

Forced alignment: places known text in time inside short audio windows with Qwen3-ForcedAligner, one window per chunk, all in one job.

*Model:* An aligner id (GET /v1/catalog): `qwen3-aligner`.

**Params**

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `language` | string | yes | — | The spoken language as an ISO code: en, de, fr, es, it, pt, ru, ja, ko, zh or yue. Anything else is refused; there is no fallback to English. |
| `chunks` | array of AlignChunk | yes | — | One per audio input, each window at most 300 s; at least one, indexes unique. |

*Inputs:* One audio file per chunk, named `<index>.<ext>` (e.g. `0.flac`), each at most 300 s, any format ffmpeg decodes. Every chunk needs a file and every file a chunk.

*Returns:* `alignment.json`: model, revision, language, and `chunks`, each `{index, items: [{text, start, end}]}` in seconds from that window's start, or `{index, error}`. Items are the model's own tokens, not your words. A `cue` event goes out as each chunk lands. The `done` event gives `chunks`, `failed` (the failed indexes) and `resident` (what is on the card as the job ends: the model while a session, a claim, a chat or a waiting call holds the card, otherwise null, because the server took it off first and said so in a `note`).

- `language` is one of en, de, fr, es, it, pt, ru, ja, ko, zh, yue; there is no fallback to English.
- A chunk that fails is reported alone in `failed`; the job still ends `done`.
- Cutting a long recording into windows is the caller's job; for a whole audiobook use `align-longform`.
- Needs ffmpeg on the server (`ffmpeg_missing`).
- The model comes off the card when the job ends unless something holds it. To run several jobs without a reload between them, open a queue session first (POST /v1/queue/sessions, docs/QUEUE.md) and close it at the end.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.

```json
{
  "type": "align",
  "model": "qwen3-aligner",
  "params": {
    "language": "en",
    "chunks": [
      {
        "index": 0,
        "text": "He had been walking for some time."
      }
    ]
  },
  "inputs": {
    "0.flac": {
      "blob_id": "<from POST /v1/uploads>"
    }
  }
}
```

### `unload-aligner`

Takes the resident aligner off the card now, whichever job put it there (align).

*Model:* The id of the aligner that is resident (GET /v1/activity's `resident`, or the resident rows of GET /v1/catalog).

*Params:* none (`{}`).

*Inputs:* None.

*Returns:* No artifacts. The `done` event gives `resident`: what is on the card afterwards, normally null.

- Refused 409 `aligner_not_resident` when that id is not the resident aligner; the refusal says what is resident instead.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.
- An unload of the subject the server is already clearing is admitted and ends `done`.

```json
{
  "type": "unload-aligner",
  "model": "qwen3-aligner",
  "params": {}
}
```

### `align-longform`

Aligns a whole audiobook to its book text: a rough faster-whisper transcript places each sentence, then Qwen3-ForcedAligner places the words, and out comes a WebVTT with one cue per sentence.

*Model:* The aligner id (GET /v1/catalog): `qwen3-aligner`. The rough pass's Whisper model is `params.rough_model`.

**Params**

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `language` | string | yes | — | The narration's language as an ISO code: en, de, fr, es, it, pt, ru, ja, ko, zh or yue; anything else is refused. |
| `sentences` | array of LongformSentence | yes | — | The whole book, one row per sentence, in reading order; not empty. |
| `rough_model` | string | no | `'small'` | The faster-whisper ASR id for the rough transcript that places each sentence, e.g. `whisper-large-v3-turbo`; it must be installed here. The default `small` names no shipped manifest (`unknown_rough_model`), so send one. |
| `chunk_s` | number | no | `240.0` | Seconds of audio per window the aligner places words in; above 0 and at most 300, the aligner's limit. |
| `hole_min_s` | number | no | `2.0` | Seconds, 0 or more. Accepted and checked, but not read by this build's stages. |
| `snap_silence_s` | number | no | `0.35` | Seconds, 0 or more. Accepted and checked, but not read by this build's stages. |
| `silence_source` | string | no | `'decoded'` | Only `decoded` (the audio this job was given) is accepted; any other value is refused. |

*Inputs:* Exactly one audio file, the whole audiobook (the m4b as it is). The book never crosses: its sentences are in `params.sentences`.

*Returns:* `alignment.vtt` (a cue per placed sentence; `kind: "heading"` cues carry a `NOTE heading`) and `align-report.json` (`sentences`, `placed`, `dropped`, `rate_tokens_per_second`, `chunks`, `capped`, `duration_s`, `rough_model`, `aligner`). Progress events name the stage: `transcribe`, `coarse-align`, `align`, `write`.

- Send `rough_model`: it names a faster-whisper ASR manifest installed on this server (e.g. `whisper-large-v3-turbo`, `whisper-tiny`). The default, `small`, names no manifest this build ships, so leaving it out is refused `unknown_rough_model`; `rough_model_not_installed` and `rough_model_not_on_this_backend` are the other refusals.
- Sentence indexes must be unique and in reading order, and text not blank.
- `chunk_s` (default 240) is at most 300, the aligner's window.
- `nothing_narrated` / `no_cues` mean no sentence could be placed: the wrong book or the wrong language.

```json
{
  "type": "align-longform",
  "model": "qwen3-aligner",
  "params": {
    "language": "en",
    "rough_model": "whisper-large-v3-turbo",
    "sentences": [
      {
        "index": 0,
        "text": "Chapter One",
        "kind": "heading"
      },
      {
        "index": 1,
        "text": "He had been walking for some time."
      }
    ]
  },
  "inputs": {
    "book.m4b": {
      "blob_id": "<from POST /v1/uploads>"
    }
  }
}
```

### `rvc`

Voice conversion: re-voices every input through an RVC model, keeping each input's container, sample format and exact duration.

*Model:* An RVC voice id (GET /v1/catalog), e.g. `sigma`, `deathstalker-rvc-v3`.

**Params**

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `index_rate` | number | yes | — | How much of the model's .index feature retrieval is blended in, 0 to 1; a model with no .index takes only 0 (`model_has_no_index`). |
| `protect_rate` | number | yes | — | Consonant and breath protection, 0 to 0.5, inverted: lower protects more and 0.5 turns it off. Only matters when retrieval runs. |
| `n_semitones` | integer | yes | — | Pitch shift in semitones, -24 to 24. |
| `f0_method` | string or null | no | `null` | The pitch extractor urvc uses, e.g. `rmvpe`; null leaves urvc's own default. |
| `hop_length` | integer or null | no | `null` | urvc's pitch-extraction hop length in samples, 1 to 512; null leaves urvc's own default. |
| `piece_s` | number or null | no | `null` | The longest piece in seconds a long input is cut into at quiet points, 10 to 600; null is 60. Memory is bounded by a piece. |
| `overlap_s` | number or null | no | `null` | Seconds of real audio converted on each side of a piece and then dropped, 0 to 5 and under half a piece; null is 0.5. |
| `crossfade_s` | number or null | no | `null` | The fade in seconds at each seam, 0 to 1 and at most twice `overlap_s`; null is 0.02. |
| `output_rate` | `'native'` or `'input'` | no | `'native'` | `native`: the higher of the input's rate and the rate urvc writes (48 kHz for the published models); `input`: the input's rate. |
| `output_channels` | `'input'` or `'mono'` | no | `'input'` | `input`: the input's channel count, the one converted voice in every channel; `mono`: one channel. |

*Inputs:* One or more audio files of any length (WAV, FLAC, OGG, MP3 or AIFF, read from the bytes, so names need no extension). Each is converted on its own.

*Returns:* One artifact per input, under the input's name, in its container and sample format (a WAV past 4 GiB comes back RF64), at `output_rate` (`native`: max of the input rate and the model's, 48 kHz for the published models; `input`: the input's) and `output_channels` (`input` or `mono`). The `done` event gives `files`, `pieces`, `model_name`, and `outputs`: per input its `frames`, `sample_rate`, `channels`, `format` and `subtype`.

- `index_rate`, `protect_rate` and `n_semitones` are required. `protect_rate` is inverted: lower protects more, 0.5 turns protection off.
- A model with no .index refuses `index_rate` above 0 (`model_has_no_index`).
- Long inputs are cut at quiet points into pieces of `piece_s` (default 60, 10 to 600) with `overlap_s` each side (default 0.5, under half a piece) and joined with a `crossfade_s` fade (default 0.02, at most twice the overlap), so a 12-hour master can be sent whole.
- A stereo input gets one converted voice in both channels; it is not a per-channel conversion.
- If any input produces no output the job fails `rvc_output_missing`, naming them.
- Needs ffmpeg and ffprobe on the server (`ffmpeg_missing`).
- Send many files as one job: up to 96 pieces share one conversion process and one load of the voice. See Throughput.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.

```json
{
  "type": "rvc",
  "model": "sigma",
  "params": {
    "index_rate": 0.3,
    "protect_rate": 0.1,
    "n_semitones": -2,
    "f0_method": "rmvpe"
  },
  "inputs": {
    "c000.flac": {
      "blob_id": "<from POST /v1/uploads>"
    }
  }
}
```

### `denoise`

Separates audio into stems with a source-separation model: `vocals-roformer` splits vocals from the instrumental, `denoise-roformer` splits dry speech from noise.

*Model:* A separator id: `vocals-roformer` or `denoise-roformer` (GET /v1/catalog lists them and whether each is installed).

**Params**

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `stems` | `'primary'` or `'all'` | no | `'primary'` | "primary" returns only the separator's answer (the vocals, the dry voice); "all" also returns what was separated from it (the instrumental, the noise), the primary first. |

*Inputs:* Exactly one audio file libsndfile reads (WAV, FLAC), at the separator's own rate, 44100 Hz. Convert a video's audio first, e.g. `ffmpeg -i in.mp4 -vn -ar 44100 -c:a pcm_s16le in.wav`.

*Returns:* WAV stems at 44100 Hz with the input's channel count and exactly its frame count, so they line up sample for sample. `stems: "primary"` (the default) returns only the primary stem (`…_(Vocals)_….wav` or `…_(Dry)_….wav`); `stems: "all"` returns every stem, the primary first. The `done` event names `primary_stem`, lists `stems`, and gives `sample_rate`, `frames`, `separate_seconds` and `load_seconds`.

- Nothing is resampled: another sample rate is refused by name.
- A model not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.
- `vocals` means every voice in the track, singing included.
- The separator stays loaded between jobs; send a file whole rather than in chunks. See Throughput.

```json
{
  "type": "denoise",
  "model": "vocals-roformer",
  "params": {
    "stems": "all"
  },
  "inputs": {
    "song.wav": {
      "blob_id": "<from POST /v1/uploads>"
    }
  }
}
```

### `unload-denoiser`

Takes the resident separator off the card now, whichever job put it there (denoise).

*Model:* The id of the separator that is resident (GET /v1/activity's `resident`, or the resident rows of GET /v1/catalog).

*Params:* none (`{}`).

*Inputs:* None.

*Returns:* No artifacts. The `done` event gives `resident`: what is on the card afterwards, normally null.

- Refused 409 `separator_not_resident` when that id is not the resident separator; the refusal says what is resident instead.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.
- An unload of the subject the server is already clearing is admitted and ends `done`.

```json
{
  "type": "unload-denoiser",
  "model": "vocals-roformer",
  "params": {}
}
```

### `image`

Makes one picture from a prompt (text-to-image), redraws an input picture (image-to-image), or regenerates the masked region of one (inpainting and outpainting).

*Model:* An image model id (GET /v1/catalog): `qwen-image-2.1`.

**Params**

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `prompt` | string | yes | — | The picture to make, described in full (docs/IMAGE.md); not blank. |
| `negative_prompt` | string or null | no | `null` | What to guide away from; only read, and only allowed, with `guidance` above 1.0. |
| `width` | integer | no | `1024` | Pixels, 256 to 2048, a multiple of 16 (32 on cuda-linux); width x height at most 1,048,576. |
| `height` | integer | no | `1024` | Pixels, 256 to 2048, a multiple of 16 (32 on cuda-linux); width x height at most 1,048,576. |
| `seed` | integer or null | no | `null` | 0 to 4294967295; null lets the server choose one and report it. The same seed and params give the same picture on the same backend. |
| `steps` | integer | no | `40` | Denoising steps, 1 to 100; fewer is faster and rougher. |
| `guidance` | number | no | `1.0` | 1.0 to 10.0. Above 1.0 runs true classifier-free guidance (two passes per step, twice the time) and needs `negative_prompt`. |
| `image_strength` | number or null | no | `null` | Image-to-image: how much of the one input image survives, between 0 and 1 (useful range 0.03 to 0.3). With `mask` it is optional and applies to the masked region only. |
| `mask` | string or null | no | `null` | Inpainting and outpainting: the name of the input that carries the mask (white, 128 and up, is regenerated). The job then carries exactly the image and the mask. |
| `mask_blur` | integer or null | no | `null` | Pixels inside the mask's edge over which the new picture fades into the kept one, 0 to 256; only with `mask`, where null is 8. |

*Inputs:* None for text-to-image. Exactly one PNG, JPEG or WebP with `image_strength`. With `mask`, exactly two: the image and the mask, `mask` naming the mask input; the mask is the image's exact size, white (128 and up) is regenerated.

*Returns:* `image.png`, and with a mask also `generated.png` (the model's picture before the paste-back). The `done` event gives `image`, every effective parameter (seed included) plus timings and memory, and `resident` (what is on the card as the job ends: the model while a session, a claim, a chat or a waiting call holds the card, otherwise null, because the server took it off first and said so in a `note`).

- Width and height are 256 to 2048 and multiples of 16 (32 on cuda-linux, `image_size_not_supported`), at most 1,048,576 pixels (`image_too_large`).
- `guidance` above 1.0 needs `negative_prompt`, and `negative_prompt` needs it; `mask_blur` needs `mask`.
- Inputs that do not match the mode are `invalid_inputs`; a mask of a different size is `mask_size_mismatch`, one with no white `mask_empty`.
- The seed reproduces a picture on the same backend only.
- The model comes off the card when the job ends unless something holds it. To run several jobs without a reload between them, open a queue session first (POST /v1/queue/sessions, docs/QUEUE.md) and close it at the end.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.

```json
{
  "type": "image",
  "model": "qwen-image-2.1",
  "params": {
    "prompt": "An ordinary documentary photograph of a kitchen table with one red apple on it, natural window light. No text.",
    "width": 1280,
    "height": 720,
    "seed": 1,
    "steps": 40
  }
}
```

### `unload-image`

Takes the resident generator off the card now, whichever job put it there (image or load-image).

*Model:* The id of the generator that is resident (GET /v1/activity's `resident`, or the resident rows of GET /v1/catalog).

*Params:* none (`{}`).

*Inputs:* None.

*Returns:* No artifacts. The `done` event gives `resident`: what is on the card afterwards, normally null.

- Refused 409 `generator_not_resident` when that id is not the resident generator; the refusal says what is resident instead.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.
- An unload of the subject the server is already clearing is admitted and ends `done`.

```json
{
  "type": "unload-image",
  "model": "qwen-image-2.1",
  "params": {}
}
```

### `load-image`

Puts an image model on the card and leaves it there, so the first job that uses it starts at once. Nothing is generated.

*Model:* An image model id (GET /v1/catalog): `qwen-image-2.1`.

*Params:* none (`{}`).

*Inputs:* None.

*Returns:* No artifacts. The `done` event gives `resident`, the image model now on the card.

- A load that ends `done` leaves the subject resident and held by nothing: the next job to end, a queue session closing or the last chat returning takes it off unless something holds it. Load inside a queue session to keep it for the session.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.

```json
{
  "type": "load-image",
  "model": "qwen-image-2.1",
  "params": {}
}
```

### `audio`

Makes sound from words: sound effects and instrumental music with Stable Audio 3, songs with sung vocals (or instrumentals) with YuE2.

*Model:* An audio model id (GET /v1/catalog): `stable-audio-3-small-sfx` (sound effects), `stable-audio-3-medium` (music), `yue2-3b` (songs).

**Params**

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `prompt` | string or null | no | `null` | Sound effects and music (Stable Audio): the sound described (docs/AUDIO.md); required there, refused by a song model. |
| `tags` | string or null | no | `null` | Songs (YuE2): the style as comma-separated genre, instruments, voice, language and tempo; required there, refused by Stable Audio. |
| `lyrics` | string or null | no | `null` | Songs (YuE2): sections tagged [Verse], [Chorus] and so on, separated by blank lines; required unless `instrumental`, where only section tags are allowed. |
| `negative_prompt` | string or null | no | `null` | Taken only by a model whose manifest lists it; the shipped models refuse it by name. |
| `duration_s` | number or null | no | `null` | Seconds of sound, for models that take it (Stable Audio: at most 120 sfx, 380 music); null is the model's default. A song's length follows its lyrics. |
| `seed` | integer or null | no | `null` | 0 to 4294967295; null lets the server choose one and report it. |
| `steps` | integer or null | no | `null` | Denoising steps, for models that take them (Stable Audio, up to its ceiling); null is the model's default. |
| `cfg` | number or null | no | `null` | Guidance toward the tags and lyrics, for models that take it (YuE2, up to its ceiling); above 1 runs the model twice per token. Null is the model's default. |
| `instrumental` | boolean or null | no | `null` | Songs (YuE2): true renders the planned melody on an instrument, so nothing is sung. Null is false. |
| `format` | `'flac'` or `'wav'` or `'mp3'` | no | `'flac'` | The artifact: `flac` (24-bit), `wav` (24-bit PCM) or `mp3` (192 kbps CBR). |

*Inputs:* None; an audio job reads no files (`invalid_inputs`).

*Returns:* `audio.flac` (24-bit, the default), `audio.wav` or `audio.mp3` (192 kbps), per `format`; a song adds `score.abc`, the score YuE2 writes first. The `done` event gives `audio`, every effective parameter (seed included) plus timings and memory, and `resident` (what is on the card as the job ends: the model while a session, a claim, a chat or a waiting call holds the card, otherwise null, because the server took it off first and said so in a `note`).

- Which params a model takes is its own: Stable Audio reads `prompt` and takes `duration_s` and `steps`; YuE2 reads `tags` and `lyrics` (sections like `[Verse]`, optional with `instrumental: true`) and takes `cfg`. Anything else is refused `audio_param_unsupported` with the list it does take; a missing one `audio_param_missing`.
- Past a model's ceiling: `audio_too_long` (120 s sfx, 380 s music), `audio_param_out_of_range`. A song's length follows its lyrics.
- A host with `[audio] low_vram = true` in its config holds only half of YuE2 on the card at a time; Crucible turns it on by itself on a card too small to hold YuE2 whole (an 8 GiB card), and `audio.low_vram` in the `done` event says which ran (docs/AUDIO.md).
- The Stable Audio repos are gated: until the licence is accepted on Hugging Face and the server has a token, `409 model_gated` says what to do (docs/AUDIO.md).
- The model comes off the card when the job ends unless something holds it. To run several jobs without a reload between them, open a queue session first (POST /v1/queue/sessions, docs/QUEUE.md) and close it at the end.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.

```json
{
  "type": "audio",
  "model": "stable-audio-3-small-sfx",
  "params": {
    "prompt": "TrackType: SFX. A heavy oak door creaks open slowly in a stone hallway, close mic, dry",
    "duration_s": 4,
    "seed": 1
  }
}
```

### `unload-audio`

Takes the resident audio generator off the card now, whichever job put it there (audio or load-audio).

*Model:* The id of the audio generator that is resident (GET /v1/activity's `resident`, or the resident rows of GET /v1/catalog).

*Params:* none (`{}`).

*Inputs:* None.

*Returns:* No artifacts. The `done` event gives `resident`: what is on the card afterwards, normally null.

- Refused 409 `audio_generator_not_resident` when that id is not the resident audio generator; the refusal says what is resident instead.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.
- An unload of the subject the server is already clearing is admitted and ends `done`.

```json
{
  "type": "unload-audio",
  "model": "stable-audio-3-medium",
  "params": {}
}
```

### `load-audio`

Puts an audio model on the card and leaves it there, so the first job that uses it starts at once. Nothing is generated.

*Model:* An audio model id (GET /v1/catalog), e.g. `stable-audio-3-small-sfx`.

*Params:* none (`{}`).

*Inputs:* None.

*Returns:* No artifacts. The `done` event gives `resident`, the audio model now on the card.

- A load that ends `done` leaves the subject resident and held by nothing: the next job to end, a queue session closing or the last chat returning takes it off unless something holds it. Load inside a queue session to keep it for the session.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.

```json
{
  "type": "load-audio",
  "model": "stable-audio-3-small-sfx",
  "params": {}
}
```

### `segment`

Makes a mask from a picture: the main subject by itself (`birefnet`, background removal), or the object under your points or inside your box (`sam2.1-hiera-large`).

*Model:* A segment model id (GET /v1/catalog): `birefnet` or `sam2.1-hiera-large`.

**Params**

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `points` | array of SegmentPoint or null | no | `null` | `sam2.1-hiera-large` only: 1 to 64 clicks in the input's pixels, at least one label 1 unless there is a box. `birefnet` refuses it. |
| `box` | array of number or null | no | `null` | `sam2.1-hiera-large` only: [x0, y0, x1, y1] in the input's pixels, top-left first, x1 > x0 and y1 > y0. `birefnet` refuses it. |

*Inputs:* Exactly one PNG, JPEG or WebP of at most 40,000,000 pixels.

*Returns:* `mask.png` (8-bit grey, 255 = selected; soft edges from birefnet, hard from SAM) and `cutout.png` (the input as RGBA with the mask as alpha), both at the input's size. The `done` event gives `segment` (effective parameters, `score`, `coverage`, timings, memory) and `resident` (what is on the card as the job ends: the model while a session, a claim, a chat or a waiting call holds the card, otherwise null, because the server took it off first and said so in a `note`).

- `birefnet` takes no params (`segment_param_unsupported`); `sam2.1-hiera-large` needs `points` (1 to 64, `label` 1 keeps, 0 leaves out), `box` `[x0, y0, x1, y1]`, or both (`segment_param_missing`).
- Coordinates are the input's stored pixels from the top-left; EXIF orientation is not applied. Outside the picture is `segment_prompt_outside_picture`.
- The model comes off the card when the job ends unless something holds it. To run several jobs without a reload between them, open a queue session first (POST /v1/queue/sessions, docs/QUEUE.md) and close it at the end.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.

```json
{
  "type": "segment",
  "model": "sam2.1-hiera-large",
  "params": {
    "points": [
      {
        "x": 412,
        "y": 300,
        "label": 1
      }
    ]
  },
  "inputs": {
    "photo.jpg": {
      "blob_id": "<from POST /v1/uploads>"
    }
  }
}
```

### `unload-segment`

Takes the resident segmenter off the card now, whichever job put it there (segment or load-segment).

*Model:* The id of the segmenter that is resident (GET /v1/activity's `resident`, or the resident rows of GET /v1/catalog).

*Params:* none (`{}`).

*Inputs:* None.

*Returns:* No artifacts. The `done` event gives `resident`: what is on the card afterwards, normally null.

- Refused 409 `segmenter_not_resident` when that id is not the resident segmenter; the refusal says what is resident instead.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.
- An unload of the subject the server is already clearing is admitted and ends `done`.

```json
{
  "type": "unload-segment",
  "model": "birefnet",
  "params": {}
}
```

### `load-segment`

Puts a segment model on the card and leaves it there, so the first job that uses it starts at once. Nothing is generated.

*Model:* A segment model id (GET /v1/catalog): `birefnet` or `sam2.1-hiera-large`.

*Params:* none (`{}`).

*Inputs:* None.

*Returns:* No artifacts. The `done` event gives `resident`, the segment model now on the card.

- A load that ends `done` leaves the subject resident and held by nothing: the next job to end, a queue session closing or the last chat returning takes it off unless something holds it. Load inside a queue session to keep it for the session.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.

```json
{
  "type": "load-segment",
  "model": "birefnet",
  "params": {}
}
```

### `video`

Makes a video clip with its own sound from a prompt (text-to-video), or brings a start picture to life (image-to-video), with LTX-2.5.

*Model:* A video model id (GET /v1/catalog): `ltx-2.5-distilled`.

**Params**

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `prompt` | string | yes | — | The shot, the motion, the light and the sound, in one paragraph (docs/VIDEO.md); not blank. |
| `negative_prompt` | string or null | no | `null` | Refused by name (`video_param_unsupported`): the distilled checkpoint runs without guidance and would never read it. |
| `width` | integer or null | no | `null` | Frame width in pixels, sent with `height` or not at all (null: the model's default, 1280x704). A multiple of 32 (64 on a Mac), sides 256 to 1280, at most 901,120 pixels. |
| `height` | integer or null | no | `null` | Frame height in pixels, sent with `width` or not at all; the same rules as `width`. |
| `duration_s` | number or null | no | `null` | Seconds of clip, rounded to the model's 8k+1 frame grid; null is the model's default (5). Not with `num_frames`. |
| `num_frames` | integer or null | no | `null` | The exact frame count instead of `duration_s`; must be 8k+1 (49, 97, 121, …). |
| `fps` | integer or null | no | `null` | Frames per second, 24 or 25 for ltx-2.5-distilled; null is the model's default (24). |
| `seed` | integer or null | no | `null` | 0 to 4294967295; null lets the server choose one and report it. A seed reproduces a clip on the machine that made it. |
| `steps` | integer or null | no | `null` | Must equal the model's fixed step count (8 for the distilled checkpoint) or be left out; any other value is refused. |
| `audio` | boolean | no | `True` | false makes a silent clip: the sound is generated with the picture but not decoded. |

*Inputs:* None for text-to-video, or exactly one PNG, JPEG or WebP: the first frame, cropped to the clip's shape and resized.

*Returns:* `video.mp4`: H.264 (yuv420p) with AAC stereo at 48 kHz, faststart. The `done` event gives `video`, every effective parameter (mode, size, `num_frames`, `fps`, `duration_s`, seed, …) plus timings and GPU-busy figures, and `resident` (what is on the card as the job ends: the model while a session, a claim, a chat or a waiting call holds the card, otherwise null, because the server took it off first and said so in a `note`).

- Send `width` and `height` together or neither (default 1280x704); `duration_s` or `num_frames`, not both. `num_frames` is 8k+1 (`video_frames_not_supported`).
- Every limit is refused by name before anything loads: `video_size_not_supported` (multiples of 32, 64 on a Mac; sides 256 to 1280), `video_too_large`, `video_too_long`, and `video_param_unsupported` (`negative_prompt`, `steps` other than 8, `fps` other than 24 or 25). docs/VIDEO.md has the per-backend figures.
- The model repo is gated: `409 model_gated` says how to accept the licence.
- The model comes off the card when the job ends unless something holds it. To run several jobs without a reload between them, open a queue session first (POST /v1/queue/sessions, docs/QUEUE.md) and close it at the end.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.

```json
{
  "type": "video",
  "model": "ltx-2.5-distilled",
  "params": {
    "prompt": "A red fox trots through fresh snow at dawn, the camera tracking alongside at knee height; each step crunches.",
    "width": 1280,
    "height": 704,
    "duration_s": 5
  }
}
```

### `unload-video`

Takes the resident video generator off the card now, whichever job put it there (video or load-video).

*Model:* The id of the video generator that is resident (GET /v1/activity's `resident`, or the resident rows of GET /v1/catalog).

*Params:* none (`{}`).

*Inputs:* None.

*Returns:* No artifacts. The `done` event gives `resident`: what is on the card afterwards, normally null.

- Refused 409 `video_generator_not_resident` when that id is not the resident video generator; the refusal says what is resident instead.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.
- An unload of the subject the server is already clearing is admitted and ends `done`.

```json
{
  "type": "unload-video",
  "model": "ltx-2.5-distilled",
  "params": {}
}
```

### `load-video`

Puts a video model on the card and leaves it there, so the first job that uses it starts at once. Nothing is generated.

*Model:* A video model id (GET /v1/catalog): `ltx-2.5-distilled`.

*Params:* none (`{}`).

*Inputs:* None.

*Returns:* No artifacts. The `done` event gives `resident`, the video model now on the card.

- A load that ends `done` leaves the subject resident and held by nothing: the next job to end, a queue session closing or the last chat returning takes it off unless something holds it. Load inside a queue session to keep it for the session.
- A model or env not installed yet is installed on submit where the server allows it: the submit answers 409 `installing`; submit again when it is done.
- Refused 409 `engine_in_use` while something holds the card (a TTS stream session); the refusal names the holder.

```json
{
  "type": "load-video",
  "model": "ltx-2.5-distilled",
  "params": {}
}
```

## Features

`GET /v1/info` answers `features`, the names below, so an app checks for what it needs instead of comparing versions. A name says the routes and fields exist in this build; whether a job type is enabled on this host is `job_types`. Defined in `crucible/features.py`.

| feature | what it is |
| --- | --- |
| `accelerator` | GET /v1/accelerator: what is on the card and which holders are Crucible's own. |
| `activity` | GET /v1/activity: what the server is doing, in one read. |
| `align` | The `align` job: words placed on audio, chunk by chunk. |
| `align.longform` | The `align-longform` job: a whole book's text aligned to its audio, with cues. |
| `asr` | The `asr` job: speech to text. |
| `audio` | The `audio` job: sound effects, music and songs (docs/AUDIO.md). |
| `catalog` | GET /v1/catalog and DELETE /v1/catalog/{kind}/{id}: what this build can serve, what is on disk, and reclaiming it. |
| `chat` | POST /v1/openai/chat/completions: chat on the resident model, OpenAI-shaped. |
| `decide` | POST /v1/decide with `questions`: answer distributions read off the resident model's next-token logprobs. |
| `decide.items` | POST /v1/decide with `items`: one choice answer per item, in one request. |
| `denoise` | The `denoise` job: speech separated from what is behind it. |
| `events` | GET /v1/events: one SSE stream of every change on the server, opening with a snapshot and resumable with Last-Event-ID (docs/EVENTS.md). |
| `events.topics` | GET /v1/events?topics=job,queue,...: only the named topics. |
| `image` | The `image` job: pictures from words (docs/IMAGE.md). |
| `image.inpaint` | `mask` on the `image` job: only the white region of a start picture is regenerated (docs/IMAGE.md). |
| `jobs.events` | GET /v1/jobs/{id}/events: one job's own SSE stream, resumable with Last-Event-ID. |
| `jobs.hold` | `hold` on a submit and /v1/jobs/{id}/hold: a job's artifacts are kept until the client lets go of them. |
| `jobs.resume` | `params.resume` and /v1/resumable: a resumable job continues the journal an earlier run left (docs/RESUMABLE-JOBS.md). |
| `playground` | GET /v1/playground: the pages the operator page's playground draws. |
| `queue.calls` | A chat or a decision is held open in the same line until the resident model has a slot (docs/QUEUE.md). |
| `queue.default` | Every request that can wait (a job, a chat, a decision, a TTS stream) waits in the line by default; `"queue": false` refuses at once instead, and `{"max_wait_s": N}` sets the wait. `"queue": {}` is refused (docs/QUEUE.md). |
| `queue.events` | GET /v1/queue/events: the waiting line's own SSE stream. |
| `queue.jobs` | POST /v1/jobs: a busy server queues the job instead of refusing it; GET/DELETE /v1/queue and its heartbeat (docs/QUEUE.md). |
| `queue.sessions` | /v1/queue/sessions: an app's session holds the machine for a run of requests, waits its turn in the line, and ends on close or idle (docs/QUEUE.md). |
| `rvc` | The `rvc` job: voice conversion. |
| `segment` | The `segment` job: subject cutouts and point-and-box selections, as masks (docs/SEGMENT.md). |
| `settings` | GET/PUT /v1/settings: the one door apps configure Crucible through. |
| `tasks` | POST /v1/tasks: operator tasks (pull, install, module, engine, engine-restart) with their own SSE stream. |
| `tts` | The `tts` job: a render of text to audio with a narration voice. |
| `tts.stream` | /v1/tts/stream: a long-lived narration session, text in and audio back over SSE. |
| `uploads` | POST /v1/uploads: bytes too big for a request body, named by `blob_id`. |
| `video` | The `video` job: clips with sound from words or a start picture (docs/VIDEO.md). |
| `voices` | /v1/voices: the narration voices, their manifests and updates. |

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
| `session` | SessionState or null | yes | — |  |
| `slots` | ActivitySlots | yes | — | Every lane this server admits work through. |
| `updating` | object or null | no | — |  |
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
| `cancelling` | boolean or null | no | — |  |
| `waited_s` | integer or number or null | no | — |  |
| `max_wait_s` | integer or null | no | — |  |
| `waiting_for` | QueueWaitingFor or null | no | — |  |

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

### `Body_save_playground_preset_v1_playground_presets__model___name__put`

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `params` | object | yes | — |  |

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
| `model` | string | yes | — | The Crucible model id. One that is not resident is loaded for the decision while it waits in the line (`409 model_not_resident` with `"queue": false`). An upstream id (`<upstream>/<id>`) is refused `400 decide_needs_logprobs`: no upstream returns a distribution. |
| `state` | State | yes | — | What the questions are about: a string, used verbatim, or any other JSON value, serialised as compact JSON. Required and never null; may be `""` only when `images` carry the state. |
| `questions` | object of ChoiceQuestion or ScoreQuestion or YesNoQuestion or null | no | — | Question name to question. Names are single path members (no `/`, `\`, leading dot) and key the answers. Answers come back in this order. Exactly one of `questions` and `items` is sent. |
| `instructions` | string or null | no | — | The items form's ask, written under every item's text: "Which of the categories listed above does the speaker do in this passage?". Each item's question is `text`, a newline, then this; absent, `text` alone. Refused with `questions`. |
| `options` | object of string or null | no | — | The items form's shared options (name to one-line description, letters A, B, C… in the order given, 2 to 26). An item without its own `options` uses these. |
| `items` | array of DecideItem or null | no | — | The items form: an ordered list of choice questions about ONE state, each answered exactly as a lone choice question would be (it sees the state and its own question, never another item), in one request: on the Mac the shared state runs once and every item continues from its cache. Answers come back as a list in this order. At most 512 (`too_many_items`); token caps in docs/internals/api.md. |
| `images` | array of string or null | no | — | Base64 image files (PNG, JPEG, GIF or WebP; standard alphabet, padded, no whitespace, no `data:` prefix), read as part of the state, after its text. At most 8 (`too_many_images`), and only on a model whose manifest declares `image` (`400 model_text_only` otherwise). `[]` is the same as none. |
| `missing` | `'refuse'` or `'report'` | no | `'refuse'` | What to do when a label is not among the top tokens the engine returned. `refuse` (the default): the decision is `502 label_not_in_probs` naming the question and the letter. `report`: the door never invents a number — that option's probability and log-probability are null, it is named in the answer's `missing_labels`, and the renormalisation, `confidence`, `score` and `label_mass` run over the letters actually returned. A question whose EVERY label is missing is refused in both modes: there is no answer to report. |
| `queue` | QueueRequest or `False` | no | — | Absent: while the model is not resident, or every slot on its engine is taken, the request is held open in the server's queue (docs/QUEUE.md) up to an hour, and the model is loaded for it when its turn comes. `{"max_wait_s": N}` changes the wait. `false`: refused at once instead (`409 model_not_resident`, `503 chat_queue_full`, `409 session_open`). |

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
| `prime` | ForwardTiming or null | yes | — | The shared prefix sent alone first — present when the decision had more than one question, null when it had one, and null when the engine read every question in one batched request (mlx-lm: each question's timing is then that one request). |

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

### `HoldRequest`

`POST /v1/server/updating`.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `seconds` | integer | no | `600` | How long the hold stands if nothing restarts the server or lets it go. |
| `release` | string or null | no | — | The release about to be installed, for the refusals to name. |

### `Info`

`GET /v1/info`: who this server is, what it runs on and what it serves.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `server` | object | yes | — |  |
| `role` | string | yes | — |  |
| `managed_by` | object of string or null | yes | — |  |
| `host` | object | yes | — |  |
| `features` | array of string | yes | — | What this server's API offers, by name (crucible/features.py; the list is in docs/API.md under Features). Check for a name rather than comparing versions. It says the routes exist in this build, not that a job type is enabled here: `job_types` says that. |
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
| `queue` | QueueRequest or `False` | no | — | Left out, a busy lane queues the job: it waits (status `queued`) up to an hour, or up to a day as an item of the open queue session. `{"max_wait_s": N}` changes the wait; `false` refuses at once with `409 server_busy` instead of waiting. |

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
| `reason` | `'operator'` or `'client'` or `'expired'` or `'server_restart'` or `'session_closed'` | yes | — |  |
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
| `sampling` | object or null | no | — |  |
| `removal` | JobRemoval or null | no | — |  |

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
| `session` | string or null | yes | — |  |
| `kind` | `'job'` or `'call'` or `'session'` | yes | — |  |
| `waiting_for` | QueueWaitingFor or null | no | — |  |

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
| `status` | `'removed'` or `'closed'` | yes | — |  |
| `reason` | `'operator'` | yes | — |  |

### `QueueRequest`

How long this request may wait in the server's line.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `max_wait_s` | integer | yes | — | How long the request may wait for its turn before it is removed `expired`. |

### `QueueWaitingFor`

Why an item at the front is not offered the lane yet although it is free: memory on the accelerator is held by a process this Crucible does not own. `message` is the guard's sentence naming the holder (pid, name and bytes, or the unattributed bytes); the item is checked again at `next_check_at` and leaves the line `expired` when its `max_wait_s` runs out.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `code` | `'accelerator_busy'` | yes | — |  |
| `message` | string | yes | — |  |
| `details` | object or null | yes | — |  |
| `since` | string | yes | — |  |
| `next_check_at` | string | yes | — |  |

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
| `details` | JobBusyDetails or CardHeldDetails or SessionBusyDetails | yes | — |  |

### `ServiceCommand`

A command, typed on the server itself, that runs it as a machine service.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `command` | string | yes | — |  |
| `does` | string | yes | — |  |

### `SessionBusyDetails`

`server_busy` (or `session_open`) because a queue session holds the server: nothing but its own items runs until it closes.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `door` | `'session'` | yes | — |  |
| `holder` | string or null | yes | — |  |
| `session_id` | string | yes | — |  |
| `type` | `'session'` | yes | — |  |
| `act` | string | yes | — |  |
| `model` | string or null | yes | — |  |
| `status` | string | yes | — |  |
| `since` | string | yes | — |  |

### `SessionOpen`

`POST /v1/queue/sessions`: ask for the server for a run of requests. `act` has no default.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `act` | string | yes | — | The capability class the run is for, as the act header names it. |
| `model` | string or null | no | — | A model to have resident when the session opens; it is loaded for the session (a load-model job attributed to it) when it is not. |
| `idle_s` | integer | no | `300` | Close the session after this long with nothing in flight, no item and no touch. A running job or an answer in flight always counts as activity. |
| `max_wait_s` | integer | no | `3600` | How long it may wait in the line to open before it is removed `expired`. |

### `SessionState`

A queue session: one client's claim on the server for a run of requests. Not a TTS stream session.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | string | yes | — |  |
| `status` | `'queued'` or `'open'` or `'closed'` | yes | — |  |
| `act` | string | yes | — |  |
| `client` | string or null | yes | — |  |
| `model` | string or null | yes | — |  |
| `position` | integer or null | yes | — |  |
| `idle_s` | integer | yes | — |  |
| `max_wait_s` | integer | yes | — |  |
| `created` | string | yes | — |  |
| `opened_at` | string or null | yes | — |  |
| `idle_deadline` | string or null | yes | — |  |
| `max_hold_deadline` | string or null | yes | — |  |
| `items_run` | integer | yes | — |  |
| `in_flight` | array of object | yes | — |  |
| `stream_session` | object or null | yes | — |  |
| `load_job` | string or null | yes | — |  |
| `closed_at` | string or null | yes | — |  |
| `reason` | string or null | yes | — |  |
| `message` | string or null | yes | — |  |
| `error` | object or null | yes | — |  |

### `SessionTicket`

`POST /v1/queue/sessions`: the session asked for, and where it stands.

| field | type | required | default | what it is |
| --- | --- | --- | --- | --- |
| `session_id` | string | yes | — |  |
| `status` | `'queued'` or `'open'` or `'closed'` | yes | — |  |
| `position` | integer or null | yes | — |  |

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
| `idle_s` | integer | no | `900` | When the client holds no queue session, the stream opens one for itself with this idle_s: no row being said, no op and no touch for this long closes the session and the stream with it. Ignored inside the client's own session. |
| `queue` | QueueRequest or `False` | no | — | Left out, the stream's queue session waits in the line to open, up to an hour; `{"max_wait_s": N}` changes the wait. `false`: refuse (`session_open`, `server_busy`) rather than wait when the server is not free now. |

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
| `serving` | object or null | yes | — | The server narrator starts for this voice (`[voice.serving]`, Higgs v3 only): `max_num_seqs`, `mem_fraction`, `context_length` with their notes, and `stall_guard`, the EFFECTIVE runaway-silence guard: `{enabled, frames, rate, max, window, env, basis, note}`, where `env` is the exact `HIGGS_STALL_GUARD` value narrator is started with (`off` when disabled) and `basis` is `default` or `manifest`. |
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
