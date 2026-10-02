# The queue: for app authors

Crucible runs one job at a time. A job sent while another runs used to be refused
`409 server_busy`, and every app wrote its own retry loop. Then an app could ask to wait.
Now **waiting is the default**: every request that can wait (a job, a chat, a decision, a
TTS stream) takes a place in the server's first-come, first-served line when the server
is busy, and runs when its turn comes. An app that would rather be told "busy" at once
says so with `"queue": false`.

The TypeScript SDK waits by default too, and its `session()` is the queue session below
(see "In the SDK").

## The `queue` member

Every request that can wait takes the same optional member:

| `queue` | means |
|---|---|
| left out | wait in the line while the server is busy, up to an hour (a day for a job that is an item of the open queue session) |
| `{"max_wait_s": N}` | wait, up to N seconds (10 to 86400) |
| `false` | do not wait: refuse at once (`409 server_busy`, `409 session_open`, `409 model_not_resident`, `503 chat_queue_full`) |

`{}` (the old opt-in, which every SDK before 1.0.78 sends) also waits with the default:
it can only mean "wait". `true`, `null` and an out-of-range `max_wait_s` are
`400 invalid_request` with a sentence naming the shapes.

`max_wait_s` is how long the request may wait for its turn; when it runs out the request
is removed `expired` (below). It is not a deadline on the work itself: once the job is on
the lane, or the chat has its slot, it runs as long as it runs. A held-open chat or
decision sends nothing before its answer, so give the HTTP request a read timeout that
covers `max_wait_s` plus the answer.

## Submit a job

```json
POST /v1/jobs
{"type": "tts", "model": "sigma", "params": {...}, "inputs": {...}}
```

The answer is `202` either way:

```json
{"job_id": "5f0c…", "resume_id": null, "queued": true, "position": 3}
```

`queued: false` means the server was free and the job is already on the lane. `queued:
true` means it is waiting, `position` places away from the front (1 is next). A queued job
is a normal job: its status is `queued`, it holds its inputs (an upload it names is moved
into it at once), and `GET /v1/jobs/{id}` and `DELETE /v1/jobs/{id}` work on it.

A submit can still be refused straight away, by name, for anything that is not "busy": an
unknown job type or model, bad inputs, a model that is not installed (`409 installing`,
as before), and two queue limits: `409 queue_full` when the client already has 50 jobs
waiting (`details.scope: "client"`) or the server has 200 (`details.scope: "server"`).

## Follow the job's events

`GET /v1/jobs/{id}/events`, the same stream as always, carries three events for a job
that may wait (every job not sent with `"queue": false`):

| event | data | means |
|---|---|---|
| `queued` | `{position, of}` | where it stands; sent when it joins and again whenever it moves |
| `waiting` | `{code, message, details, since}` | it is at the front and the lane is free, but the card is held by a process Crucible does not own (below); sent when that starts and whenever who holds it changes |
| `started` | `{waited_s}` | it reached the lane; `progress`, `done` and `failed` follow as usual |
| `removed` | `{reason, message, waited_s, at}` | it left the queue without running; **terminal** |

`removed` has four reasons:

- `operator`: someone removed it (the desktop app's Queue section, or `DELETE /v1/queue/{id}`).
- `client`: your app cancelled it with `DELETE /v1/jobs/{id}`.
- `expired`: it waited `max_wait_s`, or nobody was following it (below).
- `server_restart`: the server stopped or restarted while it waited. The server does not
  run a queued job hours later for a client that may be gone.

After `removed`, `GET /v1/jobs/{id}` says `status: "removed"`, `error: null`, and
`removal: {reason, message, waited_s, at}`. That holds after a restart too: the record is
on disk.

## Handle `removed`: it is not a failure

Show it and offer to send the job again. Do not show it as an error, and do not resubmit
by yourself on `operator` (a person removed it on purpose). On `expired` or
`server_restart`, resubmitting automatically is reasonable if the user is still waiting.

When the job reaches the front of the queue, it goes through exactly the checks a fresh
submit meets. If one of them refuses it (not enough memory, a model missing), the job ends
`failed` with that refusal as its error, as if it had been refused at submit. It is not
queued again. Only "busy" refusals (`server_busy`, `engine_in_use`) and a card held by
someone else (`accelerator_busy`, next section) mean "not yet", and the job keeps its
place.

## When the card is held by someone else

Before Crucible loads a model it checks the card. If memory on it is held by a process
this Crucible does not own (a training run, another program, or an engine an earlier
Crucible left running), the check refuses `409 accelerator_busy`, naming the holder: its
pid, name and memory, or under WSL2, where the driver names nobody, how many bytes no
process accounts for. Crucible never evicts another process.

That is contention, not a mistake in the request, so a request that may wait **waits for
the card**, the same way it waits for the lane:

- A job keeps its place at the front of the line. The lane stays free; the job is checked
  again every 5 s (each check is a full admission, so it is paced, never per tick) and runs
  as soon as the memory is let go. It says why it waits: a `waiting` event on its own
  stream, `waiting_for` on its row in `GET /v1/queue` and in `GET /v1/activity`'s
  `queued`, and `queue.waiting` on the queue streams. Each is said once, and again only
  when who holds the card changes.
- A job submitted to an idle server while the card is held joins the line at once
  (`queued: true`, `position: 1`) and waits the same way.
- A queue session opening on its `model`, and a queued chat or decision whose model must
  be loaded, wait the same way at the front: the load is not started until the card is
  free. A load that the pump did start and that met a holder on the lane (one appeared
  between the check and the start) does not end the session `load_failed` or the call
  `queued_load_failed`; the item goes back to waiting for the card.
- The wait is bounded by the request's own `max_wait_s`. When it runs out the item is
  removed `expired`, and the message says it was waiting for the accelerator and who held
  it.
- `"queue": false` still refuses at once: `409 accelerator_busy`.

`waiting_for` is `{code: "accelerator_busy", message, details, since, next_check_at}`, or
null for an item that waits only for its turn. `details` is the guard's: `processes`
(pid, name, `used_bytes`), or `unattributed_bytes`, and `left_by_previous_run` when an
earlier Crucible left the holder.

An engine an earlier Crucible left running waits like any other holder. The server asks
it to stop when it starts (SIGTERM, then the engine's stop budget), so the first requests
after a restart usually find it already exiting; one that will not exit is named with the
command to stop it (`kill <pid>`, never -9), and the waiting request goes ahead the moment
it has gone.

What still fails by name, because no wait changes it: a model larger than the host or the
card's free memory with nothing foreign on it (`insufficient_memory`), a card measured
without a feature the engine needs (`card_lacks_feature`), and a card that cannot be read
(`accelerator_unreadable`).

The line stays first come, first served while its front waits for the card: what is
behind it (another client's job, a chat for the resident model) waits behind it. The only
job types that never touch the card are `echo` and the `unload-*` jobs, and letting those
past would reorder a client's own intent (its unload ahead of its load). A chat sent with
`"queue": false` to a resident model with a free slot is not held back.

One edge stays a failure: a job that passed the check at the front and met a holder in
the milliseconds before its engine started ends `failed` `accelerator_busy`, as before. A
job that has started is never put back in the line (`running` never becomes `queued`
again). The loads the server starts for a session or a queued call are the exception
above, because the waiting item is not the job.

`removed` has a fifth reason for a job that was an item of a queue session (below):
`session_closed`, when its session ended before the job reached the lane.

## Stay present

The server removes a waiting job as `expired` when nobody has asked about it for five
minutes. Any of these count as asking:

- an open event stream on the job, **or on any other job from the same client** (so a
  client that queues fifty jobs and follows them one at a time keeps all fifty);
- `GET /v1/jobs/{id}` on the job;
- `POST /v1/queue/{id}/heartbeat`, for a client that does neither.

The client is who the job was submitted as (`X-Crucible-Client`, else `User-Agent`).

## The open session goes first

While a queue session is open (below), its items go ahead of everything waiting and
nothing from anyone else runs. Everyone else is first come, first served.

## The whole queue

- `GET /v1/queue`: `{items: [{position, job_id, type, model, client, client_ref, submitted,
  waited_s, max_wait_s, expires_at, session, kind, waiting_for}], depth, limits}`.
  `waiting_for` says the front item waits for a held card (above), else null. `kind` is `"job"`,
  `"call"` (a queued chat or decision) or `"session"` (a queue session waiting to open,
  `job_id` `ses-…`). `session` names the queue session an item belongs to, or is null.
- `DELETE /v1/queue/{job_id}`: remove one (reason `operator`). Given the open session's
  id, it ends that session (reason `operator`).
- `GET /v1/queue/events`: server-wide SSE for dashboards. A `snapshot {items, depth}`
  first, then `added`, `moved {position}`, `waiting {code, message}` (the front waits for
  a held card), `started {waited_s}` and `removed {reason}` with `job_id` and `depth` on
  every event. A job refused at the front shows here as `removed`
  with `reason: "refused"` and its `error`; on its own stream it is `failed`.
- `GET /v1/activity`: `queued` lists the waiting jobs in order (with `waited_s`,
  `max_wait_s` and `waiting_for`), and `slots.accelerated.queue_depth` counts the job on the lane plus every
  waiting job.
- `GET /v1/events`: the same announcements as `queue.added`, `queue.moved`,
  `queue.waiting`, `queue.started` and `queue.removed`, each with its `kind`, on the one stream that also
  carries jobs, the card, chats, tasks and settings. A dashboard that would otherwise poll
  `/v1/queue`, `/v1/activity` and `/v1/tasks` follows that instead (docs/EVENTS.md).

While anything waits, a submit with `"queue": false` is refused `409 server_busy` even if
the lane is momentarily free, with `details.queue_depth`, so it cannot jump the line.

A job sent with `"queue": false` gets no `queued`/`position` in its receipt and no
`started` event: it either went straight onto the lane or was refused.

## Chats and decisions wait too

`POST /v1/openai/chat/completions` (and `/openai/v1/...`) and `POST /v1/decide` take the
same member in their body. With `"queue": false` they are refused at once: `409
model_not_resident` when the model is not loaded, `503 chat_queue_full` when every slot on
its engine is taken. Otherwise (the default):

- **The model is resident with a free slot, and nothing is waiting:** the request goes
  straight through.
- **Otherwise** the request is held open and takes a place in the same line as queued jobs.
  It shows in `GET /v1/queue` with `kind: "call"`, `type: "chat"` or `"decide"`, and a
  `job_id` of the form `call-…` (there is no job record behind it; `GET /v1/jobs/{id}` does
  not know it). Job rows carry `kind: "job"`.
- **At the front, its model not resident:** when the lane is free and no chat is in flight
  on another model, the server submits a `load-model` job for it (your client name,
  `client_ref: "for the queued chat call-…"`). Every call behind it for the same model
  then fills the engine's slots together once it is loaded. A load that fails ends the
  call `502 queued_load_failed` naming the job; a card held by someone else is not a
  failure, the call waits for it ("When the card is held by someone else").
- **At the front, every slot taken:** it gets the next slot that frees.
- The answer is the normal completion (or stream, or decision). Nothing is sent before it,
  so give the HTTP request a read timeout that covers `max_wait_s` plus the answer.

A queued call leaves the line:

| how | your request gets |
|---|---|
| an operator removes it (`DELETE /v1/queue/call-…`, the desktop Queue) | `409 removed_from_queue`, `details.reason: "operator"` |
| it waits `max_wait_s` | `409 removed_from_queue`, `details.reason: "expired"` |
| the server stops | `409 removed_from_queue`, `details.reason: "server_restart"` |
| you close the connection | it is removed (`client`); nothing more is sent |

A call is never abandoned for lack of polling: the open request is its presence. A
decision is checked for everything it can be refused for (its questions, items and image
count) before it joins the line, so a malformed one never waits.

Two things change for queued jobs alongside this. A queued job that would change what is
on the card (any load, unload, or a job that brings its own model) waits while chats are in
flight instead of taking the model out from under them. And while a call waits for the
resident model, the server does not unload that model between completions.

A chat sent with `"queue": false` goes straight to a resident model with a free slot even
while something waits in the line, unless another client's queue session is open (below):
then it is refused `409 session_open`.

A chat for an upstream model (`<upstream>/<id>`) never waits: it is forwarded at once and
its `queue` member, if any, is dropped. Nothing about it uses this server's card.

## Queue sessions: the server to yourself for a run

A **queue session** is one client's claim on the server for a run of requests it cannot know
in advance: a Briefcase video analysis (ASR, then many chats and decisions), a BookForge
chapter loop, a ContentStudio render beside its editor's own calls. While the session is
open its items run back to back, nothing from any other client runs in between, and what
they leave on the card stays there for the next item. One session is open at a time.

(Not to be confused with a TTS *stream* session, `POST /v1/tts/stream`, which runs inside
a queue session; see "Streams run inside a session" below.)

### Ask for one

```json
POST /v1/queue/sessions
{"act": "analysis", "model": "qwen3.5-9b", "idle_s": 300, "max_wait_s": 3600}
```

- `act` (required): the capability class the run is for, as `X-Crucible-Act` names it.
- `model`: a model to have resident when the session opens. If it is not resident, the
  server loads it for the session (a `load-model` job with your client name and
  `client_ref: "opening session ses-…"`) and reports the session open only once the load
  is done. A load that fails ends the session (`removed`, reason `load_failed`, with the
  error); a card held by someone else is not a failure, the session waits for it at the
  front (a `waiting` event, "When the card is held by someone else"). Leave it out to open on whatever is resident. An upstream model is refused
  `409 upstream_never_resident`; an unknown one `404 unknown_model`.
- `idle_s` (default 300, 10 to 86400): see "How a session ends".
- `max_wait_s` (as the queue's): how long it may wait in the line to open.

It answers `202` at once, never blocking:

```json
{"session_id": "ses-5f0c…", "status": "open", "position": null}
{"session_id": "ses-5f0c…", "status": "queued", "position": 2}
```

A queued session waits in the same line as everything else (`kind: "session"` in
`GET /v1/queue`) and opens at the front, once the lane is free. It is presence-checked
like a queued job: follow its events, read it, or touch it at least every five minutes,
or it is removed `expired`.

### Follow it

`GET /v1/queue/sessions/{id}/events` is its own SSE stream:

| event | data | means |
|---|---|---|
| `queued` | `{position, of}` | it joined the line |
| `moved` | `{position, of}` | its place changed |
| `waiting` | `{code, message, details, since}` | at the front, its model's load waits for a card held by someone else |
| `opened` | `{opened_at, model, load_job}` | it is open; send its items |
| `closed` | `{reason, message, items_run, held_s}` | it ended; **terminal** |
| `removed` | `{reason, message, error?}` | it ended without ever opening; **terminal** |

`GET /v1/queue/sessions/{id}` reads it: `status` (`queued`, `open`, `closed`), `position`,
`opened_at`, `items_run`, `in_flight` (what it has running or waiting: jobs, chats,
queued calls, stream rows being said), `stream_session` (the TTS stream open in it, if any),
`idle_deadline` (null while anything is in flight), `max_hold_deadline` (null unless the
server sets a maximum), `load_job`, and `reason`/`message`/`error` once it is closed.

### Send its items

Every request from the client that holds the open session is one of its items. Name it
explicitly with the header `X-Crucible-Session: ses-…`, or let your client name
(`X-Crucible-Client`, else `User-Agent`) say it: a request from the same client is an
**implicit item**, header or not. That is so an app can make standalone calls beside its
own long run (an editor's title, a frame check) without them waiting behind itself.

**Give each install its own client name.** Membership is by name, so two installs that send
the same `clientName` (BookForge on the Mac and on the PC, both `"bookforge"`) ride each
other's sessions. Name each install distinctly and stably, for example
`"bookforge@<hostname>"`; give a CLI or an embedded runner its own name too when it is a
separate app. Nothing on the server keys on a particular name; it is identity and display
only. A client that opens a second session while its first is open waits in line behind its
own first one (one session is open at a time, never merged): share one session for one
install's work instead.

- `POST /v1/jobs`: admitted ahead of everything waiting. Items still run one at a time on
  the lane: an item submitted while another of the session's jobs runs waits *inside* the
  session, first come first served, ahead of everyone else (it answers `queued: true`
  with its position; it waits up to a day unless its own `queue` says otherwise, and
  `"queue": false` refuses it `server_busy` instead).
- `POST /v1/openai/chat/completions` (and `/openai/v1/...`), `POST /v1/decide`: as usual.
  One that must wait (its model not resident, every slot taken) waits ahead of the line,
  up to an hour or its own `max_wait_s`, and its model is loaded for it with the
  session's priority.
- `POST /v1/tts/stream`: see below.

A header naming a session that is not open is refused by name: `404
unknown_queue_session`, `409 session_not_open` (still waiting; send its items after
`opened`), `409 session_closed` (with `details.reason`). A header naming another client's
session is `409 session_not_yours`.

### What everyone else sees while it is open

Nothing from any other client runs:

| their request | gets |
|---|---|
| a job, chat or decision | waits in the line until the session closes |
| a job with `"queue": false` | `409 server_busy`, `details.door: "session"`, naming the holder |
| a chat or decision with `"queue": false` | `409 session_open`, naming the holder and the session |
| another queue session | waits in the line |
| a TTS stream | waits in the line (or `409 session_open` with `"queue": false`) |

There is no pre-emption. A person who wants the server back ends the session (the desktop
app's Queue, or `DELETE /v1/queue/{id}`).

### How a session ends

| reason | when |
|---|---|
| `client` | its client sends `DELETE /v1/queue/sessions/{id}` (a queued one leaves the line) |
| `idle` | `idle_s` passes with no item arriving, nothing in flight, no running job and no touch |
| `operator` | an operator ends it: `DELETE /v1/queue/{id}`, or the desktop Queue's End |
| `max_hold` | only if the server sets `[queue] max_session_hold_s` in config.toml; none by default |
| `server_restart` | the server stops |

Anything in flight is presence: a session running a day-long job never idles out. For a
long gap on your side with nothing in flight (a NAS copy between two steps), send
`POST /v1/queue/sessions/{id}/touch`; it is a timestamp in memory, so every 30 s is fine.

When a session closes, its items still waiting leave the line (`removed`, reason
`session_closed`), a TTS stream open in it closes, and the card is settled: unloaded
unless something else holds it. What is already running finishes.

### Streams run inside a session

A TTS stream (`POST /v1/tts/stream`) has no claim of its own and no priority. It opens
inside a queue session held by its client: the one its `X-Crucible-Session` names, or the
open one its client holds. A client that holds none gets one opened for the stream
(`act: "tts"`, the stream-open's `idle_s`, default 900), which waits in the line behind
other sessions like any other.

The stream-open is a held-open request, like a queued chat: it answers `201` once the
session is open and the voice is resident (a voice that is not is loaded by a `load-voice`
job inside the session), with `queue_session_id` and `queue_session_opened_for_stream`
beside the stream's own fields. `"queue": false` refuses instead of waiting (`409
session_open`, or `server_busy`). A session that ends before it opens answers `409
session_closed` with `details.reason`.

Every stream op (`say`, `cancel`, `cancel_all`, attaching the events) is activity of its
session; an open stream with no row being said is not, so `idle_s` runs out and closes
the session and the stream with it. The stream's `closed` frame then carries `code:
"session_closed"`, `session_reason` (e.g. `idle`) and `queue_session_id`. Closing the
stream closes a session that was opened for it, but never one the client opened itself.

## More than one server

The queue is per server: a session waits in one server's line, and nothing on the server side
knows about any other server. An app that can reach several (the PC and the Mac) asks each of
them for a session and keeps the first that opens. The rules that keep that fair:

- Ask only servers that can serve the run: `queue.sessions` in `GET /v1/info` `features` and,
  with a `model`, that model in `GET /v1/models` with `backend_supported` and `installed`.
- The moment one session is `opened`, `DELETE /v1/queue/sessions/{id}` every other one (reason
  `client`): a queued one leaves its line, and one that opened in the same instant closes and
  settles its card. A loser must never hold its machine.
- If the request is abandoned while a `POST /v1/queue/sessions` is still unanswered, wait for
  its ticket and delete the session it names: the server may already have made, or opened, it.

There is no priority and no pre-emption across servers; the first to open wins. The TS SDK's
`fleetSession()` does all of this (below).

## In the SDK

```ts
const crucible = new CrucibleClient({ url, token, clientName: 'bookforge' });
const id = await crucible.render({ voice: 'sigma', ... });   // queues by default
for await (const event of crucible.events(id)) {
  if (event.event === 'queued') show(`waiting, number ${event.data.position}`);
  if (event.event === 'removed') offerResubmit(event.data.reason, event.data.message);
}
```

- Every request that can wait (`submit()`, every job helper, `chat`, `chatStream`,
  `decide`, `decideItems`, `stream`) waits by default, and sends no `queue` member to do
  it. `new CrucibleClient({..., queue: false})` makes them refuse at once instead;
  `queue: {maxWaitS: 600}` changes the wait. A request's own `queue` wins. This SDK has no
  `true` and no `{}`; the server still reads an older SDK's `{}` as "wait".
- There is no "busy, then ask again with the queue" step to write: a `CrucibleBusy` (or
  `session_open`, `model_not_resident`, `chat_queue_full`) now only reaches a request
  sent with `queue: false`. The SDK never sends a request a second time.
- A queued chat or decision that is removed throws `CrucibleRefused` with code
  `removed_from_queue` and `details.reason`.
- `job()` returns `status: 'removed'` and `removal`; `cancel()` on a waiting job answers
  `status: 'removed'`. `queue()`, `removeFromQueue()`, `queueHeartbeat()` and
  `queueEvents()` cover the routes above.
- `session({act, model?, idleS?, maxWaitS?, onQueue?, signal?})` asks for a queue session
  and answers once it is open (following its stream, and calling `onQueue({position, of})`,
  while it waits). The `CrucibleSession` it returns is the client plus the session's header:
  every method sends `X-Crucible-Session`, its job helpers send no `queue`, and its
  `closed` promise resolves with the reason when the session ends. `touch()`, `state()` and
  `close()` cover the session's routes. A session that ends before it opens throws
  `CrucibleSessionClosed` with its `reason`; another client's session refuses as
  `CrucibleSessionHeld`.
- `fleetSession(clients, {act, model?, idleS?, maxWaitS?, onQueue?, signal?, probeTimeoutMs?})`
  asks every capable server for a session at once and answers `{session, client, index,
  dropouts}` with the first that opens; the rest are removed at once. `onQueue({client,
  position, places})` hears every server's place. With no session to give it throws
  `CrucibleFleetUnavailable`, whose `servers` names each server and why.
- `events({topics?, lastEventId?, signal?})` follows `GET /v1/events` (docs/EVENTS.md),
  reconnecting by itself.
- sdk/ts/MIGRATION.md maps every lease call an app made to its replacement.
