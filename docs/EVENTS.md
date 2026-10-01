# The event stream: for app authors

`GET /v1/events` is one Server-Sent Events stream of every change on the server: jobs,
the waiting line, the card, chats in flight, tasks, settings, and the server stopping. An
app opens it once and keeps it open instead of polling `/v1/activity`, `/v1/tasks`,
`/v1/queue` or `/v1/health` every few seconds. It is a protected route: send the token
and `X-Crucible-Api: 1` like any other.

Check that a server has it with `"events" in info.features` (`GET /v1/info`), not by
comparing versions.

## What arrives

Every frame is an ordinary SSE event:

```
id: 1790000000000123
event: job.done
data: {"job_id":"…","type":"echo","model":null,"client":"bookforge", …,"at":"2026-10-01T…"}
```

- `id` is an integer that only grows. Ids are not reused across restarts: they start at
  the moment the server started, in microseconds.
- `event` is `<topic>.<what>`. Every `data` carries `at`, when the server published it.
- Comment lines (`: keepalive`) arrive every 15 s while nothing happens, so a proxy does
  not time the stream out. Ignore them.

### First: `snapshot`

The stream opens with one `snapshot`, whose `id` is where the events after it start:

| field | what it is |
| --- | --- |
| `gap` | `false` on a first connect. `true` when you reconnected with a Last-Event-ID the server no longer has (see Resuming): what you missed is folded into this snapshot. |
| `topics` | The topics this stream carries. |
| `activity` | The body of `GET /v1/activity`, exactly. |
| `queue` | `{items, depth}`, as `GET /v1/queue` lists them. |
| `tasks` | The recent tasks, newest first, as `GET /v1/tasks` lists them. |

Draw the dashboard from the snapshot, then apply each event to it.

## Topics

`?topics=job,queue,card` sends only those topics. Leave it out for all of them. An
unknown name is refused `400 unknown_topic` with the known ones in `details.known`.
`server` is always sent, and so are `snapshot` and `overflow`.

| topic | events |
| --- | --- |
| `job` | `job.queued`, `job.running`, `job.progress`, `job.done`, `job.failed`, `job.cancelled`, `job.interrupted`, `job.removed` |
| `queue` | `queue.added`, `queue.moved`, `queue.started`, `queue.removed` |
| `card` | `card.warming`, `card.warming_ended`, `card.loaded`, `card.unloading`, `card.unloaded` |
| `chat` | `chat.in_flight` |
| `task` | `task.running`, `task.step`, `task.progress`, `task.done`, `task.failed`, `task.cancelled` |
| `settings` | `settings.written` |
| `server` | `server.stopping` |

## The events

### Jobs

A job says `job.<status>` once each time its status changes, so the name is always the
status it has just entered. Every one carries `job_id`, `type`, `model`, `client`,
`client_ref` and `status`, and:

| event | also carries |
| --- | --- |
| `job.queued` | `position`; `waiting`: `true` when it waits in the line (it was submitted with `queue` while the lane was busy), `false` when it went straight to the lane. Moves in the line are `queue.moved`, not more `job.queued`. |
| `job.running` | `started`. |
| `job.progress` | Only `job_id`, `fraction` (0 to 1) and `message`. At most one a second per job, only when either changed, and the latest one always arrives unless the job ends first. |
| `job.done` | `artifacts`: the names to fetch from `GET /v1/jobs/{id}/artifacts/{name}`. |
| `job.failed` | `error`: `{code, message}`. |
| `job.cancelled` | nothing more. |
| `job.interrupted` | `interrupted_at`: the server stopped while it ran (docs/RESUMABLE-JOBS.md). |
| `job.removed` | `removal`: `{reason, message, waited_s, at}`; `reason` is `operator`, `client`, `expired` or `server_restart` (docs/QUEUE.md). |

Everything a job says on its own stream (`GET /v1/jobs/{id}/events`: notes, warming
messages, cues, each artifact as it lands) stays there. Follow that stream when a single
job's detail matters.

### The queue

The waiting line's own announcements, the same as `GET /v1/queue/events`. Each carries
`job_id`, `depth` (how many wait after this change) and `kind`: `"job"`, `"call"` (a
queued chat or decision) or `"session"` (an app's session waiting for its turn).

| event | also carries |
| --- | --- |
| `queue.added` | `position`, `type`, `model`, `client`, `submitted`, `max_wait_s`. |
| `queue.moved` | `position`. |
| `queue.started` | `waited_s`: it left the line for the lane or for a chat slot. |
| `queue.removed` | `reason` and `message`; a job refused at the front has `reason: "refused"` and `error` instead of `message`. |

### Sessions

Every change to a queue session (docs/QUEUE.md), the same events its own
`GET /v1/queue/sessions/{id}/events` stream sends, named `session.<event>`. Each carries
`session_id`, `client` and `act`.

| event | also carries |
| --- | --- |
| `session.queued` | `position`, `of`: it waits for its turn. |
| `session.moved` | `position`, `of`. |
| `session.opened` | The session holds the machine. |
| `session.closed` | `reason` (`client`, `idle`, `operator`, `max_hold`, `server_restart`), `message`, `items_run`, `held_s`. |
| `session.removed` | `reason`, `message`, `error`: it left the line without opening. |

### The card

What is resident. Each carries `subject` (the model, voice or other subject id), `kind`
(`llm`, `tts`, `align`, `denoise`, `image`, `audio`, `segment`, `video`) and `engine` (the
engine's name; `null` while warming, and for an aligner or a separator, which have none).

| event | also carries |
| --- | --- |
| `card.warming` | A load began. |
| `card.warming_ended` | The load stopped warming: a `card.loaded` follows it at once if it succeeded; if none does, the load failed and its job says why. |
| `card.loaded` | `memory_bytes_estimate`, `since`. |
| `card.unloading` | `since`, `pids`: it was asked to stop. |
| `card.unloaded` | `since`, `pids`: its processes have exited and the card is free. |

### Chats

`chat.in_flight`: `in_flight` (completions and decisions being answered right now) and
`by_model` (`{model: count}`). Coalesced: at most one a second, only when the counts
changed, and the latest count always arrives, so a burst of chats ends with the true
number. `max_in_flight` is in the snapshot's `activity.chat`.

### Tasks

| event | carries |
| --- | --- |
| `task.running` | The task as `GET /v1/tasks/{id}` shows it: `task_id`, `type`, `request`, `state`, `created`, `started`, … |
| `task.step` | `task_id`, `name`, `index`, `total`. |
| `task.progress` | `task_id` and either `line` (a line of the installer's output) or `bytes_done`, `bytes_total`, `file` (a download). At most one a second per task; the latest arrives. |
| `task.done`, `task.failed`, `task.cancelled` | The task as `GET /v1/tasks/{id}` shows it, with `error` and `finished`. |

### Settings

`settings.written`: `act`, `client` and `changed` (the keys a `PUT /v1/settings` wrote),
as `activity.settings.writes` lists them. Read `GET /v1/settings` for the new values.

### The server

`server.stopping`: `reason`. Sent the moment the server is asked to stop, and the stream
ends after it. Reconnect with backoff; a new stream is refused `503 server_stopping`
until the server is back.

Every other SSE stream ends the same way: a job's (`/v1/jobs/{id}/events`), a task's,
the queue's (`/v1/queue/events`) and a narration session's each end with one
`server.stopping {reason}` frame. That frame has no `id`, so the stream's Last-Event-ID
stays on its last real event and a reconnect after the restart resumes from there.

## Resuming

Browsers' `EventSource` sends `Last-Event-ID` by itself when it reconnects; any other
client sends the last `id` it saw in that header. The server keeps the last 1000 events
in memory:

- When the id is still in that history, every event after it arrives first, in order,
  and no snapshot is sent.
- When it is not (the history has rolled past it, or the server restarted since), the
  stream opens with a `snapshot` whose `gap` is `true`. Replace what you drew with it.

## A slow reader is dropped, never waited for

Each stream has its own buffer of 2000 events. A client that stops reading while the
server keeps publishing fills it, and its stream then ends with one `overflow` event:
`last_event_id` (the last event it was sent), `limit` and a `message`. Nothing on the
server ever waits for a reader. Reconnect with `Last-Event-ID: <last_event_id>`; if that
is still in the history you lose nothing.

## Adding an event (Crucible authors)

The hub is `crucible/events.py`. Each owner of the server's state holds the app's hub as
`.events` and publishes at its own single point of truth:

```python
self.events.publish(QUEUE, "queue.added", {...})                    # once per change
self.events.publish_throttled(JOB, "job.progress", key, {...})       # at most 1/s per key
self.events.forget(key)                                              # the subject ended
```

`publish` never blocks and may be called from any thread. A new topic is added to
`TOPICS` in crucible/events.py and to the tables above, in the same change. The hooks
today: `JobStore.append_event` (crucible/jobs/queue.py), `WaitingLine._announce`
(crucible/jobs/line.py), `Residency` (warming, `occupy`, `unload`, the dying engine's
exit), `InFlight.open`/`close`, `TaskStore.append_event`/`_finish`, and
`settings.History.record`.
