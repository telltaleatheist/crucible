# Queue sessions (they replaced leases)

Status: **built, 2026-10-01.** Proposed the same day as "queue groups"; Owen renamed the
concept to *session* ("the queue's turn-taking unit is an app's session") and ruled on the
open questions. The app-facing guide is docs/QUEUE.md; this page is how it is built.

History: until this change a client kept the card between its requests with a **lease**
(`POST /v1/models/{id}/lease`, a heartbeat, a TTL, `409 leased` refusals, `params.lease`
on load jobs). A lease pinned one subject, sat beside the queue rather than in it, and
needed a heartbeat. Owen: *"we might be able to get rid of leases if we have the queue"*,
and then *"we dont need to worry about legacy anything"*. Leases are gone; nothing wraps
them.

## Owen's rulings

- **Nothing in between.** While a session is open, nothing from any other client runs.
- **One open session per server.** Other sessions wait in the line.
- **No maximum hold by default.** Some runs take a day. `[queue] max_session_hold_s` in
  config.toml sets one (absent or 0 is none); a session past it ends `max_hold`.
- **Liveness is `idle_s` alone** (default 300, up to 86400). Anything in flight counts as
  presence, so a day-long job inside a session never idles it out. `touch` is a timestamp
  in memory.
- **Same-client requests are members.** The `X-Crucible-Session` header is the explicit
  form; a request from the client holding the open session is an implicit item of it.
- **TTS streams run inside a session, with no priority** and no claim of their own. A
  client with no session gets one opened for its stream (`idle_s` 900 by default).

## The pieces

| file | what |
|---|---|
| `crucible/queuesessions.py` | `QueueSession` (one session's state) and `QueueSessions` (the registry: the open one, the queued ones, the last 200 closed). Membership (`member`, `item`), refusals (`refuse_if_held` → `server_busy`, `refuse_call_if_held` → `session_open`), `due()` (idle, max hold, its stream gone), `state()`. **Every state change goes through `QueueSessions._say`**, which records the event on the session's own stream and wakes its followers: that is where a server-wide publish hook belongs. |
| `crucible/sessionqueue.py` | `admit_session` (the pump opening the session at the front, loading its model first) and `SessionCloser.end`, the one way an open session ends: its waiting items leave the line `session_closed`, a stream inside it closes, the card is settled. |
| `crucible/jobs/line.py` | the line holds a waiting session as a `WaitingSession` (kind `session`); `ordered()` puts the open session's items first; every item knows its `session`. |
| `crucible/queuepump.py` | each step expires the line, closes a session that is `due()`, then walks the line; while a session is open it offers only that session's items. Steps never overlap (`POST /v1/queue/sessions` takes a step of its own so an idle server answers `open` at once). |
| `crucible/admission.py` | `JobRequest.session`; `refuse_if_held` names the open session for anyone else's job (a queued one waits); a session item waits only behind the session's own items. The job records its session (`job.session`). |
| `crucible/callqueue.py` | queued chats and decisions carry their session; `take_a_turn` lets a session item past the line. |
| `crucible/settle.py` | the open session is a holder (`Held("a session", …)`), checked right after "a job"; a waiting session whose model is resident counts like a waiting call. |
| `crucible/api/routes/sessions.py` | the five routes. |
| `crucible/api/caller.py` | `queue_session(request, sessions)`: the session a request is an item of. |
| `crucible/api/streamturn.py` | a TTS stream's turn: its session (opened for it if need be), then its voice (loaded in the session). |

### What counts as in flight

`api/app.py:_session_in_flight` reads it fresh every time: the session's jobs that have not
ended (queued in the line or on the lane), chats and decisions answering for it
(`InFlight` entries carry `session`), its calls waiting in the line, and rows its TTS stream
is saying. An open stream with nothing being said is *not* in flight, so the session's
`idle_s` still runs out and the stream closes with it.

### Opening

At the front of the line, `admit_session` waits for a free lane and no open stream. With a
`model` that is not resident it submits `load-model` through ordinary admission
(`from_the_line`, attributed to the session, `client_ref: "opening session ses-…"`) and
keeps the session's place; the session counts as a holder of that model while it waits
(`calls_waiting`), so nothing settles it away between the load finishing and the opening.
A load that ends anything but `done` fails the session (`removed`, `load_failed`, error
`session_load_failed`), except a card held by a process Crucible does not own: a load
refused `accelerator_busy` at admission is never started, and one that ended
`accelerator_busy` on the lane (`callqueue.card_was_held`) does not count against
`MAX_LOADS`. Either way the session keeps its place (`WaitingLine.not_yet` records the
sentence, says `waiting` on the session's stream, and paces the next try by
`CARD_RECHECK_S`); its `max_wait_s` bounds the wait. Queued calls do the same in
`callqueue.admit_call` / `_load_for`.

### Streams

`StreamManager` no longer settles the card when a stream closes: the session is the one
holder. Its residency claim stays, but as what it always protected: narrator has one
stdin and one stdout, so a render job and a stream cannot converse with it at once. A
session's own tts job submitted while its stream is open therefore waits inside the
session (`engine_in_use` keeps it waiting) until the stream closes.

Closing a stream closes a session opened for it (`opened_for_stream`), immediately from
the route, or from the pump's next step when the stream died another way (its grace window
ran out). It never closes a session the client opened itself.

## Notes

- Every session change also publishes on the server-wide event stream (`GET /v1/events`,
  topic `session`, docs/EVENTS.md) from `QueueSessions._say`, and `queue.sessions` is listed
  in `GET /v1/info` `features`.
- Upstream chats (`<upstream>/<model>`) never touch the card, so a session does not hold
  them back.
