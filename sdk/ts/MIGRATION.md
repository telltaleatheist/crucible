# Migrating from `@crucible/client` 1.0.55 to 1.0.75

For app authors (BookForge, Briefcase, ContentStudio, Foundry). Owen's ruling for this
release: **no legacy compatibility**. Leases are gone from the server and from this SDK, and
nothing wraps them; the SDK also no longer asks an older server again without `queue`. Move to
queue sessions and the event stream as you touch each call site.

## What was removed, and what replaces it

| Removed | Replacement |
|---|---|
| `lease(subject, {act, ttlSeconds, queue?})` | `session({act, model?, idleS?, maxWaitS?, onQueue?, signal?})` |
| `heartbeat(leaseId)` | nothing for work the server is doing (it counts as activity); `session.touch()` across a long gap on the app's side |
| `release(leaseId)` | `session.close()` |
| `lease` option on `loadModel`, `loadVoice`, `image`, `audio`, `segment`, `video` | run them inside a `session()`; the session holds what they load |
| `loadImage(m, {lease})`, `loadAudio(m, {lease})`, `loadSegment(m, {lease})`, `loadVideo(m, {lease})` | `loadImage(m)`, `loadAudio(m)`, `loadSegment(m)`, `loadVideo(m)` (the options types are gone) |
| `Lease`, `ActivityLease`, `LeaseOnLoad`, `LoadImageOptions`, `LoadAudioOptions`, `LoadSegmentOptions`, `LoadVideoOptions` | `QueueSessionState`, `QueueSessionEnd`, `SessionOptions` |
| `Activity.lease` | `Activity.session: QueueSessionState \| null` |
| `JobStatus.leaseId`; `leaseId` on `ImageResult`, `AudioResult`, `SegmentResult`, `VideoResult` | nothing: a job belongs to the session it was sent in |
| `QueueItem.leaseHolder`, `kind: 'lease'` | `QueueItem.session: string \| null`, `kind: 'session'` |
| `CrucibleLeased`, `LEASED` (`409 leased`) | `CrucibleSessionHeld` (`409 server_busy` door `session`, or `session_open`) |
| `LEASE_NOT_NEEDED` | `UPSTREAM_NEVER_RESIDENT` (`upstream_never_resident`) |
| the retry that sent a job, decision or lease again without `queue` when an older server refused the field | none: that refusal is thrown like any other |

## Leases to sessions

A lease pinned one model on the card beside the queue and needed a heartbeat. A session is your
app's **turn holding the machine**: it waits in the same line as everything else, and while it
is open nothing from any other client runs. One session is open at a time per server, and
there is no priority — first come, first served.

### Lease + heartbeat → `session()`

Before:

```ts
const lease = await crucible.lease('qwen3.5-9b', { act: 'clean', ttlSeconds: 120 });
const timer = setInterval(() => void crucible.heartbeat(lease.leaseId), 30_000);
try {
  for (const chunk of book) await crucible.chat({ model: 'qwen3.5-9b', messages: chunk });
} finally {
  clearInterval(timer);
  await crucible.release(lease.leaseId);
}
```

After:

```ts
const session = await crucible.session({ act: 'clean', model: 'qwen3.5-9b' });
try {
  for (const chunk of book) await session.chat({ model: 'qwen3.5-9b', messages: chunk });
} finally {
  await session.close();
}
```

`model` makes the server load it for the session when it is not resident; `session()` answers
only once the session is open with it loaded. A load that fails ends the session before it
opens: `session()` throws `CrucibleSessionClosed` with `reason: 'load_failed'`.

### Parking on `409 leased` → `session()` waiting with `onQueue`

Before, an app that met `CrucibleLeased` waited and tried again. Now ask for a session and let
the server keep your place:

```ts
const session = await crucible.session({
  act: 'analysis',
  maxWaitS: 3600,
  onQueue: ({ position, of }) => status(`Waiting for the server: ${position} of ${of}`),
  signal: cancelButton.signal,            // aborting takes it out of the line
});
```

While it waits, the SDK follows the session's own stream, which is also what keeps it in the
line (the server removes a waiting session nobody follows after five minutes, `expired`).

### Polling `/v1/activity`, `/v1/tasks`, `/v1/health` → `events()`

```ts
for await (const event of crucible.events({ signal })) {
  if (event.event === 'snapshot') redraw(event.activity, event.queue, event.tasks);
  else apply(event);   // job.*, queue.*, session.*, card.*, chat.in_flight, task.*, settings.written
}
```

It opens with a snapshot, reconnects by itself with `Last-Event-ID`, yields a `gap` snapshot
when it cannot resume, and waits out `server.stopping`. Check for it with
`await crucible.has('events')`, not a version comparison; `has()` reads `info().features`, which
is new in this release (`queue.sessions`, `queue.calls`, `queue.jobs`, `events`, …).

## Same-client membership

Every request from the client that holds the open session is an item of it, with or without
the `X-Crucible-Session` header. The server matches on the client name (`clientName` here;
`X-Crucible-Client` on the wire). So the plain client's calls made beside your own run — an
editor's title, a frame check — never wait behind your session. Items of the session go ahead
of everything waiting; they run one at a time on the lane, so a job sent while the session's
own job runs waits inside the session, not in the line.

**Give each install its own client name.** Membership is by name, so two installs that send
the same `clientName` (BookForge on the Mac and on the PC, both `"bookforge"`) ride each
other's sessions. Name each install distinctly and stably, for example
`"bookforge@<hostname>"`; give a CLI or an embedded runner its own name too when it is a
separate app. Nothing on the server keys on a particular name; it is identity and display
only. A client that opens a second session while its first is open waits in line behind its
own first one (one session is open at a time, never merged): share one session for one
install's work instead.

A `CrucibleSession` sends the header explicitly on every request. Its job helpers send no
`queue` (the server lets a session's job wait up to a day behind the session's own work); its
chats and decisions still send the client's `queue`, because a session's call that must wait
for its model or a free slot waits ahead of the line only with it.

## `idleS`, and what counts as activity

`idleS` (default 300, 10–86400) closes the session after that long with nothing in flight, no
item and no touch.

- **Running work counts.** A job the session is running or has waiting, a chat or decision
  being answered, a queued call, a stream row being said: all are activity. A day-long job
  never idles a session out.
- **Work on your side does not.** A cloud model call, an upload to a NAS, a person reading a
  draft: the server sees nothing. Across such a gap call `session.touch()` (a timestamp on the
  server; every 30 s is plenty), or choose an `idleS` that covers it.
- There is no maximum hold unless the operator configures one (`max_hold`).

When the server ends the session — `idle`, `operator`, `max_hold`, `server_restart` — the
session's `closed` promise resolves with `{reason, message, itemsRun, heldS}` by itself, and
every later call on the session throws `CrucibleSessionClosed` without reaching the server.
Open a new session if there is more to do.

## New error codes

All arrive as `CrucibleRefused` (read `code`), two with their own classes:

| Code | Class | Means |
|---|---|---|
| `session_open` | `CrucibleSessionHeld` | another client's session holds the server; a chat or decision without `queue` is refused rather than waiting |
| `server_busy` with `details.door: "session"` | `CrucibleSessionHeld` | the same, for a job without `queue` |
| `session_closed` | `CrucibleSessionClosed` (`reason`) | the session ended; nothing more runs in it |
| `session_not_open` | `CrucibleRefused` | the session is still waiting in the line; send its items after it opens (`session()` already waits) |
| `session_not_yours` | `CrucibleRefused` | the header names another client's session |
| `unknown_queue_session` | `CrucibleRefused` | the server does not know the session: it restarted, or forgot it long after it closed |

A queued job can also end `removed` with reason `session_closed`: it was an item of a session
that closed before it reached the lane.

## TTS streams inside sessions

`stream()` no longer opens a claim of its own. A stream runs inside a queue session: your
`CrucibleSession`'s, or the one your client holds open; otherwise the server opens one for it
(`act: 'tts'`, `idleS` default 900) that waits in the line and closes with the stream. The
voice no longer has to be resident first: it is loaded inside the session. `stream()` resolves
once the session is open and the voice is resident, and the result says `queueSessionId` and
`openedForStream`. New options: `idleS`, `queue` (`false` refuses instead of waiting) and
`signal`. An open stream with no row being said is not activity, so its session's `idleS` can
run out; the stream's `closed` frame then says `code: "session_closed"`.

## Smaller changes

- `info().features: string[]` and `has(feature)`.
- `closeSession(id)` closes a session your app recorded before a restart (reason `client`);
  `removeFromQueue(id)` would record it as `operator`.
- `QueueRemoved.status` is `'removed' | 'closed'` (`removeFromQueue()` given the open session's
  id ends it).
- `RemovalReason` includes `session_closed`.
- A job's (or task's, or the queue's) event stream that the server ends because it is stopping
  now throws `CrucibleUnreachable` naming the last id to resume from, instead of a protocol
  error about an id-less frame.
- `await using` is not offered for sessions (this package's TypeScript target does not declare
  `Symbol.asyncDispose`); close a session in `finally`.
