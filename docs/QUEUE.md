# The queue: for app authors

Crucible runs one job at a time. A job sent while another runs used to be refused
`409 server_busy`, and every app wrote its own retry loop. Now an app can ask to wait:
the server keeps the job in a first-come, first-served queue and runs it when its turn
comes. Old apps are not affected: a submit without `queue` is refused exactly as before.

The TypeScript SDK's high-level helpers queue by default from the release that carries
this page (see "In the SDK" below).

## Submit with `queue`

```json
POST /v1/jobs
{"type": "tts", "model": "sigma", "params": {...}, "inputs": {...},
 "queue": {"max_wait_s": 3600}}
```

`"queue": {}` takes the default wait of an hour. `max_wait_s` is 10 to 86400 seconds;
anything else is `400 invalid_request`.

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

`GET /v1/jobs/{id}/events`, the same stream as always, adds three events for a job
submitted with `queue`:

| event | data | means |
|---|---|---|
| `queued` | `{position, of}` | where it stands; sent when it joins and again whenever it moves |
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
queued again. Only "busy" refusals (`server_busy`, `leased`, `engine_in_use`) mean "not
yet", and the job keeps its place.

## Stay present

The server removes a waiting job as `expired` when nobody has asked about it for five
minutes. Any of these count as asking:

- an open event stream on the job, **or on any other job from the same client** (so a
  client that queues fifty jobs and follows them one at a time keeps all fifty);
- `GET /v1/jobs/{id}` on the job;
- `POST /v1/queue/{id}/heartbeat`, for a client that does neither.

The client is who the job was submitted as (`X-Crucible-Client`, else `User-Agent`).

## The lease holder goes first

A client holding the open lease (`POST /v1/models/{id}/lease`) is running a batch on the
resident model. Its queued jobs go ahead of everyone else's while the lease is open, and
its plain submits are not held behind the queue. Everyone else is first come, first served.

## The whole queue

- `GET /v1/queue`: `{items: [{position, job_id, type, model, client, client_ref, submitted,
  waited_s, max_wait_s, expires_at, lease_holder}], depth, limits}`.
- `DELETE /v1/queue/{job_id}`: remove one (reason `operator`).
- `GET /v1/queue/events`: server-wide SSE for dashboards. A `snapshot {items, depth}`
  first, then `added`, `moved {position}`, `started {waited_s}` and `removed {reason}` with
  `job_id` and `depth` on every event. A job refused at the front shows here as `removed`
  with `reason: "refused"` and its `error`; on its own stream it is `failed`.
- `GET /v1/activity`: `queued` lists the waiting jobs in order (with `waited_s` and
  `max_wait_s`), and `slots.accelerated.queue_depth` counts the job on the lane plus every
  waiting job.

While anything waits, a submit without `queue` is refused `409 server_busy` even if the
lane is momentarily free, with `details.queue_depth`, so an old app cannot jump the line.

## In the SDK

```ts
const crucible = new CrucibleClient({ url, token, clientName: 'bookforge' });
const id = await crucible.render({ voice: 'sigma', ... });   // queues by default
for await (const event of crucible.events(id)) {
  if (event.event === 'queued') show(`waiting, number ${event.data.position}`);
  if (event.event === 'removed') offerResubmit(event.data.reason, event.data.message);
}
```

- The high-level helpers (`render`, `asr`, `align`, `image`, `audio`, `segment`, `video`
  and every load and unload) queue by default. `new CrucibleClient({..., queue: false})`
  turns that off; `queue: {maxWaitS: 600}` changes the wait. A request's own `queue` wins.
- `submit()` queues only when its request says `queue: true` or `queue: {maxWaitS}`.
- Against a server older than the queue, the SDK sends the job again without `queue`, so
  the app sees `CrucibleBusy` as it always did.
- `job()` returns `status: 'removed'` and `removal`; `cancel()` on a waiting job answers
  `status: 'removed'`. `queue()`, `removeFromQueue()`, `queueHeartbeat()` and
  `queueEvents()` cover the routes above.
