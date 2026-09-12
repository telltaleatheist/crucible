# `@crucible/client`

The TypeScript client for a [Crucible](https://github.com/telltaleatheist/crucible)
inference server. Servers, jobs, artifacts — and no GPU or platform code at all: the
client speaks HTTP even to a server it just started on localhost.

**Zero runtime dependencies.** It uses `fetch`, `ReadableStream`, `TextDecoder`,
`FormData` and `Blob`, which are globals in Node 20+, bun, and the Electron main
process. ESM and CommonJS builds ship side by side, with `.d.ts` for both.

## Install

From the GitHub Release (there is no npm registry publish):

```bash
npm install https://github.com/telltaleatheist/crucible/releases/download/v0.1.0/crucible-client-0.1.0.tgz
```

Pin the URL in `package.json` so the version is a fact, not a resolution.

## Example

```ts
import { CrucibleClient } from '@crucible/client';

const crucible = new CrucibleClient({
  url: 'http://127.0.0.1:7100',   // the server's base URL, without /v1
  token: process.env.CRUCIBLE_TOKEN!,  // what `crucible token --show` prints
  clientName: 'bookforge',        // lands in User-Agent, so the server log names you
});

console.log(await crucible.ping());          // { crucible: true, name, apiVersion }

const jobId = await crucible.submit({
  type: 'echo',
  params: { delay_ms: 25 },
  inputs: { 'note.txt': { inline: new TextEncoder().encode('hello crucible') } },
});

for await (const event of crucible.events(jobId)) {
  console.log(event.id, event.event, event.data);   // ends on done / failed / cancelled
}

const bytes = await crucible.artifact(jobId, 'note.txt');
const provenance = await crucible.provenance(jobId, 'note.txt');
console.log(new TextDecoder().decode(bytes), provenance.server, provenance.backend);
```

## The surface

| Method | Route | Returns |
|---|---|---|
| `ping()` | `GET /v1/ping` (no auth) | `Ping` |
| `info()` | `GET /v1/info` | `ServerInfo` |
| `health()` | `GET /v1/health` | `Health` |
| `upload(bytes \| blob, {filename})` | `POST /v1/uploads` | `UploadResult` — `{blobId, bytes, sha256}` |
| `submit({type, model?, params, inputs})` | `POST /v1/jobs` | the job id |
| `job(id)` | `GET /v1/jobs/{id}` | `JobStatus` |
| `events(id, {lastEventId?})` | `GET /v1/jobs/{id}/events` | `AsyncIterable<JobEvent>` |
| `artifact(id, name)` | `GET /v1/jobs/{id}/artifacts/{name}` | `Uint8Array` |
| `provenance(id, name)` | the `.provenance.json` sibling | `Provenance` |
| `cancel(id)` | `DELETE /v1/jobs/{id}` | `CancelResult` |

Every authenticated call sends `Authorization: Bearer <token>` and `X-Crucible-Api: 1`.
`ping()` deliberately sends neither, so it can tell "wrong token" from "not a Crucible".

Each job input is either `{ blobId }` (from `upload`) or `{ inline: Uint8Array }` (the
client base64-encodes it for the wire). Use `upload` for anything large.

### Events

`events()` yields typed events with the server's monotonic `id`:
`queued`, `warming`, `progress {fraction, message}`, `artifact {name}`, `done {artifacts}`,
`failed {error}`, `cancelled {status}`. The iterator ends after the first terminal event
(`done`, `failed`, `cancelled`).

To resume after a dropped connection, pass the last id you saw — the server replays
everything after it, so nothing is lost and nothing is repeated:

```ts
let lastEventId: number | undefined;
try {
  for await (const event of crucible.events(jobId, { lastEventId })) {
    lastEventId = event.id;
    // ...
  }
} catch (error) {
  if (error instanceof CrucibleUnreachable) {
    // reconnect with the same lastEventId
  }
}
```

A stream that closes *without* a terminal event throws `CrucibleUnreachable`: "the job
finished" and "the socket died" must never look the same to a caller.

### Provenance is verbatim

Every other type this client returns is camelCase, because it is a TypeScript API.
`Provenance` is the exception: it is the sidecar document as the server wrote it
(`job_type`, not `jobType`). You are meant to persist that file beside the artifact —
a finished audiobook says which server rendered it — and rewriting its keys would
corrupt the thing being persisted.

## Errors

No call ever returns a degraded result, and nothing is retried. Each failure has its own
type, carrying the server's own `code` and `message` where the server sent one:

| Error | When |
|---|---|
| `CrucibleConfigError` | a required option is missing or unusable. Names the option. |
| `CrucibleUnreachable` | connection refused, DNS or TLS failure, a socket that died mid-stream |
| `CrucibleNotACrucible` | something answered `/v1/ping` but did not say `{"crucible": true}` |
| `CrucibleAuthError` | 401 — wrong or missing token |
| `CrucibleVersionError` | 426 — carries `serverApiVersion` and `clientApiVersion` |
| `CrucibleRefused` | any other 4xx — carries the named reason (`unknown_job_type`, `unknown_model`, `unknown_blob`, ...) |
| `CrucibleServerError` | 5xx |
| `CrucibleProtocolError` | a response API v1 does not describe: a missing field, an unknown SSE event name |

All of them extend `CrucibleError`.

## Building and testing

```bash
npm ci
npm run build        # dist/esm + dist/cjs + the .d.ts for both
npm run test:unit    # the error map, against a tiny in-process http fixture
```

The test that matters is the end-to-end run against a real server, from the repo root:

```bash
./scripts/e2e.sh                 # Linux / macOS: server and client on this host
./scripts/e2e-from-windows.sh    # Windows (Git Bash): server in WSL2, client native
```

Both exit non-zero on any failure. The e2e suite needs `CRUCIBLE_URL` and
`CRUCIBLE_TOKEN` and **fails by name** if either is missing — it never skips.

### TypeScript note

The emitted `.d.ts` refers to the `Blob` global (the `upload` parameter). Node's
`@types/node`, `bun-types` and the DOM lib all provide it; a project with none of the
three will need one.
