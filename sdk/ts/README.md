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
| `models()` | `GET /v1/models` | `ModelInfo[]` |
| `loadModel(id)` | `POST /v1/jobs {type: "load-model"}` | the job id |
| `unloadModel(id)` | `POST /v1/jobs {type: "unload-model"}` | the job id |
| `chat(options)` | `POST /v1/openai/chat/completions` | `ChatResponse` |
| `chatStream(options)` | the same, streamed | `AsyncIterable<string>` of content deltas |

Every authenticated call sends `Authorization: Bearer <token>` and `X-Crucible-Api: 1`.
`ping()` deliberately sends neither, so it can tell "wrong token" from "not a Crucible".

Each job input is either `{ blobId }` (from `upload`) or `{ inline: Uint8Array }` (the
client base64-encodes it for the wire). Use `upload` for anything large.

### Events

`events()` yields typed events with the server's monotonic `id`:
`queued {position}`, `warming {message}`, `progress {fraction, message}`, `artifact {name}`,
`done`, `failed {error}`, `cancelled {status}`. The iterator ends after the first terminal
event (`done`, `failed`, `cancelled`).

`done` is one record with two optional fields, `{artifacts?, resident?}`: a producing job
(`echo`, later `tts`) reports the artifacts it wrote, and `load-model` reports the model
that is now resident. It is a record rather than a union because there is no discriminant
inside the frame — the caller already knows which job it submitted — and a union would
force every existing caller to narrow before reading `artifacts`. It is still checked, not
loose: `artifacts` must be an array of strings and `resident` a string wherever either
appears, and a `done` frame carrying **neither** is a `CrucibleProtocolError`.

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

## llm

One model is resident at a time. Loading is a job you watch; chatting is a request that
is refused — never satisfied by a silent load — if you name a model that is not the
resident one.

```ts
import { CrucibleClient, CrucibleRefused } from '@crucible/client';

const crucible = new CrucibleClient({ url, token, clientName: 'bookforge' });

for (const model of await crucible.models()) {
  console.log(model.id, model.installed, model.resident, model.loadable, model.reason);
}

const loading = await crucible.loadModel('qwen3.5-9b');
for await (const event of crucible.events(loading)) {
  if (event.event === 'warming') console.log(event.data.message);   // the engine warming up
  if (event.event === 'done') console.log('resident:', event.data.resident);
  if (event.event === 'failed') throw new Error(event.data.error.code);
}

const answer = await crucible.chat({
  model: 'qwen3.5-9b',
  messages: [{ role: 'system', content: 'Be terse.' }, { role: 'user', content: 'Light it.' }],
  temperature: 0.2,
});
console.log(answer.content, answer.finishReason, answer.usage.totalTokens);

for await (const delta of crucible.chatStream({ model: 'qwen3.5-9b', messages: [...] })) {
  process.stdout.write(delta);          // ends on OpenAI's [DONE]
}

await crucible.unloadModel('qwen3.5-9b');   // a job too; watch it the same way
```

### `models()`

`ModelInfo` carries four booleans that are four different facts, and none of them implies
another: `backendSupported` (the manifest has a block for this host's backend),
`installed` (the weights are on disk), `resident` (an engine is serving it right now), and
`loadable` (asking for it now would succeed). The last also depends on the accelerator
guard, so a model can be installed and supported and still not loadable because someone
else's process holds the card. A model that is **not** loadable always carries `reason` in
the server's own words; a row that says `loadable: false` and gives no reason is a
`CrucibleProtocolError`, because an operator cannot act on a refusal with no cause.

The rest is the manifest: `id`, `family`, `paramsB`, `memoryBytesEstimate` (weights plus KV
at the default context, measured on the host, not guessed) and two numbers about context
that are not the same number. `contextDefault` is the manifest's intent — what this host
would serve the model at. `maxModelLen` is what is being served **right now**, which for
the resident model is the context its engine was actually started with. Size a request
against `maxModelLen`: it is what the engine measures your prompt plus `maxTokens` against.

`fingerprint` is `<id>@<revision>` and it is what to write down. A model id alone does not
identify weights — the same id serves a different repo on each backend, and a manifest can
be re-pinned — so an id in a record cannot say afterwards what actually produced the
output. The server assembles the string so every client files the same weights under the
same name. `fingerprint`, `revision`, `memoryBytesEstimate` and `maxModelLen` are all
`null` together on a model this host's backend cannot serve.

### `loadModel()` and `unloadModel()`

Both return a job id; watch it with `events()` like any other job. A load streams
`queued`, then a `warming {message}` per line of the engine's own readiness, then
`done {resident}`. The server refuses before queuing, by name, when the model is unknown,
not installed, unsupported on this backend, larger than the free VRAM, or the card is busy
with work Crucible does not own — `unknown_model`, `model_not_installed`,
`backend_unsupported`, `insufficient_memory`, `accelerator_busy`, `env_missing`. Nothing is
ever evicted to make room.

### `chat()` and `chatStream()`

Both take `{model, messages, temperature?, topP?, maxTokens?, stop?, seed?,
responseFormat?, thinking?, signal?}` and post to `/v1/openai/chat/completions` with the
bearer and `X-Crucible-Api` headers, like every other authenticated call. `model` and
`messages` are required and refused by name when missing; every other knob is simply left
out of the body when you do not pass it, so the engine's own default applies and this
client invents nothing.

`chat()` returns `ChatResponse {id, model, content, finishReason, usage: {promptTokens,
completionTokens, totalTokens}}` — OpenAI's `chat.completion` read down to the parts a
caller uses, from the first (and only) choice.

`finishReason` is the engine's own word, surfaced and never normalised — a plain string, so
a value this client did not anticipate still reaches you. `length` means the answer is
truncated; check it before you use `content`.

### Structured output

`responseFormat` is OpenAI's `response_format`, forwarded to the engine exactly as given:

```ts
const verdict = await crucible.chat({
  model: 'qwen3.5-9b',
  messages: [{ role: 'user', content: passage }],
  temperature: 0,
  maxTokens: 128,
  thinking: false,              // a grammar on a thinking model fills `reasoning` instead
  responseFormat: {
    type: 'json_schema',
    json_schema: {
      name: 'verdict',
      strict: true,
      schema: { type: 'object', properties: { supported: { type: 'boolean' } },
                required: ['supported'], additionalProperties: false },
    },
  },
});
if (verdict.finishReason === 'length') throw new Error('truncated, not malformed');
const answer = JSON.parse(verdict.content);
```

The `schema` is yours. This client checks only what a typo makes an engine answer plausibly
and wrongly — a `type` outside `text | json_object | json_schema`, or a `json_schema`
without a `name` or a `schema` — and reads nothing inside the grammar. Which dialect of
JSON Schema an engine supports is the engine's to accept or refuse, and a refusal arrives
as that engine's own 400, relayed with the message naming the part to fix.

`seed` is the engine's sampling seed: the same seed and the same sampling give the same
answer from the same engine, and say nothing across engines or backends.

`chatStream()` is an `AsyncIterable<string>` of the content deltas in order; concatenating
everything it yields gives the text `chat()` would have returned. Chunks that carry no
content — the opening frame with only a `role`, the closing one with only a
`finish_reason`, a usage-only trailer with no choice at all — yield nothing rather than an
empty string sentinel. The iterator ends on OpenAI's `data: [DONE]`; a stream that ends
*without* one throws `CrucibleUnreachable`, because a truncated answer and a finished
answer must never look the same.

**Naming a model that is not resident** is a 409 and surfaces as `CrucibleRefused` with
`code === 'model_not_resident'`, its `serverMessage` naming what is resident instead:

```ts
try {
  await crucible.chat({ model: 'qwen3.8-27b', messages });
} catch (error) {
  if (error instanceof CrucibleRefused && error.code === 'model_not_resident') {
    // load it first, or go to the host that has it. Never retried, never auto-loaded.
  }
}
```

**Aborting.** `signal` aborts the request. The rejection is the DOM `AbortError` itself,
thrown straight through — out of `chat()`, and out of the `for await` in `chatStream()` —
never wrapped in `CrucibleUnreachable`: your own cancellation is not a dead server and is
not reported as one. Deltas already yielded before the abort stay yielded.

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

The `llm` surface has its own live suite, which touches the GPU and so is run by hand
against a server that has the job type installed:

```bash
CRUCIBLE_URL=... CRUCIBLE_TOKEN=... CRUCIBLE_LLM_MODEL=qwen3.5-9b npm run test:e2e-llm
```

It loads the model, chats, streams, proves the 409, and unloads. It fails by name if any
of the three variables is missing. Before running it, `nvidia-smi` must show only the
desktop: Crucible never evicts anyone else's work, so a busy card makes the load refuse
with `accelerator_busy` — the contract working, not the test failing.

### TypeScript note

The emitted `.d.ts` refers to the `Blob` global (the `upload` parameter). Node's
`@types/node`, `bun-types` and the DOM lib all provide it; a project with none of the
three will need one.
