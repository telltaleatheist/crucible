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
| `capability()` | `GET /v1/capability` | `CapabilityRecord` — one verdict per class |
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
| `voices()` | `GET /v1/voices` | `VoiceInfo[]` |
| `loadVoice(id)` | `POST /v1/jobs {type: "load-voice"}` | the job id |
| `unloadVoice(id)` | `POST /v1/jobs {type: "unload-voice"}` | the job id |
| `render(options)` | `POST /v1/jobs {type: "tts"}` | the job id |
| `writeArtifactsTo(id, dir, {...})` | `events()` + the artifact route | `AsyncIterable<ArtifactWrite>` |
| `stream(options)` | `POST /v1/tts/stream` and its three companions | a `TtsStreamSession` |
| `accelerator()` | `GET /v1/accelerator` | `AcceleratorState` |
| `asr(options)` | `POST /v1/jobs {type: "asr"}` | the job id |
| `setup()` | `GET /v1/setup` | `ServerSetup` — urls, token, one pairing line per url |
| `catalog()` | `GET /v1/catalog` | `CatalogRow[]` — every subject this backend can hold |
| `submitTask(request)` | `POST /v1/tasks` | the task id |
| `task(id)` | `GET /v1/tasks/{id}` | `TaskStatus` |
| `tasks()` | `GET /v1/tasks` | `TaskStatus[]`, newest first |
| `taskEvents(id, {lastEventId?})` | `GET /v1/tasks/{id}/events` | `AsyncIterable<TaskEvent>` |
| `cancelTask(id)` | `DELETE /v1/tasks/{id}` | `TaskCancelResult` |
| `parsePairing(line)` | *(pure — no server)* | `{name, url, token}` |

Every authenticated call sends `Authorization: Bearer <token>` and `X-Crucible-Api: 1`.
`ping()` deliberately sends neither, so it can tell "wrong token" from "not a Crucible".

Each job input is either `{ blobId }` (from `upload`) or `{ inline: Uint8Array }` (the
client base64-encodes it for the wire). Use `upload` for anything large.

`info()` answers two different questions with two different lists. `capabilities` is what the
server can **serve** — one entry per capability, each model or voice described once, in one
shape. `jobTypes` is what you may **post**, and it is not the same list: `llm` is a capability
and is not a job type, while `load-model` and `unload-model` are job types and are not
capabilities.

### Events

`events()` yields typed events with the server's monotonic `id`:
`queued {position}`, `warming {message}`, `progress {fraction, message, extra}`,
`chunk {index, seconds, chars, charsPerSec, tokens, capped, take, guard}`, `artifact {name}`,
`done`, `failed {error}`, `cancelled {status}`. The iterator ends after the first terminal
event (`done`, `failed`, `cancelled`).

`progress.extra` is every other key the job type put on that frame, verbatim — server
spelling, server types. A job type may send its own measurements beside the fraction
because a fraction is not always the useful number: `asr` sends
`{stage, processed_s, total_s, cues}`, so a client shows a moving position six minutes into
an eighteen-hour book while the percentage is still rounding to zero. Those keys are one job
type's vocabulary rather than the API's, so they are carried rather than modelled, and
`extra` is `{}` on a frame that had none.

`done` is one record with two optional fields and one that is always there,
`{artifacts?, resident?, extra}`: a producing job (`echo`, `tts`) reports the artifacts it
wrote, and `load-model` reports the model that is now resident. It is a record rather than a
union because there is no discriminant inside the frame — the caller already knows which job
it submitted — and a union would force every existing caller to narrow before reading
`artifacts`. It is still checked, not loose: `artifacts` must be an array of strings and
`resident` a string wherever either appears, and a `done` frame carrying **neither** is a
`CrucibleProtocolError`.

`done.extra` is every other key the job type put on the frame, verbatim, exactly like
`progress.extra` and for the same reason. The server builds the frame as
`{"artifacts": [...], **job.done_extra}` and a job type puts its own terminal news there:
`tts` sends `{rendered, failed: [{index, message}], take, sample_rate}` and `load-voice` sends
a `fingerprint` beside its `resident`. Until 2026-09-13 this client read the two modelled
keys and dropped the rest, which lost a render's authoritative list of which chunks to ask
for again. `readRenderResult(done)` reads a `tts` job's out of it.

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

`modalities` is the exception to that: it is **never null, on any host**. It says what a
client may put in a chat request's content parts (`text`, `image`) — what the model is
offered *for*, which is the same answer on a host whose backend cannot serve it at all. A
page reader picks an image-capable model off this rather than knowing one by name.

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

## tts

A voice is to `tts` what a model is to `llm`, and the shape of the surface is the same:
list them, load one, unload it. One card holds one thing, and since phase 3 that thing may
be a voice or a model — `health().residentKind` is `"llm"`, `"tts"` or `null`, and it is
what tells you which door to knock on.

### `voices()`

`GET /v1/voices`. The same rows the `tts` capability carries in `info()`, from the same
producer, so a voice has one description wherever you find it (narrow with
`isTtsCapability`).

```ts
for (const voice of await crucible.voices()) {
  if (!voice.loadable) {
    console.log(`${voice.id}: ${voice.reason}`);   // always present when it cannot load
    continue;
  }
  console.log(voice.id, voice.fingerprint, voice.maxChars, voice.pace);
}
```

`revision`, `fingerprint`, `memoryBytesEstimate`, `estimateBasis` and `maxChars` are `null`
together when `backendSupported` is false: all five live in the backend block this host does
not have, and `0` would read as "needs nothing" where `""` would read as a pin. `sampleRate`,
`takes` and `pace` are facts about the voice and are never null.

Two things differ from a `ModelInfo` row and both are deliberate:

- **`reason` is always present**, `null` when the voice is loadable — where a model row omits
  the key. The client reads each route as it is rather than making the two look alike. The
  rule that does not differ: a voice that cannot load and does not say why is a
  `CrucibleProtocolError`.
- **`maxChars` is characters, not tokens.** It is *the* cap certificate for this
  (voice, backend): the most text the voice may be handed in one chunk. Nothing in `tts`
  carries a token cap on the wire — the engine derives its frame budget per chunk from the
  text it is actually given.

`pace` is the whole block, because a client that is going to pack needs all of it. The three
rates are always there; the packing shape is one of three arrangements, told apart by which
of the other three are null — a band (`safeMinChars`/`safeMaxChars`), a single `targetChars`,
or neither, which means pack to `maxChars`.

What is **not** on the row is not an omission. Sampling, the EOS levers, the token-budget
formula and the engine flags are engine tuning, they are the server's, and publishing them
would invite a client to send them back.

### `loadVoice()` and `unloadVoice()`

Job ids, watched with `events()` exactly like `loadModel()`: `queued`, a `warming {message}`
per line of the engine's readiness, then `done {resident}` — `null` after an unload. Nothing
loads a voice implicitly anywhere else, and a load is refused by name before it is queued
for every reason a model load is, plus `env_missing` when the tts env for that voice's
narrator engine is not installed.

### `render()`

`POST /v1/jobs {type: "tts"}` — text in, one `<index>.flac` per chunk out. It returns a job
id, like every other queueing call, and `events()` is how you watch it.

```ts
const jobId = await crucible.render({
  voice: 'deathstalker',
  language: 'en',
  take: 0,
  chunks: [
    { index: 41, text: 'He had been walking for some time.' },
    { index: 42, text: 'The road did not appear to end.' },
  ],
});
```

**The voice need not be resident.** A render job owns the exclusive lane for its whole
duration and is an operator's explicit order, so it loads its own voice if it has to,
emitting `warming` as `loadVoice()` does. That is the one asymmetry with `llm`, where a
fine-grained unattended chat never loads.

**Chunking is yours.** Pack to the voice's `pace` and `maxChars` before you get here: a chunk
over the cap is refused (`chunk_too_long`) and never re-split, because a server that quietly
cut a chunk in half would return two files where one was asked for. The client keeps no copy
of `maxChars` — it is per (voice, backend) and it is on the row you already read.

The `index` is yours too, and nothing renumbers it. It goes out, comes back on the retiring
row, and becomes the artifact's name.

### The `chunk` event, and the nulls that are load-bearing

```ts
for await (const event of crucible.events(jobId)) {
  if (event.event !== 'chunk') continue;
  const { index, seconds, chars, charsPerSec, tokens, capped, take, guard } = event.data;
  if (guard !== null) record(index, guard);   // what the ENGINE decided about it
}
```

**The model judges, the server forwards, the client orders** (Owen's ruling of 2026-09-13,
`docs/PHASE6-REMOTE-RENDER.md`, amending PHASE3-TTS.md). The server measures `seconds`,
`chars` and `charsPerSec` and still decides nothing — it never retakes and never re-splits —
but the judging is not yours either. A chunk that arrives has already been through the
engine's own retake ladder and been accepted; `guard` is the verdict it reached, forwarded
verbatim.

This README used to print `if (capped === true) retake(index)` here and call the seven
measured fields "the whole guard interface", with "your PaceTracker is the thing that
judges". Both were wrong in the same way: the PaceTracker was never in this path, and until
narrator grew a guarded batch driver the door a Crucible render drives had no guard at all.
Do not retake on what you read on this event.

**`guard` is an opaque object or `null`, and it is not modelled.** Its contents are
narrator's — today `{verdict, clean, parts, band, takes}`, where `verdict` is the ladder's own
last action and `parts` says whether it split the chunk — and both this client and the server
carry it across unopened, because a schema for it would break the first time the ladder grows
a rung, at the first guard fire on a real book rather than at compile time. Narrow what you
need yourself, and treat a word you do not recognise as news. A `null` guard means narrator
sent **no verdict** and is never to be read as "the take was clean".

**`capped` and `tokens` are `boolean | null` and `number | null`, and `null` means "narrator
did not say" — never `false`, never `0`.** narrator does not put the frame cap on its wire at
the pinned sha, so against a current server `capped` is `null` on *every* chunk. A client
that read that as `false` would report every runaway as a long sentence, which is precisely
the distinction this event exists to carry. Compare against `true` and `false` explicitly;
never write `if (chunk.capped)`.

A `chunk` frame that *omits* `capped` — or `guard` — is a `CrucibleProtocolError`, not a null:
"narrator did not say" has to be something the server said.

No `chunk` event and no artifact is produced for a row that rendered nothing. **A failed
chunk is reported and the run continues** — one bad sentence never sinks the other 1,399 —
so a job can finish `done` with failures in it:

```ts
const result = readRenderResult(doneEvent.data);
for (const { index, message } of result.failed) askAgainFor(index, message);
```

### `writeArtifactsTo()`

`events()`, plus the batch writer: every artifact the job publishes is fetched and written
into `dir`, with its provenance sidecar beside it, while the job is still running.

```ts
for await (const entry of crucible.writeArtifactsTo(jobId, 'Z:\\books\\mutineer\\sentences')) {
  if (entry.kind === 'event') watch(entry.event);          // nothing is swallowed
  else console.log('wrote', entry.written.path, entry.written.bytes);
}
```

It exists because **there is no shared mount, ever.** The library lives on `Z:`, WSL cannot
mount a network drive, and that single fact is why whole-m4b alignment cannot run on Owen's
PC today. So Crucible writes on its own host, the client fetches over HTTP — even from a
server on localhost — and writes where assembly and resume already look.

- Each artifact is fetched **as its `artifact` event lands**, overlapped with the next chunk
  still generating. A writer that waited for `done` would turn a streaming server back into a
  batch one.
- Each file lands **atomically**: bytes to a sibling temporary, then a rename. BookForge's
  resume test is "the file exists and exceeds 1024 bytes", so a half-written FLAC reads as a
  finished chunk and that sentence is silently missing from the book.
- The **sidecar is written first**, so the existence of `<index>.flac` implies the existence
  of `<index>.flac.provenance.json`. Its bytes are the server's own, not a re-serialisation.
- Fetches are bounded (default 4, `concurrency` to change it). `events()` replays a job's
  whole history first, so attaching to a nearly-finished 1,400-chunk render would otherwise
  open 2,800 sockets in a tick.
- A **write** that fails throws out of the iterator. A failed *chunk* is the job's ordinary
  news; a failed *write* means this client cannot do the one thing it was asked to do.
- Resuming with `lastEventId` writes only what arrives after it. Without one, the whole
  history replays and `done`'s `artifacts` list is reconciled against what was seen.

It needs a Node-like runtime and loads `node:fs/promises` lazily, from a specifier assembled
at run time — see "TypeScript note" below. Nothing else in this client touches the
filesystem.

### `stream()` — the live session

`render()` is the book; this is the Listen path, the Play button and the browser extension.
They are different in kind rather than in buffer size: sub-sentence audio emitted while a row
is still generating, rows retiring out of order, and a cancel that aborts work in flight.

```ts
const session = await crucible.stream({ voice: 'deathstalker', language: 'en' });

void session.say('r1', 'He had been walking for some time.');
void session.say('r2', 'The road did not appear to end.');

for await (const event of session) {
  if (event.kind === 'audio') speaker.write(event.pcm);        // Int16Array, mono, 24 kHz
  else if (event.kind === 'restart') discardBelow(event.id, event.fromSeq);
  else if (event.kind === 'done') console.log(event.id, event.seconds, event.cancelled);
  else if (event.kind === 'error') askAgainFor(event.id);
}
```

**The voice must already be resident.** This door never loads one — it behaves like chat, not
like a render job — and refuses `voice_not_resident` naming what *is* resident. Call
`loadVoice()` first.

**One session per server.** A second is refused `stream_session_open`, and a `render()`
submitted while one is open is refused `engine_in_use`: narrator has one stdin and one
stdout, and two conversations on it read each other's replies.

**`say` answers with the row's id, not the audio.** The audio comes out of the iterator.

**`stream()` resolves attached.** The server refuses a `say` on a session whose event stream
has never been opened (`stream_not_attached` — a row said into nothing has nowhere for its
audio to go), so the client opens the stream and reads the server's `ready` frame *before*
`stream()` resolves. The example above is exactly the order it may be called in: the first
`say` is never early, and there is no second call to make and no signal to wait for. Audio
for a row said before the loop begins waits in the stream and comes out of the first
iteration; breaking out of a `for await` detaches nothing, and a later loop resumes where the
last one stopped. The `ready` frame is also checked, field by field, against what the open
reply said — a stream announcing another voice, merge, sample rate or backend is a protocol
error, not a session. (Until 2026-09-14 the stream attached lazily on the first iteration and
a `say` before it was refused, which is what BookForge's polling stopgap was for.)

**`cancel(id)` tells you what it cost.** `dropped` — the row had not reached the engine, so
nothing was lost. `aborting_batch` — the engine is generating it, and narrator has no per-row
cancel, so its whole batch is thrown away to stop it. `already_finished` — it retired first,
which is the ordinary race on a live connection and not an error.

The rows that die alongside a cancelled one come back on their own, behind a **`restart`**:

```ts
if (event.kind === 'restart') {
  // Everything already received for event.id below event.fromSeq is void.
  // seq never restarts, so the good audio is everything at or above it.
}
```

On a `higgs-v3` voice — which is every voice that ships, and the only narrator engine a
Crucible names — the batch width is 1, the in-flight row *is* the batch, and `restart` never
fires. On an engine whose ramp dispatches N rows, up to N-1 of them regenerate.

**A dropped connection is reattached for you**, with `Last-Event-ID`, for as long as the
server's 15-second grace window could still be open. That is the one thing this SDK does that
`events()` deliberately does not, and the reason is the difference between the two: a job runs
on whether anybody is watching, while a session that misses its window loses the rows in
flight and the listener's place in the paragraph. A **refusal** is never retried —
`unknown_session` and `replay_unavailable` are the server saying the window closed, or that
replaying would hand you audio with a hole in it, and both are yours to see.

`close()` ends the session and frees the voice; the iterator ends on the server's `closed`
frame either way.

## `capability()`

`GET /v1/capability` — what this server can hold, per capability class, and why not. The
read to make *before* deciding what to ask for: phase 9 made the act-to-model mapping a
per-host fact, so a 24 GB box serves `translate` with a 4-bit 27B and a 12 GB box does not
serve it at all, and a client handed a model id by configuration is carrying one this server
may have refused.

```ts
const record = await crucible.capability();
const translate = record.classes.find((row) => row.capability === 'translate')!;
if (translate.enabled) submitWith(translate.selected);   // the id this host picked
else explain(translate.reason, translate.shortfallBytes); // "cannot", with the number
```

- **`enabled: false` is an answer, not an error.** A server that cannot translate says so
  with the shortfall that decided it. `selected` is `''` and `shortfallBytes` is `0` where
  they do not apply — the server's config is TOML, which has no null — so branch on
  `enabled`, never on the emptiness of `selected`.
- **it is a record, not an authority.** `[jobs] enable_*` stays the one owner of what the
  server offers; this says what the numbers were when somebody decided. `totalBytes` is the
  card the decision was made on, which is how a swapped GPU is noticed.
- **a server that has decided nothing throws** `CrucibleCapabilityUndecided` (503
  `capability_undecided`) rather than answering with empty rows, which would read as
  "probed, and nothing fit". The fix is the operator's — `crucible capability --write` — and
  `info()` still says what the server offers meanwhile. It extends `CrucibleServerError`, so
  an existing 5xx handler still catches it.

## `accelerator()`

`GET /v1/accelerator` — what is on the card right now, and which of it is Crucible's own.
It is the `nvidia-smi --query-compute-apps` the load guard runs, plus the free and total
figures, plus what Crucible has resident, plus a flag per holder saying whether that pid is
one of this server's engines. **It reports and it never evicts.**

```ts
const state = await crucible.accelerator();
```

Three things to read carefully before concluding the card is free:

- **`holders[].bytes` is `number | null`,** and the null is the driver declining to answer
  (WDDM, permissions) — *not zero*. Render it as 0 and a queue is told a process holding
  8 GB is holding none.
- **an empty `holders` is not an idle card either.** Under WSL2 the driver shim answers the
  compute-app query with an empty list while a process inside that VM holds 17 GB, which is
  why `unattributedBytes` exists: VRAM in use that no listed holder accounts for, past the
  declared desktop allowance. It is `null` on `mlx-darwin`, where the question cannot be
  asked, and never negative — the server clamps it, because "VRAM nothing accounts for" cannot
  be less than none.
- **a probe that cannot read the card throws** `CrucibleAcceleratorUnreadable` (503
  `accelerator_unreadable`) rather than returning zeroes. That distinction is the whole point
  of the route: a client polling for a free GPU reads it as *ask again*, never as *it is
  free*. It extends `CrucibleServerError`, so an existing 5xx handler still catches it.

## `asr()`

One audio file in, one transcript out. Returns a job id; the transcript arrives as the
artifact `transcript.json`.

```ts
const { blobId } = await crucible.upload(bytes, { filename: 'book.m4b' });
const jobId = await crucible.asr({
  model: 'faster-whisper-base',
  audio: { blobId },
  filename: 'book.m4b',
  language: 'en',
  vadFilter: true,
  wordTimestamps: true,
});
```

**Every field is required and this client supplies none of them.** There is no default model
— an ASR pass at the wrong size is a transcript that looks fine, is worse, and has nothing in
it to say so — and no default for either switch, because a transcript quietly produced under
rules the caller did not choose looks exactly like one produced under the rules they did.

`language` is a faster-whisper code or the literal `"auto"`, which is a *value* meaning
"detect it" rather than an absence. The client does not keep its own copy of the code list:
the server checks against the tokenizer's own and refuses naming the code, before the job is
queued.

`filename` names the file on the server's disk, and **the extension is load-bearing** —
ffmpeg reads the container from it.

Progress arrives as `progress {fraction, message, extra}` with
`extra = {stage, processed_s, total_s, cues}`. The decode drives no fraction at all: it is
real work with a real position, but none of the transcript exists yet.

A failed window fails the job, naming every bad stretch, and publishes nothing — a
fifteen-minute hole in the middle of a transcript looks exactly like a transcript without one.

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
| `CrucibleAcceleratorUnreadable` | 503 `accelerator_unreadable` — a `CrucibleServerError` with a narrower name, because "I cannot see the card" must never be read as "the card is free" |
| `CrucibleProtocolError` | a response API v1 does not describe: a missing field, an unknown SSE event name |

All of them extend `CrucibleError`.

## Building and testing

```bash
npm ci
npm run build        # dist/esm + dist/cjs + the .d.ts for both
npm run test:unit    # every reader and every refusal, against in-process http fixtures
```

The unit suite needs **no server**. It answers each route from a `node:http` fixture, which
is how it covers the cases a healthy Crucible never produces on a good day: a 503 that must
not read as an idle card, a holder whose memory the driver would not report, a voice row
that refuses to load and does not say why, an SSE stream that stops without a terminal
event, a `chunk` frame whose `capped` is `null` (which is what the pinned narrator sends for
every chunk), a `chunk` frame carrying an engine verdict whose vocabulary this client has
never heard of (which must survive unchanged) and one carrying none at all, an artifact fetch
that fails after its sidecar arrived, and every option this client refuses by name before it
sends anything.

Two of the render tests are timing proofs rather than shape proofs. "Fetches each artifact as
its event lands" **gates the SSE stream on the artifact GET arriving**, so a writer that
waited for `done` deadlocks it rather than failing an assertion — the honest shape of that
failure — which is why it carries a timeout. The concurrency-ceiling test holds each artifact
response open for a few milliseconds so that overlap is measured rather than inferred. Both
were mutation-tested: breaking the behaviour fails the test.

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

### Bundler note: the one place this touches `node:fs`

`writeArtifactsTo()` writes files, and it loads `node:fs/promises` and `node:path` from a
specifier **assembled at run time**, inside that method, the first time it is called.

Node's builtins are not a dependency in the sense the zero-dependency rule means — nothing is
installed to get them — but a *static* `import ... from 'node:fs/promises'` would put fs into
the module graph of `import {CrucibleClient}` itself, and a bundler targeting a browser-ish
runtime resolves that specifier at build time and fails on it. Building it at run time keeps
it out of static analysis, so a bundle that never calls `writeArtifactsTo()` never resolves
it. webpack will warn about an expression as a dependency; that warning is the mechanism
working, and `/* webpackIgnore: true */` is on the import for it.

The cost is that a dynamic `import()` of a non-literal hands back `any`, so the four fs calls
and the one path call the writer makes are declared as an interface and checked at the seam.
A runtime that has neither module gets a `CrucibleError` naming what is missing and what to
do instead (fetch each artifact with `artifact(jobId, name)`), not a `TypeError` about
`undefined`.
