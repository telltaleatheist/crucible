/**
 * Unit tests for the render door's client: `render()`, the `chunk` event, the
 * `done` frame's carried extras, and `writeArtifactsTo()` — the batch writer.
 *
 * Against an in-process `node:http` fixture and a real temporary directory,
 * following `test/unit-voices.test.ts`: the SDK's unit suite needs no live
 * server, which is the only way to assert the cases a healthy server never
 * produces on a good day. Here those are a `chunk` frame whose `capped` is null
 * (which is what the *pinned* narrator produces, every time, and which must never
 * read as `false`), an artifact fetch that fails after its sidecar arrived, an
 * artifact the server names `../escape`, and a job whose `done` lists a file no
 * `artifact` event announced.
 *
 * Two of these tests are timing proofs rather than shape proofs, and they are the
 * ones that matter most:
 *
 * - "fetches each artifact as its event lands" gates the SSE stream on the
 *   artifact GET arriving. A writer that waited for `done` would deadlock it
 *   rather than fail an assertion, which is the honest failure for "this turned a
 *   streaming server back into a batch one".
 * - "a failed fetch leaves nothing resume would count" is the reason the writes
 *   are atomic at all: BookForge's resume test is "the file exists and exceeds
 *   1024 bytes", so a half-written FLAC is a sentence silently missing from a
 *   book.
 *
 * Run: `npm run test:unit` (exits non-zero on any failure).
 */

import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, beforeEach, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleConfigError,
  CrucibleProtocolError,
  CrucibleRefused,
  readRenderResult,
  type ArtifactWrite,
  type JobEvent,
} from '../src/index.js';

// ------------------------------------------------------------------ fixture

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

/** What `GET /v1/jobs/{id}/artifacts/{name}` serves, by name. */
const artifacts = new Map<string, Buffer>();
/** Names this fixture answers 404 for, however they are asked for. */
const missingArtifacts = new Set<string>();
/** Every artifact path the client asked for, in order. */
let artifactRequests: string[] = [];
/** Resolved as each artifact GET arrives, so a test can gate the SSE stream on it. */
let onArtifactRequest: (name: string) => void = () => undefined;
/**
 * How long the fixture holds an artifact response open, and the high-water mark
 * of how many it held at once. Zero means answer immediately; a few milliseconds
 * makes overlap observable, which is how the concurrency ceiling is measured
 * rather than inferred.
 */
let holdArtifactsMs = 0;
let inFlightArtifacts = 0;
let peakInFlightArtifacts = 0;

let handle: Handler = (_request, response) => {
  response.writeHead(500, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: { code: 'no_handler', message: 'the test set none' } }));
};

let lastPath = '';
let lastMethod = '';
let lastBody = '';

let server: Server;
let base = '';
let workspace = '';
let directory = '';
let directories = 0;

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-render' });
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

function answers(status: number, body: unknown): void {
  handle = (_request, response) => json(response, status, body);
}

function openSse(response: ServerResponse): void {
  response.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });
}

/** One SSE frame, in the shape the server writes them. */
function frame(id: number, event: string, data: unknown): string {
  return `id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/** Serve these events as the job's stream; serve artifacts from the map. */
function streams(...events: string[]): void {
  handle = (_request, response) => {
    openSse(response);
    for (const event of events) response.write(event);
    response.end();
  };
}

before(async () => {
  server = createServer((request, response) => {
    lastPath = request.url ?? '';
    lastMethod = request.method ?? '';
    const chunks: Buffer[] = [];
    request.on('data', (chunk: Buffer) => chunks.push(chunk));
    request.on('end', () => {
      lastBody = Buffer.concat(chunks).toString('utf8');
      // Artifacts are served by the fixture itself rather than by `handle`, so a
      // test can set one handler for the event stream and still have its files
      // fetched — which is the whole point of the batch writer.
      const artifact = /^\/v1\/jobs\/[^/]+\/artifacts\/(.+)$/.exec(lastPath);
      if (artifact !== null) {
        const name = decodeURIComponent(artifact[1]!);
        artifactRequests.push(name);
        inFlightArtifacts += 1;
        peakInFlightArtifacts = Math.max(peakInFlightArtifacts, inFlightArtifacts);
        onArtifactRequest(name);
        const send = (): void => {
          inFlightArtifacts -= 1;
          const bytes = artifacts.get(name);
          if (bytes === undefined || missingArtifacts.has(name)) {
            json(response, 404, {
              error: { code: 'unknown_artifact', message: `no artifact ${name}` },
            });
            return;
          }
          response.writeHead(200, { 'Content-Type': 'application/octet-stream' });
          response.end(bytes);
        };
        if (holdArtifactsMs > 0) setTimeout(send, holdArtifactsMs);
        else send();
        return;
      }
      handle(request, response, lastBody);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  workspace = await mkdtemp(join(tmpdir(), 'crucible-sdk-render-'));
});

after(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
  await rm(workspace, { recursive: true, force: true });
});

beforeEach(() => {
  artifacts.clear();
  missingArtifacts.clear();
  artifactRequests = [];
  onArtifactRequest = () => undefined;
  holdArtifactsMs = 0;
  inFlightArtifacts = 0;
  peakInFlightArtifacts = 0;
  directories += 1;
  directory = join(workspace, `run-${directories}`);
});

// --------------------------------------------------------------- provenance

const PROVENANCE = {
  server: { name: 'crucible@owens-pc', version: '0.2.0' },
  backend: 'cuda-linux',
  job_type: 'tts',
  model: {
    id: 'deathstalker',
    revision: '9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c',
    fingerprint: 'deathstalker@9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c',
  },
  params: { language: 'en', take: 0 },
  started: '2026-09-13T04:11:02Z',
  finished: '2026-09-13T04:19:44Z',
};

/** The server writes its sidecar with `indent=2` and a trailing newline. */
const PROVENANCE_BYTES = Buffer.from(`${JSON.stringify(PROVENANCE, null, 2)}\n`, 'utf8');

/** A believable FLAC: only its length and its bytes matter to this client. */
function flac(index: number, size = 4096): Buffer {
  const bytes = Buffer.alloc(size, index % 251);
  bytes.write('fLaC', 0, 'ascii');
  return bytes;
}

/** Publish `<index>.flac` and its sidecar on the fixture. */
function publish(index: number, size?: number): Buffer {
  const bytes = flac(index, size);
  artifacts.set(`${index}.flac`, bytes);
  artifacts.set(`${index}.flac.provenance.json`, PROVENANCE_BYTES);
  return bytes;
}

// ------------------------------------------------------------------ render

const CHUNKS = [
  { index: 41, text: 'He had been walking for some time.' },
  { index: 42, text: 'The road did not appear to end.' },
];

test('render() posts the tts job the contract describes', async () => {
  answers(200, { job_id: 'job-tts-1' });
  const jobId = await client().render({
    voice: 'deathstalker',
    language: 'en',
    take: 0,
    chunks: CHUNKS,
  });

  assert.equal(jobId, 'job-tts-1');
  assert.equal(lastMethod, 'POST');
  assert.equal(lastPath, '/v1/jobs');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'tts',
    // For tts the model IS the voice: a Higgs v3 voice is the merged checkpoint
    // the engine runs, so the wire needs no second word for it.
    model: 'deathstalker',
    params: {
      language: 'en',
      take: 0,
      // Spelled `index`, which is the only key TtsChunk accepts — it forbids
      // extras, and narrator's own `i` never appears on this wire.
      chunks: [
        { index: 41, text: 'He had been walking for some time.' },
        { index: 42, text: 'The road did not appear to end.' },
      ],
    },
    inputs: {},
  });
});

test('render() returns a job id, and nothing invents a second way to watch one', async () => {
  // The same shape as loadModel/loadVoice/asr: queueing returns an id and
  // `events()` is how a job is watched. A handle with its own iterator would be
  // a second clock on the same stream.
  answers(200, { job_id: 'job-tts-2' });
  const returned = await client().render({
    voice: 'deathstalker',
    language: 'en',
    take: 0,
    chunks: CHUNKS,
  });
  assert.equal(typeof returned, 'string');
});

test('every render option is required and is refused by name', async () => {
  const complete = { voice: 'deathstalker', language: 'en', take: 0, chunks: CHUNKS };
  for (const option of ['voice', 'language', 'take', 'chunks'] as const) {
    const { [option]: _dropped, ...missing } = complete;
    await assert.rejects(
      client().render(missing as never),
      (error: unknown) => {
        assert.ok(error instanceof CrucibleConfigError, `${option}: got ${String(error)}`);
        assert.equal(error.option, option);
        return true;
      },
      `expected ${option} to be refused by name`,
    );
  }
});

test('render() refuses two chunks sharing an index, and names both', async () => {
  // An index is an artifact name. Two chunks sharing one would be two renders
  // writing the same FLAC on the server and two writes racing for the same path
  // in the caller's library.
  await assert.rejects(
    client().render({
      voice: 'deathstalker',
      language: 'en',
      take: 0,
      chunks: [
        { index: 41, text: 'first' },
        { index: 42, text: 'second' },
        { index: 41, text: 'third' },
      ],
    }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
      assert.equal(error.option, 'chunks[2].index');
      assert.match(error.message, /chunks\[0\] already claimed/);
      return true;
    },
  );
});

test('render() refuses a blank chunk, which would end the whole batch', async () => {
  // narrator answers an empty generate with a WHOLE-REQUEST error, so one blank
  // chunk takes the other 1,399 with it.
  await assert.rejects(
    client().render({
      voice: 'deathstalker',
      language: 'en',
      take: 0,
      chunks: [{ index: 41, text: '   ' }],
    }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
      assert.equal(error.option, 'chunks[0].text');
      assert.match(error.message, /whole-request error/);
      return true;
    },
  );
});

test('render() refuses an empty chunk list rather than queueing a job with no work', async () => {
  await assert.rejects(
    client().render({ voice: 'deathstalker', language: 'en', take: 0, chunks: [] }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
      assert.equal(error.option, 'chunks');
      return true;
    },
  );
});

test('a take or an index that is not a whole count is refused, never rounded', async () => {
  for (const [option, options] of [
    ['take', { voice: 'v', language: 'en', take: -1, chunks: CHUNKS }],
    ['take', { voice: 'v', language: 'en', take: 1.5, chunks: CHUNKS }],
    ['chunks[0].index', { voice: 'v', language: 'en', take: 0, chunks: [{ index: -3, text: 'x' }] }],
  ] as const) {
    await assert.rejects(client().render(options as never), (error: unknown) => {
      assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
      assert.equal(error.option, option);
      assert.match(error.message, /non-negative integer/);
      return true;
    });
  }
});

test('render() keeps no copy of max_chars; the voice row is the authority', async () => {
  // The cap is per (voice, backend) and rides on the voice row. A second copy in
  // the client is a second thing to drift — the same reason asr() keeps no copy
  // of faster-whisper's language list. So an over-long chunk goes out and comes
  // back as the server's own refusal, naming the index and the cap.
  answers(400, {
    error: {
      code: 'chunk_too_long',
      message:
        "1 chunk(s) are longer than the 800-character cap for 'deathstalker' on " +
        'cuda-linux: index 41 is 1200. Chunking is the client\'s, so this is a ' +
        'refusal and not a re-split',
    },
  });
  await assert.rejects(
    client().render({
      voice: 'deathstalker',
      language: 'en',
      take: 0,
      chunks: [{ index: 41, text: 'x'.repeat(1200) }],
    }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
      assert.equal(error.code, 'chunk_too_long');
      return true;
    },
  );
  assert.match(lastBody, /"index":41/, 'the chunk went out unaltered');
});

test("render()'s signal aborts the submit, and the abort is the caller's own error", async () => {
  // A 1,400-chunk book is a large POST. Aborting it is not a dead server and
  // must not be reported as one.
  handle = () => undefined; // never answers
  const controller = new AbortController();
  const pending = client().render({
    voice: 'deathstalker',
    language: 'en',
    take: 0,
    chunks: CHUNKS,
    signal: controller.signal,
  });
  controller.abort();
  await assert.rejects(pending, (error: unknown) => {
    assert.equal((error as Error).name, 'AbortError', `got ${String(error)}`);
    return true;
  });
});

// ------------------------------------------------------------ chunk events

const CHUNK_FRAME = {
  index: 41,
  seconds: 2.04,
  chars: 33,
  chars_per_sec: 16.176470588235293,
  tokens: null,
  capped: null,
  take: 0,
};

async function drain(jobId: string): Promise<JobEvent[]> {
  const events: JobEvent[] = [];
  for await (const event of client().events(jobId)) events.push(event);
  return events;
}

test('a chunk event is typed, narrowable, and read with the server spelling', async () => {
  streams(
    frame(1, 'chunk', CHUNK_FRAME),
    frame(2, 'done', { artifacts: ['41.flac'], rendered: 1, failed: [], take: 0, sample_rate: 24000 }),
  );
  const events = await drain('job-tts-1');
  const first = events[0]!;
  assert.equal(first.event, 'chunk');
  assert.deepEqual(first.event === 'chunk' ? first.data : null, {
    index: 41,
    seconds: 2.04,
    chars: 33,
    // camelCase here, `chars_per_sec` on the wire: everything this client models
    // is camelCase, and only Provenance keeps the server's spelling.
    charsPerSec: 16.176470588235293,
    tokens: null,
    capped: null,
    take: 0,
  });
});

test('capped: null stays null, and is never softened into false', async () => {
  // This is what the PINNED narrator sends for every single chunk: the frame cap
  // it computed never leaves the engine. A client that read this as `false` would
  // report every runaway as a long sentence, which is exactly what the field
  // exists to prevent — and it would do it silently, on every book.
  streams(
    frame(1, 'chunk', { ...CHUNK_FRAME, tokens: null, capped: null }),
    frame(2, 'done', { artifacts: ['41.flac'], rendered: 1, failed: [], take: 0, sample_rate: 24000 }),
  );
  const events = await drain('job-tts-1');
  const chunk = events[0]!;
  assert.ok(chunk.event === 'chunk');
  assert.equal(chunk.data.capped, null);
  assert.notEqual(chunk.data.capped, false);
  assert.equal(chunk.data.tokens, null);
  assert.notEqual(chunk.data.tokens, 0);
});

test('capped: false and capped: true both survive, and are different news', async () => {
  streams(
    frame(1, 'chunk', { ...CHUNK_FRAME, index: 41, tokens: 120, capped: false }),
    frame(2, 'chunk', { ...CHUNK_FRAME, index: 42, tokens: 900, capped: true }),
    frame(3, 'done', { artifacts: ['41.flac', '42.flac'], rendered: 2, failed: [], take: 0, sample_rate: 24000 }),
  );
  const events = await drain('job-tts-1');
  const finished = events[0]!;
  const runaway = events[1]!;
  assert.ok(finished.event === 'chunk' && runaway.event === 'chunk');
  assert.equal(finished.data.capped, false);
  assert.equal(finished.data.tokens, 120);
  assert.equal(runaway.data.capped, true);
  assert.equal(runaway.data.tokens, 900);
});

test('a chunk frame with no capped key at all is a protocol error, not a null', async () => {
  // `null` is narrator declining to say; an absent key is a server that does not
  // speak the field. Only the first is something the server stated, and this
  // client will not infer the statement from an absence.
  const { capped: _capped, ...withoutCapped } = CHUNK_FRAME;
  streams(frame(1, 'chunk', withoutCapped));
  await assert.rejects(drain('job-tts-1'), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /event 1 \(chunk\) has no field "capped"/);
    return true;
  });
});

test('a chunk frame whose capped is neither a boolean nor null is a protocol error', async () => {
  streams(frame(1, 'chunk', { ...CHUNK_FRAME, capped: 'maybe' }));
  await assert.rejects(drain('job-tts-1'), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /capped is neither a boolean nor null/);
    return true;
  });
});

test('a chunk frame missing a measurement is a protocol error, not a zero', async () => {
  const { chars_per_sec: _rate, ...withoutRate } = CHUNK_FRAME;
  streams(frame(1, 'chunk', withoutRate));
  await assert.rejects(drain('job-tts-1'), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /has no field "chars_per_sec"/);
    return true;
  });
});

// --------------------------------------------------------- done, and extras

const RENDER_DONE = {
  artifacts: ['41.flac'],
  rendered: 1,
  failed: [{ index: 42, message: 'No audio generated' }],
  take: 0,
  sample_rate: 24000,
};

test("a done frame's own terminal news reaches the caller instead of being dropped", async () => {
  // Until 2026-09-13 this client read `artifacts` and `resident` and silently
  // discarded the rest, which lost the render's authoritative failure list.
  streams(frame(1, 'done', RENDER_DONE));
  const events = await drain('job-tts-1');
  const done = events[0]!;
  assert.ok(done.event === 'done');
  assert.deepEqual(done.data.artifacts, ['41.flac']);
  assert.deepEqual(done.data.extra, {
    rendered: 1,
    failed: [{ index: 42, message: 'No audio generated' }],
    take: 0,
    sample_rate: 24000,
  });
});

test("load-voice's fingerprint survives in done.extra beside its resident", async () => {
  // Which merge of a fine-tune is on the card, which is a different fact from
  // which fine-tune. It was being dropped by the same bug.
  streams(
    frame(1, 'done', {
      artifacts: [],
      resident: 'deathstalker',
      fingerprint: 'deathstalker@9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c',
    }),
  );
  const events = await drain('job-load-voice-1');
  const done = events[0]!;
  assert.ok(done.event === 'done');
  assert.equal(done.data.resident, 'deathstalker');
  assert.deepEqual(done.data.extra, {
    fingerprint: 'deathstalker@9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c',
  });
});

test('a done frame with only the modelled keys carries an empty extra', async () => {
  // `{}` is the true answer to "what else was on the frame", not a default.
  streams(frame(1, 'done', { artifacts: ['transcript.json'] }));
  const events = await drain('job-asr-1');
  const done = events[0]!;
  assert.ok(done.event === 'done');
  assert.deepEqual(done.data.extra, {});
});

test('readRenderResult reads the authoritative list of chunks to ask for again', async () => {
  streams(frame(1, 'done', RENDER_DONE));
  const events = await drain('job-tts-1');
  const done = events[0]!;
  assert.ok(done.event === 'done');
  assert.deepEqual(readRenderResult(done.data), {
    rendered: 1,
    // A successful job can still have failures: one bad sentence never sinks the
    // other 1,399, and this is how a client learns which index is missing.
    failed: [{ index: 42, message: 'No audio generated' }],
    take: 0,
    sampleRate: 24000,
    artifacts: ['41.flac'],
  });
});

test('readRenderResult refuses a done frame without a failed list, rather than reading it as clean', async () => {
  const { failed: _failed, ...withoutFailed } = RENDER_DONE;
  streams(frame(1, 'done', withoutFailed));
  const events = await drain('job-tts-1');
  const done = events[0]!;
  assert.ok(done.event === 'done');
  assert.throws(
    () => readRenderResult(done.data),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
      assert.match(error.message, /has no field "failed"/);
      return true;
    },
  );
});

test('readRenderResult on another job type\'s done says which field is not there', async () => {
  streams(frame(1, 'done', { artifacts: [], resident: 'qwen3.5-9b' }));
  const events = await drain('job-load-1');
  const done = events[0]!;
  assert.ok(done.event === 'done');
  assert.throws(() => readRenderResult(done.data), CrucibleProtocolError);
});

// ------------------------------------------------------------ batch writer

async function collect(
  jobId: string,
  dir: string,
  options?: { lastEventId?: number; concurrency?: number },
): Promise<ArtifactWrite[]> {
  const seen: ArtifactWrite[] = [];
  for await (const entry of client().writeArtifactsTo(jobId, dir, options ?? {})) {
    seen.push(entry);
  }
  return seen;
}

test('writeArtifactsTo writes every artifact and its sidecar, and yields both kinds', async () => {
  const first = publish(41);
  const second = publish(42);
  streams(
    frame(1, 'chunk', { ...CHUNK_FRAME, index: 41 }),
    frame(2, 'artifact', { name: '41.flac' }),
    frame(3, 'chunk', { ...CHUNK_FRAME, index: 42 }),
    frame(4, 'artifact', { name: '42.flac' }),
    frame(5, 'done', { artifacts: ['41.flac', '42.flac'], rendered: 2, failed: [], take: 0, sample_rate: 24000 }),
  );

  const seen = await collect('job-tts-1', directory);

  // The job's own events are all there, unchanged: the writer reports its
  // progress without swallowing anything.
  const events = seen.filter((entry) => entry.kind === 'event');
  assert.deepEqual(
    events.map((entry) => (entry.kind === 'event' ? entry.event.event : null)),
    ['chunk', 'artifact', 'chunk', 'artifact', 'done'],
  );

  const written = seen.flatMap((entry) => (entry.kind === 'written' ? [entry.written] : []));
  assert.deepEqual(written.map((one) => one.name).sort(), ['41.flac', '42.flac']);
  assert.equal(written[0]!.provenance.model!.id, 'deathstalker');

  assert.deepEqual(await readFile(join(directory, '41.flac')), first);
  assert.deepEqual(await readFile(join(directory, '42.flac')), second);
  // The sidecar's bytes are the server's own, not a re-serialisation: the
  // document is meant to be persisted, and round-tripping it through this
  // client's reader would rewrite whitespace it did not author.
  assert.deepEqual(await readFile(join(directory, '41.flac.provenance.json')), PROVENANCE_BYTES);
});

test('writeArtifactsTo leaves no temporary behind, only the finished names', async () => {
  publish(41);
  streams(
    frame(1, 'artifact', { name: '41.flac' }),
    frame(2, 'done', { artifacts: ['41.flac'], rendered: 1, failed: [], take: 0, sample_rate: 24000 }),
  );
  await collect('job-tts-1', directory);
  assert.deepEqual((await readdir(directory)).sort(), ['41.flac', '41.flac.provenance.json']);
});

// A timeout, because the failure this test detects is a DEADLOCK rather than a
// bad value: a writer that waited for `done` would never fetch, so the gated
// stream would never reach `done`, and `node --test` imposes no deadline of its
// own. Ten seconds is far past anything an in-process fixture needs.
test('a rerun replaces an artifact that is already on disk', async () => {
  // A retake, or a resumed run that re-asks for a chunk. The rename lands on an
  // existing name, and Node replaces it on Windows as well as on POSIX — no
  // unlink first, which would be a window where neither file exists and where a
  // crash would leave resume with nothing.
  const stale = publish(41, 2048);
  streams(
    frame(1, 'artifact', { name: '41.flac' }),
    frame(2, 'done', { artifacts: ['41.flac'], rendered: 1, failed: [], take: 0, sample_rate: 24000 }),
  );
  await collect('job-tts-1', directory);
  assert.deepEqual(await readFile(join(directory, '41.flac')), stale);

  const fresh = publish(41, 6144);
  assert.notDeepEqual(fresh, stale);
  streams(
    frame(1, 'artifact', { name: '41.flac' }),
    frame(2, 'done', { artifacts: ['41.flac'], rendered: 1, failed: [], take: 0, sample_rate: 24000 }),
  );
  await collect('job-tts-2', directory);
  assert.deepEqual(await readFile(join(directory, '41.flac')), fresh);
  assert.deepEqual((await readdir(directory)).sort(), ['41.flac', '41.flac.provenance.json']);
});

test('writeArtifactsTo fetches each artifact as its event lands, not at done', { timeout: 10_000 }, async () => {
  // The proof that a streaming server stays a streaming one. The stream refuses
  // to emit the next frame until the artifact GET for the previous one has
  // arrived, so a writer that waited for `done` would deadlock here rather than
  // fail an assertion — which is the honest shape of that failure.
  publish(41);
  publish(42);
  handle = (_request, response) => {
    openSse(response);
    const waitFor = (name: string): Promise<void> =>
      new Promise((resolve) => {
        onArtifactRequest = (asked) => {
          if (asked === name) resolve();
        };
      });
    void (async () => {
      const firstFetched = waitFor('41.flac');
      response.write(frame(1, 'artifact', { name: '41.flac' }));
      await firstFetched;
      const secondFetched = waitFor('42.flac');
      response.write(frame(2, 'artifact', { name: '42.flac' }));
      await secondFetched;
      response.write(
        frame(3, 'done', {
          artifacts: ['41.flac', '42.flac'],
          rendered: 2,
          failed: [],
          take: 0,
          sample_rate: 24000,
        }),
      );
      response.end();
    })();
  };

  const seen = await collect('job-tts-1', directory);
  assert.equal(seen.filter((entry) => entry.kind === 'written').length, 2);
  assert.deepEqual(
    artifactRequests.filter((name) => name.endsWith('.flac')),
    ['41.flac', '42.flac'],
  );
});

test('a failed fetch leaves nothing resume would count as a finished chunk', async () => {
  // BookForge's resume test is "the file exists and exceeds 1024 bytes". A
  // partial FLAC passes it, so resume never asks again and the sentence is gone
  // from the book. Nothing is written unless everything arrived.
  publish(41);
  publish(42);
  missingArtifacts.add('42.flac');
  streams(
    frame(1, 'artifact', { name: '41.flac' }),
    frame(2, 'artifact', { name: '42.flac' }),
    frame(3, 'done', { artifacts: ['41.flac', '42.flac'], rendered: 2, failed: [], take: 0, sample_rate: 24000 }),
  );

  await assert.rejects(collect('job-tts-1', directory), (error: unknown) => {
    assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
    assert.equal(error.code, 'unknown_artifact');
    return true;
  });

  const files = await readdir(directory);
  assert.ok(!files.includes('42.flac'), `42.flac must not exist; found ${files.join(', ')}`);
  assert.ok(
    !files.some((name) => name.endsWith('.part')),
    `no temporary may survive; found ${files.join(', ')}`,
  );
});

test('a write failure throws out of the iterator rather than being yielded', async () => {
  // A failed CHUNK is the job's ordinary news and the server already reports it.
  // A failed WRITE means this client cannot do the one thing it was asked to do,
  // and a caller who learned about it from a yielded value would be free to
  // ignore it.
  publish(41);
  missingArtifacts.add('41.flac');
  streams(
    frame(1, 'artifact', { name: '41.flac' }),
    frame(2, 'done', { artifacts: ['41.flac'], rendered: 1, failed: [], take: 0, sample_rate: 24000 }),
  );
  await assert.rejects(collect('job-tts-1', directory), CrucibleRefused);
});

test('a zero-byte artifact is a protocol error, not an empty file on disk', async () => {
  artifacts.set('41.flac', Buffer.alloc(0));
  artifacts.set('41.flac.provenance.json', PROVENANCE_BYTES);
  streams(
    frame(1, 'artifact', { name: '41.flac' }),
    frame(2, 'done', { artifacts: ['41.flac'], rendered: 1, failed: [], take: 0, sample_rate: 24000 }),
  );
  await assert.rejects(collect('job-tts-1', directory), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /zero bytes/);
    return true;
  });
});

test('an artifact name that is not a single path member is refused, never joined', async () => {
  // The name comes off the wire and is about to become a path in a real library
  // directory. The server validates it too; that is not a property of this
  // machine's filesystem.
  for (const name of ['../escape.flac', 'sub/41.flac', '.hidden']) {
    artifacts.set(name, flac(1));
    artifacts.set(`${name}.provenance.json`, PROVENANCE_BYTES);
    streams(
      frame(1, 'artifact', { name }),
      frame(2, 'done', { artifacts: [name], rendered: 1, failed: [], take: 0, sample_rate: 24000 }),
    );
    await assert.rejects(
      collect('job-tts-1', directory),
      (error: unknown) => {
        assert.ok(error instanceof CrucibleProtocolError, `${name}: got ${String(error)}`);
        assert.match(error.message, /not a single path member/);
        return true;
      },
      `expected ${name} to be refused`,
    );
  }
});

test('a sidecar that is not a provenance document is a protocol error, and nothing lands', async () => {
  artifacts.set('41.flac', flac(41));
  artifacts.set('41.flac.provenance.json', Buffer.from('{"server": {"name": "x"}}', 'utf8'));
  streams(
    frame(1, 'artifact', { name: '41.flac' }),
    frame(2, 'done', { artifacts: ['41.flac'], rendered: 1, failed: [], take: 0, sample_rate: 24000 }),
  );
  await assert.rejects(collect('job-tts-1', directory), CrucibleProtocolError);
  const files = await readdir(directory);
  assert.deepEqual(files, [], `nothing may land; found ${files.join(', ')}`);
});

test("writeArtifactsTo reconciles against done's list when it replayed the whole stream", async () => {
  // `done` is the authority on what the job published. With the whole history
  // seen, a name there that produced no `artifact` frame is a gap, and a gap is
  // a book with a hole in it.
  publish(41);
  publish(42);
  streams(
    frame(1, 'artifact', { name: '41.flac' }),
    // 42.flac is announced only by `done`.
    frame(2, 'done', { artifacts: ['41.flac', '42.flac'], rendered: 2, failed: [], take: 0, sample_rate: 24000 }),
  );
  const seen = await collect('job-tts-1', directory);
  const written = seen.flatMap((entry) => (entry.kind === 'written' ? [entry.written.name] : []));
  assert.deepEqual(written.sort(), ['41.flac', '42.flac']);
  assert.deepEqual((await readdir(directory)).sort(), [
    '41.flac',
    '41.flac.provenance.json',
    '42.flac',
    '42.flac.provenance.json',
  ]);
});

test('resuming from an event id writes only what arrives after it', async () => {
  // The prefix the caller chose not to replay is theirs — it is what they
  // already had when they recorded that id. Re-fetching a finished book's worth
  // of FLACs on every reconnect would be a worse bug than the one it guards.
  publish(41);
  publish(42);
  streams(
    frame(2, 'artifact', { name: '42.flac' }),
    frame(3, 'done', { artifacts: ['41.flac', '42.flac'], rendered: 2, failed: [], take: 0, sample_rate: 24000 }),
  );
  const seen = await collect('job-tts-1', directory, { lastEventId: 1 });
  const written = seen.flatMap((entry) => (entry.kind === 'written' ? [entry.written.name] : []));
  assert.deepEqual(written, ['42.flac']);
  assert.deepEqual((await readdir(directory)).sort(), ['42.flac', '42.flac.provenance.json']);
});

test('a replayed artifact event never writes the same file twice', async () => {
  publish(41);
  streams(
    frame(1, 'artifact', { name: '41.flac' }),
    frame(2, 'artifact', { name: '41.flac' }),
    frame(3, 'done', { artifacts: ['41.flac'], rendered: 1, failed: [], take: 0, sample_rate: 24000 }),
  );
  const seen = await collect('job-tts-1', directory);
  assert.equal(seen.filter((entry) => entry.kind === 'written').length, 1);
  assert.deepEqual(
    artifactRequests.filter((name) => name === '41.flac'),
    ['41.flac'],
  );
});

test('writeArtifactsTo holds the fetches to its ceiling', async () => {
  // `events()` replays a job's whole history before it follows live, so
  // attaching to a nearly-finished 1,400-chunk render delivers 1,400 artifact
  // frames in one burst. Unbounded, that is 2,800 sockets in a tick.
  const indices = [1, 2, 3, 4, 5, 6, 7, 8];
  for (const index of indices) publish(index);
  // Hold every artifact response open for a few milliseconds so that overlap is
  // observable at all: with instant replies the ceiling would be met by the
  // event loop rather than by the writer, and the meter would prove nothing.
  holdArtifactsMs = 5;
  streams(
    ...indices.map((index, at) => frame(at + 1, 'artifact', { name: `${index}.flac` })),
    frame(indices.length + 1, 'done', {
      artifacts: indices.map((index) => `${index}.flac`),
      rendered: indices.length,
      failed: [],
      take: 0,
      sample_rate: 24000,
    }),
  );
  await collect('job-tts-1', directory, { concurrency: 2 });
  // Two artifacts at a time, each fetching its own sidecar alongside it. The
  // lower bound proves the meter ran and that the fetches really do overlap —
  // without it, a writer that fetched one file at a time would also pass.
  assert.ok(
    peakInFlightArtifacts >= 2 && peakInFlightArtifacts <= 4,
    `expected 2..4 requests in flight at concurrency 2, saw ${peakInFlightArtifacts} ` +
      `over ${artifactRequests.length} request(s)`,
  );
  assert.equal(artifactRequests.length, indices.length * 2);
  assert.equal((await readdir(directory)).length, indices.length * 2);
});

test('writeArtifactsTo refuses a concurrency below one rather than rounding it up', async () => {
  await assert.rejects(collect('job-tts-1', directory, { concurrency: 0 }), (error: unknown) => {
    assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
    assert.equal(error.option, 'concurrency');
    return true;
  });
});

test('writeArtifactsTo refuses an empty job id or directory by name', async () => {
  for (const [option, call] of [
    ['jobId', () => collect('', directory)],
    ['dir', () => collect('job-tts-1', '')],
  ] as const) {
    await assert.rejects(call(), (error: unknown) => {
      assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
      assert.equal(error.option, option);
      return true;
    });
  }
});

test('writeArtifactsTo creates the directory it was pointed at', async () => {
  publish(41);
  const nested = join(directory, 'chapter-07', 'sentences');
  streams(
    frame(1, 'artifact', { name: '41.flac' }),
    frame(2, 'done', { artifacts: ['41.flac'], rendered: 1, failed: [], take: 0, sample_rate: 24000 }),
  );
  await collect('job-tts-1', nested);
  assert.deepEqual((await readdir(nested)).sort(), ['41.flac', '41.flac.provenance.json']);
});

test('a caller that stops early leaves no temporary behind', async () => {
  publish(41);
  publish(42);
  streams(
    frame(1, 'artifact', { name: '41.flac' }),
    frame(2, 'artifact', { name: '42.flac' }),
    frame(3, 'done', { artifacts: ['41.flac', '42.flac'], rendered: 2, failed: [], take: 0, sample_rate: 24000 }),
  );
  for await (const entry of client().writeArtifactsTo('job-tts-1', directory)) {
    if (entry.kind === 'written') break;
  }
  // The generator's `finally` settles whatever was in flight, so the directory
  // holds finished files or nothing — never a half-written one.
  const files = await readdir(directory);
  assert.ok(
    !files.some((name) => name.endsWith('.part')),
    `no temporary may survive an early break; found ${files.join(', ')}`,
  );
});

test('the stream ending without a terminal event still throws, writer or not', async () => {
  // "The job finished" and "the socket died" must never look the same, and
  // wrapping events() must not soften that.
  publish(41);
  handle = (_request, response) => {
    openSse(response);
    response.write(frame(1, 'artifact', { name: '41.flac' }));
    response.end();
  };
  await assert.rejects(collect('job-tts-1', directory), (error: unknown) => {
    assert.match((error as Error).message, /without a terminal event/);
    return true;
  });
});
