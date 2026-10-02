/**
 * `stream({onQueue})` — a stream's place in the line while its queue session waits.
 *
 * Against an in-process `node:http` fixture, like `unit-stream.test.ts`: it plays a server that
 * answers the ticketed open with `202`, a queue session's own stream that moves and opens (or is
 * removed), and the second open that names the session. What it proves is the order on the wire —
 * the ticket header only when `onQueue` is given, the session header on the second open, the line
 * left on an abort — and that an older server, which ignores the header and holds the request
 * open, still opens the stream.
 *
 * Run: `npm run test:unit` (exits non-zero on any failure).
 */

import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleSessionClosed,
  type QueuePosition,
  type TtsStreamSession,
} from '../src/index.js';

// ------------------------------------------------------------------ fixture

interface Seen {
  method: string;
  path: string;
  body: string;
  ticket?: string;
  session?: string;
}

let seen: Seen[] = [];
let server: Server;
let base = '';

const QUEUE_SESSION = 'ses-for-stream';

const STREAM = {
  session_id: 'cafebabe',
  voice: 'deathstalker',
  fingerprint: 'deathstalker@0123456789abcdef0123456789abcdef01234567',
  sample_rate: 24000,
  backend: 'cuda-linux',
  queue_session_id: QUEUE_SESSION,
  queue_session_opened_for_stream: true,
};

const READY = {
  voice: STREAM.voice,
  fingerprint: STREAM.fingerprint,
  sample_rate: 24000,
  backend: 'cuda-linux',
};

/** How the fake server answers the first open: with a ticket, or held open like an old server. */
let ticketing = true;
/** What the second open answers. */
let secondOpen: (response: ServerResponse) => void = (response) => json(response, 201, STREAM);
/** The queue session's own stream, written by the test as it goes. */
let feed: ServerResponse | null = null;
let feedId = 0;

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

function say(event: string, data: Record<string, unknown>): void {
  feedId += 1;
  feed?.write(
    `id: ${feedId}\nevent: ${event}\ndata: ${JSON.stringify({ session_id: QUEUE_SESSION, ...data })}\n\n`,
  );
}

function handle(request: IncomingMessage, response: ServerResponse): void {
  const path = request.url ?? '';
  if (path === '/v1/tts/stream' && request.method === 'POST') {
    if (request.headers['x-crucible-session'] === QUEUE_SESSION) {
      secondOpen(response);
      return;
    }
    if (ticketing && request.headers['x-crucible-queue-ticket'] === '1') {
      json(response, 202, { queue_session_id: QUEUE_SESSION, status: 'queued', position: 2 });
      return;
    }
    json(response, 201, STREAM);
    return;
  }
  if (path === `/v1/queue/sessions/${QUEUE_SESSION}/events`) {
    response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store' });
    response.flushHeaders();
    feed = response;
    feedId = 0;
    response.on('close', () => {
      if (feed === response) feed = null;
    });
    return;
  }
  if (path === `/v1/queue/sessions/${QUEUE_SESSION}` && request.method === 'DELETE') {
    json(response, 200, { session_id: QUEUE_SESSION, status: 'closed', reason: 'client' });
    return;
  }
  if (path === `/v1/tts/stream/${STREAM.session_id}/events`) {
    response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store' });
    response.write(`id: 1\nevent: ready\ndata: ${JSON.stringify(READY)}\n\n`);
    response.write(`id: 2\nevent: closed\ndata: ${JSON.stringify({ reason: 'done' })}\n\n`);
    response.end();
    return;
  }
  if (path === `/v1/tts/stream/${STREAM.session_id}` && request.method === 'DELETE') {
    json(response, 200, { session_id: STREAM.session_id, closed: true });
    return;
  }
  json(response, 500, { error: { code: 'no_handler', message: `the test set none for ${path}` } });
}

before(async () => {
  server = createServer((request, response) => {
    const chunks: Buffer[] = [];
    request.on('data', (chunk: Buffer) => chunks.push(chunk));
    request.on('end', () => {
      const entry: Seen = {
        method: request.method ?? '',
        path: request.url ?? '',
        body: Buffer.concat(chunks).toString('utf8'),
      };
      const ticket = request.headers['x-crucible-queue-ticket'];
      if (typeof ticket === 'string') entry.ticket = ticket;
      const session = request.headers['x-crucible-session'];
      if (typeof session === 'string') entry.session = session;
      seen.push(entry);
      handle(request, response);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  feed?.end();
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

function reset(): void {
  seen = [];
  ticketing = true;
  secondOpen = (response) => json(response, 201, STREAM);
  feed = null;
  feedId = 0;
}

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-stream-onqueue' });
}

async function until(condition: () => boolean, what: string): Promise<void> {
  const deadline = Date.now() + 5_000;
  while (!condition()) {
    if (Date.now() > deadline) throw new Error(`never saw ${what}`);
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

function opens(): Seen[] {
  return seen.filter((entry) => entry.method === 'POST' && entry.path === '/v1/tts/stream');
}

function leaves(): Seen[] {
  return seen.filter(
    (entry) => entry.method === 'DELETE' && entry.path === `/v1/queue/sessions/${QUEUE_SESSION}`,
  );
}

async function drain(session: TtsStreamSession): Promise<void> {
  for await (const event of session) void event;
}

// ------------------------------------------------------------------ tests

test('onQueue hears every move while the stream waits, then the stream opens in its session', async () => {
  reset();
  const moves: QueuePosition[] = [];
  const opening = client().stream({
    voice: 'deathstalker',
    language: 'en',
    onQueue: (position) => moves.push(position),
  });
  await until(() => feed !== null, 'the queue session to be followed');
  say('queued', { position: 2, of: 2 });
  say('moved', { position: 1, of: 1 });
  await until(() => moves.length === 2, 'both moves');
  assert.equal(opens().length, 1, 'nothing opens the stream while its session waits');
  say('opened', { opened_at: '2026-10-02T10:00:00+00:00', model: null, load_job: null });
  const session = await opening;

  assert.deepEqual(moves, [{ position: 2, of: 2 }, { position: 1, of: 1 }]);
  const [first, second] = opens();
  assert.equal(first?.ticket, '1', 'the first open asks for a ticket');
  assert.equal(first?.session, undefined);
  assert.equal(second?.session, QUEUE_SESSION, 'the second open names the session it waited in');
  assert.equal(second?.ticket, undefined);
  assert.deepEqual(JSON.parse(second?.body ?? '{}'), JSON.parse(first?.body ?? '{}'));
  assert.equal(session.queueSessionId, QUEUE_SESSION);
  assert.equal(session.openedForStream, true);
  await until(() => feed === null, 'the queue session follow to end once it opened');
  say('closed', { reason: 'client' });
  assert.equal(moves.length, 2, 'never called after the session opens');
  assert.equal(leaves().length, 0, 'an opened stream leaves its session alone');
  await drain(session);
});

test('without onQueue the open asks for no ticket: the held-open request, unchanged', async () => {
  reset();
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  assert.equal(opens().length, 1);
  assert.equal(opens()[0]?.ticket, undefined);
  await drain(session);
});

test('a server that answers 201 (free, or older and ignoring the header) opens the stream at once', async () => {
  reset();
  ticketing = false;
  const moves: QueuePosition[] = [];
  const session = await client().stream({
    voice: 'deathstalker',
    language: 'en',
    onQueue: (position) => moves.push(position),
  });
  assert.equal(opens().length, 1, 'one open, held until the stream was ready');
  assert.equal(session.sessionId, STREAM.session_id);
  assert.deepEqual(moves, []);
  assert.equal(feed, null, 'no queue session followed');
  await drain(session);
});

test('aborting while the session waits takes it out of the line and throws the abort', async () => {
  reset();
  const abort = new AbortController();
  const opening = client().stream({
    voice: 'deathstalker',
    language: 'en',
    onQueue: () => undefined,
    signal: abort.signal,
  });
  await until(() => feed !== null, 'the queue session to be followed');
  say('queued', { position: 1, of: 1 });
  abort.abort(new Error('the reader closed the tab'));
  await assert.rejects(opening, /the reader closed the tab/);
  await until(() => leaves().length === 1, 'the session to leave the line');
  assert.equal(opens().length, 1, 'no stream was asked for');
});

test('a session removed before it opens throws CrucibleSessionClosed with its reason', async () => {
  reset();
  const opening = client().stream({ voice: 'deathstalker', language: 'en', onQueue: () => undefined });
  await until(() => feed !== null, 'the queue session to be followed');
  say('queued', { position: 1, of: 1 });
  say('removed', { reason: 'operator', message: 'an operator removed it' });
  await assert.rejects(
    opening,
    (error: unknown) =>
      error instanceof CrucibleSessionClosed &&
      error.reason === 'operator' &&
      error.sessionId === QUEUE_SESSION,
  );
  assert.equal(opens().length, 1);
  assert.equal(leaves().length, 0, 'a removed session is already out of the line');
});

test('a second open that fails gives the session back rather than holding the server', async () => {
  reset();
  secondOpen = (response) =>
    json(response, 502, {
      error: { code: 'stream_voice_load_failed', message: 'the voice would not load' },
    });
  const opening = client().stream({ voice: 'deathstalker', language: 'en', onQueue: () => undefined });
  await until(() => feed !== null, 'the queue session to be followed');
  say('opened', { opened_at: '2026-10-02T10:00:00+00:00', model: null, load_job: null });
  await assert.rejects(opening, /the voice would not load/);
  await until(() => leaves().length === 1, 'the opened session to be closed');
});

test('onQueue that is not a function is refused before anything is sent', async () => {
  reset();
  await assert.rejects(
    () => client().stream({ voice: 'deathstalker', language: 'en', onQueue: 'yes' as never }),
    /onQueue must be a function/,
  );
  assert.equal(seen.length, 0);
});
