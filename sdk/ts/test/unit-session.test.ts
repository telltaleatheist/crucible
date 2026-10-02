/**
 * Queue sessions, from the client's side.
 *
 * A session is an app's turn holding the server for a run of requests. What a
 * client has to get right: `session()` answers only once the session is OPEN,
 * following its stream (and reporting each move) while it waits; every request
 * the session's client makes carries `X-Crucible-Session`, and its helpers send
 * no `queue`; the server ending the session (idle, an operator, a restart)
 * resolves `closed` by itself; a session that has ended refuses its own items
 * without asking the server; and a session that never opens throws its reason.
 *
 * Run: `npm run test:unit` (this file is found by `scripts/unit.mjs`).
 */

import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, beforeEach, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleSession,
  CrucibleSessionClosed,
  CrucibleSessionHeld,
  type QueuePosition,
} from '../src/index.js';

interface Seen {
  readonly method: string;
  readonly path: string;
  readonly body: string;
  readonly session: string | undefined;
  readonly lastEventId: string | undefined;
}

interface Frame {
  readonly id: number;
  readonly event: string;
  readonly data: Record<string, unknown>;
}

/** One fake session's own stream: what it has said, and who is listening. */
class Feed {
  readonly frames: Frame[] = [];
  readonly live = new Set<ServerResponse>();

  say(event: string, data: Record<string, unknown>): void {
    const frame = { id: this.frames.length + 1, event, data: { session_id: 'ses-1', ...data } };
    this.frames.push(frame);
    for (const response of this.live) response.write(sse(frame));
  }

  /** The server stopping: one id-less frame, and every stream ends. */
  stop(reason: string): void {
    for (const response of this.live) {
      response.end(`event: server.stopping\ndata: ${JSON.stringify({ reason })}\n\n`);
    }
    this.live.clear();
  }

  /** The connection breaks without the session having ended. */
  drop(): void {
    for (const response of this.live) response.end();
    this.live.clear();
  }

  attach(request: IncomingMessage, response: ServerResponse): void {
    const after = Number(request.headers['last-event-id'] ?? 0);
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    for (const frame of this.frames) if (frame.id > after) response.write(sse(frame));
    this.live.add(response);
    response.on('close', () => this.live.delete(response));
  }
}

function sse(frame: Frame): string {
  return `id: ${frame.id}\nevent: ${frame.event}\ndata: ${JSON.stringify(frame.data)}\n\n`;
}

type Handler = (request: IncomingMessage, body: string, response: ServerResponse) => boolean;

let server: Server;
let url: string;
let seen: Seen[];
let feed: Feed;
let ticket: Record<string, unknown>;
let extra: Handler;

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

function state(overrides: Record<string, unknown>): Record<string, unknown> {
  return {
    session_id: 'ses-1', status: 'open', act: 'analysis', client: 'session-test', model: null,
    position: null, idle_s: 300, max_wait_s: 3600, created: '2026-10-01T10:00:00+00:00',
    opened_at: '2026-10-01T10:00:01+00:00', idle_deadline: '2026-10-01T10:05:01+00:00',
    max_hold_deadline: null, items_run: 0, in_flight: [], stream_session: null, load_job: null,
    closed_at: null, reason: null, message: null, error: null,
    ...overrides,
  };
}

const COMPLETION = {
  id: 'chatcmpl-s',
  model: 'qwen3.5-9b',
  choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
};

before(async () => {
  server = createServer((request, response) => {
    let body = '';
    request.on('data', (chunk) => (body += chunk));
    request.on('end', () => {
      const path = request.url ?? '';
      const header = request.headers['x-crucible-session'];
      const lastEventId = request.headers['last-event-id'];
      seen.push({
        method: request.method ?? '',
        path,
        body,
        session: typeof header === 'string' ? header : undefined,
        lastEventId: typeof lastEventId === 'string' ? lastEventId : undefined,
      });
      if (extra(request, body, response)) return;
      if (request.method === 'POST' && path === '/v1/queue/sessions') {
        json(response, 202, ticket);
        return;
      }
      if (request.method === 'GET' && path === '/v1/queue/sessions/ses-1/events') {
        feed.attach(request, response);
        return;
      }
      if (request.method === 'POST' && path === '/v1/queue/sessions/ses-1/touch') {
        json(response, 200, { session_id: 'ses-1', status: 'open' });
        return;
      }
      if (request.method === 'GET' && path === '/v1/queue/sessions/ses-1') {
        json(response, 200, state({ items_run: 2, in_flight: [{ kind: 'job', job_id: 'j1' }] }));
        return;
      }
      if (request.method === 'DELETE' && path === '/v1/queue/sessions/ses-1') {
        feed.say('closed', { reason: 'client', message: 'the client closed it', items_run: 2, held_s: 9 });
        json(response, 200, state({
          status: 'closed', items_run: 2, idle_deadline: null, closed_at: '2026-10-01T10:00:10+00:00',
          reason: 'client', message: 'the client closed it',
        }));
        return;
      }
      if (request.method === 'POST' && path === '/v1/jobs') {
        json(response, 202, { job_id: 'j1', resume_id: null, queued: false, position: null });
        return;
      }
      if (request.method === 'POST' && path === '/v1/openai/chat/completions') {
        json(response, 200, COMPLETION);
        return;
      }
      json(response, 404, { error: { code: 'not_found', message: path } });
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

beforeEach(() => {
  seen = [];
  feed = new Feed();
  ticket = { session_id: 'ses-1', status: 'open', position: null };
  extra = () => false;
});

function client(): CrucibleClient {
  return new CrucibleClient({ url, token: 't', clientName: 'session-test' });
}

/** Resolves once `predicate` holds, polling the fake's record; fails after two seconds. */
async function until(predicate: () => boolean, what: string): Promise<void> {
  const deadline = Date.now() + 2_000;
  while (!predicate()) {
    if (Date.now() > deadline) assert.fail(`timed out waiting for ${what}`);
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

function watching(): boolean {
  return feed.live.size > 0;
}

test('a session the server opens at once is open when session() answers', async () => {
  feed.say('opened', { opened_at: '2026-10-01T10:00:01+00:00', model: null, load_job: null });
  const crucible = client();
  const session = await crucible.session({ act: 'analysis', idleS: 600, maxWaitS: 120 });
  try {
    assert.ok(session instanceof CrucibleSession);
    assert.equal(session.id, 'ses-1');
    assert.equal(session.act, 'analysis');
    assert.equal(session.ended, null);
    const opened = seen.find((request) => request.path === '/v1/queue/sessions');
    assert.deepEqual(JSON.parse(opened?.body ?? '{}'), { act: 'analysis', idle_s: 600, max_wait_s: 120 });
  } finally {
    await session.close();
  }
});

test('a queued session reports each move and answers once it opens', async () => {
  ticket = { session_id: 'ses-1', status: 'queued', position: 2 };
  const moves: QueuePosition[] = [];
  const opening = client().session({
    act: 'analysis',
    model: 'qwen3.5-9b',
    onQueue: (position) => moves.push(position),
  });
  await until(watching, 'the queued session to be followed');
  feed.say('queued', { position: 2, of: 2 });
  feed.say('moved', { position: 1, of: 1 });
  await until(() => moves.length === 2, 'both moves');
  feed.say('opened', { opened_at: '2026-10-01T10:00:01+00:00', model: 'qwen3.5-9b', load_job: 'j0' });
  const session = await opening;
  try {
    assert.deepEqual(moves, [{ position: 2, of: 2 }, { position: 1, of: 1 }]);
    // The background follow resumes after `opened`, not from the beginning.
    await until(() => seen.some((request) => request.lastEventId === '3'), 'the follow to resume after 3');
    assert.equal(JSON.parse(seen[0]?.body ?? '{}').model, 'qwen3.5-9b');
  } finally {
    await session.close();
  }
});

test('every item carries the session header, and helpers inside it send no queue', async () => {
  const crucible = client();
  const session = await crucible.session({ act: 'analysis' });
  try {
    seen = [];
    await session.loadModel('qwen3.5-9b');
    await session.chat({ model: 'qwen3.5-9b', messages: [{ role: 'user', content: 'hi' }] });
    await crucible.loadModel('qwen3.5-9b');
    const [load, chat, outside] = seen.filter((request) => request.method === 'POST');
    assert.equal(load?.session, 'ses-1');
    assert.ok(!('queue' in JSON.parse(load?.body ?? '{}')), 'a job inside a session waits by right');
    assert.equal(chat?.session, 'ses-1');
    // A chat that must wait for its model or a slot waits ahead of the line by default: no `queue`.
    assert.ok(!('queue' in JSON.parse(chat?.body ?? '{}')));
    // The client it came from is untouched: no header, and its helpers still wait by default.
    assert.equal(outside?.session, undefined);
    assert.ok(!('queue' in JSON.parse(outside?.body ?? '{}')));
  } finally {
    await session.close();
  }
});

test('touch, state and close go to the session routes, and close resolves closed', async () => {
  const session = await client().session({ act: 'analysis' });
  await session.touch();
  const read = await session.state();
  assert.equal(read.status, 'open');
  assert.equal(read.itemsRun, 2);
  assert.deepEqual(read.inFlight, [{ kind: 'job', job_id: 'j1' }]);
  const end = await session.close();
  assert.equal(end.reason, 'client');
  assert.equal(end.itemsRun, 2);
  assert.equal(end.heldS, 9);
  assert.deepEqual(await session.closed, end);
  assert.deepEqual(await session.close(), end, 'a second close answers how it ended');
  assert.deepEqual(
    seen.filter((request) => !request.path.endsWith('/events')).map((request) => `${request.method} ${request.path}`),
    [
      'POST /v1/queue/sessions',
      'POST /v1/queue/sessions/ses-1/touch',
      'GET /v1/queue/sessions/ses-1',
      'DELETE /v1/queue/sessions/ses-1',
    ],
  );
});

test('the server ending the session resolves closed, and its items are then refused here', async () => {
  const session = await client().session({ act: 'analysis' });
  await until(watching, 'the background follow');
  feed.say('closed', {
    reason: 'idle', message: 'nothing arrived for its idle_s (300 s)', items_run: 3, held_s: 312.5,
  });
  const end = await session.closed;
  assert.deepEqual(end, {
    reason: 'idle', message: 'nothing arrived for its idle_s (300 s)', itemsRun: 3, heldS: 312.5,
  });
  assert.deepEqual(session.ended, end);
  const before = seen.length;
  await assert.rejects(
    session.chat({ model: 'qwen3.5-9b', messages: [{ role: 'user', content: 'hi' }] }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleSessionClosed);
      assert.equal(error.code, 'session_closed');
      assert.equal(error.reason, 'idle');
      assert.equal(error.sessionId, 'ses-1');
      return true;
    },
  );
  await assert.rejects(session.touch(), CrucibleSessionClosed);
  assert.equal(seen.length, before, 'nothing was sent for a session known to have ended');
});

test('an item the server refuses session_closed ends the session for its client too', async () => {
  const session = await client().session({ act: 'analysis' });
  extra = (request, _body, response) => {
    if (request.method !== 'POST' || request.url !== '/v1/jobs') return false;
    json(response, 409, {
      error: {
        code: 'session_closed',
        message: 'session ses-1 closed (operator): an operator ended it',
        details: { session_id: 'ses-1', status: 'closed', reason: 'operator' },
      },
    });
    return true;
  };
  await assert.rejects(session.loadModel('qwen3.5-9b'), CrucibleSessionClosed);
  assert.equal((await session.closed).reason, 'operator');
});

test('a dropped follow reconnects with Last-Event-ID and still hears the end', async () => {
  feed.say('opened', { opened_at: '2026-10-01T10:00:01+00:00', model: null, load_job: null });
  const session = await client().session({ act: 'analysis' });
  await until(watching, 'the background follow');
  feed.drop();
  await until(
    () => seen.filter((request) => request.path.endsWith('/events')).length >= 2 && watching(),
    'the follow to reconnect',
  );
  assert.equal(seen.filter((request) => request.path.endsWith('/events')).at(-1)?.lastEventId, '1');
  feed.say('closed', { reason: 'operator', message: 'an operator ended it', items_run: 0, held_s: 4 });
  assert.equal((await session.closed).reason, 'operator');
});

test('the server stopping ends the session it holds', async () => {
  const session = await client().session({ act: 'analysis' });
  await until(watching, 'the background follow');
  feed.stop('the server was asked to stop');
  const end = await session.closed;
  assert.equal(end.reason, 'server_restart');
  assert.match(end.message, /the server was asked to stop/);
});

test('a session that never opens throws its reason, and is not left in the line', async () => {
  ticket = { session_id: 'ses-1', status: 'queued', position: 1 };
  const opening = client().session({ act: 'analysis', model: 'qwen3.5-9b' });
  await until(watching, 'the queued session to be followed');
  feed.say('queued', { position: 1, of: 1 });
  feed.say('removed', {
    reason: 'load_failed',
    message: 'its model would not load',
    error: { code: 'session_load_failed', message: 'qwen3.5-9b: out of memory' },
  });
  await assert.rejects(opening, (error: unknown) => {
    assert.ok(error instanceof CrucibleSessionClosed);
    assert.equal(error.code, 'session_closed');
    assert.equal(error.reason, 'load_failed');
    assert.deepEqual((error.details as { error: unknown }).error, {
      code: 'session_load_failed', message: 'qwen3.5-9b: out of memory',
    });
    return true;
  });
  assert.ok(!seen.some((request) => request.method === 'DELETE'), 'an ended session needs no DELETE');
});

test('aborting the wait takes the session out of the line', async () => {
  ticket = { session_id: 'ses-1', status: 'queued', position: 4 };
  const controller = new AbortController();
  const opening = client().session({ act: 'analysis', signal: controller.signal });
  await until(watching, 'the queued session to be followed');
  controller.abort();
  await assert.rejects(opening, (error: unknown) => (error as Error).name === 'AbortError');
  assert.ok(seen.some((request) => request.method === 'DELETE' && request.path === '/v1/queue/sessions/ses-1'));
});

test('a session cannot be opened from a session', async () => {
  const session = await client().session({ act: 'analysis' });
  try {
    await assert.rejects(session.session({ act: 'analysis' }), /already/);
  } finally {
    await session.close();
  }
});

test('another client\'s session refuses by name, as server_busy or session_open', async () => {
  const held = {
    door: 'session', holder: 'briefcase', session_id: 'ses-9', type: 'session', act: 'analysis',
    model: 'qwen3.5-9b', status: 'open', since: '2026-10-01T09:00:00+00:00',
  };
  extra = (request, body, response) => {
    if (request.method === 'POST' && request.url === '/v1/jobs') {
      json(response, 409, { error: { code: 'server_busy', message: 'a session holds it', details: held } });
      return true;
    }
    if (request.method === 'POST' && request.url === '/v1/tts/stream') {
      json(response, 409, { error: { code: 'session_open', message: 'a session holds it', details: held } });
      return true;
    }
    return false;
  };
  const crucible = client();
  await assert.rejects(crucible.submit({ type: 'echo', params: {}, inputs: {} }), (error: unknown) => {
    assert.ok(error instanceof CrucibleSessionHeld);
    assert.equal(error.code, 'server_busy');
    assert.equal(error.holder, 'briefcase');
    assert.equal(error.sessionId, 'ses-9');
    return true;
  });
  await assert.rejects(
    crucible.stream({ voice: 'sigma', language: 'en', idleS: 600, queue: false }),
    (error: unknown) => error instanceof CrucibleSessionHeld && error.code === 'session_open',
  );
  const open = seen.find((request) => request.path === '/v1/tts/stream');
  assert.deepEqual(JSON.parse(open?.body ?? '{}'), {
    voice: 'sigma', language: 'en', idle_s: 600, queue: false,
  });
});

test('activity reads the open session', async () => {
  extra = (request, _body, response) => {
    if (request.url !== '/v1/activity') return false;
    json(response, 200, {
      server: { name: 'crucible@pc', version: '1.0.75', api_version: 1, backend: 'cuda-linux', uptime_s: 1 },
      resident: null, stopping: null, warming: null, claim: null, streaming: null,
      chat: { in_flight: 0, max_in_flight: null, max_in_flight_basis: null, rows: [] },
      settings: { writes: [] }, catalog: { removals: [] },
      session: state({ client: 'briefcase', model: 'qwen3.5-9b' }),
      slots: { accelerated: { busy: 0, of: 1, queue_depth: 0, accepts_work: false } },
      running: [], queued: [],
    });
    return true;
  };
  const activity = await client().activity();
  assert.equal(activity.session?.sessionId, 'ses-1');
  assert.equal(activity.session?.client, 'briefcase');
  assert.equal(activity.session?.idleS, 300);
});

test('closeSession() closes a recorded session by its id, as its client', async () => {
  const closed = await client().closeSession('ses-1');
  assert.equal(closed.status, 'closed');
  assert.equal(closed.reason, 'client');
  const sent = seen.filter((request) => request.method === 'DELETE');
  assert.deepEqual(sent.map((request) => [request.path, request.session]), [
    ['/v1/queue/sessions/ses-1', undefined],
  ]);
});
