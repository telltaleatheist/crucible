/**
 * The server-wide event stream (`GET /v1/events`) and the feature list, from the
 * client's side.
 *
 * What a client has to get right: the snapshot comes first and is read whole;
 * each event family is typed; a dropped connection resumes with Last-Event-ID;
 * a resume the server cannot serve is a `gap` snapshot; `overflow` reconnects at
 * once from the id it names; `server.stopping` is shown, then waited out and
 * reconnected; only the caller's signal ends the iteration. And an app asks
 * `has(feature)` instead of comparing versions, with the answer read once.
 *
 * Run: `npm run test:unit` (this file is found by `scripts/unit.mjs`).
 */

import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, beforeEach, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleConfigError,
  CrucibleUnreachable,
  type ServerEvent,
  type ServerEventsOptions,
} from '../src/index.js';

/** One connection's answer: frames to write, and whether to end the response after them. */
interface Connection {
  readonly status?: number;
  readonly body?: unknown;
  readonly frames?: readonly string[];
  readonly end?: boolean;
}

let server: Server;
let url: string;
let connections: Connection[];
let requests: { path: string; lastEventId: string | undefined }[];
let infoReads: number;
let infoFails: number;

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

function frame(id: number | null, event: string, data: unknown): string {
  return `${id === null ? '' : `id: ${id}\n`}event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

const AT = '2026-10-01T10:00:00+00:00';

const ACTIVITY = {
  server: { name: 'crucible@pc', version: '1.0.75', api_version: 1, backend: 'cuda-linux', uptime_s: 1 },
  resident: null, stopping: null, warming: null, claim: null, streaming: null,
  chat: { in_flight: 0, max_in_flight: null, max_in_flight_basis: null, rows: [] },
  settings: { writes: [] }, catalog: { removals: [] }, session: null, updating: null,
  slots: { accelerated: { busy: 0, of: 1, queue_depth: 0, accepts_work: true } },
  running: [], queued: [],
};

const TASK = {
  task_id: 't1', type: 'pull', request: { type: 'pull', kind: 'model', id: 'qwen3.5-9b' },
  state: 'running', error: null, created: AT, started: AT, finished: null, unmet: [], message: null,
};

function snapshot(id: number, gap = false): string {
  return frame(id, 'snapshot', {
    gap,
    topics: ['card', 'chat', 'job', 'queue', 'server', 'session', 'settings', 'task'],
    activity: ACTIVITY,
    queue: { items: [], depth: 0 },
    tasks: [TASK],
  });
}

const INFO = {
  server: { name: 'crucible@pc', version: '1.0.75', api_version: 1 },
  role: 'orchestrator',
  engine: null,
  host: {
    platform: 'linux', arch: 'x86_64', backend: 'cuda-linux',
    gpu: { vendor: 'nvidia', name: '3090 Ti', vram_bytes: 25757220864 },
  },
  job_types: ['echo'],
  features: ['events', 'queue.calls', 'queue.jobs', 'queue.sessions'],
  capabilities: [],
};

before(async () => {
  server = createServer((request: IncomingMessage, response: ServerResponse) => {
    const path = request.url ?? '';
    if (path === '/v1/info') {
      infoReads += 1;
      if (infoFails > 0) {
        infoFails -= 1;
        json(response, 503, { error: { code: 'server_stopping', message: 'stopping' } });
        return;
      }
      json(response, 200, INFO);
      return;
    }
    const lastEventId = request.headers['last-event-id'];
    requests.push({ path, lastEventId: typeof lastEventId === 'string' ? lastEventId : undefined });
    const next = connections.shift();
    if (next === undefined) {
      // Nothing more planned: hold the stream open, as a quiet server does.
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      return;
    }
    if (next.status !== undefined) {
      json(response, next.status, next.body);
      return;
    }
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    for (const chunk of next.frames ?? []) response.write(chunk);
    if (next.end === true) response.end();
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

beforeEach(() => {
  connections = [];
  requests = [];
  infoReads = 0;
  infoFails = 0;
});

function client(): CrucibleClient {
  return new CrucibleClient({ url, token: 't', clientName: 'events-test' });
}

/** The first `count` events of the server-wide stream, then stop following it. */
async function take(count: number, options: ServerEventsOptions = {}): Promise<ServerEvent[]> {
  const seen: ServerEvent[] = [];
  const controller = new AbortController();
  for await (const event of client().events({ ...options, signal: controller.signal })) {
    seen.push(event);
    if (seen.length === count) controller.abort();
  }
  return seen;
}

test('a card wait arrives typed on the queue and session families', async () => {
  connections = [{
    frames: [
      snapshot(100),
      frame(101, 'queue.waiting', {
        job_id: 'ses-2', code: 'accelerator_busy', message: 'held by pid 4242', depth: 1,
        kind: 'session', at: AT,
      }),
      frame(102, 'session.waiting', {
        session_id: 'ses-2', client: 'briefcase', act: 'analysis', code: 'accelerator_busy',
        message: 'held by pid 4242', details: null, since: AT, next_check_at: AT, at: AT,
      }),
    ],
  }];
  const [, queued, session] = await take(3);
  assert.ok(queued?.event === 'queue.waiting');
  assert.deepEqual(queued.waiting, { code: 'accelerator_busy', message: 'held by pid 4242' });
  assert.equal(queued.position, null);
  assert.ok(session?.event === 'session.waiting');
  assert.deepEqual(session.waiting, {
    code: 'accelerator_busy', message: 'held by pid 4242', details: null, since: AT, nextCheckAt: AT,
  });
  assert.equal(session.reason, null);
});

test('the snapshot comes first, then every family arrives typed', async () => {
  connections = [{
    frames: [
      snapshot(100),
      frame(101, 'job.queued', {
        job_id: 'j1', type: 'tts', model: 'sigma', client: 'bookforge', client_ref: null,
        status: 'queued', position: 2, waiting: true, at: AT,
      }),
      frame(102, 'job.done', {
        job_id: 'j1', type: 'tts', model: 'sigma', client: 'bookforge', client_ref: null,
        status: 'done', artifacts: ['0.flac'], at: AT,
      }),
      frame(103, 'job.progress', { job_id: 'j2', fraction: 0.5, message: 'half', at: AT }),
      frame(104, 'queue.added', {
        job_id: 'ses-2', position: 1, type: 'session', model: null, client: 'briefcase',
        submitted: AT, max_wait_s: 3600, depth: 1, kind: 'session', at: AT,
      }),
      frame(105, 'session.closed', {
        session_id: 'ses-1', client: 'bookforge', act: 'tts', reason: 'idle', message: 'idle',
        items_run: 4, held_s: 600, at: AT,
      }),
      frame(106, 'card.loaded', {
        subject: 'qwen3.5-9b', kind: 'llm', engine: 'vllm', memory_bytes_estimate: 2e10, since: AT, at: AT,
      }),
      frame(107, 'chat.in_flight', { in_flight: 3, by_model: { 'qwen3.5-9b': 3 }, at: AT }),
      frame(108, 'task.step', { task_id: 't1', name: 'download', index: 1, total: 2, at: AT }),
      frame(109, 'task.done', { ...TASK, state: 'done', finished: AT, at: AT }),
      frame(110, 'settings.written', { act: 'settings', client: 'foundry', changed: ['routes'], at: AT }),
      frame(111, 'something.new', { at: AT }),
    ],
  }];
  const seen = await take(12);
  assert.deepEqual(seen.map((event) => event.event), [
    'snapshot', 'job.queued', 'job.done', 'job.progress', 'queue.added', 'session.closed',
    'card.loaded', 'chat.in_flight', 'task.step', 'task.done', 'settings.written', 'unknown',
  ]);
  const [first, queued, done, progress, added, closed, card, chat, step, task, written, unknown] = seen;
  assert.ok(first?.event === 'snapshot');
  assert.equal(first.gap, false);
  assert.equal(first.activity.server.version, '1.0.75');
  assert.equal(first.activity.session, null);
  assert.equal(first.tasks[0]?.taskId, 't1');
  assert.ok(queued?.event === 'job.queued');
  assert.deepEqual([queued.position, queued.waiting, queued.artifacts], [2, true, null]);
  assert.ok(done?.event === 'job.done');
  assert.deepEqual(done.artifacts, ['0.flac']);
  assert.ok(progress?.event === 'job.progress');
  assert.equal(progress.fraction, 0.5);
  assert.ok(added?.event === 'queue.added');
  assert.deepEqual([added.kind, added.position, added.data['client']], ['session', 1, 'briefcase']);
  assert.ok(closed?.event === 'session.closed');
  assert.deepEqual([closed.sessionId, closed.reason, closed.data['items_run']], ['ses-1', 'idle', 4]);
  assert.ok(card?.event === 'card.loaded');
  assert.deepEqual([card.subject, card.engine, card.memoryBytesEstimate, card.pids], ['qwen3.5-9b', 'vllm', 2e10, null]);
  assert.ok(chat?.event === 'chat.in_flight');
  assert.deepEqual(chat.byModel, { 'qwen3.5-9b': 3 });
  assert.ok(step?.event === 'task.step');
  assert.equal(step.step.name, 'download');
  assert.ok(task?.event === 'task.done');
  assert.equal(task.task.state, 'done');
  assert.ok(written?.event === 'settings.written');
  assert.deepEqual(written.changed, ['routes']);
  assert.ok(unknown?.event === 'unknown');
  assert.equal(unknown.kind, 'something.new');
});

test('a dropped stream resumes with Last-Event-ID and loses nothing', async () => {
  connections = [
    { frames: [snapshot(10), frame(11, 'chat.in_flight', { in_flight: 1, by_model: {}, at: AT })], end: true },
    { frames: [frame(12, 'chat.in_flight', { in_flight: 0, by_model: {}, at: AT })] },
  ];
  const seen = await take(3);
  assert.deepEqual(seen.map((event) => event.id), [10, 11, 12]);
  assert.deepEqual(requests.map((request) => request.lastEventId), [undefined, '11']);
});

test('a resume the server can no longer serve opens with a gap snapshot', async () => {
  connections = [{ frames: [snapshot(500, true)] }];
  const [first] = await take(1, { lastEventId: 42 });
  assert.ok(first?.event === 'snapshot');
  assert.equal(first.gap, true);
  assert.equal(requests[0]?.lastEventId, '42');
});

test('overflow is shown and reconnects at once from the id it names', async () => {
  connections = [
    {
      frames: [
        snapshot(10),
        frame(11, 'overflow', { last_event_id: 11, limit: 2000, message: 'fell behind' }),
      ],
    },
    { frames: [frame(12, 'settings.written', { act: null, client: null, changed: [], at: AT })] },
  ];
  const seen = await take(3);
  assert.deepEqual(seen.map((event) => event.event), ['snapshot', 'overflow', 'settings.written']);
  assert.deepEqual(requests.map((request) => request.lastEventId), [undefined, '11']);
});

test('server.stopping is shown, then the stream is waited out and reconnected', async () => {
  connections = [
    { frames: [snapshot(10), frame(11, 'server.stopping', { reason: 'asked to stop', at: AT })], end: true },
    { status: 503, body: { error: { code: 'server_stopping', message: 'stopping' } } },
    { frames: [snapshot(9000, true)] },
  ];
  const started = Date.now();
  const seen = await take(3);
  assert.deepEqual(seen.map((event) => event.event), ['snapshot', 'server.stopping', 'snapshot']);
  const stopping = seen[1];
  assert.ok(stopping?.event === 'server.stopping');
  assert.equal(stopping.reason, 'asked to stop');
  const back = seen[2];
  assert.ok(back?.event === 'snapshot');
  assert.equal(back.gap, true);
  assert.deepEqual(requests.map((request) => request.lastEventId), [undefined, '11', '11']);
  // Two waits, a quarter second and then half of one: backed off, never a spin.
  assert.ok(Date.now() - started >= 700, `reconnected after ${Date.now() - started} ms`);
});

test('the signal ends the iteration, and topics are sent and checked', async () => {
  connections = [{ frames: [snapshot(1)] }];
  const seen = await take(1, { topics: ['job', 'session'] });
  assert.equal(seen.length, 1);
  assert.equal(requests[0]?.path, '/v1/events?topics=job%2Csession');
  const iterator = client().events({ topics: ['jobs' as 'job'] });
  await assert.rejects(iterator.next(), CrucibleConfigError);
});

test('a job stream that meets server.stopping ends by name, not as a malformed frame', async () => {
  connections = [{
    frames: [
      frame(1, 'progress', { fraction: 0.1, message: 'working' }),
      frame(null, 'server.stopping', { reason: 'asked to stop' }),
    ],
    end: true,
  }];
  const crucible = client();
  const seen: string[] = [];
  await assert.rejects(
    (async () => {
      for await (const event of crucible.events('j1')) seen.push(event.event);
    })(),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleUnreachable);
      assert.match(error.message, /stopping \(asked to stop\).*lastEventId 1/);
      return true;
    },
  );
  assert.deepEqual(seen, ['progress']);
});

test('info carries the features, and has() reads them once', async () => {
  const crucible = client();
  assert.deepEqual((await crucible.info()).features, INFO.features);
  infoReads = 0;
  assert.equal(await crucible.has('queue.sessions'), true);
  assert.equal(await crucible.has('events'), true);
  assert.equal(await crucible.has('video'), false);
  assert.equal(infoReads, 1);
});

test('has() does not remember a failed read', async () => {
  infoFails = 1;
  const crucible = client();
  await assert.rejects(crucible.has('events'));
  assert.equal(await crucible.has('events'), true);
  assert.equal(infoReads, 2);
});
