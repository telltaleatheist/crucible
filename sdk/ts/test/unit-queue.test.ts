/**
 * The server's queue, from the client's side.
 *
 * A job submitted with `queue` waits on the server instead of being refused
 * `server_busy`. What a client has to get right: the raw `submit` sends the
 * field only when asked, the high-level helpers ask by default, a refusal of
 * the field is the server's word (never retried without it), and a job that
 * leaves the queue without running ends `removed` — its own terminal event and
 * status, never mistaken for `failed`.
 *
 * Run: `npm run test:unit` (this file is found by `scripts/unit.mjs`).
 */

import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, beforeEach, test } from 'node:test';

import {
  CrucibleBusy,
  CrucibleClient,
  CrucibleConfigError,
  CrucibleRefused,
  type JobEvent,
  type QueueEvent,
} from '../src/index.js';

type Handler = (request: IncomingMessage, body: string, response: ServerResponse) => void;

let server: Server;
let url: string;
let handler: Handler;
let posted: unknown[] = [];

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

before(async () => {
  server = createServer((request, response) => {
    let body = '';
    request.on('data', (chunk) => (body += chunk));
    request.on('end', () => handler(request, body, response));
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

beforeEach(() => {
  posted = [];
  handler = (request, body, response) => {
    if (request.method === 'POST' && request.url === '/v1/jobs') {
      posted.push(JSON.parse(body));
      json(response, 202, { job_id: 'j1', resume_id: null, queued: true, position: 2 });
      return;
    }
    json(response, 404, { error: { code: 'not_found', message: request.url ?? '' } });
  };
});

function client(options: { queue?: boolean | { maxWaitS?: number } } = {}): CrucibleClient {
  return new CrucibleClient({ url, token: 't', clientName: 'queue-test', ...options });
}

const ECHO = { type: 'echo', params: {}, inputs: {} } as const;

test('submit() sends no queue field unless the request asks for one', async () => {
  await client().submit(ECHO);
  await client().submit({ ...ECHO, queue: true });
  await client().submit({ ...ECHO, queue: { maxWaitS: 600 } });
  await client().submit({ ...ECHO, queue: false });
  assert.deepEqual(
    posted.map((body) => (body as { queue?: unknown }).queue),
    [undefined, {}, { max_wait_s: 600 }, undefined],
  );
});

test('a max wait the server would refuse is refused before anything is sent', async () => {
  for (const maxWaitS of [5, 86_401, 1.5]) {
    await assert.rejects(client().submit({ ...ECHO, queue: { maxWaitS } }), CrucibleConfigError);
  }
  assert.throws(() => client({ queue: { maxWaitS: 0 } }), CrucibleConfigError);
  assert.equal(posted.length, 0);
});

test('the high-level helpers queue by default, and the client option turns that off', async () => {
  await client().loadModel('qwen3.5-9b');
  await client({ queue: { maxWaitS: 120 } }).unloadModel('qwen3.5-9b');
  await client({ queue: false }).loadModel('qwen3.5-9b');
  assert.deepEqual(
    posted.map((body) => (body as { queue?: unknown }).queue),
    [{}, { max_wait_s: 120 }, undefined],
  );
});

test('a refusal of the queue field is the server\'s word, not a cue to ask again without it', async () => {
  handler = (request, body, response) => {
    posted.push(JSON.parse(body));
    json(response, 400, {
      error: {
        code: 'invalid_request',
        message: 'queue: Extra inputs are not permitted',
        details: {
          problems: [{ location: ['body', 'queue'], type: 'extra_forbidden', message: 'x' }],
        },
      },
    });
  };
  await assert.rejects(client().loadModel('qwen3.5-9b'), (error: unknown) => {
    assert.ok(error instanceof CrucibleRefused);
    assert.equal(error.code, 'invalid_request');
    return true;
  });
  assert.equal(posted.length, 1);
});

test('a busy refusal to a submit that did not queue is still CrucibleBusy', async () => {
  handler = (_request, _body, response) =>
    json(response, 409, {
      error: {
        code: 'server_busy',
        message: 'busy',
        details: {
          door: 'job', holder: 'bookforge', job_id: 'j0', type: 'tts', model: 'sigma',
          status: 'running', since: '2026-09-30T10:00:00+00:00', progress: 0.5,
          message: null, queue_depth: 3,
        },
      },
    });
  await assert.rejects(client().submit(ECHO), CrucibleBusy);
});

function stream(frames: string[]): Handler {
  return (_request, _body, response) => {
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    response.end(frames.join(''));
  };
}

test('queued, started and removed arrive typed, and removed ends the stream', async () => {
  handler = stream([
    'id: 1\nevent: queued\ndata: {"position": 3, "of": 3, "max_wait_s": 3600}\n\n',
    'id: 2\nevent: queued\ndata: {"position": 1, "of": 1}\n\n',
    'id: 3\nevent: removed\ndata: {"reason": "operator", "message": "an operator removed it",' +
      ' "waited_s": 41.5, "at": "2026-09-30T10:01:00+00:00"}\n\n',
    'id: 4\nevent: progress\ndata: {"fraction": 0, "message": "never read"}\n\n',
  ]);
  const seen: JobEvent[] = [];
  for await (const event of client().events('j1')) seen.push(event);
  assert.deepEqual(seen.map((event) => event.event), ['queued', 'queued', 'removed']);
  const [first, , last] = seen;
  assert.ok(first?.event === 'queued');
  assert.deepEqual(first.data, { position: 3, of: 3 });
  assert.ok(last?.event === 'removed');
  assert.deepEqual(last.data, {
    reason: 'operator',
    message: 'an operator removed it',
    waitedS: 41.5,
    at: '2026-09-30T10:01:00+00:00',
  });
});

test('started carries how long the job waited', async () => {
  handler = stream([
    'id: 1\nevent: queued\ndata: {"position": 1}\n\n',
    'id: 2\nevent: started\ndata: {"waited_s": 12.25}\n\n',
    'id: 3\nevent: done\ndata: {"artifacts": []}\n\n',
  ]);
  const seen: JobEvent[] = [];
  for await (const event of client().events('j1')) seen.push(event);
  const started = seen[1];
  assert.ok(started?.event === 'started');
  assert.equal(started.data.waitedS, 12.25);
  assert.deepEqual(seen.map((event) => event.event), ['queued', 'started', 'done']);
});

function jobBody(overrides: Record<string, unknown>): Record<string, unknown> {
  return {
    job_id: 'j1', type: 'echo', model: null, status: 'queued', progress: 0, position: 2,
    error: null, artifacts: [], created: '2026-09-30T10:00:00+00:00', started: null,
    finished: null, client_ref: null, interrupted_at: null, held_by: null, held_since: null,
    chunks_done: [], chunks_total: null, chunk_at: null, resume_id: null, resumed: false,
    ...overrides,
  };
}

test('a removed job reads as removed with its reason, not as failed', async () => {
  handler = (_request, _body, response) =>
    json(response, 200, jobBody({
      status: 'removed',
      position: null,
      finished: '2026-09-30T11:00:00+00:00',
      removal: {
        reason: 'server_restart', message: 'the server restarted', waited_s: null,
        at: '2026-09-30T11:00:00+00:00',
      },
    }));
  const job = await client().job('j1');
  assert.equal(job.status, 'removed');
  assert.equal(job.error, null);
  assert.equal(job.removal?.reason, 'server_restart');
  assert.equal(job.removal?.waitedS, null);
});

test('a job from a server older than the queue reads with removal null', async () => {
  handler = (_request, _body, response) => json(response, 200, jobBody({}));
  assert.equal((await client().job('j1')).removal, null);
});

test('cancelling a waiting job answers removed', async () => {
  handler = (_request, _body, response) => json(response, 200, { job_id: 'j1', status: 'removed' });
  assert.deepEqual(await client().cancel('j1'), { jobId: 'j1', status: 'removed' });
});

const ROW = {
  position: 1, job_id: 'j1', type: 'tts', model: 'sigma', client: 'bookforge crucible-client/1.0',
  client_ref: 'chapter 3', submitted: '2026-09-30T10:00:00+00:00', waited_s: 12.5,
  max_wait_s: 3600, expires_at: '2026-09-30T11:00:00+00:00', session: null, kind: 'job',
};

test('queue() lists the waiting jobs in order, and removeFromQueue() removes one', async () => {
  const seen: string[] = [];
  handler = (request, _body, response) => {
    seen.push(`${request.method} ${request.url}`);
    if (request.method === 'GET') {
      json(response, 200, {
        items: [ROW],
        depth: 1,
        limits: {
          per_client: 50, total: 200, max_wait_s: { default: 3600, min: 10, max: 86400 },
          abandon_after_s: 300,
        },
      });
      return;
    }
    json(response, 200, { job_id: 'j1', status: 'removed', reason: 'operator' });
  };
  const listed = await client().queue();
  assert.equal(listed.depth, 1);
  assert.deepEqual(listed.items[0], {
    position: 1, jobId: 'j1', type: 'tts', model: 'sigma', client: 'bookforge crucible-client/1.0',
    clientRef: 'chapter 3', submitted: '2026-09-30T10:00:00+00:00', waitedS: 12.5,
    maxWaitS: 3600, expiresAt: '2026-09-30T11:00:00+00:00', session: null, kind: 'job',
  });
  assert.equal(listed.limits.maxWaitS.default, 3600);
  assert.deepEqual(await client().removeFromQueue('j1'), {
    jobId: 'j1', status: 'removed', reason: 'operator',
  });
  assert.deepEqual(seen, ['GET /v1/queue', 'DELETE /v1/queue/j1']);
});

test('queueEvents() opens with a snapshot and then names every change', async () => {
  handler = (_request, _body, response) => {
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    response.write(`id: 1\nevent: snapshot\ndata: ${JSON.stringify({ items: [ROW], depth: 1 })}\n\n`);
    response.write('id: 2\nevent: added\ndata: {"job_id": "j2", "position": 2, "depth": 2}\n\n');
    response.end('id: 3\nevent: removed\ndata: {"job_id": "j1", "reason": "operator", "depth": 1}\n\n');
  };
  const seen: QueueEvent[] = [];
  for await (const event of client().queueEvents()) {
    seen.push(event);
    if (seen.length === 3) break;
  }
  assert.deepEqual(seen.map((event) => event.event), ['snapshot', 'added', 'removed']);
  const [snapshot, added] = seen;
  assert.ok(snapshot?.event === 'snapshot');
  assert.equal(snapshot.items[0]?.jobId, 'j1');
  assert.ok(added?.event === 'added');
  assert.equal(added.jobId, 'j2');
  assert.equal(added.data['position'], 2);
});

const COMPLETION = {
  id: 'chatcmpl-q',
  model: 'qwen3.5-9b',
  choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
};

const REVISION = '4d1b2f0c9e8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c';

const DECISION = {
  model: { id: 'qwen3.5-9b', revision: REVISION, fingerprint: `qwen3.5-9b@${REVISION}` },
  engine: 'vllm',
  answers: { urgent: { type: 'yesno', p: 0.8, logprob: -0.22, label_mass: 0.99 } },
  timing_ms: {
    total: 10,
    per_question: { urgent: { wall_ms: 5, prompt_tokens: 10, cached_tokens: null } },
    prime: { wall_ms: 5, prompt_tokens: 10, cached_tokens: null },
  },
  tokens: { per_question: { urgent: 12 }, images: 0 },
};

const URGENT = {
  model: 'qwen3.5-9b',
  state: 'Now, please.',
  questions: { urgent: { type: 'yesno', instructions: 'The message conveys urgency' } },
} as const;

test('chat() waits in the queue by default, and not when told not to', async () => {
  handler = (request, body, response) => {
    posted.push(JSON.parse(body));
    json(response, 200, COMPLETION);
  };
  const messages = [{ role: 'user', content: 'hi' }] as const;
  await client().chat({ model: 'qwen3.5-9b', messages });
  await client().chat({ model: 'qwen3.5-9b', messages, queue: { maxWaitS: 120 } });
  await client().chat({ model: 'qwen3.5-9b', messages, queue: false });
  await client({ queue: false }).chat({ model: 'qwen3.5-9b', messages });
  assert.deepEqual(
    posted.map((body) => (body as { queue?: unknown }).queue),
    [{}, { max_wait_s: 120 }, undefined, undefined],
  );
});

test('decide() waits in the queue by default, and not when told not to', async () => {
  handler = (request, body, response) => {
    posted.push(JSON.parse(body));
    json(response, 200, DECISION);
  };
  const answer = await client().decide(URGENT);
  assert.equal(answer.answers['urgent']?.type, 'yesno');
  await client().decide(URGENT, { queue: false });
  assert.deepEqual(
    posted.map((body) => (body as { queue?: unknown }).queue),
    [{}, undefined],
  );
});

test('a queue row of a session waiting to open, and an item of the open one', async () => {
  handler = (_request, _body, response) =>
    json(response, 200, {
      items: [
        { ...ROW, job_id: 'j9', session: 'ses-1' },
        { ...ROW, position: 2, job_id: 'ses-2', type: 'session', model: null, kind: 'session' },
      ],
      depth: 2,
      limits: {
        per_client: 50, total: 200, max_wait_s: { default: 3600, min: 10, max: 86400 },
        abandon_after_s: 300,
      },
    });
  const listed = await client().queue();
  assert.deepEqual(
    listed.items.map((item) => [item.jobId, item.kind, item.session]),
    [['j9', 'job', 'ses-1'], ['ses-2', 'session', null]],
  );
});

test('removeFromQueue() given the open session ends it', async () => {
  handler = (_request, _body, response) =>
    json(response, 200, { job_id: 'ses-1', status: 'closed', reason: 'operator' });
  assert.deepEqual(await client().removeFromQueue('ses-1'), {
    jobId: 'ses-1', status: 'closed', reason: 'operator',
  });
});
