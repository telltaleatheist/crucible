/**
 * A fleet session: one queue session asked of several servers at once.
 *
 * What the client has to get right: only servers that can serve the request are
 * asked (a server that does not answer, lacks `queue.sessions`, or lacks the
 * model is left out with its reason); the first session to OPEN wins and every
 * other one is taken out of its line at once, including one that opened in the
 * same instant; each server's place is reported; aborting removes every
 * session asked for, even one whose ticket was still on the wire; and when no
 * server can give a session the error names each server and why.
 *
 * Run: `npm run test:unit` (this file is found by `scripts/unit.mjs`).
 */

import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { afterEach, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleFleetUnavailable,
  FLEET_UNAVAILABLE,
  fleetSession,
  type FleetQueueUpdate,
} from '../src/index.js';

const INFO = {
  server: { name: 'crucible@fake', version: '1.0.77', api_version: 1 },
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

function modelRow(id: string, overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id, family: 'qwen3.5', params_b: 9, revision: 'abc', fingerprint: `${id}@abc`,
    modalities: ['text'], weights_of: null, orphan: null, backend_supported: true,
    installed: true, resident: false, loadable: true, memory_bytes_estimate: null,
    held_by: null, unclaimed_since: null, context_default: 12288, max_model_len: null,
    ...overrides,
  };
}

interface Frame {
  readonly id: number;
  readonly event: string;
  readonly data: Record<string, unknown>;
}

interface Seen {
  readonly method: string;
  readonly path: string;
  readonly body: string;
}

/** One fake Crucible: its info, its catalogue, and one queue session's stream. */
class Fake {
  readonly name: string;
  readonly sessionId: string;
  readonly seen: Seen[] = [];
  readonly frames: Frame[] = [];
  readonly live = new Set<ServerResponse>();
  features: string[] = [...INFO.features];
  models: Record<string, unknown>[] = [modelRow('qwen3.5-9b')];
  /** The status the ticket answers with: `open` at once, or `queued` at `position`. */
  ticket: { status: 'open' | 'queued'; position: number | null } = { status: 'queued', position: 1 };
  /** When set, the POST's answer waits for this to resolve. */
  holdTicket: Promise<void> | null = null;
  /** When true, `/v1/info` never answers. */
  silent = false;
  status: 'none' | 'queued' | 'open' | 'closed' = 'none';
  server!: Server;
  url = '';

  constructor(name: string) {
    this.name = name;
    this.sessionId = `ses-${name}`;
  }

  async start(): Promise<void> {
    this.server = createServer((request, response) => {
      let body = '';
      request.on('data', (chunk) => (body += chunk));
      request.on('end', () => void this.handle(request, body, response));
    });
    await new Promise<void>((resolve) => this.server.listen(0, '127.0.0.1', resolve));
    this.url = `http://127.0.0.1:${(this.server.address() as AddressInfo).port}`;
  }

  async stop(): Promise<void> {
    this.server.closeAllConnections();
    await new Promise<void>((resolve) => this.server.close(() => resolve()));
  }

  client(): CrucibleClient {
    return new CrucibleClient({ url: this.url, token: 't', clientName: 'fleet-test' });
  }

  say(event: string, data: Record<string, unknown>): void {
    const frame = { id: this.frames.length + 1, event, data: { session_id: this.sessionId, ...data } };
    this.frames.push(frame);
    for (const response of this.live) response.write(sse(frame));
  }

  open(): void {
    this.status = 'open';
    this.say('opened', { opened_at: '2026-10-01T10:00:01+00:00', model: null, load_job: null });
  }

  deletes(): number {
    return this.seen.filter((entry) => entry.method === 'DELETE').length;
  }

  posts(): Seen[] {
    return this.seen.filter((entry) => entry.method === 'POST' && entry.path === '/v1/queue/sessions');
  }

  async handle(request: IncomingMessage, body: string, response: ServerResponse): Promise<void> {
    const path = request.url ?? '';
    const method = request.method ?? '';
    this.seen.push({ method, path, body });
    const own = `/v1/queue/sessions/${this.sessionId}`;
    if (method === 'GET' && path === '/v1/info') {
      if (this.silent) return; // never answers
      json(response, 200, { ...INFO, features: this.features });
      return;
    }
    if (method === 'GET' && path === '/v1/models') {
      json(response, 200, this.models);
      return;
    }
    if (method === 'POST' && path === '/v1/queue/sessions') {
      if (this.holdTicket !== null) await this.holdTicket;
      this.status = this.ticket.status;
      if (this.ticket.status === 'queued') {
        this.say('queued', { position: this.ticket.position, of: this.ticket.position });
      } else {
        this.say('opened', { opened_at: '2026-10-01T10:00:01+00:00', model: null, load_job: null });
      }
      json(response, 202, { session_id: this.sessionId, ...this.ticket });
      return;
    }
    if (method === 'GET' && path === `${own}/events`) {
      const after = Number(request.headers['last-event-id'] ?? 0);
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      for (const frame of this.frames) if (frame.id > after) response.write(sse(frame));
      this.live.add(response);
      response.on('close', () => this.live.delete(response));
      return;
    }
    if (method === 'DELETE' && path === own) {
      const wasOpen = this.status === 'open';
      this.status = 'closed';
      if (wasOpen) {
        this.say('closed', { reason: 'client', message: 'the client closed it', items_run: 0, held_s: 0 });
      } else {
        this.say('removed', { reason: 'client', message: 'the client removed it' });
      }
      json(response, 200, closedState(this.sessionId, wasOpen));
      return;
    }
    json(response, 404, { error: { code: 'not_found', message: path } });
  }
}

function sse(frame: Frame): string {
  return `id: ${frame.id}\nevent: ${frame.event}\ndata: ${JSON.stringify(frame.data)}\n\n`;
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

function closedState(id: string, opened: boolean): Record<string, unknown> {
  return {
    session_id: id, status: 'closed', act: 'analysis', client: 'fleet-test', model: null,
    position: null, idle_s: 300, max_wait_s: 3600, created: '2026-10-01T10:00:00+00:00',
    opened_at: opened ? '2026-10-01T10:00:01+00:00' : null, idle_deadline: null,
    max_hold_deadline: null, items_run: 0, in_flight: [], stream_session: null, load_job: null,
    closed_at: '2026-10-01T10:00:10+00:00', reason: 'client', message: 'the client closed it',
    error: null,
  };
}

let fakes: Fake[] = [];

async function fake(name: string): Promise<Fake> {
  const made = new Fake(name);
  await made.start();
  fakes.push(made);
  return made;
}

/** A URL nothing listens on: a server that is down. */
async function deadUrl(): Promise<string> {
  const probe = createServer();
  await new Promise<void>((resolve) => probe.listen(0, '127.0.0.1', resolve));
  const port = (probe.address() as AddressInfo).port;
  await new Promise<void>((resolve) => probe.close(() => resolve()));
  return `http://127.0.0.1:${port}`;
}

afterEach(async () => {
  for (const made of fakes) await made.stop();
  fakes = [];
});

/** Resolves once `predicate` holds; fails after two seconds. */
async function until(predicate: () => boolean, what: string): Promise<void> {
  const deadline = Date.now() + 2_000;
  while (!predicate()) {
    if (Date.now() > deadline) assert.fail(`timed out waiting for ${what}`);
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

test('the first session to open wins, and every other one leaves its line', async () => {
  const pc = await fake('pc');
  const mac = await fake('mac');
  pc.ticket = { status: 'queued', position: 2 };
  mac.ticket = { status: 'queued', position: 1 };
  const pcClient = pc.client();
  const macClient = mac.client();
  const heard: FleetQueueUpdate[] = [];

  const asking = fleetSession([pcClient, macClient], {
    act: 'analysis',
    model: 'qwen3.5-9b',
    maxWaitS: 600,
    onQueue: (update) => heard.push(update),
  });
  await until(() => pc.live.size > 0 && mac.live.size > 0, 'both servers followed');
  await until(() => heard.length === 2, 'both places reported');
  mac.open();
  const won = await asking;
  try {
    assert.equal(won.client, macClient);
    assert.equal(won.index, 1);
    assert.equal(won.session.id, 'ses-mac');
    assert.equal(won.session.url, mac.url);
    assert.deepEqual(won.dropouts, []);
    await until(() => pc.deletes() === 1, 'the loser taken out of its line');
    assert.equal(pc.status, 'closed');
    assert.equal(mac.deletes(), 0, 'the winner is left open');

    for (const server of [pc, mac]) {
      const posts = server.posts();
      assert.equal(posts.length, 1);
      assert.deepEqual(JSON.parse(posts[0]!.body), {
        act: 'analysis', model: 'qwen3.5-9b', max_wait_s: 600,
      });
    }
    const last = heard[heard.length - 1]!;
    assert.deepEqual(
      last.places.map((place) => [place.client.url, place.position?.position]).sort(),
      [[mac.url, 1], [pc.url, 2]].sort(),
    );
  } finally {
    await won.session.close();
  }
  assert.equal(mac.deletes(), 1);
});

test('two servers open in the same instant: one wins, the other is closed at once', async () => {
  const pc = await fake('pc');
  const mac = await fake('mac');
  pc.ticket = { status: 'open', position: null };
  mac.ticket = { status: 'open', position: null };

  const won = await fleetSession([pc.client(), mac.client()], { act: 'analysis' });
  try {
    const loser = won.session.url === pc.url ? mac : pc;
    const winner = loser === pc ? mac : pc;
    await until(() => loser.deletes() === 1, 'the loser that opened closed');
    assert.equal(loser.status, 'closed');
    assert.equal(winner.status, 'open');
    assert.equal(winner.deletes(), 0);
  } finally {
    await won.session.close();
  }
});

test('two queued sessions opening together: still exactly one winner', async () => {
  const pc = await fake('pc');
  const mac = await fake('mac');
  const asking = fleetSession([pc.client(), mac.client()], { act: 'analysis' });
  await until(() => pc.live.size > 0 && mac.live.size > 0, 'both servers followed');
  pc.open();
  mac.open();
  const won = await asking;
  try {
    const loser = won.session.url === pc.url ? mac : pc;
    await until(() => loser.deletes() === 1, 'the loser closed');
    assert.equal(loser.status, 'closed');
    assert.equal(won.session.ended, null);
  } finally {
    await won.session.close();
  }
});

test('a server that is down, or does not answer, is left out with its reason', async () => {
  const silent = await fake('silent');
  silent.silent = true;
  const pc = await fake('pc');
  const down = new CrucibleClient({ url: await deadUrl(), token: 't', clientName: 'fleet-test' });

  const asking = fleetSession([down, silent.client(), pc.client()], {
    act: 'analysis',
    probeTimeoutMs: 200,
  });
  await until(() => pc.live.size > 0, 'the capable server followed');
  await new Promise((resolve) => setTimeout(resolve, 400)); // past the silent server's probe clock
  pc.open();
  const won = await asking;
  try {
    assert.equal(won.session.url, pc.url);
    assert.equal(won.index, 2);
    assert.deepEqual(
      won.dropouts.map((entry) => [entry.url, entry.stage]).sort(),
      [[down.url, 'probe'], [silent.url, 'probe']].sort(),
    );
    const quiet = won.dropouts.find((entry) => entry.url === silent.url)!;
    assert.equal(quiet.reason, 'it did not answer within 200 ms');
    assert.equal(silent.posts().length, 0);
  } finally {
    await won.session.close();
  }
});

test('no server can serve: the error names every server and why', async () => {
  const old = await fake('old');
  old.features = ['events', 'queue.jobs'];
  const other = await fake('other');
  other.models = [modelRow('gemma-27b')];
  const mac = await fake('mac');
  mac.models = [modelRow('qwen3.5-9b', { backend_supported: false, reason: 'no mlx-darwin block' })];
  const down = new CrucibleClient({ url: await deadUrl(), token: 't', clientName: 'fleet-test' });

  await assert.rejects(
    fleetSession([old.client(), other.client(), mac.client(), down], {
      act: 'analysis',
      model: 'qwen3.5-9b',
    }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleFleetUnavailable);
      assert.equal(error.code, FLEET_UNAVAILABLE);
      assert.deepEqual(
        error.servers.map((entry) => entry.url),
        [old.url, other.url, mac.url, down.url],
      );
      assert.ok(error.servers.every((entry) => entry.stage === 'probe'));
      assert.match(error.servers[0]!.reason, /queue\.sessions/);
      assert.match(error.servers[1]!.reason, /qwen3\.5-9b is not in its catalogue/);
      assert.match(error.servers[2]!.reason, /not supported on its backend \(no mlx-darwin block\)/);
      assert.match(error.message, /can serve a queue session for act analysis with qwen3\.5-9b/);
      return true;
    },
  );
  for (const server of [old, other, mac]) assert.equal(server.posts().length, 0);
});

test('every asked server ends its session before it opens: the error says so per server', async () => {
  const pc = await fake('pc');
  const mac = await fake('mac');
  const asking = fleetSession([pc.client(), mac.client()], { act: 'analysis' });
  await until(() => pc.live.size > 0 && mac.live.size > 0, 'both servers followed');
  pc.say('removed', { reason: 'operator', message: 'an operator removed it' });
  mac.say('removed', { reason: 'load_failed', message: 'the load failed', error: { code: 'oom', message: 'oom' } });
  await assert.rejects(asking, (error: unknown) => {
    assert.ok(error instanceof CrucibleFleetUnavailable);
    assert.deepEqual(error.servers.map((entry) => [entry.url, entry.stage]), [
      [pc.url, 'line'],
      [mac.url, 'line'],
    ]);
    assert.match(error.servers[0]!.reason, /\(operator\)/);
    assert.match(error.servers[1]!.reason, /\(load_failed\)/);
    return true;
  });
});

test('aborting takes every session out of its line, and throws the abort', async () => {
  const pc = await fake('pc');
  const mac = await fake('mac');
  const controller = new AbortController();
  const asking = fleetSession([pc.client(), mac.client()], {
    act: 'analysis',
    signal: controller.signal,
  });
  await until(() => pc.live.size > 0 && mac.live.size > 0, 'both servers followed');
  const reason = new Error('the user went home');
  controller.abort(reason);
  await assert.rejects(asking, (error: unknown) => error === reason);
  await until(() => pc.deletes() === 1 && mac.deletes() === 1, 'both sessions removed');
  assert.equal(pc.status, 'closed');
  assert.equal(mac.status, 'closed');
});

test('an abort while a ticket is still on the wire removes the session it names', async () => {
  const pc = await fake('pc');
  let answer: () => void = () => undefined;
  pc.holdTicket = new Promise<void>((resolve) => {
    answer = resolve;
  });
  pc.ticket = { status: 'open', position: null };
  const controller = new AbortController();
  const asking = fleetSession([pc.client()], { act: 'analysis', signal: controller.signal });
  await until(() => pc.posts().length === 1, 'the ticket asked for');
  controller.abort();
  await assert.rejects(asking, (error: unknown) => error instanceof Error && error.name === 'AbortError');
  assert.equal(pc.deletes(), 0, 'nothing to remove yet: the server has not answered');
  answer();
  await until(() => pc.deletes() === 1, 'the session the late ticket named removed');
  assert.equal(pc.status, 'closed');
});

test('a fleet is refused by name when it is misconfigured', async () => {
  const pc = await fake('pc');
  await assert.rejects(fleetSession([], { act: 'analysis' }), /at least one CrucibleClient/);
  await assert.rejects(
    fleetSession([pc.client(), pc.client()], { act: 'analysis' }),
    /in the fleet twice/,
  );
  await assert.rejects(fleetSession([pc.client()], { act: 'analysis', maxWaitS: 5 }), /maxWaitS/);
  assert.equal(pc.seen.length, 0);
});
