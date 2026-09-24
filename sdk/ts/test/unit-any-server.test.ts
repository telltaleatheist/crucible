/**
 * ANY CRUCIBLE THAT ANSWERS WORKS (Owen, 2026-09-24).
 *
 * *"lets modify bookforge and foundry so they dont require any particular
 * crucible server. if it can make the call to the crucible server then it
 * should work."* It replaced the lockstep ruling of 2026-09-20, under which this
 * client demanded every field it knew and a client one release ahead of its
 * server refused it by name. After the 1.0.24 repin that cost ten BookForge
 * keepers whose fake servers simply lacked the newest fields.
 *
 * The rule these tests pin, in three parts:
 *
 * 1. A server one release BEHIND this client — here, the fields 1.0.24 added
 *    removed — reads cleanly, with `null` where a field is not stated.
 * 2. A missing LOAD-BEARING field is still refused, by name.
 * 3. An informational field that is PRESENT with the wrong type is still
 *    refused: API v1 adds fields and never retypes them, so a wrong type is a
 *    broken server, never an old one.
 *
 * And the misconfigurations stay refusals (401, 426, not-a-crucible,
 * unreachable) — `unit.test.ts` pins those and nothing here loosens them.
 *
 * Other files pin the rule per door (`unit-capability`, `unit-voices`,
 * `unit-decide`, `unit-llm`, `unit-render`, `unit-stream`, ...). This one is
 * the whole-server view: an older Crucible, asked everything a probe asks.
 *
 * Run: `npm run test:unit` (exits non-zero on any failure).
 */

import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleProtocolError,
  CrucibleVersionError,
  isLlmCapability,
  isTtsCapability,
} from '../src/index.js';

/** Routes → bodies. Each test sets what it needs. */
let routes: Record<string, { status: number; body: unknown }> = {};

let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-any-server' });
}

before(async () => {
  server = createServer((request, response) => {
    const path = (request.url ?? '').split('?')[0] ?? '';
    const route = routes[path];
    if (route === undefined) {
      response.writeHead(404, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ error: { code: 'not_found', message: path } }));
      return;
    }
    response.writeHead(route.status, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify(route.body));
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

// ------------------------------------------------- a 1.0.23-shaped server

const REVISION = '4d1b2f0c9e8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c';

/**
 * A voice row as a 1.0.23 server wrote it: no `orphan`, and — going further
 * back than one release, because the rule is not "one release" — no
 * `serving`, `takes`, `display` or `estimate_basis` either.
 */
const OLD_VOICE = {
  id: 'deathstalker',
  kind: 'checkpoint',
  language: 'en',
  narrator_engine: 'higgs-v3',
  backend_supported: true,
  installed: true,
  resident: true,
  loadable: true,
  reason: null,
  revision: REVISION,
  fingerprint: `deathstalker@${REVISION}`,
  memory_bytes_estimate: 19000000000,
  max_chars: 800,
  sample_rate: 24000,
  needs_reference: false,
  pace: {
    pace_chars_per_sec: 16.64,
    max_chars_per_sec: 21.63,
    min_chars_per_sec: 12.8,
    target_chars: null,
    safe_min_chars: 600,
    safe_max_chars: 800,
  },
};

/** A model row with only the load-bearing fields and a few of the old ones. */
const OLD_MODEL = {
  id: 'qwen3.5-9b',
  family: 'qwen3.5',
  params_b: 9,
  revision: REVISION,
  modalities: ['text'],
  installed: true,
  resident: false,
  loadable: true,
  context_default: 12288,
};

const OLD_INFO = {
  server: { name: 'crucible@older', version: '1.0.23', api_version: 1 },
  host: {
    platform: 'linux',
    arch: 'x86_64',
    backend: 'cuda-linux',
    gpu: { vendor: 'nvidia', name: 'NVIDIA GeForce RTX 3090 Ti', vram_bytes: 25757220864 },
  },
  job_types: ['echo', 'tts', 'load-voice', 'load-model'],
  capabilities: [
    { job_type: 'echo', models: [] },
    { job_type: 'llm', models: [OLD_MODEL] },
    { job_type: 'tts', models: [OLD_VOICE] },
  ],
};

/** `/v1/capability` before `work`, `context_ceilings` and `route` existed. */
const OLD_CAPABILITY = {
  backend_kind: 'cuda-linux',
  total_bytes: 24 * 1024 ** 3,
  desktop_allowance_bytes: 3 * 1024 ** 3,
  classes: [
    {
      capability: 'clean',
      enabled: true,
      selected: 'qwen3.5-9b',
      reason: 'qwen3.5-9b fits',
      shortfall_bytes: 0,
    },
  ],
};

/** A job record before `chunks_total`, `chunk_at`, `client_ref` and `lease_id`. */
const OLD_JOB = {
  job_id: 'j1',
  type: 'tts',
  model: 'deathstalker',
  status: 'running',
  progress: 0.25,
  position: 0,
  error: null,
  artifacts: ['0.flac'],
  created: '2026-09-20T10:00:00Z',
  started: '2026-09-20T10:00:01Z',
  finished: null,
  chunks_done: [0],
};

test('a 1.0.23-shaped server answers every probe, with null where it says nothing', async () => {
  routes = {
    '/v1/info': { status: 200, body: OLD_INFO },
    '/v1/capability': { status: 200, body: OLD_CAPABILITY },
    '/v1/voices': { status: 200, body: [OLD_VOICE] },
    '/v1/models': { status: 200, body: [OLD_MODEL] },
    '/v1/jobs/j1': { status: 200, body: OLD_JOB },
  };
  const crucible = client();

  const info = await crucible.info();
  const llm = info.capabilities.find(isLlmCapability);
  const tts = info.capabilities.find(isTtsCapability);
  assert.ok(llm !== undefined && tts !== undefined);
  assert.deepEqual(llm.unreadableRows, []);
  assert.deepEqual(tts.unreadableRows, []);
  const [voice] = tts.models;
  assert.equal(voice?.orphan, null);
  assert.equal(voice?.serving, null);
  assert.equal(voice?.takes, null);
  assert.equal(voice?.display, null);
  assert.equal(voice?.estimateBasis, null);
  assert.equal(voice?.sampleRate, 24000);
  const [model] = llm.models;
  assert.equal(model?.weightsOf, null);
  assert.equal(model?.maxModelLen, null);
  assert.equal(model?.fingerprint, null);
  assert.equal(model?.reason, null);
  // Pre-phase-17 and pre-3.10: the vintage readings, not refusals.
  assert.equal(info.role, 'engine');
  assert.equal(info.pagesEngine, null);

  const [row] = (await crucible.capability()).classes;
  assert.equal(row?.work, null);
  assert.equal(row?.contextCeilings, null);
  // No row states a route: a server that predates routing, where every class
  // IS local.
  assert.equal(row?.route, 'local');

  assert.equal((await crucible.voices())[0]?.orphan, null);
  assert.equal((await crucible.models())[0]?.weightsOf, null);

  const job = await crucible.job('j1');
  assert.equal(job.status, 'running');
  assert.deepEqual(job.chunksDone, [0]);
  assert.equal(job.chunksTotal, null);
  assert.equal(job.chunkAt, null);
  assert.equal(job.clientRef, null);
  assert.equal(job.leaseId, null);
  assert.equal(job.interruptedAt, null);
});

// --------------------------------------- load-bearing: still refused by name

test('a job record without its status is refused by name', async () => {
  const { status: _gone, ...withoutStatus } = OLD_JOB;
  routes = { '/v1/jobs/j1': { status: 200, body: withoutStatus } };
  await assert.rejects(client().job('j1'), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /job has no field "status"/);
    return true;
  });
});

test('a job record without chunks_done still reads: null is "unknown", not "none done"', async () => {
  const { chunks_done: _gone, ...withoutDone } = OLD_JOB;
  routes = { '/v1/jobs/j1': { status: 200, body: withoutDone } };
  const job = await client().job('j1');
  assert.equal(job.status, 'running');
  assert.equal(job.chunksDone, null);
});

test('a present chunks_done with a non-integer index is still refused', async () => {
  routes = { '/v1/jobs/j1': { status: 200, body: { ...OLD_JOB, chunks_done: [0, 1.5] } } };
  await assert.rejects(client().job('j1'), /job\.chunks_done\[1\] is not an integer chunk index/);
});

test('a job status outside the lifecycle is still refused: callers branch on it', async () => {
  routes = { '/v1/jobs/j1': { status: 200, body: { ...OLD_JOB, status: 'pondering' } } };
  await assert.rejects(client().job('j1'), /job\.status is "pondering"/);
});

test('an info document without the server name is refused by name', async () => {
  routes = {
    '/v1/info': {
      status: 200,
      body: { ...OLD_INFO, server: { version: '1.0.23', api_version: 1 } },
    },
  };
  await assert.rejects(client().info(), /info\.server has no field "name"/);
});

test('an info document with no host block still reads: it describes the machine', async () => {
  const { host: _gone, ...withoutHost } = OLD_INFO;
  routes = { '/v1/info': { status: 200, body: withoutHost } };
  const info = await client().info();
  assert.equal(info.host.backend, null);
  assert.equal(info.host.gpu, null);
  assert.equal(info.server.version, '1.0.23');
});

test('a done event that says neither what it made nor what is resident is refused', async () => {
  // `done` must say what finished; `artifact` must name what to fetch.
  routes = {};
  const sse = createServer((_request, response) => {
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    response.end('id: 1\nevent: done\ndata: {"rendered": 3}\n\n');
  });
  await new Promise<void>((resolve) => sse.listen(0, '127.0.0.1', resolve));
  try {
    const url = `http://127.0.0.1:${(sse.address() as AddressInfo).port}`;
    const c = new CrucibleClient({ url, token: 't', clientName: 'unit-any-server' });
    await assert.rejects(async () => {
      for await (const _ of c.events('j1')) void _;
    }, /carries neither "artifacts" nor "resident"/);
  } finally {
    await new Promise<void>((resolve) => sse.close(() => resolve()));
  }
});

// ------------------------------------------- wrong-typed: still a protocol error

test('an informational field present with the wrong type is refused, naming it', async () => {
  routes = { '/v1/jobs/j1': { status: 200, body: { ...OLD_JOB, chunks_total: 'many' } } };
  await assert.rejects(client().job('j1'), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /job\.chunks_total is present but is not a number \(got string\)/);
    return true;
  });
});

test('an explicit null on an informational field is the same answer as its absence', async () => {
  routes = { '/v1/jobs/j1': { status: 200, body: { ...OLD_JOB, chunks_total: null, chunk_at: null } } };
  const job = await client().job('j1');
  assert.equal(job.chunksTotal, null);
  assert.equal(job.chunkAt, null);
});

// ------------------------------------------------ misconfiguration: unchanged

test('an API major the client does not speak is still a named refusal, not tolerance', async () => {
  // Version skew inside API v1 is weather. A different API is not: the
  // handshake refuses it by name, and nothing about this rule touches that.
  routes = {
    '/v1/info': {
      status: 426,
      body: {
        error: {
          code: 'api_version_mismatch',
          message: 'this server speaks api 2',
          details: { server_api_version: 2 },
        },
      },
    },
  };
  await assert.rejects(client().info(), (error: unknown) => {
    assert.ok(error instanceof CrucibleVersionError, `got ${String(error)}`);
    assert.equal(error.serverApiVersion, 2);
    return true;
  });
});
