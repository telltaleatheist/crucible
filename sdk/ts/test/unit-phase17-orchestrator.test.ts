/**
 * PHASE17-ORCHESTRATOR.md, from the client's side: the three fields `info()`
 * gained, and `engineOf`.
 *
 * The whole of what an app has to learn this phase is one function. Everything
 * else — the claim, the release, the restart — happens between two Crucible
 * processes and an app never sees it.
 *
 * Run: `npm run test:unit`.
 */

import assert from 'node:assert/strict';
import { createServer, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleProtocolError,
  engineOf,
  type ServerInfo,
} from '../src/index.js';

// ------------------------------------------------------------------ fixture

let reply: unknown = {};
let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-phase17' });
}

function json(response: ServerResponse, body: unknown): void {
  response.writeHead(200, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

before(async () => {
  server = createServer((_request, response) => json(response, reply));
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

function engineDocument(): Record<string, unknown> {
  return {
    server: { name: 'crucible@owens-pc-wsl', version: '0.5.0', api_version: 1 },
    host: {
      platform: 'linux',
      arch: 'x86_64',
      backend: 'cuda-linux',
      gpu: { vendor: 'nvidia', name: '3090 Ti', vram_bytes: 25757220864 },
    },
    job_types: ['llm', 'tts'],
    features: ['queue.sessions', 'events'],
    capabilities: [],
    pages_engine: {
      engine: null,
      installed: false,
      detail: 'dots-ocr has no cuda-linux block',
      request: {
        model: 'dots-ocr',
        dpi: 200,
        max_pixels: 11289600,
        max_tokens: 8192,
        temperature: 0.0,
        prompt: 'Please output the layout information from the PDF image.',
        dialect: 'dots-json',
        concurrency: 12,
        truncated_finish_reason: 'length',
      },
    },
  };
}

// ---------------------------------------------------------------- an engine

test('an unclaimed engine says so, and null is a complete answer', async () => {
  reply = { ...engineDocument(), role: 'engine', managed_by: null };
  const info = await client().info();
  assert.equal(info.role, 'engine');
  assert.equal(info.managedBy, null);
  assert.equal(engineOf(info), null);
});

test('a claimed engine names who manages it, and not their version', async () => {
  reply = {
    ...engineDocument(),
    role: 'engine',
    managed_by: { name: 'crucible-orchestrator@owens-pc', url: 'http://127.0.0.1:7101' },
  };
  const info = await client().info();
  assert.deepEqual(info.managedBy, {
    name: 'crucible-orchestrator@owens-pc',
    url: 'http://127.0.0.1:7101',
  });
  // Still the engine: an app talks HERE, claimed or not. A claim is a
  // statement of fact between two Crucible processes, never a gate.
  assert.equal(engineOf(info), null);
});

// ---------------------------------------------------------- an orchestrator

function orchestratorDocument(engine: unknown): Record<string, unknown> {
  return {
    server: { name: 'crucible-orchestrator@owens-pc', version: '0.6.0', api_version: 1 },
    host: {
      platform: 'win32',
      arch: 'AMD64',
      backend: 'orchestrator',
      gpu: { vendor: 'none', name: '', vram_bytes: 0 },
    },
    role: 'orchestrator',
    job_types: [],
    features: ['queue.sessions', 'events'],
    engine,
    capabilities: [],
  };
}

test('an orchestrator says it plays nothing, and where the engine is', async () => {
  reply = orchestratorDocument({
    name: 'crucible@owens-pc-wsl',
    url: 'http://127.0.0.1:7100',
    backend: 'cuda-linux',
    owner: 'wsl-unit',
  });
  const info = await client().info();
  assert.equal(info.role, 'orchestrator');
  assert.equal(info.host.backend, 'orchestrator');
  assert.deepEqual(info.jobTypes, [], 'zero job types is the DEFINITION of the role');
  assert.equal(info.managedBy, null, 'an orchestrator claims and is not claimed');

  const followed = engineOf(info);
  assert.ok(followed);
  assert.equal(followed.url, 'http://127.0.0.1:7100');
  assert.equal(followed.backend, 'cuda-linux');
  assert.equal(followed.owner, 'wsl-unit');
});

test('an engine the orchestrator cannot read still has an address', async () => {
  // PHASE17 3.2: `url` is a fact about the MACHINE rather than about the
  // engine's health, so it stands while `name` and `backend` go null. An app
  // follows it and gets the truth from the engine itself, or fails to reach
  // it and knows that too.
  reply = orchestratorDocument({
    name: null,
    url: 'http://127.0.0.1:7100',
    backend: null,
    owner: 'wsl-unit',
  });
  const followed = engineOf(await client().info());
  assert.ok(followed);
  assert.equal(followed.name, null);
  assert.equal(followed.backend, null);
  assert.equal(followed.url, 'http://127.0.0.1:7100');
});

test('an orchestrator with no engine throws orchestrator_has_no_engine', async () => {
  // A fact to show a person, next to the button that installs one.
  reply = orchestratorDocument(null);
  const info = await client().info();
  assert.equal(info.engine, null);
  assert.throws(() => engineOf(info), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError);
    assert.match(error.message, /orchestrator_has_no_engine/);
    assert.match(error.message, /crucible-orchestrator@owens-pc/);
    return true;
  });
});

test('an owner word this build has not heard of is carried, not refused', async () => {
  // `Health.residentKind`'s rule, for its reason: the set grows, and a client
  // that threw on a word the server added would break on that server.
  reply = orchestratorDocument({
    name: 'x',
    url: 'http://127.0.0.1:7100',
    backend: 'cuda-linux',
    owner: 'something-later',
  });
  const followed = engineOf(await client().info());
  assert.equal(followed?.owner, 'something-later');
});

test('a role that is neither word is refused by name', async () => {
  reply = { ...engineDocument(), role: 'conductor' };
  await assert.rejects(client().info(), CrucibleProtocolError);
});

test('an orchestrator that states its role and omits engine is a defect', async () => {
  const document = orchestratorDocument(null);
  delete document.engine;
  reply = document;
  await assert.rejects(client().info(), CrucibleProtocolError);
});

// ------------------------------------------------------------------ one hop

test('following an engine.url that names another orchestrator is refused', async () => {
  // PHASE17 section 6: ONCE, and never a chain. An app checks the second
  // document's role and refuses anything but `engine` rather than looping.
  reply = orchestratorDocument({
    name: 'not-really-an-engine',
    url: base,
    backend: 'orchestrator',
    owner: 'child',
  });
  const first = await client().info();
  const hop = engineOf(first);
  assert.ok(hop);

  // The app follows ONE hop, with the SAME token, and looks at what answered.
  const second: ServerInfo = await new CrucibleClient({
    url: hop.url,
    token: 'the-token',
    clientName: 'unit-phase17',
  }).info();
  assert.equal(second.role, 'orchestrator', 'the fixture answers itself');
  assert.notEqual(
    second.role,
    'engine',
    'so the app stops here rather than following a second time',
  );
});
