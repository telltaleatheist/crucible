/**
 * Unit tests for `upstreamModels()`, `GET /v1/openai/models`'s upstream rows (2.0.31),
 * against an in-process `node:http` fixture: the resident row is left out, every upstream
 * row and every upstream's list status is read as sent, and a row missing a field is a
 * protocol error rather than a default.
 *
 * Run: `npm run test:unit` (exits non-zero on any failure).
 */

import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import { CrucibleClient, CrucibleProtocolError } from '../src/index.js';

let answer: unknown = {};
let lastPath = '';
let server: Server;
let base = '';

before(async () => {
  server = createServer((request, response) => {
    lastPath = request.url ?? '';
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify(answer));
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 't', clientName: 'unit-upstream-models' });
}

const OPUS = {
  id: 'anthropic/claude-opus-5-5',
  object: 'model',
  owned_by: 'anthropic',
  upstream: 'anthropic',
  routed_for: [],
  listed: true,
  facts: {
    tool_choice: ['auto', 'none'],
    reasoning_efforts: ['low', 'medium', 'high', 'xhigh', 'max'],
    sampling: false,
    structured_output: true,
  },
};

test('upstreamModels() reads the upstream rows and leaves the resident model out', async () => {
  answer = {
    object: 'list',
    data: [
      { id: 'qwen3.5-9b', object: 'model', owned_by: 'crucible@pc', max_model_len: 8192 },
      OPUS,
      { id: 'openai/gpt-5', object: 'model', owned_by: 'openai', upstream: 'openai',
        routed_for: ['translate'], listed: false, facts: null },
    ],
    upstreams: {
      anthropic: { listed_at: '2026-10-25T18:00:00+00:00', stale: false, error: null },
      openai: { listed_at: null, stale: false,
        error: { code: 'upstream_unreachable', message: 'nothing answered' } },
    },
  };
  const read = await client().upstreamModels();
  assert.equal(lastPath, '/v1/openai/models');
  assert.deepEqual(read.models.map((m) => m.id), ['anthropic/claude-opus-5-5', 'openai/gpt-5']);
  assert.deepEqual(read.models[0], {
    id: 'anthropic/claude-opus-5-5',
    upstream: 'anthropic',
    routedFor: [],
    listed: true,
    facts: {
      toolChoice: ['auto', 'none'],
      reasoningEfforts: ['low', 'medium', 'high', 'xhigh', 'max'],
      sampling: false,
      structuredOutput: true,
    },
  });
  assert.equal(read.models[1]?.facts, null);
  assert.deepEqual(read.upstreams['openai'], {
    listedAt: null, stale: false, error: { code: 'upstream_unreachable', message: 'nothing answered' },
  });
});

test('a row without listed, or an answer without upstreams, is a protocol error', async () => {
  const { listed: _listed, ...partial } = OPUS;
  answer = { object: 'list', data: [partial], upstreams: {} };
  await assert.rejects(client().upstreamModels(), CrucibleProtocolError);
  answer = { object: 'list', data: [OPUS] };
  await assert.rejects(client().upstreamModels(), CrucibleProtocolError);
});
