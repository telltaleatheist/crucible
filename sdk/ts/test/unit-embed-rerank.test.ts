/**
 * Unit tests for `embed()` (`POST /v1/embed`) and `rerank()` (`POST /v1/rerank`) against an
 * in-process `node:http` fixture: what goes on the wire (snake_case, the queue and act the client
 * holds), how the reply is read and refused when it does not answer what was asked, the base64
 * vector decoder, and the new rows of `GET /v1/models` and `GET /v1/info`.
 *
 * Run: `npm run test:unit` (exits non-zero on any failure).
 */

import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleConfigError,
  CrucibleProtocolError,
  CrucibleRefused,
  decodeEmbedding,
} from '../src/index.js';
import { UNCLAIMED_ENGINE } from './engine-info.js';

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => json(response, 500, { error: { code: 'no_handler', message: 'none' } });
let lastPath = '';
let lastBody = '';
let lastHeaders: Record<string, string | string[] | undefined> = {};
let requests = 0;

let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-embed' });
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

before(async () => {
  server = createServer((request, response) => {
    requests += 1;
    lastPath = request.url ?? '';
    lastHeaders = request.headers;
    const chunks: Buffer[] = [];
    request.on('data', (chunk: Buffer) => chunks.push(chunk));
    request.on('end', () => {
      lastBody = Buffer.concat(chunks).toString('utf8');
      handle(request, response, lastBody);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

const FINGERPRINT = 'qwen3-embedding-8b@ac37281f6437:qwen3-embedding-8b-bf16.gguf:llama-server-b10970:e1';

function embedReply(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    object: 'crucible.embeddings',
    model: {
      id: 'qwen3-embedding-8b', revision: 'r'.repeat(40), file: 'qwen3-embedding-8b-bf16.gguf', form: null,
      bits: 16, engine: 'llama-server', engine_build: 'llama-server-b10970', scheme: 1,
      fingerprint: FINGERPRINT, dimensions: 4096,
    },
    dimensions: 2,
    input_type: 'query',
    instruction: 'Find it',
    encoding_format: 'float',
    embeddings: [[0.6, 0.8], [1, 0]],
    tokens: { per_input: [5, 3], total: 8 },
    timing_ms: { total: 12.5, queued: 0.4 },
    ...overrides,
  };
}

function rerankReply(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    object: 'crucible.rerank',
    model: {
      id: 'qwen3-reranker-8b', revision: 'r'.repeat(40), file: 'Qwen3-Reranker-8B-bf16.gguf', form: null,
      engine: 'llama-server', engine_build: 'llama-server-b10970', template: 'model',
      fingerprint: 'qwen3-reranker-8b@x:f:b:r1-model',
    },
    instruction: 'Given a web search query, retrieve relevant passages that answer the query',
    scores: [0.1, 0.9],
    results: [{ index: 1, relevance_score: 0.9 }, { index: 0, relevance_score: 0.1 }],
    tokens: { per_document: [40, 41], total: 162, cached: 120 },
    timing_ms: { total: 80, queued: null },
    ...overrides,
  };
}

test('embed sends its body in snake_case and reads the answer', async () => {
  handle = (_request, response) => json(response, 200, embedReply());
  const answer = await client().embed(
    {
      inputs: ['what is it', 'a'], inputType: 'query', instruction: 'Find it', model: 'qwen3-embedding-8b',
      fingerprint: FINGERPRINT, dimensions: 2, encodingFormat: 'float', maxParamsB: 8,
    },
    { act: 'decide', queue: { maxWaitS: 30 } },
  );
  assert.equal(lastPath, '/v1/embed');
  assert.deepEqual(JSON.parse(lastBody), {
    inputs: ['what is it', 'a'], input_type: 'query', instruction: 'Find it', model: 'qwen3-embedding-8b',
    max_params_b: 8, fingerprint: FINGERPRINT, dimensions: 2, encoding_format: 'float',
    queue: { max_wait_s: 30 },
  });
  assert.equal(lastHeaders['x-crucible-act'], 'decide');
  assert.equal(answer.model.fingerprint, FINGERPRINT);
  assert.equal(answer.model.engineBuild, 'llama-server-b10970');
  assert.deepEqual(answer.embeddings, [[0.6, 0.8], [1, 0]]);
  assert.deepEqual(answer.tokens, { perInput: [5, 3], total: 8 });
  assert.deepEqual(answer.timingMs, { total: 12.5, queued: 0.4 });
  assert.equal(answer.inputType, 'query');
});

test('embed with a client queue of false sends it, and leaves out what was not given', async () => {
  handle = (_request, response) => json(response, 200, embedReply({ input_type: 'document', instruction: null }));
  const quiet = new CrucibleClient({ url: base, token: 't', clientName: 'unit-embed', queue: false });
  await quiet.embed({ inputs: ['a', 'b'], inputType: 'document' });
  assert.deepEqual(JSON.parse(lastBody), { inputs: ['a', 'b'], input_type: 'document', queue: false });
});

test('embed refuses a malformed request before anything is sent', async () => {
  const before = requests;
  const c = client();
  await assert.rejects(c.embed({ inputs: [], inputType: 'query' }), CrucibleConfigError);
  await assert.rejects(c.embed({ inputs: [' '], inputType: 'query' }), CrucibleConfigError);
  await assert.rejects(
    c.embed({ inputs: new Array(257).fill('x'), inputType: 'document' }),
    CrucibleConfigError,
  );
  await assert.rejects(
    c.embed({ inputs: ['a'], inputType: 'passage' as 'query' }),
    CrucibleConfigError,
  );
  await assert.rejects(c.embed({ inputs: ['a'], inputType: 'query', dimensions: 0 }), CrucibleConfigError);
  await assert.rejects(
    c.embed({ inputs: ['a'], inputType: 'query', encodingFormat: 'hex' as 'float' }),
    CrucibleConfigError,
  );
  await assert.rejects(c.embed({ inputs: ['a'], inputType: 'query', maxParamsB: 0 }), CrucibleConfigError);
  assert.equal(requests, before);
});

test('an embed reply that does not answer what was asked is a protocol error', async () => {
  const c = client();
  for (const spoiled of [
    embedReply({ embeddings: [[0.6, 0.8]] }),
    embedReply({ embeddings: [[0.6, 0.8], [1]] }),
    embedReply({ tokens: { per_input: [5], total: 5 } }),
    embedReply({ encoding_format: 'hex' }),
    embedReply({ model: { id: 'm' } }),
    embedReply({ encoding_format: 'base64', embeddings: ['AAAA', 7] }),
  ]) {
    handle = (_request, response) => json(response, 200, spoiled);
    await assert.rejects(c.embed({ inputs: ['a', 'b'], inputType: 'query' }), CrucibleProtocolError);
  }
});

test('a base64 embed answer is strings, and decodeEmbedding reads both widths', async () => {
  const f32 = Buffer.alloc(8);
  f32.writeFloatLE(0.6, 0);
  f32.writeFloatLE(-0.8, 4);
  const f16 = Buffer.from([0xcd, 0x38, 0x66, 0xba]); // 0.6 and -0.8 in IEEE half, little-endian
  handle = (_request, response) =>
    json(response, 200, embedReply({
      encoding_format: 'base64', embeddings: [f32.toString('base64'), f32.toString('base64')],
    }));
  const answer = await client().embed({ inputs: ['a', 'b'], inputType: 'query', encodingFormat: 'base64' });
  const first = answer.embeddings[0];
  assert.equal(typeof first, 'string');
  const wide = decodeEmbedding(first as string, 'base64');
  assert.ok(Math.abs((wide[0] as number) - 0.6) < 1e-7 && Math.abs((wide[1] as number) + 0.8) < 1e-7);
  const half = decodeEmbedding(f16.toString('base64'), 'base64_float16');
  assert.ok(Math.abs((half[0] as number) - 0.6) < 1e-3 && Math.abs((half[1] as number) + 0.8) < 1e-3);
  assert.deepEqual(Array.from(decodeEmbedding([0.5, 0.25], 'float')), [0.5, 0.25]);
  assert.throws(() => decodeEmbedding('AAAA', 'base64'), CrucibleConfigError);
  assert.throws(() => decodeEmbedding([1], 'base64'), CrucibleConfigError);
});

test('a refusal the server names reaches the caller as CrucibleRefused with its code', async () => {
  for (const [status, code] of [
    [409, 'fingerprint_mismatch'],
    [409, 'package_not_installed'],
    [409, 'model_does_not_fit'],
    [400, 'model_not_for_verb'],
    [400, 'embed_input_too_long'],
    [400, 'dimensions_not_supported'],
    [400, 'instruction_not_taken'],
  ] as const) {
    handle = (_request, response) =>
      json(response, status, { error: { code, message: `refused: ${code}`, details: { model: 'm' } } });
    await assert.rejects(client().embed({ inputs: ['a'], inputType: 'query' }), (error: unknown) => {
      assert.ok(error instanceof CrucibleRefused);
      assert.equal(error.code, code);
      assert.equal(error.status, status);
      return true;
    });
  }
});

test('rerank sends its body and reads scores in document order and results sorted', async () => {
  handle = (_request, response) => json(response, 200, rerankReply());
  const controller = new AbortController();
  const answer = await client().rerank(
    { query: 'q', documents: ['d0', 'd1'], instruction: 'i', model: 'qwen3.5-9b', form: 'bf16' },
    { signal: controller.signal, act: 'decide' },
  );
  assert.equal(lastPath, '/v1/rerank');
  assert.deepEqual(JSON.parse(lastBody), {
    query: 'q', documents: ['d0', 'd1'], instruction: 'i', model: 'qwen3.5-9b', form: 'bf16',
  });
  assert.deepEqual(answer.scores, [0.1, 0.9]);
  assert.deepEqual(answer.results, [{ index: 1, relevanceScore: 0.9 }, { index: 0, relevanceScore: 0.1 }]);
  assert.deepEqual(answer.tokens, { perDocument: [40, 41], total: 162, cached: 120 });
  assert.equal(answer.model.template, 'model');
  assert.equal(answer.timingMs.queued, null);
});

test('rerank refuses a malformed request, and a reply that misses a document', async () => {
  const before = requests;
  const c = client();
  await assert.rejects(c.rerank({ query: '', documents: ['d'] }), CrucibleConfigError);
  await assert.rejects(c.rerank({ query: 'q', documents: [] }), CrucibleConfigError);
  assert.equal(requests, before);
  for (const spoiled of [
    rerankReply({ scores: [0.1] }),
    rerankReply({ results: [{ index: 1, relevance_score: 0.9 }, { index: 1, relevance_score: 0.9 }] }),
    rerankReply({ tokens: { per_document: [1, 2], total: 3 } }),
  ]) {
    handle = (_request, response) => json(response, 200, spoiled);
    await assert.rejects(c.rerank({ query: 'q', documents: ['a', 'b'] }), CrucibleProtocolError);
  }
});

test('the info reads the verbs, and an older server has none', async () => {
  const verbs = {
    embed: {
      route: 'POST /v1/embed', available: true, registered: 'qwen3-embedding-8b', reason: 'fits',
      package: 'retrieval', package_installed: true, models: ['qwen3-embedding-8b'], limits: { max_inputs: 256 },
    },
    rerank: {
      route: 'POST /v1/rerank', available: false, registered: null, reason: 'not installed',
      package: 'retrieval', package_installed: false, models: ['qwen3-reranker-8b', 'qwen3.5-4b'],
      limits: { max_documents: 256 },
    },
  };
  const info = {
    server: { name: 's', version: '1', api_version: 1 },
    host: { platform: 'linux', arch: 'x86_64', backend: 'cuda-linux', gpu: { vendor: 'nvidia', name: 'g', vram_bytes: 1 } },
    job_types: [], capabilities: [], features: ['embed'], ...UNCLAIMED_ENGINE,
  };
  handle = (_request, response) => json(response, 200, { ...info, verbs });
  const read = await client().info();
  assert.equal(read.verbs?.['embed']?.registered, 'qwen3-embedding-8b');
  assert.equal(read.verbs?.['rerank']?.packageInstalled, false);
  assert.deepEqual(read.verbs?.['rerank']?.limits, { maxInputs: null, maxDocuments: 256 });
  handle = (_request, response) => json(response, 200, info);
  assert.equal((await client().info()).verbs, null);
});

test('a model row reads its verbs, package, embed and rerank, and an older row has none', async () => {
  const row = {
    id: 'qwen3-embedding-8b', family: 'qwen3-embedding', params_b: 8, revision: 'r', fingerprint: 'f',
    modalities: ['text'], backend_supported: true, installed: false, weights_of: null, resident: false,
    loadable: false, reason: 'no weights', memory_bytes_estimate: 1, context_default: 8192,
    max_model_len: 8192, form: null, form_reason: null, forms: null,
  };
  const embed = {
    dimensions: 4096, dimensions_range: [32, 4096], matryoshka: true, pooling: 'last', normalized: true,
    input_types: ['query', 'document'], query_takes_instruction: true, default_instruction: 'd',
    query_template: 'Instruct: {instruction}\nQuery:{text}<|endoftext|>', document_template: '{text}<|endoftext|>',
    source: 's', max_inputs: 256, max_input_tokens: 8192,
  };
  const general = { template: 'crucible-general-1', default_instruction: 'd', max_documents: 256, max_tokens: 16384 };
  handle = (_request, response) =>
    json(response, 200, [
      { ...row, verbs: ['embed'], package: 'retrieval', package_installed: true, embed, rerank: null },
      { ...row, id: 'qwen3.5-9b', verbs: ['decide', 'rerank'], package: null, package_installed: null, embed: null, rerank: general },
      row,
    ]);
  const [embedder, decider, older] = await client().models();
  assert.deepEqual(embedder?.verbs, ['embed']);
  assert.equal(embedder?.packageInstalled, true);
  assert.deepEqual(embedder?.embed?.dimensionsRange, [32, 4096]);
  assert.equal(embedder?.embed?.maxInputTokens, 8192);
  assert.equal(decider?.rerank?.template, 'crucible-general-1');
  assert.equal(decider?.rerank?.prefixTemplate, null);
  assert.equal(older?.verbs, null);
  assert.equal(older?.embed, null);
  assert.equal(older?.package, null);
});
