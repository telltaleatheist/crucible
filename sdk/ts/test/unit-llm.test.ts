/**
 * Unit tests for the `llm` surface: `models`, `loadModel`, `unloadModel`,
 * `chat` and `chatStream`, against an in-process `node:http` fixture.
 *
 * These exist for the same reason the phase-1 unit tests do — they cover what a
 * healthy real server will not produce on a good day: a 409 on a model that is
 * not resident, an OpenAI stream that stops without `[DONE]`, a `/v1/models`
 * row that claims a model is unloadable and does not say why, and a caller who
 * changes their mind mid-stream. The live proof is `test/e2e-llm.test.ts`.
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
  CrucibleUnreachable,
  isLlmCapability,
  type JobEvent,
} from '../src/index.js';

// ------------------------------------------------------------------ fixture

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

/** What the fixture does with the next request; each test sets its own. */
let handle: Handler = (_request, response) => {
  response.writeHead(500, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: { code: 'no_handler', message: 'the test set none' } }));
};

/** The last request the fixture saw, for asserting on what the client sent. */
let lastPath = '';
let lastMethod = '';
let lastBody = '';
let lastHeaders: Record<string, string | string[] | undefined> = {};

let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-llm' });
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

/** Open an SSE response the way an OpenAI-compatible engine does. */
function openSse(response: ServerResponse): void {
  response.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });
}

/** One `chat.completion.chunk` frame, as `data: {...}\n\n`. */
function chunkFrame(chunk: unknown): string {
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

function contentChunk(content: string): string {
  return chunkFrame({
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: 1757640000,
    model: 'qwen3.5-9b',
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  });
}

const REVISION = '4d1b2f0c9e8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c';

const MODEL_ROW = {
  id: 'qwen3.5-9b',
  family: 'qwen3.5',
  params_b: 9,
  revision: REVISION,
  fingerprint: `qwen3.5-9b@${REVISION}`,
  modalities: ['text'],
  weights_of: null,
  orphan: null, backend_supported: true,
  installed: true,
  resident: true,
  loadable: true,
  memory_bytes_estimate: 21000000000, held_by: null, unclaimed_since: null,
  context_default: 12288,
  max_model_len: 12288,
};

const COMPLETION = {
  id: 'chatcmpl-abc123',
  object: 'chat.completion',
  created: 1757640000,
  model: 'qwen3.5-9b',
  choices: [
    {
      index: 0,
      message: { role: 'assistant', content: 'The forge is lit.' },
      finish_reason: 'stop',
    },
  ],
  usage: { prompt_tokens: 17, completion_tokens: 5, total_tokens: 22 },
};

before(async () => {
  server = createServer((request, response) => {
    lastPath = request.url ?? '';
    lastMethod = request.method ?? '';
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
  // A stream test deliberately leaves a socket open; drop it so close resolves.
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

// ------------------------------------------------------------------- models

test('models() reads every field /v1/models promises', async () => {
  handle = (_request, response) =>
    json(response, 200, [
      MODEL_ROW,
      {
        id: 'qwen3.8-27b',
        family: 'qwen3.5',
        params_b: 27,
        revision: 'b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8',
        fingerprint: 'qwen3.8-27b@b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8',
        modalities: ['text', 'image'],
        weights_of: null,
        orphan: null, backend_supported: true,
        installed: false,
        resident: false,
        loadable: false,
        reason: 'not installed: run `crucible models pull qwen3.8-27b`',
        memory_bytes_estimate: 54000000000, held_by: null, unclaimed_since: null,
        context_default: 12288,
        max_model_len: 12288,
      },
      {
        // A model this host's backend cannot serve has no revision here to
        // name: the server sends null, never the other backend's sha. Its
        // fingerprint and its max_model_len are null for the same reason.
        id: 'mac-only',
        family: 'demo',
        params_b: 1,
        revision: null,
        fingerprint: null,
        // Never null, even here: what a model is offered FOR is the same answer
        // on a host that cannot serve it at all.
        modalities: ['text'],
        weights_of: null,
        orphan: null, backend_supported: false,
        installed: false,
        resident: false,
        loadable: false,
        reason: 'mac-only.toml has no cuda-linux block; it declares [mlx-darwin]',
        memory_bytes_estimate: null,
        context_default: 4096,
        max_model_len: null,
      },
    ]);

  const models = await client().models();
  assert.equal(lastMethod, 'GET');
  assert.equal(lastPath, '/v1/models');
  assert.equal(lastHeaders['authorization'], 'Bearer the-token');
  assert.equal(lastHeaders['x-crucible-api'], '1');

  assert.deepEqual(models, [
    {
      id: 'qwen3.5-9b',
      family: 'qwen3.5',
      paramsB: 9,
      revision: REVISION,
      fingerprint: `qwen3.5-9b@${REVISION}`,
      modalities: ['text'],
      weightsOf: null,
      backendSupported: true,
      installed: true,
      resident: true,
      loadable: true,
      memoryBytesEstimate: 21000000000,
      contextDefault: 12288,
      maxModelLen: 12288,
    },
    {
      id: 'qwen3.8-27b',
      family: 'qwen3.5',
      paramsB: 27,
      revision: 'b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8',
      fingerprint: 'qwen3.8-27b@b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8',
      modalities: ['text', 'image'],
      weightsOf: null,
      backendSupported: true,
      installed: false,
      resident: false,
      loadable: false,
      reason: 'not installed: run `crucible models pull qwen3.8-27b`',
      memoryBytesEstimate: 54000000000,
      contextDefault: 12288,
      maxModelLen: 12288,
    },
    {
      id: 'mac-only',
      family: 'demo',
      paramsB: 1,
      revision: null,
      fingerprint: null,
      modalities: ['text'],
      weightsOf: null,
      backendSupported: false,
      installed: false,
      resident: false,
      loadable: false,
      reason: 'mac-only.toml has no cuda-linux block; it declares [mlx-darwin]',
      memoryBytesEstimate: null,
      contextDefault: 4096,
      maxModelLen: null,
    },
  ]);
  assert.ok(!('reason' in models[0]!), 'a loadable model carries no reason');
});

test('a model row carries weights_of: the base, or null, and never absent', async () => {
  // PHASE22-DECIDE.md section 2.9: one copy on disk, two rows.
  handle = (_request, response) =>
    json(response, 200, [MODEL_ROW, { ...MODEL_ROW, id: 'qwen3.5-9b-vl', weights_of: 'qwen3.5-9b' }]);
  const models = await client().models();
  assert.equal(models[0]!.weightsOf, null);
  assert.equal(models[1]!.weightsOf, 'qwen3.5-9b');
  const { weights_of: _dropped, ...without } = MODEL_ROW;
  handle = (_request, response) => json(response, 200, [without]);
  await assert.rejects(client().models(), CrucibleProtocolError);
});

test('a model that is not loadable and does not say why is a protocol error', async () => {
  handle = (_request, response) => json(response, 200, [{ ...MODEL_ROW, loadable: false }]);
  await assert.rejects(client().models(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /models\[0\] has no field "reason"/);
    return true;
  });
});

test('a /v1/models body that is not an array is a protocol error, not an empty list', async () => {
  handle = (_request, response) => json(response, 200, { models: [MODEL_ROW] });
  await assert.rejects(client().models(), CrucibleProtocolError);
});

test('a row without a max_model_len is a protocol error, not an unclamped model', async () => {
  // The whole reason the field exists: a client that cannot read it has no
  // clamp at all, and an unclamped request is a 400 from the engine.
  const { max_model_len: _len, ...withoutLen } = MODEL_ROW;
  handle = (_request, response) => json(response, 200, [withoutLen]);
  await assert.rejects(client().models(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /models\[0\] has no field "max_model_len"/);
    return true;
  });
});

test('a row without a fingerprint is a protocol error, not a model to file under its id', async () => {
  const { fingerprint: _fingerprint, ...withoutFingerprint } = MODEL_ROW;
  handle = (_request, response) => json(response, 200, [withoutFingerprint]);
  await assert.rejects(client().models(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /models\[0\] has no field "fingerprint"/);
    return true;
  });
});

test('a model row without a revision is a protocol error, not an unpinned model', async () => {
  const { revision: _revision, ...withoutRevision } = MODEL_ROW;
  handle = (_request, response) => json(response, 200, [withoutRevision]);
  await assert.rejects(client().models(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /models\[0\] has no field "revision"/);
    return true;
  });
});

// ---------------------------------------------------- info's llm capability

/** `GET /v1/info` as a phase-2 server answers it. */
const INFO = {
  server: { name: 'crucible@test', version: '0.1.0', api_version: 1 },
  host: {
    platform: 'linux',
    arch: 'x86_64',
    backend: 'cuda-linux',
    gpu: { vendor: 'nvidia', name: 'NVIDIA GeForce RTX 3090 Ti', vram_bytes: 25757220864 },
  },
  // What to POST, which is not the capability list: `llm` is a capability and
  // `load-model` / `unload-model` are the job types that operate it.
  job_types: ['asr', 'echo', 'load-model', 'unload-model'],
  capabilities: [
    {
      job_type: 'asr',
      // DESIGN.md section 4's row, which every capability but `llm` and `tts`
      // still uses.
      models: [
        {
          id: 'faster-whisper-base',
          revision: 'ebe41f70d5b6a1f3c2e9d8a7b6c5d4e3f2a1b0c9',
          source: 'Systran/faster-whisper-base',
          installed: true,
          resident: false,
          vram_bytes: 1685651456,
        },
      ],
    },
    { job_type: 'echo', models: [] },
    { job_type: 'llm', models: [MODEL_ROW] },
  ],
};

test("info() reads the llm capability's rows with the /models reader", async () => {
  handle = (_request, response) => json(response, 200, INFO);
  const info = await client().info();
  assert.equal(lastPath, '/v1/info');

  const llm = info.capabilities.find((capability) => capability.jobType === 'llm');
  assert.ok(llm !== undefined, 'a phase-2 server offers the llm capability');
  assert.ok(isLlmCapability(llm), 'the llm capability narrows to the /models rows');
  assert.deepEqual(llm.models, [
    {
      id: 'qwen3.5-9b',
      family: 'qwen3.5',
      paramsB: 9,
      revision: REVISION,
      fingerprint: `qwen3.5-9b@${REVISION}`,
      modalities: ['text'],
      weightsOf: null,
      backendSupported: true,
      installed: true,
      resident: true,
      loadable: true,
      memoryBytesEstimate: 21000000000,
      contextDefault: 12288,
      maxModelLen: 12288,
    },
  ]);

  // The other capabilities keep DESIGN.md section 4's row — `installed` and
  // `resident` both, and neither read as the other.
  const asr = info.capabilities.find((capability) => capability.jobType === 'asr');
  assert.ok(asr !== undefined && !isLlmCapability(asr));
  assert.deepEqual(asr.models, [
    {
      id: 'faster-whisper-base',
      revision: 'ebe41f70d5b6a1f3c2e9d8a7b6c5d4e3f2a1b0c9',
      source: 'Systran/faster-whisper-base',
      installed: true,
      resident: false,
      vramBytes: 1685651456,
    },
  ]);

  // And the phase-1 capability that serves no models still reads.
  const echo = info.capabilities.find((capability) => capability.jobType === 'echo');
  assert.ok(echo !== undefined);
  assert.deepEqual(echo.models, []);

  // `job_types` is what to POST, and is deliberately not the capability list:
  // `llm` is a capability and is not a job type; `load-model` is a job type and
  // is not a capability. One model id appears in exactly one capability.
  assert.deepEqual(info.jobTypes, ['asr', 'echo', 'load-model', 'unload-model']);
  const ids = info.capabilities.flatMap((capability) =>
    capability.models.map((model) => model.id),
  );
  assert.equal(new Set(ids).size, ids.length, 'a model is described in exactly one place');
});

test('an info body without job_types is a protocol error, not an empty list', async () => {
  // A client that cannot see what to POST would discover an unknown job type by
  // being refused one, which is what `job_types` exists to replace.
  const { job_types: _types, ...withoutTypes } = INFO;
  handle = (_request, response) => json(response, 200, withoutTypes);
  await assert.rejects(client().info(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /info has no field "job_types"/);
    return true;
  });
});

test('an llm capability row shaped like the phase-1 row is a protocol error', async () => {
  handle = (_request, response) =>
    json(response, 200, {
      ...INFO,
      capabilities: [
        {
          job_type: 'llm',
          models: [
            {
              id: 'qwen3.5-9b',
              revision: REVISION,
              source: 'Qwen/Qwen3.5-9B',
              resident: true,
              vram_bytes: 21000000000,
            },
          ],
        },
      ],
    });
  await assert.rejects(client().info(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /info\.capabilities\[0\]\.models\[0\] has no field "loadable"/);
    return true;
  });
});

// -------------------------------------------------------- load and unload

test('loadModel submits a load-model job with the model and no inputs', async () => {
  handle = (_request, response) => json(response, 200, { job_id: 'job-load-1' });
  const jobId = await client().loadModel('qwen3.5-9b');
  assert.equal(jobId, 'job-load-1');
  assert.equal(lastMethod, 'POST');
  assert.equal(lastPath, '/v1/jobs');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'load-model',
    params: {},
    inputs: {},
    model: 'qwen3.5-9b',
  });
});

test('loadModel sends a stated context as params.context, beside a lease', async () => {
  handle = (_request, response) => json(response, 200, { job_id: 'job-load-2' });
  await client().loadModel('qwen3.5-9b', {
    context: 65536,
    lease: { act: 'generate', ttlSeconds: 600 },
  });
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'load-model',
    params: { lease: { act: 'generate', ttl_seconds: 600 }, context: 65536 },
    inputs: {},
    model: 'qwen3.5-9b',
  });
});

test('loadModel sends no context key when none is stated', async () => {
  handle = (_request, response) => json(response, 200, { job_id: 'job-load-3' });
  await client().loadModel('qwen3.5-9b', {});
  assert.deepEqual(JSON.parse(lastBody).params, {});
});

test('unloadModel submits an unload-model job the same way', async () => {
  handle = (_request, response) => json(response, 200, { job_id: 'job-unload-1' });
  const jobId = await client().unloadModel('qwen3.5-9b');
  assert.equal(jobId, 'job-unload-1');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'unload-model',
    params: {},
    inputs: {},
    model: 'qwen3.5-9b',
  });
});

test('loadModel refuses an empty model id by name rather than posting it', async () => {
  handle = (_request, response) => json(response, 200, { job_id: 'never' });
  await assert.rejects(client().loadModel(''), (error: unknown) => {
    assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
    assert.equal(error.option, 'model');
    return true;
  });
});

test("a load job's events carry warming {message} and done {resident}", async () => {
  handle = (_request, response) => {
    openSse(response);
    response.write('id: 1\nevent: queued\ndata: {"position": 1}\n\n');
    response.write('id: 2\nevent: warming\ndata: {"message": "loading weights"}\n\n');
    response.write('id: 3\nevent: done\ndata: {"resident": "qwen3.5-9b"}\n\n');
    response.end();
  };

  const events: JobEvent[] = [];
  for await (const event of client().events('job-load-1')) events.push(event);

  assert.deepEqual(
    events.map((event) => event.event),
    ['queued', 'warming', 'done'],
  );
  const warming = events[1]!;
  assert.equal(warming.event === 'warming' ? warming.data.message : null, 'loading weights');
  const done = events[2]!;
  assert.equal(done.event === 'done' ? done.data.resident : null, 'qwen3.5-9b');
  assert.equal(done.event === 'done' ? done.data.artifacts : 'x', undefined);
});

test("an unload's done event says resident: null — nothing is resident now", async () => {
  handle = (_request, response) => {
    openSse(response);
    response.write('id: 1\nevent: queued\ndata: {"position": 1}\n\n');
    response.write('id: 2\nevent: progress\ndata: {"fraction": 0.0, "message": "unloading"}\n\n');
    response.write('id: 3\nevent: done\ndata: {"artifacts": [], "resident": null}\n\n');
    response.end();
  };

  const events: JobEvent[] = [];
  for await (const event of client().events('job-unload-1')) events.push(event);

  const done = events.at(-1)!;
  assert.equal(done.event, 'done');
  // Present and null: the field is answered, and the answer is "nothing".
  assert.equal(done.event === 'done' ? done.data.resident : 'x', null);
  assert.deepEqual(done.event === 'done' ? done.data.artifacts : null, []);
});

test('a done event whose resident is neither a string nor null is a protocol error', async () => {
  handle = (_request, response) => {
    openSse(response);
    response.write('id: 1\nevent: done\ndata: {"resident": 7}\n\n');
    response.end();
  };
  await assert.rejects(async () => {
    for await (const _event of client().events('job-bad-done')) {
      // drain
    }
  }, CrucibleProtocolError);
});

test('a done event that says neither artifacts nor resident is a protocol error', async () => {
  handle = (_request, response) => {
    openSse(response);
    response.write('id: 1\nevent: done\ndata: {}\n\n');
    response.end();
  };
  await assert.rejects(async () => {
    for await (const _event of client().events('job-empty-done')) {
      // drain
    }
  }, CrucibleProtocolError);
});

// --------------------------------------------------------------------- chat

test('chat posts the OpenAI body and reads the completion down to what it promises', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);

  const answer = await client().chat({
    model: 'qwen3.5-9b',
    messages: [
      { role: 'system', content: 'You are terse.' },
      { role: 'user', content: 'Light it.' },
    ],
    temperature: 0.2,
    topP: 0.9,
    maxTokens: 64,
    stop: ['\n\n'],
  });

  assert.equal(lastMethod, 'POST');
  assert.equal(lastPath, '/v1/openai/chat/completions');
  assert.equal(lastHeaders['authorization'], 'Bearer the-token');
  assert.equal(lastHeaders['x-crucible-api'], '1');
  assert.deepEqual(JSON.parse(lastBody), {
    model: 'qwen3.5-9b',
    messages: [
      { role: 'system', content: 'You are terse.' },
      { role: 'user', content: 'Light it.' },
    ],
    stream: false,
    temperature: 0.2,
    top_p: 0.9,
    max_tokens: 64,
    stop: ['\n\n'],
  });

  assert.deepEqual(answer, {
    id: 'chatcmpl-abc123',
    model: 'qwen3.5-9b',
    content: 'The forge is lit.',
    finishReason: 'stop',
    usage: { promptTokens: 17, completionTokens: 5, totalTokens: 22 },
  });
});

test('the optional sampling knobs are omitted entirely when not given', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  await client().chat({ model: 'qwen3.5-9b', messages: [{ role: 'user', content: 'hi' }] });
  assert.deepEqual(JSON.parse(lastBody), {
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'hi' }],
    stream: false,
  });
});

test('thinking: false sends the template kwarg that turns a reasoning model off', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  await client().chat({
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'hi' }],
    maxTokens: 64,
    thinking: false,
  });
  assert.deepEqual(JSON.parse(lastBody), {
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'hi' }],
    stream: false,
    max_tokens: 64,
    chat_template_kwargs: { enable_thinking: false },
  });
});

test('thinking: true sends the same field set to true', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  await client().chat({
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'hi' }],
    thinking: true,
  });
  assert.deepEqual(JSON.parse(lastBody)['chat_template_kwargs'], { enable_thinking: true });
});

test('omitting thinking sends nothing, leaving the model its own default', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  await client().chat({ model: 'qwen3.5-9b', messages: [{ role: 'user', content: 'hi' }] });
  assert.ok(
    !('chat_template_kwargs' in JSON.parse(lastBody)),
    'an omitted option must not be sent as a value',
  );
});

test('thinking must be a boolean and is refused by name when it is not', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  await assert.rejects(
    client().chat({
      model: 'qwen3.5-9b',
      messages: [{ role: 'user', content: 'hi' }],
      thinking: 'no' as never,
    }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
      assert.equal(error.option, 'thinking');
      return true;
    },
  );
});

// ---------------------------------------------------------- X-Crucible-Act

test('act is sent as X-Crucible-Act, and as a header rather than a body field', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  await client().chat({
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'hi' }],
    act: 'clean',
  });
  assert.equal(lastHeaders['x-crucible-act'], 'clean');
  // The body is OpenAI's and is proxied to the engine verbatim; the act is
  // Crucible's and stops at the proxy. A body field would reach the engine.
  assert.ok(!('act' in JSON.parse(lastBody)), 'the act must not ride in the body');
});

test('chatStream sends the act too, beside the Accept it adds', async () => {
  handle = (_request, response) => {
    openSse(response);
    response.write(contentChunk('lit.'));
    response.write('data: [DONE]\n\n');
    response.end();
  };
  const seen: string[] = [];
  for await (const delta of client().chatStream({
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'hi' }],
    act: 'translate',
  })) {
    seen.push(delta);
  }
  assert.deepEqual(seen, ['lit.']);
  assert.equal(lastHeaders['x-crucible-act'], 'translate');
  assert.equal(lastHeaders['accept'], 'text/event-stream');
});

test('omitting act sends NO header, which is how the server records "did not say"', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  await client().chat({ model: 'qwen3.5-9b', messages: [{ role: 'user', content: 'hi' }] });
  assert.equal(
    lastHeaders['x-crucible-act'],
    undefined,
    'a default act would put a name nobody chose on a bench',
  );
});

test('an empty act is refused here rather than sent as a header saying nothing', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  await assert.rejects(
    client().chat({
      model: 'qwen3.5-9b',
      messages: [{ role: 'user', content: 'hi' }],
      act: '   ',
    }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
      assert.equal(error.option, 'act');
      return true;
    },
  );
});

test('an act this server does not know is the SERVER\'s refusal, not a second vocabulary', async () => {
  // The client keeps no copy of the capability classes: the server answers
  // 400 `unknown_act` listing them, before the completion runs, and that
  // sentence is what reaches the caller.
  handle = (_request, response) =>
    json(response, 400, {
      error: {
        code: 'unknown_act',
        message: "'narrate' is not an act this server knows.",
        details: { act: 'narrate', known: ['clean', 'translate'] },
      },
    });
  await assert.rejects(
    client().chat({
      model: 'qwen3.5-9b',
      messages: [{ role: 'user', content: 'hi' }],
      act: 'narrate',
    }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
      assert.equal(error.code, 'unknown_act');
      return true;
    },
  );
  assert.equal(lastHeaders['x-crucible-act'], 'narrate');
});

// -------------------------------------------------------------- provenance

test('a provenance sidecar names the weights, not only the model', async () => {
  handle = (_request, response) => {
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(
      JSON.stringify({
        server: { name: 'crucible@owens-pc', version: '0.2.0' },
        backend: 'cuda-linux',
        job_type: 'load-model',
        model: {
          id: 'qwen3.5-9b',
          revision: REVISION,
          fingerprint: `qwen3.5-9b@${REVISION}`,
        },
        params: {},
        started: '2026-09-13T00:00:00+00:00',
        finished: '2026-09-13T00:04:00+00:00', client_ref: null, interrupted_at: null, chunks_done: [],
        chunks_total: null, chunk_at: null,
      }),
    );
  };

  const provenance = await client().provenance('job-1', 'payload.bin');
  assert.equal(lastPath, '/v1/jobs/job-1/artifacts/payload.bin.provenance.json');
  assert.deepEqual(provenance.model, {
    id: 'qwen3.5-9b',
    revision: REVISION,
    fingerprint: `qwen3.5-9b@${REVISION}`,
  });
  // The document round-trips: clients persist it verbatim beside the artifact.
  assert.equal(provenance.backend, 'cuda-linux');
});

test('a provenance model block with no fingerprint is a protocol error', async () => {
  handle = (_request, response) => {
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(
      JSON.stringify({
        server: { name: 'crucible@owens-pc', version: '0.2.0' },
        backend: 'cuda-linux',
        job_type: 'load-model',
        model: { id: 'qwen3.5-9b', revision: REVISION },
        params: {},
        started: null,
        finished: '2026-09-13T00:04:00+00:00', client_ref: null, interrupted_at: null, chunks_done: [],
        chunks_total: null, chunk_at: null,
      }),
    );
  };
  await assert.rejects(client().provenance('job-1', 'payload.bin'), CrucibleProtocolError);
});

// ----------------------------------------------- the constrained transport

/** A Foundry analyze verdict's schema, as `askConstrained` builds it. */
const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    supported: { type: 'boolean' },
    quote: { type: 'string', maxLength: 200 },
  },
  required: ['supported', 'quote'],
  additionalProperties: false,
} as const;

test('responseFormat and seed reach the wire under OpenAI\'s own names', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  await client().chat({
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'Does the passage support the claim?' }],
    temperature: 0,
    maxTokens: 128,
    seed: 1729,
    responseFormat: {
      type: 'json_schema',
      json_schema: { name: 'verdict', schema: VERDICT_SCHEMA, strict: true },
    },
    thinking: false,
  });

  assert.deepEqual(JSON.parse(lastBody), {
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'Does the passage support the claim?' }],
    stream: false,
    temperature: 0,
    max_tokens: 128,
    seed: 1729,
    response_format: {
      type: 'json_schema',
      json_schema: { name: 'verdict', schema: VERDICT_SCHEMA, strict: true },
    },
    chat_template_kwargs: { enable_thinking: false },
  });
});

test('the grammar inside responseFormat is forwarded, not read', async () => {
  // A schema using a keyword this client has never heard of: it is the engine's
  // guided-decoding backend that decides what it supports, and if it will not
  // compile this it answers its own 400 saying so.
  handle = (_request, response) => json(response, 200, COMPLETION);
  const exotic = { type: 'array', prefixItems: [{ type: 'string' }], unevaluatedItems: false };
  await client().chat({
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'hi' }],
    responseFormat: { type: 'json_schema', json_schema: { name: 'pair', schema: exotic } },
  });
  assert.deepEqual(JSON.parse(lastBody)['response_format']['json_schema']['schema'], exotic);
});

test('responseFormat: {type: "json_object"} needs no schema', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  await client().chat({
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'hi' }],
    responseFormat: { type: 'json_object' },
  });
  assert.deepEqual(JSON.parse(lastBody)['response_format'], { type: 'json_object' });
});

test('a malformed responseFormat is refused by name, before anything is posted', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  const cases: Array<[unknown, string]> = [
    [{ type: 'jsonschema' }, 'responseFormat.type'],
    [{ type: 'json_schema' }, 'responseFormat.json_schema'],
    [{ type: 'json_schema', json_schema: { schema: {} } }, 'responseFormat.json_schema.name'],
    [
      { type: 'json_schema', json_schema: { name: 'verdict' } },
      'responseFormat.json_schema.schema',
    ],
    [
      { type: 'json_schema', json_schema: { name: 'verdict', schema: 'an object, please' } },
      'responseFormat.json_schema.schema',
    ],
    ['json', 'responseFormat'],
  ];
  for (const [responseFormat, option] of cases) {
    await assert.rejects(
      client().chat({
        model: 'qwen3.5-9b',
        messages: [{ role: 'user', content: 'hi' }],
        responseFormat: responseFormat as never,
      }),
      (error: unknown) => {
        assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
        assert.equal(error.option, option);
        return true;
      },
      `expected ${option} to be refused by name`,
    );
  }
});

test('seed must be an integer and is refused by name when it is not', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  await assert.rejects(
    client().chat({
      model: 'qwen3.5-9b',
      messages: [{ role: 'user', content: 'hi' }],
      seed: 1.5,
    }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
      assert.equal(error.option, 'seed');
      return true;
    },
  );
});

test('finishReason is the engine\'s own word, surfaced rather than normalised', async () => {
  // `length` is the one that matters: both apps record it as a degradation
  // instead of using the answer, which only works if it arrives. `content` is
  // present here — a truncated answer is still an answer that was begun.
  for (const reason of ['stop', 'length', 'content_filter', 'something_new']) {
    handle = (_request, response) =>
      json(response, 200, {
        ...COMPLETION,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: '{"supported": tr' },
            finish_reason: reason,
          },
        ],
      });
    const answer = await client().chat({
      model: 'qwen3.5-9b',
      messages: [{ role: 'user', content: 'hi' }],
    });
    assert.equal(answer.finishReason, reason);
    assert.equal(answer.content, '{"supported": tr');
  }
});

test('a completion that is all reasoning and no content says why, and how to fix it', async () => {
  // What mlx-lm returns for Qwen3.5 when the ceiling lands inside the thinking:
  // a message with `reasoning` and no `content` key at all.
  handle = (_request, response) =>
    json(response, 200, {
      ...COMPLETION,
      choices: [
        {
          index: 0,
          message: { role: 'assistant', reasoning: 'The user wants a colour. Let me think' },
          finish_reason: 'length',
        },
      ],
    });

  await assert.rejects(
    client().chat({
      model: 'qwen3.5-9b',
      messages: [{ role: 'user', content: 'Name one colour.' }],
      maxTokens: 8,
    }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
      assert.match(error.message, /has no "content"/);
      assert.match(error.message, /reasoning/);
      assert.match(error.message, /token ceiling/);
      assert.match(error.message, /raise maxTokens, or pass thinking: false/);
      return true;
    },
  );
});

test('a completion with no content and no reasoning is still the plain missing-field error', async () => {
  handle = (_request, response) =>
    json(response, 200, {
      ...COMPLETION,
      choices: [{ index: 0, message: { role: 'assistant' }, finish_reason: 'stop' }],
    });
  await assert.rejects(
    client().chat({ model: 'qwen3.5-9b', messages: [{ role: 'user', content: 'hi' }] }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
      assert.match(error.message, /chat\.choices\[0\]\.message has no field "content"/);
      return true;
    },
  );
});

test('chat requires model and messages, and names the one that is missing', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  const cases: Array<[Record<string, unknown>, string]> = [
    [{ messages: [{ role: 'user', content: 'hi' }] }, 'model'],
    [{ model: 'qwen3.5-9b' }, 'messages'],
    [{ model: 'qwen3.5-9b', messages: [] }, 'messages'],
    [{ model: 'qwen3.5-9b', messages: [{ role: 'oracle', content: 'hi' }] }, 'messages[0].role'],
    [{ model: 'qwen3.5-9b', messages: [{ role: 'user' }] }, 'messages[0].content'],
  ];
  for (const [options, option] of cases) {
    await assert.rejects(
      client().chat(options as never),
      (error: unknown) => {
        assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
        assert.equal(error.option, option);
        return true;
      },
      `expected ${option} to be refused by name`,
    );
  }
});

test('409 model_not_resident is a CrucibleRefused naming the resident model', async () => {
  handle = (_request, response) =>
    json(response, 409, {
      error: {
        code: 'model_not_resident',
        message: "qwen3.8-27b is not resident; qwen3.5-9b is. Load it first.",
        details: { requested: 'qwen3.8-27b', resident: 'qwen3.5-9b' },
      },
    });

  await assert.rejects(
    client().chat({ model: 'qwen3.8-27b', messages: [{ role: 'user', content: 'hi' }] }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
      assert.equal(error.status, 409);
      assert.equal(error.code, 'model_not_resident');
      assert.match(error.serverMessage, /qwen3\.5-9b is\./);
      assert.deepEqual(error.details, { requested: 'qwen3.8-27b', resident: 'qwen3.5-9b' });
      return true;
    },
  );
});

test('a completion missing usage is a protocol error, not a zero count', async () => {
  const { usage: _usage, ...withoutUsage } = COMPLETION;
  handle = (_request, response) => json(response, 200, withoutUsage);
  await assert.rejects(
    client().chat({ model: 'qwen3.5-9b', messages: [{ role: 'user', content: 'hi' }] }),
    CrucibleProtocolError,
  );
});

test('an already-aborted signal rejects with the AbortError, not an unreachable server', async () => {
  handle = (_request, response) => json(response, 200, COMPLETION);
  await assert.rejects(
    client().chat({
      model: 'qwen3.5-9b',
      messages: [{ role: 'user', content: 'hi' }],
      signal: AbortSignal.abort(),
    }),
    (error: unknown) => {
      assert.equal((error as Error).name, 'AbortError', `got ${String(error)}`);
      assert.ok(!(error instanceof CrucibleUnreachable), 'a caller abort is not a dead server');
      return true;
    },
  );
});

// --------------------------------------------------------------- chatStream

test('chatStream assembles the deltas of an OpenAI transcript and ends on [DONE]', async () => {
  handle = (_request, response) => {
    openSse(response);
    // The opening frame carries only the role; the closing one only a reason.
    response.write(
      chunkFrame({
        id: 'chatcmpl-stream',
        object: 'chat.completion.chunk',
        created: 1757640000,
        model: 'qwen3.5-9b',
        choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
      }),
    );
    response.write(contentChunk('The '));
    response.write(contentChunk('forge '));
    response.write(contentChunk('is lit.'));
    response.write(
      chunkFrame({
        id: 'chatcmpl-stream',
        object: 'chat.completion.chunk',
        created: 1757640000,
        model: 'qwen3.5-9b',
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      }),
    );
    // vLLM's usage-only trailer, which carries no choice at all.
    response.write(
      chunkFrame({
        id: 'chatcmpl-stream',
        object: 'chat.completion.chunk',
        created: 1757640000,
        model: 'qwen3.5-9b',
        choices: [],
        usage: { prompt_tokens: 17, completion_tokens: 5, total_tokens: 22 },
      }),
    );
    response.write('data: [DONE]\n\n');
    response.end();
  };

  const deltas: string[] = [];
  for await (const delta of client().chatStream({
    model: 'qwen3.5-9b',
    messages: [{ role: 'user', content: 'Light it.' }],
  })) {
    deltas.push(delta);
  }

  assert.deepEqual(deltas, ['The ', 'forge ', 'is lit.']);
  assert.equal(deltas.join(''), 'The forge is lit.');
  assert.equal(JSON.parse(lastBody).stream, true, 'a streamed chat asks the engine to stream');
  assert.match(String(lastHeaders['accept']), /text\/event-stream/);
});

test('a stream that stops without [DONE] is a truncated answer, not a finished one', async () => {
  handle = (_request, response) => {
    openSse(response);
    response.write(contentChunk('The '));
    response.write(contentChunk('forge'));
    response.end();
  };

  const deltas: string[] = [];
  await assert.rejects(
    async () => {
      for await (const delta of client().chatStream({
        model: 'qwen3.5-9b',
        messages: [{ role: 'user', content: 'Light it.' }],
      })) {
        deltas.push(delta);
      }
    },
    (error: unknown) => {
      assert.ok(error instanceof CrucibleUnreachable, `got ${String(error)}`);
      assert.match(error.message, /\[DONE\]/);
      return true;
    },
  );
  assert.deepEqual(deltas, ['The ', 'forge']);
});

test('aborting mid-stream throws the AbortError out of the iterator', async () => {
  // One frame, then the socket is held open: the abort is the only thing that
  // can end this stream, so the test is not racing the fixture.
  handle = (_request, response) => {
    openSse(response);
    response.write(contentChunk('The '));
  };

  const controller = new AbortController();
  const deltas: string[] = [];
  await assert.rejects(
    async () => {
      for await (const delta of client().chatStream({
        model: 'qwen3.5-9b',
        messages: [{ role: 'user', content: 'Light it.' }],
        signal: controller.signal,
      })) {
        deltas.push(delta);
        controller.abort();
      }
    },
    (error: unknown) => {
      assert.equal((error as Error).name, 'AbortError', `got ${String(error)}`);
      return true;
    },
  );
  assert.deepEqual(deltas, ['The '], 'everything delivered before the abort still arrived');
});

test('a chunk that is not JSON is a protocol error', async () => {
  handle = (_request, response) => {
    openSse(response);
    response.write('data: not json at all\n\n');
    response.write('data: [DONE]\n\n');
    response.end();
  };
  await assert.rejects(async () => {
    for await (const _delta of client().chatStream({
      model: 'qwen3.5-9b',
      messages: [{ role: 'user', content: 'hi' }],
    })) {
      // drain
    }
  }, CrucibleProtocolError);
});

test('a refusal on a streamed chat is read from the error body, not the stream', async () => {
  handle = (_request, response) =>
    json(response, 409, {
      error: { code: 'model_not_resident', message: 'nothing is resident' },
    });
  await assert.rejects(
    async () => {
      for await (const _delta of client().chatStream({
        model: 'qwen3.5-9b',
        messages: [{ role: 'user', content: 'hi' }],
      })) {
        // drain
      }
    },
    (error: unknown) => {
      assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
      assert.equal(error.code, 'model_not_resident');
      return true;
    },
  );
});
