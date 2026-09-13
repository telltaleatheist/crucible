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

const MODEL_ROW = {
  id: 'qwen3.5-9b',
  family: 'qwen3.5',
  params_b: 9,
  backend_supported: true,
  installed: true,
  resident: true,
  loadable: true,
  memory_bytes_estimate: 21000000000,
  context_default: 12288,
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
        id: 'qwen3.5-27b',
        family: 'qwen3.5',
        params_b: 27,
        backend_supported: true,
        installed: false,
        resident: false,
        loadable: false,
        reason: 'not installed: run `crucible models pull qwen3.5-27b`',
        memory_bytes_estimate: 54000000000,
        context_default: 12288,
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
      backendSupported: true,
      installed: true,
      resident: true,
      loadable: true,
      memoryBytesEstimate: 21000000000,
      contextDefault: 12288,
    },
    {
      id: 'qwen3.5-27b',
      family: 'qwen3.5',
      paramsB: 27,
      backendSupported: true,
      installed: false,
      resident: false,
      loadable: false,
      reason: 'not installed: run `crucible models pull qwen3.5-27b`',
      memoryBytesEstimate: 54000000000,
      contextDefault: 12288,
    },
  ]);
  assert.ok(!('reason' in models[0]!), 'a loadable model carries no reason');
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
        message: "qwen3.5-27b is not resident; qwen3.5-9b is. Load it first.",
        details: { requested: 'qwen3.5-27b', resident: 'qwen3.5-9b' },
      },
    });

  await assert.rejects(
    client().chat({ model: 'qwen3.5-27b', messages: [{ role: 'user', content: 'hi' }] }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
      assert.equal(error.status, 409);
      assert.equal(error.code, 'model_not_resident');
      assert.match(error.serverMessage, /qwen3\.5-9b is\./);
      assert.deepEqual(error.details, { requested: 'qwen3.5-27b', resident: 'qwen3.5-9b' });
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
