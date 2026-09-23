/**
 * Unit tests for `decide()`, `POST /v1/decide` (PHASE22-DECIDE.md, 2026-09-23),
 * against an in-process `node:http` fixture.
 *
 * The request half is pinned to the contract's worked example byte for byte in
 * meaning: what goes on the wire is the ORDER (state, questions, options) and
 * nothing the SDK made up. The reply half is the lockstep rule (Owen,
 * 2026-09-20): every field the contract names is demanded, `cached_tokens: null`
 * is a statement and not a missing key, and a reply that answers a different
 * question than the one asked is a protocol error rather than a result.
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
  CrucibleServerError,
  type DecideRequest,
} from '../src/index.js';

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => json(response, 500, { error: { code: 'no_handler', message: 'none' } });
let lastPath = '';
let lastMethod = '';
let lastBody = '';
let lastHeaders: Record<string, string | string[] | undefined> = {};
let requests = 0;

let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-decide' });
}

function json(response: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}): void {
  response.writeHead(status, { 'Content-Type': 'application/json', ...headers });
  response.end(JSON.stringify(body));
}

before(async () => {
  server = createServer((request, response) => {
    requests += 1;
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
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

// The contract's worked example, section 2.2, verbatim.
const EXAMPLE: DecideRequest = {
  model: 'qwen3.5-9b',
  state: 'I was charged twice for March, please refund one.',
  questions: {
    team: {
      type: 'choice',
      instructions: 'Which team should handle this?',
      options: { billing: 'Payment and invoice issues', technical: 'Bugs and errors' },
    },
    anger: {
      type: 'score',
      instructions: 'How frustrated is the customer?',
      levels: ['Calm', 'Frustrated but civil', 'Very angry'],
    },
    urgent: { type: 'yesno', instructions: 'The message conveys urgency' },
  },
};

const REVISION = '4d1b2f0c9e8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c';

function reply(): Record<string, any> {
  return {
    model: { id: 'qwen3.5-9b', revision: REVISION, fingerprint: `qwen3.5-9b@${REVISION}` },
    engine: 'vllm',
    answers: {
      team: {
        type: 'choice',
        choice: 'billing',
        probabilities: { billing: 0.91, technical: 0.09 },
        confidence: 0.91,
        label_mass: 0.998,
      },
      anger: {
        type: 'score',
        score: 1.4,
        level: 'Calm',
        probabilities: { Calm: 0.62, 'Frustrated but civil': 0.36, 'Very angry': 0.02 },
        confidence: 0.62,
        label_mass: 0.997,
      },
      urgent: { type: 'yesno', p: 0.83, label_mass: 0.99 },
    },
    timing_ms: {
      total: 84.0,
      per_question: {
        team: { wall_ms: 21.3, prompt_tokens: 61, cached_tokens: 75 },
        anger: { wall_ms: 19.0, prompt_tokens: 70, cached_tokens: null },
        urgent: { wall_ms: 12.7, prompt_tokens: 50, cached_tokens: 75 },
      },
      prime: { wall_ms: 31.0, prompt_tokens: 75, cached_tokens: null },
    },
    tokens: { per_question: { team: 136, anger: 145, urgent: 125 }, images: 0 },
  };
}

test('decide() posts the worked example to /v1/decide with no act header when none is given', async () => {
  handle = (_request, response) => json(response, 200, reply());
  await client().decide(EXAMPLE);
  assert.equal(lastMethod, 'POST');
  assert.equal(lastPath, '/v1/decide');
  assert.deepEqual(JSON.parse(lastBody), {
    model: 'qwen3.5-9b',
    state: 'I was charged twice for March, please refund one.',
    questions: {
      team: {
        type: 'choice',
        instructions: 'Which team should handle this?',
        options: { billing: 'Payment and invoice issues', technical: 'Bugs and errors' },
      },
      anger: {
        type: 'score',
        instructions: 'How frustrated is the customer?',
        levels: ['Calm', 'Frustrated but civil', 'Very angry'],
      },
      urgent: { type: 'yesno', instructions: 'The message conveys urgency' },
    },
  });
  // Option order IS the letter order on the server; the wire must keep it.
  assert.ok(lastBody.indexOf('"billing"') < lastBody.indexOf('"technical"'));
  assert.equal(lastHeaders['x-crucible-act'], undefined);
  assert.equal(lastHeaders['authorization'], 'Bearer the-token');
  assert.equal(lastHeaders['x-crucible-api'], '1');
  assert.equal(lastHeaders['content-type'], 'application/json');
});

test('decide() sends the act as X-Crucible-Act, and images only when given', async () => {
  handle = (_request, response) => json(response, 200, reply());
  await client().decide({ ...EXAMPLE, images: ['aGVsbG8='] }, { act: 'analysis' });
  assert.equal(lastHeaders['x-crucible-act'], 'analysis');
  const body = JSON.parse(lastBody);
  assert.deepEqual(body.images, ['aGVsbG8=']);
  assert.deepEqual(Object.keys(body), ['model', 'state', 'images', 'questions']);
  assert.equal('act' in body, false, 'the act is a header, never a body field');
});

test('decide() reads every field the contract names', async () => {
  handle = (_request, response) => json(response, 200, reply());
  const result = await client().decide(EXAMPLE);
  assert.deepEqual(result, {
    model: { id: 'qwen3.5-9b', revision: REVISION, fingerprint: `qwen3.5-9b@${REVISION}` },
    engine: 'vllm',
    answers: {
      team: {
        type: 'choice',
        choice: 'billing',
        probabilities: { billing: 0.91, technical: 0.09 },
        confidence: 0.91,
        labelMass: 0.998,
      },
      anger: {
        type: 'score',
        score: 1.4,
        level: 'Calm',
        probabilities: { Calm: 0.62, 'Frustrated but civil': 0.36, 'Very angry': 0.02 },
        confidence: 0.62,
        labelMass: 0.997,
      },
      urgent: { type: 'yesno', p: 0.83, labelMass: 0.99 },
    },
    timingMs: {
      total: 84.0,
      perQuestion: {
        team: { wallMs: 21.3, promptTokens: 61, cachedTokens: 75 },
        anger: { wallMs: 19.0, promptTokens: 70, cachedTokens: null },
        urgent: { wallMs: 12.7, promptTokens: 50, cachedTokens: 75 },
      },
      prime: { wallMs: 31.0, promptTokens: 75, cachedTokens: null },
    },
    tokens: { perQuestion: { team: 136, anger: 145, urgent: 125 }, images: 0 },
  });
});

test('a null prime is a statement and reads as null', async () => {
  const body = reply();
  body.timing_ms.prime = null;
  handle = (_request, response) => json(response, 200, body);
  const result = await client().decide(EXAMPLE);
  assert.equal(result.timingMs.prime, null);
});

/** Every demanded field, removed one at a time, must be a protocol error naming it. */
const DEMANDED: Array<[string, string, (body: Record<string, any>) => void]> = [
  ['answers.urgent', 'label_mass', (b) => delete b.answers.urgent.label_mass],
  ['answers.team', 'label_mass', (b) => delete b.answers.team.label_mass],
  ['timing_ms.per_question.team', 'cached_tokens', (b) => delete b.timing_ms.per_question.team.cached_tokens],
  ['timing_ms', 'prime', (b) => delete b.timing_ms.prime],
  ['', 'engine', (b) => delete b.engine],
  ['model', 'fingerprint', (b) => delete b.model.fingerprint],
  ['tokens', 'images', (b) => delete b.tokens.images],
  ['answers.anger', 'confidence', (b) => delete b.answers.anger.confidence],
  ['answers.anger', 'score', (b) => delete b.answers.anger.score],
  ['answers.urgent', 'p', (b) => delete b.answers.urgent.p],
];

for (const [path, name, strip] of DEMANDED) {
  test(`a reply without ${path ? `${path}.` : ''}${name} is a protocol error, never a default`, async () => {
    const body = reply();
    strip(body);
    handle = (_request, response) => json(response, 200, body);
    await assert.rejects(
      client().decide(EXAMPLE),
      (error: unknown) =>
        error instanceof CrucibleProtocolError && error.detail.includes(`"${name}"`),
    );
  });
}

test('a null revision is refused: a decision is read only from a resident engine, which has one', async () => {
  const body = reply();
  body.model.revision = null;
  handle = (_request, response) => json(response, 200, body);
  await assert.rejects(client().decide(EXAMPLE), /decide\.model\.revision is not a string/);
});

test('a reply that answers a different question than the one asked is a protocol error', async () => {
  const missing = reply();
  delete missing.answers.urgent;
  handle = (_request, response) => json(response, 200, missing);
  await assert.rejects(client().decide(EXAMPLE), /decide\.answers does not match the questions asked; missing \["urgent"\]/);

  const wrongType = reply();
  wrongType.answers.urgent = { ...wrongType.answers.team };
  handle = (_request, response) => json(response, 200, wrongType);
  await assert.rejects(client().decide(EXAMPLE), /but the question asked was a yesno/);

  const strayOption = reply();
  strayOption.answers.team.probabilities.other = 0.0;
  handle = (_request, response) => json(response, 200, strayOption);
  await assert.rejects(client().decide(EXAMPLE), /not asked \["other"\]/);
});

test('a 409 model_not_resident surfaces as CrucibleRefused with the code and details', async () => {
  handle = (_request, response) =>
    json(response, 409, {
      error: {
        code: 'model_not_resident',
        message: "qwen3.5-9b is not resident; resident is 'qwen3.8-27b'",
        details: { resident: 'qwen3.8-27b' },
      },
    });
  await assert.rejects(client().decide(EXAMPLE), (error: unknown) => {
    assert.ok(error instanceof CrucibleRefused);
    assert.equal(error.status, 409);
    assert.equal(error.code, 'model_not_resident');
    assert.deepEqual(error.details, { resident: 'qwen3.8-27b' });
    return true;
  });
});

test('a 400 too_many_options surfaces as CrucibleRefused with its own name', async () => {
  handle = (_request, response) =>
    json(response, 400, { error: { code: 'too_many_options', message: 'team has 27 options; at most 26' } });
  await assert.rejects(
    client().decide(EXAMPLE),
    (error: unknown) => error instanceof CrucibleRefused && error.code === 'too_many_options',
  );
});

test('a 503 chat_queue_full is a CrucibleServerError carrying retry_after', async () => {
  handle = (_request, response) =>
    json(
      response,
      503,
      {
        error: {
          code: 'chat_queue_full',
          message: 'this server already has 2 chat completion(s) open',
          details: { model: 'qwen3.5-9b', engine: 'llama-server', max_in_flight: 2, max_in_flight_basis: '--parallel 1', retry_after: 3 },
        },
      },
      { 'Retry-After': '3' },
    );
  await assert.rejects(client().decide(EXAMPLE), (error: unknown) => {
    assert.ok(error instanceof CrucibleServerError);
    assert.equal(error.status, 503);
    assert.equal(error.code, 'chat_queue_full');
    assert.equal((error.details as { retry_after: number }).retry_after, 3);
    return true;
  });
});

test('a 502 label_not_in_probs keeps its name and its details', async () => {
  handle = (_request, response) =>
    json(response, 502, {
      error: { code: 'label_not_in_probs', message: "letter 'Z' of team is not in the engine's top-30", details: { question: 'team', label: 'Z' } },
    });
  await assert.rejects(client().decide(EXAMPLE), (error: unknown) => {
    assert.ok(error instanceof CrucibleServerError);
    assert.equal(error.code, 'label_not_in_probs');
    assert.deepEqual(error.details, { question: 'team', label: 'Z' });
    return true;
  });
});

test('a question with a field this client does not know is refused before any request', async () => {
  const before = requests;
  await assert.rejects(
    client().decide({
      ...EXAMPLE,
      questions: { urgent: { type: 'yesno', instructions: 'x', options: { a: 'b' } } as never },
    }),
    (error: unknown) => error instanceof CrucibleConfigError && error.option === 'questions.urgent.options',
  );
  await assert.rejects(
    client().decide({ model: 'qwen3.5-9b', questions: EXAMPLE.questions } as unknown as DecideRequest),
    (error: unknown) => error instanceof CrucibleConfigError && error.option === 'state',
  );
  assert.equal(requests, before, 'nothing reached the server');
});
