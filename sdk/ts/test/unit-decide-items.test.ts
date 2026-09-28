import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleConfigError,
  CrucibleProtocolError,
  CrucibleRefused,
  type DecideItemsRequest,
} from '../src/index.js';

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => json(response, 500, { error: { code: 'no_handler', message: 'none' } });
let lastPath = '';
let lastBody = '';
let lastHeaders: Record<string, string | string[] | undefined> = {};
let requests = 0;
let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-decide-items' });
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

const FLAGS = { hate: 'Hate', conspiracy: 'Conspiracy', none: 'None of these' };

const THUMBNAIL: DecideItemsRequest = {
  model: 'qwen3.5-9b-vl',
  state: '',
  images: ['aGVsbG8='],
  instructions: 'Answer about the image.',
  items: [
    { text: 'Is a face visible?', options: { yes: 'Yes', no: 'No' } },
    { text: 'Desktop or video?', options: { desktop: 'A desktop', video: 'A video frame' } },
  ],
};

const REVISION = '27cdb77d9381d508abfe5d4e06bcb95471e312ac';

function reply(): Record<string, any> {
  return {
    model: { id: 'qwen3.5-9b-vl', revision: REVISION, fingerprint: `qwen3.5-9b-vl@${REVISION}` },
    engine: 'mlx-vlm',
    answers: [
      {
        type: 'choice', choice: 'yes', probabilities: { yes: 0.62, no: 0.38 },
        logprobs: { yes: -0.478, no: -0.968 }, confidence: 0.62, label_mass: 0.999,
      },
      {
        type: 'choice', choice: 'video', probabilities: { desktop: 0.085, video: 0.915 },
        logprobs: { desktop: -2.465, video: -0.089 }, confidence: 0.915, label_mass: 0.998,
      },
    ],
    timing_ms: { total: 1910.0, engine_requests: 1 },
    tokens: { shared: 280, per_item: [325, 341], images: 1 },
  };
}

test('decideItems() posts the items form to /v1/decide with each item\'s own options', async () => {
  handle = (_request, response) => json(response, 200, reply());
  await client().decideItems(THUMBNAIL, { act: 'thumbnails' });
  assert.equal(lastPath, '/v1/decide');
  assert.equal(lastHeaders['x-crucible-act'], 'thumbnails');
  const body = JSON.parse(lastBody);
  assert.deepEqual(Object.keys(body), ['model', 'state', 'images', 'instructions', 'items']);
  assert.deepEqual(body.items, THUMBNAIL.items);
  assert.equal('questions' in body, false);
});

test('decideItems() reads the answers as a list in item order', async () => {
  handle = (_request, response) => json(response, 200, reply());
  const result = await client().decideItems(THUMBNAIL);
  assert.deepEqual(result.answers.map((answer) => answer.choice), ['yes', 'video']);
  assert.deepEqual(result.answers[1], {
    type: 'choice', choice: 'video', probabilities: { desktop: 0.085, video: 0.915 },
    logprobs: { desktop: -2.465, video: -0.089 }, confidence: 0.915, labelMass: 0.998,
  });
  assert.deepEqual(result.timingMs, { total: 1910.0, engineRequests: 1 });
  assert.deepEqual(result.tokens, { shared: 280, perItem: [325, 341], images: 1 });
  assert.equal(result.engine, 'mlx-vlm');
});

test('shared options travel once and items without their own send only text', async () => {
  const body = reply();
  body.answers = [
    { type: 'choice', choice: 'none', probabilities: { hate: 0.1, conspiracy: 0.1, none: 0.8 },
      logprobs: { hate: -2.3, conspiracy: -2.3, none: -0.22 }, confidence: 0.8, label_mass: 0.99,
      missing_labels: [] },
  ];
  body.tokens = { shared: null, per_item: [5290], images: 0 };
  handle = (_request, response) => json(response, 200, body);
  const result = await client().decideItems({
    model: 'qwen3.5-9b', state: 'a transcript', options: FLAGS, missing: 'report',
    items: [{ text: 'Passage: "a sentence"' }],
  });
  const sent = JSON.parse(lastBody);
  assert.deepEqual(sent.options, FLAGS);
  assert.deepEqual(sent.items, [{ text: 'Passage: "a sentence"' }]);
  assert.equal(sent.missing, 'report');
  assert.deepEqual(result.answers[0]?.missingLabels, []);
  assert.equal(result.tokens.shared, null);
});

test('an answer list of the wrong length is a protocol error', async () => {
  const body = reply();
  body.answers = body.answers.slice(0, 1);
  handle = (_request, response) => json(response, 200, body);
  await assert.rejects(client().decideItems(THUMBNAIL), CrucibleProtocolError);
});

test('an answer naming an option the item did not have is a protocol error', async () => {
  const body = reply();
  body.answers[0].choice = 'maybe';
  handle = (_request, response) => json(response, 200, body);
  await assert.rejects(client().decideItems(THUMBNAIL), CrucibleProtocolError);
});

test('an item with no options and no shared options is refused before any request', async () => {
  const before = requests;
  await assert.rejects(
    client().decideItems({ model: 'm', state: 's', items: [{ text: 'x' }] }),
    (error: unknown) => error instanceof CrucibleConfigError && /items\[0\]\.options/.test(String(error)),
  );
  await assert.rejects(client().decideItems({ model: 'm', state: 's', options: FLAGS, items: [] }), CrucibleConfigError);
  assert.equal(requests, before);
});

test('a 400 too_many_items surfaces as CrucibleRefused with its numbers', async () => {
  handle = (_request, response) =>
    json(response, 400, {
      error: { code: 'too_many_items', message: 'the request carries 513 items', details: { items: 513, max_items: 512 } },
    });
  await assert.rejects(client().decideItems(THUMBNAIL), (error: unknown) => {
    assert.ok(error instanceof CrucibleRefused);
    assert.equal(error.code, 'too_many_items');
    assert.deepEqual(error.details, { items: 513, max_items: 512 });
    return true;
  });
});
