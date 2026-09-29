import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import { CrucibleClient, CrucibleConfigError, CrucibleProtocolError, readImageResult } from '../src/index.js';

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => {
  response.writeHead(500, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: { code: 'no_handler', message: 'the test set none' } }));
};

let lastPath = '';
let lastBody = '';
let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-image' });
}

function answer(status: number, body: unknown): void {
  handle = (_request, response) => {
    response.writeHead(status, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify(body));
  };
}

before(async () => {
  server = createServer((request, response) => {
    let body = '';
    request.on('data', (chunk) => {
      body += chunk;
    });
    request.on('end', () => {
      lastPath = request.url ?? '';
      lastBody = body;
      handle(request, response, body);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  await new Promise<void>((resolve, reject) =>
    server.close((error) => (error ? reject(error) : resolve())),
  );
});

const DONE_IMAGE = {
  model: 'qwen-image-2.1',
  hf_repo: 'Qwen/Qwen-Image-2.1',
  revision: '790c92633540aa0cb11d9abf19eb46d861714758',
  backend: 'mlx-darwin',
  engine: 'mflux',
  dtype: 'bfloat16',
  prompt: 'a kitchen table with one red apple',
  negative_prompt: null,
  width: 1280,
  height: 720,
  seed: 1,
  steps: 40,
  guidance: 1.0,
  image_strength: null,
  input: null,
  seconds: 301.5,
  stage_seconds: { encoding: 9.1, denoising: 280.2, decoding: 12.2 },
  peak_bytes: 16_000_000_000,
  stage_peak_bytes: { encoding: 16_000_000_000, denoising: 15_100_000_000, decoding: 5_000_000_000 },
  memory_bytes_estimate: 17_000_000_000,
  memory_basis: 'measured',
};

test('image() posts one image job with snake_case params and only what the caller set', async () => {
  answer(200, { job_id: 'job-image-1' });
  const jobId = await client().image({
    model: 'qwen-image-2.1',
    prompt: 'a kitchen table with one red apple',
    width: 1280,
    height: 720,
    seed: 1,
  });
  assert.equal(jobId, 'job-image-1');
  assert.equal(lastPath, '/v1/jobs');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'image',
    model: 'qwen-image-2.1',
    params: { prompt: 'a kitchen table with one red apple', width: 1280, height: 720, seed: 1 },
    inputs: {},
  });
});

test('image-to-image sends the picture as one input beside image_strength', async () => {
  answer(200, { job_id: 'job-image-2' });
  await client().image({
    model: 'qwen-image-2.1',
    prompt: 'the same room at dusk',
    negativePrompt: 'glossy',
    guidance: 4,
    imageStrength: 0.6,
    image: { blobId: 'abc' },
    imageName: 'room.jpg',
  });
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'image',
    model: 'qwen-image-2.1',
    params: {
      prompt: 'the same room at dusk',
      negative_prompt: 'glossy',
      guidance: 4,
      image_strength: 0.6,
    },
    inputs: { 'room.jpg': { blob_id: 'abc' } },
  });
});

test('an image without a strength, or a strength without an image, is refused before sending', async () => {
  await assert.rejects(
    client().image({ model: 'qwen-image-2.1', prompt: 'x', image: { blobId: 'abc' } }),
    CrucibleConfigError,
  );
  await assert.rejects(
    client().image({ model: 'qwen-image-2.1', prompt: 'x', imageStrength: 0.5 }),
    CrucibleConfigError,
  );
});

test('readImageResult reads the effective parameters a picture can be made again from', () => {
  const result = readImageResult({ artifacts: ['image.png'], extra: { image: DONE_IMAGE } });
  assert.equal(result.seed, 1);
  assert.equal(result.revision, DONE_IMAGE.revision);
  assert.deepEqual([result.width, result.height, result.steps], [1280, 720, 40]);
  assert.equal(result.negativePrompt, null);
  assert.deepEqual(result.stageSeconds, DONE_IMAGE.stage_seconds);
  assert.equal(result.stagePeakBytes.encoding, 16_000_000_000);
  assert.equal(result.memoryBasis, 'measured');
  assert.deepEqual(result.artifacts, ['image.png']);
});

test('readImageResult refuses a done frame with no image block', () => {
  assert.throws(() => readImageResult({ artifacts: ['image.png'], extra: {} }), CrucibleProtocolError);
});
