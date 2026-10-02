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

test('inpainting sends the picture and the mask as two inputs and names the mask in params', async () => {
  answer(200, { job_id: 'job-image-mask' });
  await client().image({
    model: 'qwen-image-2.1',
    prompt: 'a kitchen table with a blue vase',
    width: 1024,
    height: 768,
    image: { blobId: 'photo' },
    imageName: 'photo.png',
    mask: { blobId: 'selection' },
    maskBlur: 12,
  });
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'image',
    model: 'qwen-image-2.1',
    params: {
      prompt: 'a kitchen table with a blue vase',
      width: 1024,
      height: 768,
      mask_blur: 12,
      mask: 'mask.png',
    },
    inputs: { 'photo.png': { blob_id: 'photo' }, 'mask.png': { blob_id: 'selection' } },
  });
  await client().image({
    model: 'qwen-image-2.1',
    prompt: 'x',
    image: { blobId: 'photo' },
    mask: { blobId: 'selection' },
    maskName: 'sel.png',
    imageStrength: 0.2,
  });
  const body = JSON.parse(lastBody);
  assert.deepEqual(body.params, { prompt: 'x', image_strength: 0.2, mask: 'sel.png' });
  assert.deepEqual(Object.keys(body.inputs).sort(), ['input.png', 'sel.png']);
});

test('a mask without an image, a maskBlur without a mask, or one name for both is refused before sending', async () => {
  await assert.rejects(
    client().image({ model: 'qwen-image-2.1', prompt: 'x', mask: { blobId: 'm' } }),
    CrucibleConfigError,
  );
  await assert.rejects(
    client().image({ model: 'qwen-image-2.1', prompt: 'x', maskBlur: 4 }),
    CrucibleConfigError,
  );
  await assert.rejects(
    client().image({
      model: 'qwen-image-2.1',
      prompt: 'x',
      image: { blobId: 'a' },
      imageName: 'same.png',
      mask: { blobId: 'm' },
      maskName: 'same.png',
    }),
    CrucibleConfigError,
  );
});

test('readImageResult reads the mask fields, and null from an older server', () => {
  const masked = readImageResult({
    artifacts: ['image.png'],
    extra: {
      image: {
        ...DONE_IMAGE,
        input: 'photo.png',
        mask: 'mask.png',
        mask_blur: 8,
        mask_coverage: 0.25,
        mask_blend_steps: 30,
        mask_outside_drift: 2.5,
      },
    },
  });
  assert.deepEqual([masked.input, masked.mask, masked.maskBlur, masked.maskCoverage], ['photo.png', 'mask.png', 8, 0.25]);
  assert.deepEqual([masked.maskBlendSteps, masked.maskOutsideDrift], [30, 2.5]);
  const older = readImageResult({ artifacts: ['image.png'], extra: { image: DONE_IMAGE } });
  assert.deepEqual(
    [older.mask, older.maskBlur, older.maskCoverage, older.maskBlendSteps, older.maskOutsideDrift],
    [null, null, null, null, null],
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

test('loadImage() queues a load-image job', async () => {
  answer(200, { job_id: 'job-load-image' });
  assert.equal(await client().loadImage('qwen-image-2.1'), 'job-load-image');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'load-image',
    model: 'qwen-image-2.1',
    params: {},
    inputs: {},
  });
});

test('readImageResult reads prompt_cache, and null when the server does not say', () => {
  const fresh = readImageResult({
    artifacts: ['image.png'],
    extra: { image: { ...DONE_IMAGE, prompt_cache: 'hit' } },
  });
  assert.equal(fresh.promptCache, 'hit');
  const unsaid = readImageResult({ artifacts: ['image.png'], extra: { image: DONE_IMAGE } });
  assert.equal(unsaid.promptCache, null);
});

test('readImageResult refuses a prompt_cache it does not know', () => {
  assert.throws(
    () => readImageResult({ artifacts: ['image.png'], extra: { image: { ...DONE_IMAGE, prompt_cache: 'maybe' } } }),
    CrucibleProtocolError,
  );
});
