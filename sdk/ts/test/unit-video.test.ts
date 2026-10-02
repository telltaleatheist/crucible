import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import { CrucibleClient, CrucibleConfigError, CrucibleProtocolError, readVideoResult } from '../src/index.js';

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => {
  response.writeHead(500, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: { code: 'no_handler', message: 'the test set none' } }));
};

let lastBody = '';
let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-video' });
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

const DONE_CLIP = {
  model: 'ltx-2.5-distilled',
  hf_repo: 'Lightricks/LTX-2.5-Diffusers',
  revision: '426936f8b22dc28e4def61e515478b0b7e4a53cc',
  transformer: {
    hf_repo: 'Abiray/LTX-2.5-Distilled-GGUF',
    revision: '7b0c2025441f1bf12c18eac375ad21f5e3d3c9e0',
    file: 'LTX-2.5-Distilled-Q6_K.gguf',
    sha256: 'ee8835ff8f11e4f59fa4be7bf31b1200172659364e724de444d790ddf4869a58',
  },
  backend: 'cuda-linux',
  engine: 'ltx',
  dtype: 'bfloat16',
  quantization: { text_encoder: 'torchao int8 weight-only', transformer: 'GGUF Q6_K' },
  mode: 'image-to-video',
  prompt: 'The woman in the photo turns toward the window and smiles',
  input: 'start.png',
  width: 1280,
  height: 704,
  num_frames: 73,
  fps: 24,
  duration_s: 3.042,
  video_tokens: 8800,
  seed: 11,
  steps: 8,
  audio: true,
  audio_seconds: 3.04,
  audio_sample_rate: 48000,
  audio_channels: 2,
  artifact: 'video.mp4',
  bytes: 2_900_000,
  encoder: 'libopenh264',
  seconds: 120.5,
  stage_seconds: { encoding: 40.1, connecting: 10.2, conditioning: 3.3, denoising: 50.0, decoding: 14.0 },
  peak_bytes: 19_500_000_000,
  stage_peak_bytes: { denoising: 19_500_000_000, decoding: 7_000_000_000 },
  memory_bytes_estimate: 20_500_000_000,
  memory_basis: 'declared',
  stage_memory_bytes: { encoding: 16_000_000_000, denoising: 20_500_000_000 },
  prompt_cache: 'miss',
  versions: { diffusers: '0.41.0.dev0' },
};

test('video() posts one video job with snake_case params and only what the caller set', async () => {
  answer(200, { job_id: 'job-video-1' });
  const jobId = await client().video({
    model: 'ltx-2.5-distilled',
    prompt: 'A red fox trots through fresh snow at dawn',
    width: 768,
    height: 512,
    durationS: 4,
    audio: false,
  });
  assert.equal(jobId, 'job-video-1');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'video',
    model: 'ltx-2.5-distilled',
    params: {
      prompt: 'A red fox trots through fresh snow at dawn',
      width: 768,
      height: 512,
      duration_s: 4,
      audio: false,
    },
    inputs: {},
  });
});

test('a start picture goes as the one input', async () => {
  answer(200, { job_id: 'job-video-2' });
  await client().video({
    model: 'ltx-2.5-distilled',
    prompt: 'She turns and smiles',
    numFrames: 73,
    image: { blobId: 'picture-1' },
  });
  const sent = JSON.parse(lastBody);
  assert.deepEqual(sent.params, {
    prompt: 'She turns and smiles',
    num_frames: 73,
  });
  assert.deepEqual(Object.keys(sent.inputs), ['start.png']);
});

test('video() refuses a length said twice and a lone side before sending', async () => {
  await assert.rejects(
    client().video({ model: 'ltx-2.5-distilled', prompt: 'x', durationS: 2, numFrames: 49 }),
    CrucibleConfigError,
  );
  await assert.rejects(client().video({ model: 'ltx-2.5-distilled', prompt: 'x', width: 768 }), CrucibleConfigError);
  await assert.rejects(client().video({ model: 'ltx-2.5-distilled', prompt: '' }), CrucibleConfigError);
});

test('loadVideo() queues a load-video job', async () => {
  answer(200, { job_id: 'job-load-video' });
  assert.equal(await client().loadVideo('ltx-2.5-distilled'), 'job-load-video');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'load-video',
    model: 'ltx-2.5-distilled',
    params: {},
    inputs: {},
  });
});

test('readVideoResult reads what the clip can be made again from', () => {
  const result = readVideoResult({ artifacts: ['video.mp4'], extra: { video: DONE_CLIP } });
  assert.deepEqual([result.mode, result.input, result.numFrames, result.fps], ['image-to-video', 'start.png', 73, 24]);
  assert.deepEqual([result.seed, result.steps, result.videoTokens, result.audio], [11, 8, 8800, true]);
  assert.equal(result.transformer?.file, 'LTX-2.5-Distilled-Q6_K.gguf');
  assert.equal(result.refineSteps, null);
  assert.equal(result.stageMemoryBytes.denoising, 20_500_000_000);
  assert.equal(result.stageSeconds.conditioning, 3.3);
  assert.deepEqual([result.encoder, result.promptCache], ['libopenh264', 'miss']);
  assert.deepEqual(result.artifacts, ['video.mp4']);
});

test('readVideoResult reads a Mac clip: no separate transformer, a refining pass', () => {
  const mac = {
    ...DONE_CLIP,
    hf_repo: 'dgrauet/ltx-2.5-mlx-q8',
    revision: '746ca9aacb697d2c739f544d68b584214dedcc75',
    transformer: null,
    backend: 'mlx-darwin',
    engine: 'ltx-2-mlx',
    refine_steps: 3,
    encoder: 'h264_videotoolbox',
    stage_peak_bytes: { denoising: 23_000_000_000, refining: 25_000_000_000 },
  };
  const result = readVideoResult({ artifacts: ['video.mp4'], extra: { video: mac } });
  assert.equal(result.transformer, null);
  assert.deepEqual([result.engine, result.steps, result.refineSteps], ['ltx-2-mlx', 8, 3]);
  assert.equal(result.stagePeakBytes.refining, 25_000_000_000);
});

test('readVideoResult refuses a missing video block and a mode it does not know', () => {
  assert.throws(() => readVideoResult({ artifacts: [], extra: {} }), CrucibleProtocolError);
  assert.throws(
    () => readVideoResult({ artifacts: [], extra: { video: { ...DONE_CLIP, mode: 'video-to-video' } } }),
    CrucibleProtocolError,
  );
});
