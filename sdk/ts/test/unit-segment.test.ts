import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleConfigError,
  CrucibleProtocolError,
  readSegmentResult,
  type SegmentOptions,
} from '../src/index.js';

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => {
  response.writeHead(500, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: { code: 'no_handler', message: 'the test set none' } }));
};

let lastBody = '';
let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-segment' });
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

const DONE_SELECT = {
  model: 'sam2.1-hiera-large',
  kind: 'select',
  hf_repo: 'facebook/sam2.1-hiera-large',
  revision: '665f8e2ad61cf5f53d65644ff27c8ee525124610',
  backend: 'cuda-linux',
  engine: 'sam2',
  dtype: 'float32',
  input: 'photo.jpg',
  width: 1600,
  height: 1200,
  points: [{ x: 412, y: 300, label: 1 }, { x: 520.5, y: 610, label: 0 }],
  box: [120, 80, 700, 590],
  mask: 'mask.png',
  cutout: 'cutout.png',
  score: 0.97,
  multimask: false,
  coverage: 0.184,
  seconds: 0.62,
  stage_seconds: { reading: 0.03, segmenting: 0.41, saving: 0.18 },
  peak_bytes: 2_100_000_000,
  stage_peak_bytes: { segmenting: 2_100_000_000 },
  memory_bytes_estimate: 4_000_000_000,
  memory_basis: 'declared',
  versions: { torch: '2.14.0' },
};

test('segment() posts one segment job: the picture as its input, points and box as params', async () => {
  answer(200, { job_id: 'job-segment-1' });
  const jobId = await client().segment({
    model: 'sam2.1-hiera-large',
    image: { blobId: 'blob-7' },
    imageName: 'photo.jpg',
    points: [{ x: 412, y: 300, label: 1 }],
    box: [120, 80, 700, 590],
    lease: { act: 'select', ttlSeconds: 300 },
  });
  assert.equal(jobId, 'job-segment-1');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'segment',
    model: 'sam2.1-hiera-large',
    params: {
      points: [{ x: 412, y: 300, label: 1 }],
      box: [120, 80, 700, 590],
      lease: { act: 'select', ttl_seconds: 300 },
    },
    inputs: { 'photo.jpg': { blob_id: 'blob-7' } },
    queue: {},
  });
});

test('a cutout sends no params and names its input input.png by default', async () => {
  answer(200, { job_id: 'job-segment-2' });
  await client().segment({ model: 'birefnet', image: { inline: new Uint8Array([137, 80, 78, 71]) } });
  const sent = JSON.parse(lastBody);
  assert.deepEqual(sent.params, {});
  assert.deepEqual(Object.keys(sent.inputs), ['input.png']);
  assert.equal(sent.inputs['input.png'].inline_base64, 'iVBORw==');
});

test('a mask made by an earlier job can be the input', async () => {
  answer(200, { job_id: 'job-segment-3' });
  await client().segment({
    model: 'birefnet',
    image: { artifact: { jobId: 'job-image-1', name: 'image.png' } },
  });
  assert.deepEqual(Object.values(JSON.parse(lastBody).inputs), [
    { artifact: { job_id: 'job-image-1', name: 'image.png' } },
  ]);
});

test('segment() without a picture is refused before sending', async () => {
  await assert.rejects(
    client().segment({ model: 'birefnet' } as unknown as SegmentOptions),
    CrucibleConfigError,
  );
});

test('loadSegment() queues a load-segment job, with or without a lease', async () => {
  answer(200, { job_id: 'job-load-segment' });
  assert.equal(
    await client().loadSegment('sam2.1-hiera-large', { lease: { act: 'select', ttlSeconds: 120 } }),
    'job-load-segment',
  );
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'load-segment',
    model: 'sam2.1-hiera-large',
    params: { lease: { act: 'select', ttl_seconds: 120 } },
    inputs: {},
    queue: {},
  });
  await client().loadSegment('birefnet');
  assert.deepEqual(JSON.parse(lastBody).params, {});
});

test('readSegmentResult reads the selection, its score and its artifacts', () => {
  const result = readSegmentResult({
    artifacts: ['mask.png', 'cutout.png'],
    extra: { segment: DONE_SELECT, lease_id: 'lease-3' },
  });
  assert.deepEqual([result.kind, result.width, result.height], ['select', 1600, 1200]);
  assert.deepEqual(result.points, [{ x: 412, y: 300, label: 1 }, { x: 520.5, y: 610, label: 0 }]);
  assert.deepEqual(result.box, [120, 80, 700, 590]);
  assert.deepEqual([result.mask, result.cutout, result.score, result.multimask], [
    'mask.png',
    'cutout.png',
    0.97,
    false,
  ]);
  assert.equal(result.stageSeconds.segmenting, 0.41);
  assert.equal(result.leaseId, 'lease-3');
});

test('readSegmentResult reads a cutout, whose prompts and score are null', () => {
  const result = readSegmentResult({
    artifacts: ['mask.png', 'cutout.png'],
    extra: {
      segment: { ...DONE_SELECT, model: 'birefnet', kind: 'cutout', points: null, box: null, score: null, multimask: null },
    },
  });
  assert.deepEqual([result.points, result.box, result.score, result.multimask], [null, null, null, null]);
  assert.equal(result.leaseId, null);
});

test('readSegmentResult refuses a missing block, an unknown kind and a bad label', () => {
  assert.throws(() => readSegmentResult({ artifacts: [], extra: {} }), CrucibleProtocolError);
  assert.throws(
    () => readSegmentResult({ artifacts: [], extra: { segment: { ...DONE_SELECT, kind: 'depth' } } }),
    CrucibleProtocolError,
  );
  assert.throws(
    () =>
      readSegmentResult({
        artifacts: [],
        extra: { segment: { ...DONE_SELECT, points: [{ x: 1, y: 1, label: 2 }] } },
      }),
    CrucibleProtocolError,
  );
});
