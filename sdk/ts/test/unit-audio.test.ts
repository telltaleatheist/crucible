import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import { CrucibleClient, CrucibleConfigError, CrucibleProtocolError, readAudioResult } from '../src/index.js';

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => {
  response.writeHead(500, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: { code: 'no_handler', message: 'the test set none' } }));
};

let lastBody = '';
let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-audio' });
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

const DONE_SONG = {
  model: 'yue2-3b',
  kind: 'song',
  hf_repo: 'm-a-p/YuE2-3B',
  revision: 'c044757a011169583f363168348ae380946efff8',
  backend: 'cuda-linux',
  engine: 'yue2',
  dtype: 'bfloat16',
  prompt: null,
  tags: 'English, warm piano pop, 88 BPM',
  lyrics: '[Verse]\nThe kettle sings\n',
  duration_s: null,
  seed: 3,
  steps: null,
  cfg: 1.0,
  format: 'flac',
  artifact: 'audio.flac',
  score: 'score.abc',
  audio_seconds: 182.4,
  sample_rate: 48000,
  channels: 2,
  seconds: 71.2,
  stage_seconds: { scoring: 9.1, composing: 50.2, synthesizing: 8.4, decoding: 2.1, saving: 0.4 },
  peak_bytes: 12_000_000_000,
  stage_peak_bytes: { scoring: 9_000_000_000, composing: 12_000_000_000 },
  memory_bytes_estimate: 16_000_000_000,
  memory_basis: 'declared',
};

test('audio() posts one audio job with snake_case params and only what the caller set', async () => {
  answer(200, { job_id: 'job-audio-1' });
  const jobId = await client().audio({
    model: 'stable-audio-3-small-sfx',
    prompt: 'TrackType: SFX. A door creaks open',
    durationS: 4,
    seed: 9,
  });
  assert.equal(jobId, 'job-audio-1');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'audio',
    model: 'stable-audio-3-small-sfx',
    params: { prompt: 'TrackType: SFX. A door creaks open', duration_s: 4, seed: 9 },
    inputs: {},
    queue: {},
  });
});

test('a song sends tags, lyrics, cfg and format', async () => {
  answer(200, { job_id: 'job-audio-2' });
  await client().audio({
    model: 'yue2-3b',
    tags: 'English, piano pop',
    lyrics: '[Verse]\nla la\n',
    cfg: 1.2,
    format: 'wav',
  });
  assert.deepEqual(JSON.parse(lastBody).params, {
    tags: 'English, piano pop',
    lyrics: '[Verse]\nla la\n',
    cfg: 1.2,
    format: 'wav',
  });
});

test('audio() without a prompt or tags is refused before sending', async () => {
  await assert.rejects(client().audio({ model: 'stable-audio-3-medium' }), CrucibleConfigError);
});

test('loadAudio() queues a load-audio job', async () => {
  answer(200, { job_id: 'job-load-audio' });
  assert.equal(await client().loadAudio('stable-audio-3-medium'), 'job-load-audio');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'load-audio',
    model: 'stable-audio-3-medium',
    params: {},
    inputs: {},
    queue: {},
  });
});

test('readAudioResult reads what a song can be made again from, score included', () => {
  const result = readAudioResult({
    artifacts: ['audio.flac', 'score.abc'],
    extra: { audio: DONE_SONG },
  });
  assert.deepEqual([result.kind, result.seed, result.cfg, result.durationS], ['song', 3, 1.0, null]);
  assert.deepEqual([result.sampleRate, result.channels, result.score], [48000, 2, 'score.abc']);
  assert.equal(result.stageSeconds.composing, 50.2);
  assert.deepEqual(result.artifacts, ['audio.flac', 'score.abc']);
});

test('readAudioResult refuses a missing audio block and a kind it does not know', () => {
  assert.throws(() => readAudioResult({ artifacts: [], extra: {} }), CrucibleProtocolError);
  assert.throws(
    () => readAudioResult({ artifacts: [], extra: { audio: { ...DONE_SONG, kind: 'speech' } } }),
    CrucibleProtocolError,
  );
});
