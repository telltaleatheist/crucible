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
  low_vram: false,
  decode_stages: {
    scoring: {
      tokens: 1180, cap: 4096, ended: 'eos', execution: 'cuda_graph', attention: 'sdpa',
      low_vram: false, prefix_tokens: 212, cfg_branches: 1, seconds: 9.0,
      prefill_seconds: 0.1, tokens_per_second: 131.1,
    },
    composing: {
      tokens: 9000, cap: 9000, ended: 'cap', execution: 'cuda_graph', attention: 'sdpa',
      low_vram: false, prefix_tokens: 1395, cfg_branches: 1, seconds: 50.0,
      prefill_seconds: 0.2, tokens_per_second: 180.0,
    },
  },
  stages_at_cap: ['composing'],
  planning_lyrics: null,
  length: {
    min_duration_s: 120, max_duration_s: 180, score_seconds: 144.0, in_range: true,
    attempts: [
      {
        attempt: 1, body_sections: null, structure: null, lines: null, score_seconds: 144.0,
        score: { seconds: 144.0, bars: 48, quarters: 192.0, bpm: 80.0, meter: '4/4' },
        unread: null, score_tokens: 1180, score_ended: 'eos', in_range: true,
      },
    ],
  },
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

test('readAudioResult reads how each token stage ended, and which ran to its cap', () => {
  const result = readAudioResult({ artifacts: ['audio.flac'], extra: { audio: DONE_SONG } });
  assert.deepEqual(result.stagesAtCap, ['composing']);
  assert.equal(result.lowVram, false);
  const composing = result.decodeStages?.['composing'];
  assert.deepEqual(
    [composing?.tokens, composing?.cap, composing?.ended, composing?.execution, composing?.tokensPerSecond],
    [9000, 9000, 'cap', 'cuda_graph', 180.0],
  );
  assert.equal(result.decodeStages?.['scoring']?.ended, 'eos');
  const sfx = readAudioResult({
    artifacts: ['audio.flac'],
    extra: { audio: { ...DONE_SONG, decode_stages: null, stages_at_cap: null } },
  });
  assert.deepEqual([sfx.decodeStages, sfx.stagesAtCap], [null, null]);
  const bad = { ...DONE_SONG.decode_stages.composing, ended: 'length' };
  assert.throws(
    () => readAudioResult({
      artifacts: [],
      extra: { audio: { ...DONE_SONG, decode_stages: { composing: bad } } },
    }),
    CrucibleProtocolError,
  );
});

test('an instrumental sends its planning lyrics, and the result says what the score was planned from', async () => {
  answer(200, { job_id: 'job-instrumental' });
  const words = '[Verse]' + String.fromCharCode(10) + 'Stone on stone the wall goes up' + String.fromCharCode(10);
  await client().audio({ model: 'yue2-3b', tags: 'Instrumental, piano', instrumental: true, planningLyrics: words });
  assert.deepEqual(JSON.parse(lastBody).params, {
    tags: 'Instrumental, piano',
    instrumental: true,
    planning_lyrics: words,
  });

  const pooled = { source: 'pool', id: 'harbor', requested: false, resized: false, lyrics: words };
  const result = readAudioResult({ artifacts: [], extra: { audio: { ...DONE_SONG, planning_lyrics: pooled } } });
  assert.deepEqual(result.planningLyrics, {
    source: 'pool', id: 'harbor', requested: false, resized: false, lyrics: words,
  });
  const own = readAudioResult({
    artifacts: [],
    extra: {
      audio: {
        ...DONE_SONG,
        planning_lyrics: { source: 'request', id: null, requested: null, resized: false, lyrics: words },
      },
    },
  });
  assert.deepEqual([own.planningLyrics?.source, own.planningLyrics?.id], ['request', null]);
  assert.equal(readAudioResult({ artifacts: [], extra: { audio: DONE_SONG } }).planningLyrics, null);
  assert.throws(
    () => readAudioResult({
      artifacts: [],
      extra: { audio: { ...DONE_SONG, planning_lyrics: { ...pooled, source: 'random' } } },
    }),
    CrucibleProtocolError,
  );
});

test('readAudioResult refuses a missing audio block and a kind it does not know', () => {
  assert.throws(() => readAudioResult({ artifacts: [], extra: {} }), CrucibleProtocolError);
  assert.throws(
    () => readAudioResult({ artifacts: [], extra: { audio: { ...DONE_SONG, kind: 'speech' } } }),
    CrucibleProtocolError,
  );
});

test('a song asks a length range and a planning set, and reads back what its score measured', async () => {
  answer(200, { job_id: 'job-ranged' });
  await client().audio({
    model: 'yue2-3b', tags: 'Instrumental, piano', instrumental: true, planningSet: 'harbor',
    minDurationS: 120, maxDurationS: 180,
  });
  assert.deepEqual(JSON.parse(lastBody).params, {
    tags: 'Instrumental, piano', instrumental: true, planning_set: 'harbor',
    min_duration_s: 120, max_duration_s: 180,
  });
  const length = readAudioResult({ artifacts: [], extra: { audio: DONE_SONG } }).length;
  assert.deepEqual(
    [length?.minDurationS, length?.maxDurationS, length?.scoreSeconds, length?.inRange],
    [120, 180, 144.0, true],
  );
  assert.deepEqual(length?.attempts[0], {
    attempt: 1, bodySections: null, structure: null, lines: null, scoreSeconds: 144.0,
    unread: null, scoreTokens: 1180, scoreEnded: 'eos', inRange: true,
  });
  assert.equal(readAudioResult({ artifacts: [], extra: { audio: { ...DONE_SONG, length: null } } }).length, null);
});
