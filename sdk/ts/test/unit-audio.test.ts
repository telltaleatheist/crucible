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
    min_duration_s: 120, max_duration_s: 180, sized: true, body_sections: 3,
    structure: ['verse', 'chorus', 'verse'], lines: 12, score_seconds: 144.0,
    score: { seconds: 144.0, bars: 48, quarters: 192.0, bpm: 80.0, meter: '4/4' },
    unread: null, in_range: true,
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
  assert.deepEqual(length, {
    minDurationS: 120, maxDurationS: 180, sized: true, bodySections: 3,
    structure: ['verse', 'chorus', 'verse'], lines: 12, scoreSeconds: 144.0, unread: null,
    inRange: true,
  });
  assert.equal(readAudioResult({ artifacts: [], extra: { audio: { ...DONE_SONG, length: null } } }).length, null);
});

test('a song writes its CFG negative branch and reads back which one it composed against', async () => {
  answer(200, { job_id: 'job-negative' });
  await client().audio({
    model: 'yue2-3b', tags: 'Instrumental, piano', instrumental: true, cfg: 2.0,
    negativeTags: 'lead vocals, singing, choir', negativeLyrics: '[Verse]\nla la\n',
  });
  assert.deepEqual(JSON.parse(lastBody).params, {
    tags: 'Instrumental, piano', instrumental: true, cfg: 2.0,
    negative_tags: 'lead vocals, singing, choir', negative_lyrics: '[Verse]\nla la\n',
  });
  const written = readAudioResult({
    artifacts: [],
    extra: { audio: { ...DONE_SONG, negative: { tags: 'singing', lyrics: '' }, negative_prefix_tokens: 640 } },
  });
  assert.deepEqual(written.negative, { tags: 'singing', lyrics: '' });
  assert.equal(written.negativePrefixTokens, 640);
  const stock = readAudioResult({
    artifacts: [], extra: { audio: { ...DONE_SONG, negative: null, negative_prefix_tokens: 610 } },
  });
  assert.equal(stock.negative, null);
  assert.equal(stock.negativePrefixTokens, 610);
  // A server older than the knob sends neither field.
  const older = readAudioResult({ artifacts: [], extra: { audio: DONE_SONG } });
  assert.equal(older.negative, null);
  assert.equal(older.negativePrefixTokens, null);
  assert.throws(
    () => readAudioResult({ artifacts: [], extra: { audio: { ...DONE_SONG, negative: { tags: 'x' } } } }),
    CrucibleProtocolError,
  );
});

const NL = String.fromCharCode(10);
const ABC = ['X:1', 'T:', 'M:4/4', 'L:1/32', 'Q:1/4=66', 'V:Vocal', '% verse', 'C8D8E8F8|', ''].join(NL);

test('an instrumental cover uploads its score: the input carries it and params.score names it', async () => {
  answer(200, { job_id: 'job-cover-a' });
  const crucible = client();
  await crucible.audio({
    model: 'yue2-3b',
    tags: 'Instrumental, warm piano and strings, no vocals, no singing, 72 BPM',
    instrumental: true,
    score: { blobId: 'blob-abc' },
    seed: 3,
  });
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'audio',
    model: 'yue2-3b',
    params: {
      tags: 'Instrumental, warm piano and strings, no vocals, no singing, 72 BPM',
      instrumental: true,
      score: 'score.abc',
      seed: 3,
    },
    inputs: { 'score.abc': { blob_id: 'blob-abc' } },
  });

  answer(200, { job_id: 'job-cover-a-inline' });
  await crucible.audio({
    model: 'yue2-3b',
    tags: 'Instrumental, piano',
    instrumental: true,
    score: { inline: new TextEncoder().encode(ABC) },
    chords: 'strip',
  });
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'audio',
    model: 'yue2-3b',
    params: { tags: 'Instrumental, piano', instrumental: true, chords: 'strip', score: 'score.abc' },
    inputs: { 'score.abc': { inline_base64: Buffer.from(ABC, 'utf8').toString('base64') } },
  });
});

test('a sung cover takes a held job\'s score by reference, under the name it chose', async () => {
  answer(200, { job_id: 'job-cover-b' });
  const crucible = client();
  const lyrics = '[Verse]' + NL + 'The lantern by the harbour wall' + NL;
  await crucible.audio({
    model: 'yue2-3b',
    tags: 'English, intimate acoustic folk, warm male voice, 72 BPM',
    lyrics,
    score: crucible.artifactRef('job-score-1', 'score.abc'),
    scoreName: 'melody.abc',
  });
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'audio',
    model: 'yue2-3b',
    params: {
      tags: 'English, intimate acoustic folk, warm male voice, 72 BPM',
      lyrics,
      score: 'melody.abc',
    },
    inputs: { 'melody.abc': { artifact: { job_id: 'job-score-1', name: 'score.abc' } } },
  });

  answer(200, { job_id: 'job-cover-b-upload' });
  await crucible.audio({
    model: 'yue2-3b', tags: 'English, folk', lyrics, score: { blobId: 'blob-melody' }, scoreName: 'melody.abc',
    chords: 'keep',
  });
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'audio',
    model: 'yue2-3b',
    params: { tags: 'English, folk', lyrics, chords: 'keep', score: 'melody.abc' },
    inputs: { 'melody.abc': { blob_id: 'blob-melody' } },
  });
});

test('chords and scoreName without a score are refused before sending', async () => {
  lastBody = '';
  await assert.rejects(
    client().audio({ model: 'yue2-3b', tags: 'Instrumental', instrumental: true, chords: 'keep' }),
    CrucibleConfigError,
  );
  await assert.rejects(
    client().audio({ model: 'yue2-3b', tags: 'Instrumental', instrumental: true, scoreName: 'melody.abc' }),
    CrucibleConfigError,
  );
  assert.equal(lastBody, '');
});

const COVER_REFERENCE = {
  kind: 'sung', source: 'reference', input: 'melody.abc',
  reference: { job_id: 'job-score-1', name: 'score.abc' },
  given: 'given-score.abc', cot: 'melody', chords: 'stripped', chords_chosen_by: 'default',
  score_tokens: 1180, score_seconds: 95.0,
  score: { seconds: 95.0, bars: 26, quarters: 104.0, bpm: 66, meter: '4/4' },
};

test('readAudioResult reads how a cover was made, and null for a song that is not one', () => {
  const sung = readAudioResult({
    artifacts: ['audio.flac', 'score.abc', 'given-score.abc'],
    extra: { audio: { ...DONE_SONG, cover: COVER_REFERENCE } },
  });
  assert.deepEqual(sung.cover, {
    kind: 'sung', source: 'reference', input: 'melody.abc',
    reference: { jobId: 'job-score-1', name: 'score.abc' },
    given: 'given-score.abc', cot: 'melody', chords: 'stripped', chordsChosenBy: 'default',
    scoreTokens: 1180, scoreSeconds: 95.0,
    score: { seconds: 95.0, bars: 26, quarters: 104.0, bpm: 66, meter: '4/4' },
  });
  assert.deepEqual(sung.artifacts, ['audio.flac', 'score.abc', 'given-score.abc']);

  const uploaded = readAudioResult({
    artifacts: [],
    extra: {
      audio: {
        ...DONE_SONG,
        cover: {
          ...COVER_REFERENCE, kind: 'instrumental', source: 'upload', input: 'score.abc', reference: null,
          cot: 'full', chords: 'kept', chords_chosen_by: 'request',
        },
      },
    },
  });
  assert.deepEqual(
    [uploaded.cover?.kind, uploaded.cover?.source, uploaded.cover?.reference, uploaded.cover?.cot, uploaded.cover?.chordsChosenBy],
    ['instrumental', 'upload', null, 'full', 'request'],
  );

  // A planned song on a current server says null; an older server sends no key at all.
  assert.equal(readAudioResult({ artifacts: [], extra: { audio: { ...DONE_SONG, cover: null } } }).cover, null);
  assert.equal(readAudioResult({ artifacts: [], extra: { audio: DONE_SONG } }).cover, null);
});

test('readAudioResult refuses a cover record that is missing a field or contradicts itself', () => {
  const read = (cover: unknown) => () => readAudioResult({ artifacts: [], extra: { audio: { ...DONE_SONG, cover } } });
  const { score_seconds: _seconds, ...noSeconds } = COVER_REFERENCE;
  assert.throws(read(noSeconds), CrucibleProtocolError);
  assert.throws(read({ ...COVER_REFERENCE, kind: 'remix' }), CrucibleProtocolError);
  assert.throws(read({ ...COVER_REFERENCE, reference: null }), CrucibleProtocolError);
  assert.throws(read({ ...COVER_REFERENCE, source: 'upload' }), CrucibleProtocolError);
  assert.throws(read({ ...COVER_REFERENCE, score: { ...COVER_REFERENCE.score, meter: undefined } }), CrucibleProtocolError);
  assert.throws(read('given-score.abc'), CrucibleProtocolError);
});
