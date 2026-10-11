import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import { CrucibleClient, CrucibleProtocolError, CrucibleRefused, readAudioResult } from '../src/index.js';

// The playground routes and a song's `instrumental`, as B-Side (the first app outside the
// operator page to draw a playground form) uses them.

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => {
  response.writeHead(500, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: { code: 'no_handler', message: 'the test set none' } }));
};

let last: { method: string; url: string; body: string } = { method: '', url: '', body: '' };
let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-playground' });
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
      last = { method: request.method ?? '', url: request.url ?? '', body };
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

const SONG_PAGE = {
  job_type: 'audio',
  id: 'yue2-3b',
  name: 'YuE2 3B',
  media: 'audio',
  kind: 'song',
  makes: 'songs',
  standing: 'ready',
  available: true,
  reason: null,
  download_bytes: null,
  fields: [
    {
      name: 'tags', label: 'Style tags', kind: 'tags', required: true, default: null,
      placeholder: 'English, warm piano pop, 88 BPM', hint: 'type a phrase and a comma',
      suggestions: [{ group: 'Genre', tags: ['pop', 'jazz'] }],
      conflicts: { 'male vocal': [{ tag: 'female vocal', why: 'one lead voice' }] },
    },
    { name: 'instrumental', label: 'Instrumental (no vocals)', kind: 'boolean', required: false, default: false },
    { name: 'cfg', label: 'Guidance', kind: 'number', required: false, default: 1.0, min: 0.5, max: 3, step: 0.1 },
  ],
};

test('playground() reads each page: its standing, and a song form with suggestions and conflicts', async () => {
  answer(200, { pages: [SONG_PAGE] });
  const [page] = await client().playground();
  assert.equal(last.url, '/v1/playground');
  assert.deepEqual([page?.id, page?.standing, page?.available, page?.downloadBytes], ['yue2-3b', 'ready', true, null]);
  const tags = page?.fields[0];
  assert.deepEqual(tags?.suggestions, [{ group: 'Genre', tags: ['pop', 'jazz'] }]);
  assert.deepEqual(tags?.conflicts, { 'male vocal': [{ tag: 'female vocal', why: 'one lead voice' }] });
  const cfg = page?.fields[2];
  assert.deepEqual([cfg?.default, cfg?.min, cfg?.max, cfg?.step, cfg?.suggestions], [1.0, 0.5, 3, 0.1, null]);
});

test('playground() refuses a standing it does not know, rather than drawing a guess', async () => {
  answer(200, { pages: [{ ...SONG_PAGE, standing: 'maybe' }] });
  await assert.rejects(client().playground(), CrucibleProtocolError);
});

test('presets: list, save (the params as given, in a {params} body) and delete, on the model\'s path', async () => {
  answer(200, { model: 'yue2-3b', presets: [{ name: 'Porch', params: { tags: 'folk', instrumental: true }, saved_at: '2026-10-04T06:00:00Z' }] });
  const listed = await client().playgroundPresets('yue2-3b');
  assert.equal(last.url, '/v1/playground/presets/yue2-3b');
  assert.deepEqual(listed, [{ name: 'Porch', params: { tags: 'folk', instrumental: true }, savedAt: '2026-10-04T06:00:00Z' }]);

  answer(200, { model: 'yue2-3b', preset: { name: 'Rainy day', params: { tags: 'lo-fi', cfg: 1.2 }, saved_at: '2026-10-04T06:01:00Z' } });
  const saved = await client().savePlaygroundPreset('yue2-3b', 'Rainy day', { tags: 'lo-fi', cfg: 1.2 });
  assert.equal(last.method, 'PUT');
  assert.equal(last.url, '/v1/playground/presets/yue2-3b/Rainy%20day');
  assert.deepEqual(JSON.parse(last.body), { params: { tags: 'lo-fi', cfg: 1.2 } });
  assert.equal(saved.name, 'Rainy day');

  answer(200, { model: 'yue2-3b', deleted: 'Rainy day' });
  await client().deletePlaygroundPreset('yue2-3b', 'Rainy day');
  assert.equal(last.method, 'DELETE');
});

test('a preset the server says is missing is refused with its code', async () => {
  answer(404, { error: { code: 'preset_not_found', message: "yue2-3b has no preset 'x'" } });
  await assert.rejects(client().deletePlaygroundPreset('yue2-3b', 'x'), (error: unknown) =>
    error instanceof CrucibleRefused && error.code === 'preset_not_found');
});

test('audio() sends instrumental; readAudioResult reads it, and null from a server older than it', async () => {
  answer(200, { job_id: 'job-song' });
  await client().audio({ model: 'yue2-3b', tags: 'lo-fi, jazz', instrumental: true });
  assert.deepEqual(JSON.parse(last.body).params, { tags: 'lo-fi, jazz', instrumental: true });

  const done = {
    model: 'yue2-3b', kind: 'song', hf_repo: 'm-a-p/YuE2-3B', revision: 'r', backend: 'cuda-linux',
    engine: 'yue2', dtype: 'bfloat16', prompt: null, tags: 'lo-fi', lyrics: null, duration_s: null,
    seed: 7, steps: null, cfg: 1.0, format: 'flac', artifact: 'audio.flac', score: null,
    audio_seconds: 30, sample_rate: 44100, channels: 2, seconds: 60, stage_seconds: null,
    peak_bytes: null, stage_peak_bytes: null, memory_bytes_estimate: 16e9, memory_basis: 'declared',
    low_vram: false, decode_stages: null, stages_at_cap: null, planning_lyrics: null,
  };
  assert.equal(readAudioResult({ artifacts: [], extra: { audio: { ...done, instrumental: true } } }).instrumental, true);
  assert.equal(readAudioResult({ artifacts: [], extra: { audio: done } }).instrumental, null);
});
