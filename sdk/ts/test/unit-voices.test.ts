/**
 * Unit tests for the four surfaces phase 3 and phase 4 added: `voices`,
 * `loadVoice` / `unloadVoice`, the accelerator probe, and the `asr` submit.
 *
 * These exist against an in-process `node:http` fixture rather than against a
 * live server, and that is the point of the file. The previous builder shipped
 * the accelerator route with **no SDK method at all**, reasoning that the SDK's
 * only tests needed a real Crucible and "every behaviour gets a test" outranked
 * the convenience. The reasoning was right and the premise was not: a fixture
 * answers a route with whatever bytes a test wants, which is how you prove the
 * cases a healthy server never produces on a good day — a 503 that must not read
 * as an idle card, a holder whose memory the driver would not report, a voice row
 * that refuses to load and does not say why, and an `asr` submit missing a switch
 * that has no default.
 *
 * The live proof is still `test/e2e.test.ts`; these are the shapes it cannot ask
 * a working server for.
 *
 * Run: `npm run test:unit` (exits non-zero on any failure).
 */

import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import {
  CrucibleAcceleratorUnreadable,
  CrucibleClient,
  CrucibleConfigError,
  CrucibleProtocolError,
  CrucibleServerError,
  isTtsCapability,
  type JobEvent,
} from '../src/index.js';

// ------------------------------------------------------------------ fixture

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => {
  response.writeHead(500, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: { code: 'no_handler', message: 'the test set none' } }));
};

let lastPath = '';
let lastMethod = '';
let lastBody = '';
let lastHeaders: Record<string, string | string[] | undefined> = {};

let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-voices' });
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

/** Answer whatever is asked with this body, and record what was asked. */
function answers(status: number, body: unknown): void {
  handle = (_request, response) => json(response, status, body);
}

function openSse(response: ServerResponse): void {
  response.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });
}

before(async () => {
  server = createServer((request, response) => {
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

// ------------------------------------------------------------------- voices

const VOICE_REVISION = '9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c';

/** `GET /v1/voices`' row for a voice this host can serve (PHASE3-TTS.md § 2). */
const VOICE_ROW = {
  id: 'deathstalker',
  display: 'Deathstalker',
  kind: 'checkpoint',
  language: 'en',
  narrator_engine: 'higgs-v3',
  backend_supported: true,
  installed: true,
  resident: false,
  loadable: true,
  reason: null,
  revision: VOICE_REVISION,
  fingerprint: `deathstalker@${VOICE_REVISION}`,
  memory_bytes_estimate: 19000000000,
  estimate_basis: 'declared',
  max_chars: 800,
  sample_rate: 24000,
  takes: 1,
  pace: {
    pace_chars_per_sec: 16.64,
    max_chars_per_sec: 21.63,
    min_chars_per_sec: 12.8,
    target_chars: null,
    safe_min_chars: 600,
    safe_max_chars: 800,
  },
};

/** The same voice on a host whose backend its manifest has no block for. */
const UNSUPPORTED_VOICE_ROW = {
  ...VOICE_ROW,
  id: 'mac-only-voice',
  kind: 'zeroshot',
  backend_supported: false,
  installed: false,
  loadable: false,
  reason: 'mac-only-voice.toml has no cuda-linux block; it declares [mlx-darwin]',
  revision: null,
  fingerprint: null,
  memory_bytes_estimate: null,
  estimate_basis: null,
  max_chars: null,
  pace: {
    pace_chars_per_sec: 15.0,
    max_chars_per_sec: 20.0,
    min_chars_per_sec: 14.5,
    target_chars: 600,
    safe_min_chars: null,
    safe_max_chars: null,
  },
};

test('voices() reads every field /v1/voices promises, on the authed route', async () => {
  answers(200, [VOICE_ROW, UNSUPPORTED_VOICE_ROW]);
  const voices = await client().voices();

  assert.equal(lastMethod, 'GET');
  assert.equal(lastPath, '/v1/voices');
  assert.equal(lastHeaders['authorization'], 'Bearer the-token');
  assert.equal(lastHeaders['x-crucible-api'], '1');

  assert.deepEqual(voices, [
    {
      id: 'deathstalker',
      display: 'Deathstalker',
      kind: 'checkpoint',
      language: 'en',
      narratorEngine: 'higgs-v3',
      backendSupported: true,
      installed: true,
      resident: false,
      loadable: true,
      reason: null,
      revision: VOICE_REVISION,
      fingerprint: `deathstalker@${VOICE_REVISION}`,
      memoryBytesEstimate: 19000000000,
      estimateBasis: 'declared',
      maxChars: 800,
      sampleRate: 24000,
      takes: 1,
      pace: {
        paceCharsPerSec: 16.64,
        maxCharsPerSec: 21.63,
        minCharsPerSec: 12.8,
        targetChars: null,
        safeMinChars: 600,
        safeMaxChars: 800,
      },
    },
    {
      id: 'mac-only-voice',
      display: 'Deathstalker',
      kind: 'zeroshot',
      language: 'en',
      narratorEngine: 'higgs-v3',
      backendSupported: false,
      installed: false,
      resident: false,
      loadable: false,
      reason: 'mac-only-voice.toml has no cuda-linux block; it declares [mlx-darwin]',
      // All five null together, because they live in a backend block this host
      // does not have. Never 0, which would read as "needs nothing".
      revision: null,
      fingerprint: null,
      memoryBytesEstimate: null,
      estimateBasis: null,
      maxChars: null,
      // These three are facts about the voice and survive the missing block.
      sampleRate: 24000,
      takes: 1,
      pace: {
        paceCharsPerSec: 15.0,
        maxCharsPerSec: 20.0,
        minCharsPerSec: 14.5,
        targetChars: 600,
        safeMinChars: null,
        safeMaxChars: null,
      },
    },
  ]);
});

test('a voice row carries reason: null when it is loadable, and the key is always there', async () => {
  // The one shape difference from a model row, and this client reads it as it
  // is rather than making the two routes look alike.
  const { reason: _reason, ...withoutReason } = VOICE_ROW;
  answers(200, [withoutReason]);
  await assert.rejects(client().voices(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /voices\[0\] has no field "reason"/);
    return true;
  });
});

test('a voice that is not loadable and does not say why is a protocol error', async () => {
  answers(200, [{ ...VOICE_ROW, loadable: false, reason: null }]);
  await assert.rejects(client().voices(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /not loadable and its "reason" is null/);
    return true;
  });
});

test('a voice kind outside the manifest vocabulary is a protocol error', async () => {
  answers(200, [{ ...VOICE_ROW, kind: 'improvised' }]);
  await assert.rejects(client().voices(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /voices\[0\]\.kind is "improvised"/);
    return true;
  });
});

test('an estimate_basis that is neither measured nor declared is a protocol error', async () => {
  // The basis rides on the row so nothing downstream can mistake a reservation
  // for a measurement. A third word would defeat that silently.
  answers(200, [{ ...VOICE_ROW, estimate_basis: 'guessed' }]);
  await assert.rejects(client().voices(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /voices\[0\]\.estimate_basis is "guessed"/);
    return true;
  });
});

test('a pace block missing a rate is a protocol error, not a voice packed to a guess', async () => {
  const { max_chars_per_sec: _rate, ...lamePace } = VOICE_ROW.pace;
  answers(200, [{ ...VOICE_ROW, pace: lamePace }]);
  await assert.rejects(client().voices(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /voices\[0\]\.pace has no field "max_chars_per_sec"/);
    return true;
  });
});

test('a voice row without a sample_rate is a protocol error, not 24000', async () => {
  // 24000 everywhere in today's catalog is the coincidence this field exists to
  // stop becoming a constant.
  const { sample_rate: _rate, ...withoutRate } = VOICE_ROW;
  answers(200, [withoutRate]);
  await assert.rejects(client().voices(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /voices\[0\] has no field "sample_rate"/);
    return true;
  });
});

test('a /v1/voices body that is not an array is a protocol error, not an empty list', async () => {
  answers(200, { voices: [VOICE_ROW] });
  await assert.rejects(client().voices(), CrucibleProtocolError);
});

test('loadVoice and unloadVoice post their own job types and return the job id', async () => {
  answers(200, { job_id: 'job-load-voice-1' });
  const loadId = await client().loadVoice('deathstalker');
  assert.equal(loadId, 'job-load-voice-1');
  assert.equal(lastMethod, 'POST');
  assert.equal(lastPath, '/v1/jobs');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'load-voice',
    model: 'deathstalker',
    params: {},
    inputs: {},
  });

  answers(200, { job_id: 'job-unload-voice-1' });
  const unloadId = await client().unloadVoice('deathstalker');
  assert.equal(unloadId, 'job-unload-voice-1');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'unload-voice',
    model: 'deathstalker',
    params: {},
    inputs: {},
  });
});

test('loadVoice and unloadVoice refuse an empty voice id by name', async () => {
  for (const call of [
    () => client().loadVoice(''),
    () => client().unloadVoice(undefined as never),
  ]) {
    await assert.rejects(call(), (error: unknown) => {
      assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
      assert.equal(error.option, 'voice');
      return true;
    });
  }
});

// --------------------------------------------------- info's tts capability

test("info() reads the tts capability's rows with the /voices reader", async () => {
  // Before this landed, a server with `enable_tts` broke `info()` outright: the
  // tts rows were read with DESIGN.md section 4's descriptor, which demands a
  // `source` and a `vram_bytes` that a voice row does not carry.
  answers(200, {
    server: { name: 'crucible@test', version: '0.2.0', api_version: 1 },
    host: {
      platform: 'linux',
      arch: 'x86_64',
      backend: 'cuda-linux',
      gpu: { vendor: 'nvidia', name: 'NVIDIA GeForce RTX 3090 Ti', vram_bytes: 25757220864 },
    },
    job_types: ['echo', 'load-voice', 'unload-voice'],
    capabilities: [
      { job_type: 'echo', models: [] },
      { job_type: 'tts', models: [VOICE_ROW] },
    ],
  });

  const info = await client().info();
  const tts = info.capabilities.find((capability) => capability.jobType === 'tts');
  assert.ok(tts !== undefined, 'a tts-enabled server offers the tts capability');
  assert.ok(isTtsCapability(tts), 'the tts capability narrows to the /voices rows');
  assert.equal(tts.models.length, 1);
  assert.equal(tts.models[0]!.fingerprint, `deathstalker@${VOICE_REVISION}`);
  assert.equal(tts.models[0]!.pace.safeMaxChars, 800);
});

// -------------------------------------------------------------- health row

test('health() reads resident_kind beside the ids', async () => {
  answers(200, {
    status: 'ok',
    queue_depth: 0,
    resident_models: ['deathstalker'],
    resident_kind: 'tts',
  });
  const health = await client().health();
  assert.deepEqual(health, {
    status: 'ok',
    queueDepth: 0,
    residentModels: ['deathstalker'],
    // One id is one id whatever kind of thing it names; this is which door to
    // knock on, and `chat()` here would be model_not_resident.
    residentKind: 'tts',
  });
});

test('health() reports a kind this client has never heard of rather than refusing it', async () => {
  // The set grows with the job types — phase 4's aligner is next — and a client
  // that threw here would be broken by the server that added one.
  answers(200, {
    status: 'busy',
    queue_depth: 2,
    resident_models: ['qwen3-forced-aligner'],
    resident_kind: 'align',
  });
  const health = await client().health();
  assert.equal(health.residentKind, 'align');
});

test('a health body without resident_kind is a protocol error, not a null kind', async () => {
  answers(200, { status: 'ok', queue_depth: 0, resident_models: [] });
  await assert.rejects(client().health(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /health has no field "resident_kind"/);
    return true;
  });
});

// ------------------------------------------------------------- accelerator

/** `GET /v1/accelerator` on the PC, with one foreign holder. */
const ACCELERATOR = {
  backend: 'cuda-linux',
  gpu: { vendor: 'nvidia', name: 'NVIDIA GeForce RTX 3090 Ti', total_bytes: 25757220864 },
  free_bytes: 24297504768,
  used_bytes: 1459716096,
  desktop_allowance_bytes: 3221225472,
  // Zero is what an idle card reports: the allowance already covers everything
  // the driver can see. The server clamps here, and published a negative until
  // 2026-09-13.
  unattributed_bytes: 0,
  resident: {
    kind: 'llm',
    id: 'qwen3.5-9b',
    since: '2026-09-13T04:11:02Z',
    memory_bytes_estimate: 20950548480,
  },
  holders: [{ pid: 44503, name: 'python', bytes: 1249902592, owned_by_crucible: false }],
  detail: '22.6 GiB free of 24.0 GiB, 1 compute app(s), desktop allowance 3.0 GiB',
};

test('accelerator() reads the probe, on the authed route', async () => {
  answers(200, ACCELERATOR);
  const state = await client().accelerator();

  assert.equal(lastMethod, 'GET');
  assert.equal(lastPath, '/v1/accelerator');
  assert.equal(lastHeaders['authorization'], 'Bearer the-token');
  assert.equal(lastHeaders['x-crucible-api'], '1');

  assert.deepEqual(state, {
    backend: 'cuda-linux',
    gpu: { vendor: 'nvidia', name: 'NVIDIA GeForce RTX 3090 Ti', totalBytes: 25757220864 },
    freeBytes: 24297504768,
    usedBytes: 1459716096,
    desktopAllowanceBytes: 3221225472,
    unattributedBytes: 0,
    resident: {
      kind: 'llm',
      id: 'qwen3.5-9b',
      since: '2026-09-13T04:11:02Z',
      memoryBytesEstimate: 20950548480,
    },
    holders: [{ pid: 44503, name: 'python', bytes: 1249902592, ownedByCrucible: false }],
    detail: '22.6 GiB free of 24.0 GiB, 1 compute app(s), desktop allowance 3.0 GiB',
  });
});

test('a holder whose bytes the driver would not report stays null, and never becomes 0', async () => {
  // WDDM and permissions both produce this. A queue shown 0 for a process
  // holding 8 GB concludes the card is idle, which is the one conclusion this
  // route exists to prevent.
  answers(200, {
    ...ACCELERATOR,
    holders: [
      { pid: 6120, name: 'python.exe', bytes: null, owned_by_crucible: false },
      { pid: 9001, name: 'sglang', bytes: 19000000000, owned_by_crucible: true },
    ],
  });
  const state = await client().accelerator();
  assert.equal(state.holders[0]!.bytes, null);
  assert.equal(state.holders[1]!.bytes, 19000000000);
  assert.equal(state.holders[1]!.ownedByCrucible, true);
});

test('a holder with no bytes field at all is a protocol error, not an unknown one', async () => {
  // `null` is the driver declining to answer; an absent key is a server that did
  // not answer, and the two are not the same news.
  answers(200, {
    ...ACCELERATOR,
    holders: [{ pid: 6120, name: 'python.exe', owned_by_crucible: false }],
  });
  await assert.rejects(client().accelerator(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /accelerator\.holders\[0\] has no field "bytes"/);
    return true;
  });
});

test('a negative unattributed figure is surfaced, not quietly clamped here', async () => {
  // The server clamps at zero, because "VRAM nothing accounts for, past the
  // allowance" cannot be less than none. An older server that still publishes a
  // negative has a bug, and this client shows it where it is instead of hiding
  // it — correcting it here would put the fix in the wrong repository and make
  // the real one unfindable.
  answers(200, { ...ACCELERATOR, unattributed_bytes: -1509949440 });
  const state = await client().accelerator();
  assert.equal(state.unattributedBytes, -1509949440);
});

test('mlx-darwin answers unattributed_bytes: null, and it stays null', async () => {
  answers(200, {
    ...ACCELERATOR,
    backend: 'mlx-darwin',
    gpu: { vendor: 'apple', name: 'Apple M2 Ultra', total_bytes: 137438953472 },
    unattributed_bytes: null,
    resident: null,
    holders: [],
    detail: 'unified memory',
  });
  const state = await client().accelerator();
  // "Unanswerable" and "nobody unaccounted for" are different answers, and a 0
  // here would be the second one.
  assert.equal(state.unattributedBytes, null);
  assert.equal(state.resident, null);
  assert.deepEqual(state.holders, []);
});

test('an empty holders list with unattributed bytes is a busy card, not an idle one', async () => {
  // The WSL2 case, measured on Owen's PC 2026-09-12: the driver shim lists no
  // compute apps while a process inside that VM holds 17 GB.
  answers(200, {
    ...ACCELERATOR,
    free_bytes: 7516192768,
    used_bytes: 18241028096,
    unattributed_bytes: 17000000000,
    resident: null,
    holders: [],
  });
  const state = await client().accelerator();
  assert.deepEqual(state.holders, []);
  assert.equal(state.unattributedBytes, 17000000000);
});

test('503 accelerator_unreadable is its own type, and is still a server error', async () => {
  answers(503, {
    error: {
      code: 'accelerator_unreadable',
      message: 'this server cannot read its accelerator: nvidia-smi timed out after 5s',
    },
  });
  await assert.rejects(client().accelerator(), (error: unknown) => {
    assert.ok(error instanceof CrucibleAcceleratorUnreadable, `got ${String(error)}`);
    // Still a 5xx, so every phase-2 handler that catches one still catches this.
    assert.ok(error instanceof CrucibleServerError);
    assert.equal(error.status, 503);
    assert.equal(error.code, 'accelerator_unreadable');
    assert.match(error.serverMessage, /nvidia-smi timed out/);
    return true;
  });
});

test('an unreadable probe throws rather than answering with zeroes', async () => {
  // The distinction the whole route is for: a client polling for a free GPU must
  // read this as "ask again", never as "it is free". If the call ever returned a
  // state here, this assertion is what catches it.
  answers(503, {
    error: { code: 'accelerator_unreadable', message: 'nvidia-smi is not on PATH' },
  });
  let returned: unknown = 'nothing';
  try {
    returned = await client().accelerator();
  } catch (error) {
    assert.ok(error instanceof CrucibleAcceleratorUnreadable);
  }
  assert.equal(returned, 'nothing', 'an unreadable probe must not resolve to a state');
});

test('a 5xx that is not the probe refusal stays a plain server error', async () => {
  answers(503, { error: { code: 'overloaded', message: 'the lane is wedged' } });
  await assert.rejects(client().accelerator(), (error: unknown) => {
    assert.ok(error instanceof CrucibleServerError, `got ${String(error)}`);
    assert.ok(
      !(error instanceof CrucibleAcceleratorUnreadable),
      'only accelerator_unreadable gets the narrower type',
    );
    return true;
  });
});

// --------------------------------------------------------------------- asr

test('asr() posts the job the contract describes, with the params spelled the server way', async () => {
  answers(200, { job_id: 'job-asr-1' });
  const jobId = await client().asr({
    model: 'faster-whisper-base',
    audio: { blobId: 'b7c6d5e4f3a2b1c0' },
    filename: 'audiobook.m4b',
    language: 'en',
    vadFilter: true,
    wordTimestamps: true,
  });

  assert.equal(jobId, 'job-asr-1');
  assert.equal(lastMethod, 'POST');
  assert.equal(lastPath, '/v1/jobs');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'asr',
    model: 'faster-whisper-base',
    params: { language: 'en', vad_filter: true, word_timestamps: true },
    // Keyed by the caller's filename: the input's name becomes the file's name
    // on disk and ffmpeg reads the container off the extension.
    inputs: { 'audiobook.m4b': { blob_id: 'b7c6d5e4f3a2b1c0' } },
  });
});

test('asr() carries inline bytes as base64 under the caller\'s filename', async () => {
  answers(200, { job_id: 'job-asr-2' });
  await client().asr({
    model: 'faster-whisper-tiny',
    audio: { inline: new Uint8Array([0x52, 0x49, 0x46, 0x46]) },
    filename: 'clip.wav',
    language: 'auto',
    vadFilter: false,
    wordTimestamps: false,
  });
  const body = JSON.parse(lastBody) as {
    params: Record<string, unknown>;
    inputs: Record<string, { inline_base64: string }>;
  };
  assert.equal(body.inputs['clip.wav']!.inline_base64, 'UklGRg==');
  // `auto` is a value meaning "detect it", and false is a choice, not an absence.
  assert.deepEqual(body.params, {
    language: 'auto',
    vad_filter: false,
    word_timestamps: false,
  });
});

test('every asr option is required and is refused by name', async () => {
  const complete = {
    model: 'faster-whisper-base',
    audio: { blobId: 'b7c6d5e4f3a2b1c0' },
    filename: 'audiobook.m4b',
    language: 'en',
    vadFilter: true,
    wordTimestamps: true,
  };
  const options: Array<keyof typeof complete> = [
    'model',
    'audio',
    'filename',
    'language',
    'vadFilter',
    'wordTimestamps',
  ];
  for (const option of options) {
    const { [option]: _dropped, ...missing } = complete;
    await assert.rejects(
      client().asr(missing as never),
      (error: unknown) => {
        assert.ok(error instanceof CrucibleConfigError, `${option}: got ${String(error)}`);
        assert.equal(error.option, option);
        return true;
      },
      `expected ${option} to be refused by name`,
    );
  }
});

test('asr() does not coerce a switch that is not a boolean', async () => {
  // The whole reason both switches are required: a transcript made under rules
  // the caller did not choose looks exactly like one made under the rules they
  // did.
  await assert.rejects(
    client().asr({
      model: 'faster-whisper-base',
      audio: { blobId: 'b7c6d5e4f3a2b1c0' },
      filename: 'audiobook.m4b',
      language: 'en',
      vadFilter: 'yes' as never,
      wordTimestamps: true,
    }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
      assert.equal(error.option, 'vadFilter');
      assert.match(error.message, /must be a boolean, got string/);
      return true;
    },
  );
});

test('asr() sends no language list of its own; the server is the authority', async () => {
  // A second copy of faster-whisper's hundred codes in this client is a second
  // thing to drift. A code this client has never seen goes out, and comes back
  // as the server's 400 naming it.
  answers(400, {
    error: {
      code: 'invalid_params',
      message: "asr params are not valid: language: 'kw' is not a language faster-whisper knows",
    },
  });
  await assert.rejects(
    client().asr({
      model: 'faster-whisper-base',
      audio: { blobId: 'b7c6d5e4f3a2b1c0' },
      filename: 'audiobook.m4b',
      language: 'kw',
      vadFilter: true,
      wordTimestamps: true,
    }),
    (error: unknown) => {
      assert.match((error as Error).message, /is not a language faster-whisper knows/);
      return true;
    },
  );
  assert.match(lastBody, /"language":"kw"/, 'the code went out unaltered');
});

// -------------------------------------------------------- progress extras

test("a progress frame's own measurements reach the caller instead of being dropped", async () => {
  // `asr` sends these so a client shows a moving position six minutes into an
  // eighteen-hour book, while the fraction is still rounding to zero.
  handle = (_request, response) => {
    openSse(response);
    response.write(
      'id: 1\nevent: progress\ndata: ' +
        JSON.stringify({
          fraction: 0.0,
          message: 'decoding audiobook.m4b',
          stage: 'decoding',
          processed_s: 0.0,
          total_s: 0.0,
          cues: 0,
        }) +
        '\n\n',
    );
    response.write(
      'id: 2\nevent: progress\ndata: ' +
        JSON.stringify({
          fraction: 0.02,
          message: 'window 1 of 72',
          stage: 'transcribing',
          processed_s: 900.0,
          total_s: 64800.0,
          cues: 214,
        }) +
        '\n\n',
    );
    response.write('id: 3\nevent: done\ndata: {"artifacts": ["transcript.json"]}\n\n');
    response.end();
  };

  const events: JobEvent[] = [];
  for await (const event of client().events('job-asr-1')) events.push(event);

  const first = events[0]!;
  assert.equal(first.event, 'progress');
  assert.deepEqual(first.event === 'progress' ? first.data.extra : null, {
    // Verbatim: the server's spelling and the server's types, because these keys
    // are one job type's vocabulary rather than API v1's.
    stage: 'decoding',
    processed_s: 0.0,
    total_s: 0.0,
    cues: 0,
  });

  const second = events[1]!;
  assert.equal(second.event === 'progress' ? second.data.extra['cues'] : null, 214);
  assert.equal(second.event === 'progress' ? second.data.fraction : null, 0.02);
});

test('a progress frame with no measurements of its own carries an empty extra', async () => {
  // Not a substituted default: `{}` is the true answer to "what else was on the
  // frame", and it is what every job type that sends nothing extra produces.
  handle = (_request, response) => {
    openSse(response);
    response.write('id: 1\nevent: progress\ndata: {"fraction": 0.5, "message": "half"}\n\n');
    response.write('id: 2\nevent: done\ndata: {"artifacts": []}\n\n');
    response.end();
  };

  const events: JobEvent[] = [];
  for await (const event of client().events('job-plain-1')) events.push(event);
  const first = events[0]!;
  assert.equal(first.event, 'progress');
  assert.deepEqual(first.event === 'progress' ? first.data.extra : null, {});
});

test('a progress frame without a fraction is still a protocol error', async () => {
  // Carrying the extras must not loosen the two fields API v1 pins.
  handle = (_request, response) => {
    openSse(response);
    response.write('id: 1\nevent: progress\ndata: {"message": "half", "cues": 7}\n\n');
    response.end();
  };
  await assert.rejects(async () => {
    for await (const _event of client().events('job-bad-progress')) {
      // drain
    }
  }, CrucibleProtocolError);
});
