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
  orphan: null, backend_supported: true,
  installed: true,
  resident: false,
  loadable: true,
  reason: null,
  revision: VOICE_REVISION,
  fingerprint: `deathstalker@${VOICE_REVISION}`,
  memory_bytes_estimate: 19000000000, held_by: null, unclaimed_since: null,
  estimate_basis: 'declared',
  max_chars: 800,
  sample_rate: 24000,
  takes: 2,
  needs_reference: false,
  serving: {
    max_num_seqs: 16,
    max_num_seqs_note: "vllm-omni's own stage-0 value, measured at 0.35 + 0.10.",
    mem_fraction: null,
    mem_fraction_note: null,
    context_length: null,
    context_length_note: null,
  },
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
  needs_reference: true,
  orphan: null, backend_supported: false,
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
      takes: 2,
      serving: {
        maxNumSeqs: 16,
        maxNumSeqsNote: "vllm-omni's own stage-0 value, measured at 0.35 + 0.10.",
        memFraction: null,
        memFractionNote: null,
        contextLength: null,
        contextLengthNote: null,
      },
      needsReference: false,
      orphan: null,
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
      takes: 2,
      serving: {
        maxNumSeqs: 16,
        maxNumSeqsNote: "vllm-omni's own stage-0 value, measured at 0.35 + 0.10.",
        memFraction: null,
        memFractionNote: null,
        contextLength: null,
        contextLengthNote: null,
      },
      // A zero-shot row says a load must carry a clip, whatever this host can
      // serve: it is a fact about the KIND, not about the backend block.
      needsReference: true,
      orphan: null,
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

// ANY CRUCIBLE THAT ANSWERS WORKS (Owen, 2026-09-24). A voice row's id,
// residency, loadability, sample rate, clip requirement and pace block are
// what a caller loads and packs by; the rest describes the voice and reads as
// null when a server does not state it.

test('a 1.0.23-shaped voice row — no orphan — reads cleanly, with null there', async () => {
  // The exact failure the ruling was made over: `voices[0] has no field
  // "orphan"` in every BookForge fake after the 1.0.24 repin.
  const { orphan: _orphan, ...older } = VOICE_ROW;
  answers(200, [older]);
  const [voice] = await client().voices();
  assert.equal(voice!.orphan, null);
  assert.equal(voice!.id, 'deathstalker');
  assert.equal(voice!.sampleRate, 24000);
});

test('a voice row without reason reads null, and a refusal with no reason still reads', async () => {
  const { reason: _reason, ...withoutReason } = VOICE_ROW;
  answers(200, [withoutReason]);
  assert.equal((await client().voices())[0]!.reason, null);
  // `loadable: false` is the fact a caller acts on; the reason is for a person.
  answers(200, [{ ...VOICE_ROW, loadable: false, reason: null }]);
  const [voice] = await client().voices();
  assert.equal(voice!.loadable, false);
  assert.equal(voice!.reason, null);
});

test('a voice kind outside today\'s vocabulary is carried as the server\'s word', async () => {
  // Whether a load needs a clip is `needs_reference`'s to say, so a fourth
  // kind from a newer server is news for a display, never a lost row.
  answers(200, [{ ...VOICE_ROW, kind: 'improvised' }]);
  assert.equal((await client().voices())[0]!.kind, 'improvised');
});

test('an estimate_basis outside today\'s vocabulary is carried as the server\'s word', async () => {
  answers(200, [{ ...VOICE_ROW, estimate_basis: 'guessed' }]);
  assert.equal((await client().voices())[0]!.estimateBasis, 'guessed');
});

test('a voice kind that is present but not a string is a protocol error', async () => {
  answers(200, [{ ...VOICE_ROW, kind: 3 }]);
  await assert.rejects(client().voices(), /voices\[0\]\.kind is present but is not a string/);
});

test('a pace block missing a rate is a half-stated band, refused by name', async () => {
  // An absent rate reads as "this voice states none" — and two of three is
  // still the wire disagreeing with itself, which is refused whatever the
  // vintage.
  const { max_chars_per_sec: _rate, ...lamePace } = VOICE_ROW.pace;
  answers(200, [{ ...VOICE_ROW, pace: lamePace }]);
  await assert.rejects(client().voices(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /voices\[0\]\.pace states 2 of its 3 rates/);
    return true;
  });
});

test('a voice row without its pace block is a protocol error: it is what a client packs to', async () => {
  const { pace: _pace, ...withoutPace } = VOICE_ROW;
  answers(200, [withoutPace]);
  await assert.rejects(client().voices(), /voices\[0\] has no field "pace"/);
});

// ── a voice nobody measured ──────────────────────────────────────────────────
//
// The three rates became optional AS A GROUP on 2026-09-18. `higgs-default` and
// `zeroshot` are the base weights and no length ladder has ever been run on
// either, so both manifests had been satisfying a required triple by copying
// narrator's own Higgs v3 constants back to it — a pace of 15.0 that is the
// DIVISOR `cap_frames()` sizes against and was never measured as a speaking
// rate, inside a band written around a real book pace nearer 17.2. narrator
// re-centres a band's RATIOS on the running median, so that combination judged
// healthy chunks run-ons and sent them to MAX_DEPTH.
//
// So a manifest states what was measured or states nothing, `to_dict()` sends
// all three as `null`, and the derivation of a centre keeps its one owner —
// narrator's. This client derives none either.

test('a voice with no measured pace parses, and states three nulls rather than a guess', async () => {
  answers(200, [{
    ...VOICE_ROW,
    id: 'higgs-default',
    pace: {
      pace_chars_per_sec: null,
      max_chars_per_sec: null,
      min_chars_per_sec: null,
      target_chars: null,
      safe_min_chars: null,
      safe_max_chars: null,
    },
  }]);
  const [voice] = await client().voices();
  assert.equal(voice!.pace.paceCharsPerSec, null);
  assert.equal(voice!.pace.maxCharsPerSec, null);
  assert.equal(voice!.pace.minCharsPerSec, null);
});

test('a HALF-stated band is refused by name: all three or none, on both sides', async () => {
  // The manifest loader refuses a partial triple (`_PACE_RATES`), so a document
  // carrying one is a server disagreeing with itself — and a client that read
  // two of three would pack to an edge with no centre, which is the shape that
  // caused the run-on cascade in the first place.
  answers(200, [{
    ...VOICE_ROW,
    pace: { ...VOICE_ROW.pace, pace_chars_per_sec: null },
  }]);
  await assert.rejects(client().voices(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /voices\[0\]\.pace states 2 of its 3 rates/);
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

test('loadVoice carries a zero-shot reference in params, and omits an absent name', async () => {
  // PHASE3-TTS.md § 5. The clip is the CLIENT's — a per-client choice like the
  // voice pick — so it travels with the load. `name` is omitted rather than
  // sent as null when there is none: the load door forbids unknown keys and a
  // null label is not a label.
  answers(200, { job_id: 'job-load-zeroshot-1' });
  await client().loadVoice('zeroshot', {
    reference: { data: 'UklGRiQAAABXQVZF', transcript: 'He had been walking.' },
  });
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'load-voice',
    model: 'zeroshot',
    params: {
      reference: { data: 'UklGRiQAAABXQVZF', transcript: 'He had been walking.' },
    },
    inputs: {},
  });

  answers(200, { job_id: 'job-load-zeroshot-2' });
  await client().loadVoice('zeroshot', {
    reference: {
      data: 'UklGRiQAAABXQVZF',
      transcript: 'He had been walking.',
      name: 'the stranger',
    },
  });
  assert.deepEqual(JSON.parse(lastBody).params, {
    reference: {
      data: 'UklGRiQAAABXQVZF',
      transcript: 'He had been walking.',
      name: 'the stranger',
    },
  });
});

test('loadVoice refuses a reference with no data or no transcript by name', async () => {
  // Refused HERE, before a round trip, for the reason narrator states about
  // the transcript: a clone conditioned on an absent one is a whole book in a
  // subtly wrong voice, reported as success.
  answers(200, { job_id: 'never-submitted' });
  for (const [reference, option] of [
    [{ data: '', transcript: 'Rain.' }, 'reference.data'],
    [{ data: 'UklGRiQAAABXQVZF', transcript: '   ' }, 'reference.transcript'],
  ] as const) {
    await assert.rejects(
      client().loadVoice('zeroshot', { reference }),
      (error: unknown) => {
        assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
        assert.equal(error.option, option);
        return true;
      },
    );
  }
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
  assert.deepEqual(tts.unreadableRows, []);
});

test('info() with one malformed voice row still returns, and says which row it could not read', async () => {
  // Until 2026-09-24 one voice row this build could not read took the whole
  // probe down — `info()` is where the voice rows ride. Now the row is
  // carried aside with its raw data and the reason, and everything else in
  // the document reads (Owen: "if it can make the call to the crucible server
  // then it should work").
  const { sample_rate: _rate, ...noRate } = { ...VOICE_ROW, id: 'broken' };
  const wrongType = { ...VOICE_ROW, id: 'banana', max_chars: 'lots' };
  answers(200, {
    server: { name: 'crucible@test', version: '1.0.24', api_version: 1 },
    host: {
      platform: 'linux',
      arch: 'x86_64',
      backend: 'cuda-linux',
      gpu: { vendor: 'nvidia', name: 'NVIDIA GeForce RTX 3090 Ti', vram_bytes: 25757220864 },
    },
    job_types: ['echo', 'load-voice', 'unload-voice'],
    capabilities: [
      { job_type: 'echo', models: [] },
      { job_type: 'tts', models: [VOICE_ROW, noRate, wrongType] },
    ],
  });

  const info = await client().info();
  const tts = info.capabilities.find(isTtsCapability);
  assert.ok(tts !== undefined);
  assert.deepEqual(tts.models.map((voice) => voice.id), ['deathstalker']);
  assert.deepEqual(
    tts.unreadableRows.map((row) => [row.index, row.id]),
    [[1, 'broken'], [2, 'banana']],
  );
  assert.deepEqual(tts.unreadableRows[0]!.raw, noRate);
  assert.match(
    tts.unreadableRows[0]!.unreadable,
    /info\.capabilities\[1\]\.models\[1\] has no field "sample_rate"/,
  );
  assert.match(tts.unreadableRows[1]!.unreadable, /max_chars is present but is not a number/);
  // The rest of the document is intact.
  assert.equal(info.server.name, 'crucible@test');
  assert.deepEqual(info.jobTypes, ['echo', 'load-voice', 'unload-voice']);

  // The direct read stays strict about the same row, by name.
  answers(200, [VOICE_ROW, noRate]);
  await assert.rejects(client().voices(), /voices\[1\] has no field "sample_rate"/);
});

// -------------------------------------------------------------- health row

test('health() reads resident_kind beside the ids', async () => {
  answers(200, {
    status: 'ok',
    queue_depth: 0,
    resident_models: ['deathstalker'],
    resident_kind: 'tts',
    stopping: null,
  });
  const health = await client().health();
  assert.deepEqual(health, {
    status: 'ok',
    queueDepth: 0,
    residentModels: ['deathstalker'],
    stopping: null,
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
    stopping: null,
  });
  const health = await client().health();
  assert.equal(health.residentKind, 'align');
});

test('a health body without resident_kind reads it as null', async () => {
  answers(200, { status: 'ok', queue_depth: 0, resident_models: [] });
  const health = await client().health();
  assert.equal(health.residentKind, null);
  assert.equal(health.status, 'ok');
});

test('a health status outside today\'s three words is carried, not refused', async () => {
  answers(200, { status: 'draining', queue_depth: 0, resident_models: [], resident_kind: null, stopping: null });
  assert.equal((await client().health()).status, 'draining');
});

test('a health body without resident_models is a protocol error: loads are decided on it', async () => {
  answers(200, { status: 'ok', queue_depth: 0, resident_kind: null, stopping: null });
  await assert.rejects(client().health(), /health has no field "resident_models"/);
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
    memory_bytes_estimate: 20950548480, held_by: null, unclaimed_since: null,
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
  assert.ok(state.holders !== null);
  assert.equal(state.holders[0]!.bytes, null);
  assert.equal(state.holders[1]!.bytes, 19000000000);
  assert.equal(state.holders[1]!.ownedByCrucible, true);
});

test('a holder with no bytes field at all reads as null — unknown, never 0', async () => {
  // The driver declining and a server that does not say are the same news to
  // a caller: not known. Neither is ever zero.
  answers(200, {
    ...ACCELERATOR,
    holders: [{ pid: 6120, name: 'python.exe', owned_by_crucible: false }],
  });
  const state = await client().accelerator();
  assert.equal(state.holders?.[0]?.bytes, null);
  assert.notEqual(state.holders?.[0]?.bytes, 0);
});

test('a holder with no pid is a protocol error: it is what an operator acts on', async () => {
  answers(200, {
    ...ACCELERATOR,
    holders: [{ name: 'python.exe', bytes: null, owned_by_crucible: false }],
  });
  await assert.rejects(client().accelerator(), /accelerator\.holders\[0\] has no field "pid"/);
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
    model: 'whisper-tiny',
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
    model: 'whisper-tiny',
    params: { language: 'en', vad_filter: true, word_timestamps: true },
    // Keyed by the caller's filename: the input's name becomes the file's name
    // on disk and ffmpeg reads the container off the extension.
    inputs: { 'audiobook.m4b': { blob_id: 'b7c6d5e4f3a2b1c0' } },
  });
});

test('asr() carries inline bytes as base64 under the caller\'s filename', async () => {
  answers(200, { job_id: 'job-asr-2' });
  await client().asr({
    model: 'whisper-tiny',
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

test('asr() sends initial_prompt only when the caller states it', async () => {
  const base = {
    model: 'whisper-large-v3-turbo',
    audio: { blobId: 'b7c6d5e4f3a2b1c0' },
    filename: 'episode.m4a',
    language: 'en',
    vadFilter: true,
    wordTimestamps: true,
  };
  // Left out: the three keys every existing caller sends, and nothing else.
  answers(200, { job_id: 'job-asr-3' });
  await client().asr(base);
  assert.deepEqual((JSON.parse(lastBody) as { params: unknown }).params, {
    language: 'en',
    vad_filter: true,
    word_timestamps: true,
  });
  // Stated as a string: sent verbatim under the server's name.
  answers(200, { job_id: 'job-asr-4' });
  await client().asr({ ...base, initialPrompt: 'Episode 12: Kaladin and Syl.' });
  assert.deepEqual((JSON.parse(lastBody) as { params: unknown }).params, {
    language: 'en',
    vad_filter: true,
    word_timestamps: true,
    initial_prompt: 'Episode 12: Kaladin and Syl.',
  });
  // Stated as null: sent as null, which is "no prompt" said out loud.
  answers(200, { job_id: 'job-asr-5' });
  await client().asr({ ...base, initialPrompt: null });
  assert.equal(
    (JSON.parse(lastBody) as { params: Record<string, unknown> }).params['initial_prompt'],
    null,
  );
});

test('asr() refuses an initialPrompt that is not a non-blank string, by name', async () => {
  for (const bad of [5, ['Kaladin'], '', '   ']) {
    await assert.rejects(
      client().asr({
        model: 'whisper-tiny',
        audio: { blobId: 'b7c6d5e4f3a2b1c0' },
        filename: 'audiobook.m4b',
        language: 'en',
        vadFilter: true,
        wordTimestamps: true,
        initialPrompt: bad as never,
      }),
      (error: unknown) => {
        assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
        assert.equal(error.option, 'initialPrompt');
        return true;
      },
      `expected ${JSON.stringify(bad)} to be refused`,
    );
  }
});

test('asr() sends a stated context under the server name, and only then', async () => {
  const base = {
    model: 'qwen3-asr-1.7b',
    audio: { blobId: 'b7c6d5e4f3a2b1c0' },
    filename: 'stream.m4a',
    language: 'en',
    vadFilter: false,
    wordTimestamps: true,
  };
  answers(200, { job_id: 'job-asr-q1' });
  await client().asr(base);
  assert.equal(
    'context' in (JSON.parse(lastBody) as { params: Record<string, unknown> }).params,
    false,
  );
  answers(200, { job_id: 'job-asr-q2' });
  await client().asr({ ...base, context: 'Verbatim transcript. um, uh.' });
  assert.deepEqual((JSON.parse(lastBody) as { params: unknown }).params, {
    language: 'en',
    vad_filter: false,
    word_timestamps: true,
    context: 'Verbatim transcript. um, uh.',
  });
  for (const bad of [5, '', '   ']) {
    await assert.rejects(
      client().asr({ ...base, context: bad as never }),
      (error: unknown) => {
        assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
        assert.equal(error.option, 'context');
        return true;
      },
    );
  }
});

test('every asr option is required and is refused by name', async () => {
  const complete = {
    model: 'whisper-tiny',
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
      model: 'whisper-tiny',
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
      model: 'whisper-tiny',
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

test('a progress frame without a fraction reads it as null, and keeps its extras', async () => {
  // The fraction is drawn, never acted on (Owen, 2026-09-24).
  handle = (_request, response) => {
    openSse(response);
    response.write('id: 1\nevent: progress\ndata: {"message": "half", "cues": 7}\n\n');
    response.write('id: 2\nevent: done\ndata: {"artifacts": []}\n\n');
    response.end();
  };
  const seen = [];
  for await (const event of client().events('job-sparse-progress')) seen.push(event);
  const [progress] = seen;
  assert.ok(progress !== undefined && progress.event === 'progress');
  assert.equal(progress.data.fraction, null);
  assert.deepEqual(progress.data.extra, { cues: 7 });
});

test('a progress fraction that is present but not a number is a protocol error', async () => {
  handle = (_request, response) => {
    openSse(response);
    response.write('id: 1\nevent: progress\ndata: {"fraction": "half", "message": "x"}\n\n');
    response.end();
  };
  await assert.rejects(async () => {
    for await (const _event of client().events('job-bad-progress')) {
      // drain
    }
  }, /fraction is present but is not a number/);
});
