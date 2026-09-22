/**
 * Three seams PHASE15 moved, each found by a real client reading a real
 * server. Not hypotheticals — each of these is a thing that broke.
 *
 * 1. **`capability()` against a PRE-PHASE-15 server.** Owen's live WSL engine
 *    answers eleven rows and not one of them carries `route`, and this client
 *    threw `CrucibleProtocolError: capability.classes[0] has no field "route"`
 *    at it. Section 3.3's last bullet is the rule and it is a READING, not a
 *    default: no row has it → every class IS local; some have it and one does
 *    not → `capability_route_missing`; a word outside the vocabulary →
 *    `capability_route_unknown`.
 * 2. **A clock on the three probes.** Foundry draws a tooltip from
 *    `capability()` with a 3 s deadline, because a sleeping Mac answers
 *    nothing at all and a fetch with no deadline hangs for minutes behind it.
 * 3. **`cruciblePairingPath()` on win32.** It answered `~/.crucible/pairing`,
 *    which on Windows is a directory nothing writes to; 3.6's table pins
 *    `%LOCALAPPDATA%\Crucible\pairing`, with `CRUCIBLE_HOME` overriding
 *    everywhere. Same rule as `crucible/config.py`'s `crucible_home()`.
 * 8. **A voice row with no `needs_reference`** (section 8 below). Foundry read
 *    a server one commit older than the field and BOTH `info()` and `voices()`
 *    threw on its first call. Same rule as 1, same shape, applied to the voice
 *    row wherever it arrives: no row states it → a pre-field server, every
 *    voice a checkpoint; some do and one does not →
 *    `voices_needs_reference_missing`; not a boolean →
 *    `voices_needs_reference_unknown`.
 *
 * Sections 4-7 are the rest of PHASE15's seams, added as they were built.
 *
 * Run: `npm run test:unit`.
 */

import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { createServer, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, before, test } from 'node:test';

import {
  CAPABILITY_ROUTE_MISSING,
  CAPABILITY_ROUTE_UNKNOWN,
  CrucibleClient,
  CruciblePairingFileError,
  SUBJECT_IN_USE,
  SUBJECT_NOT_INSTALLED,
  SUBJECT_REMOVE_FAILED,
  SUBJECT_UNKNOWN,
  CrucibleProtocolError,
  PAIRING_FILE_MALFORMED,
  VOICES_NEEDS_REFERENCE_MISSING,
  VOICES_NEEDS_REFERENCE_UNKNOWN,
  WINDOWS_HOME_DIRNAME,
  cruciblePairingPath,
  isTtsCapability,
  readPairingFile,
} from '../src/index.js';

// ------------------------------------------------------------------ fixture

let reply: { status: number; body: unknown } = { status: 500, body: {} };
/** Milliseconds the fixture waits before answering. The sleeping Mac. */
let stall = 0;
let lastMethod = '';
let lastPath = '';

let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-phase15' });
}

function answers(status: number, body: unknown): void {
  reply = { status, body };
  stall = 0;
}

function json(response: ServerResponse, status: number, body: unknown): void {
  if (status === 204) {
    // A 204 carries no body and no content type, which is what the route
    // really answers and what `removeSubject` has to cope with.
    response.writeHead(204);
    response.end();
    return;
  }
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

before(async () => {
  server = createServer((request, response) => {
    lastMethod = request.method ?? '';
    lastPath = request.url ?? '';
    if (stall > 0) {
      const timer = setTimeout(() => json(response, reply.status, reply.body), stall);
      request.on('aborted', () => clearTimeout(timer));
      return;
    }
    json(response, reply.status, reply.body);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

const GIB = 1024 ** 3;

/**
 * Eleven classes with NO `route` on any of them — the shape a server
 * initialised before phase 15 sent.
 *
 * It was read as "every class is local" until 2026-09-16. Owen ruled that
 * population out of existence (nothing is released, so no such server exists
 * for anyone but us), and inventing a route is inventing the one fact that
 * decides whether a run costs GPU-minutes or money — so it is a REFUSAL now,
 * whole document or half. The fixture stays, because what it produces is worth
 * asserting; only the answer changed.
 */
function preRouteRecord(): Record<string, unknown> {
  const classes = [
    'echo',
    'clean',
    'translate',
    'simplify',
    'analysis',
    'pages',
    'tts',
    'asr',
    'align',
    'rvc',
    'denoise',
  ].map((capability) => ({
    capability,
    enabled: capability !== 'denoise',
    selected: capability === 'clean' ? 'qwen3.5-9b' : '',
    reason: `${capability}: whatever this server said before phase 15`,
    shortfall_bytes: 0,
  }));
  return {
    backend_kind: 'cuda-linux',
    total_bytes: 24 * GIB,
    desktop_allowance_bytes: 3 * GIB,
    classes,
  };
}

// ------------------------------- 1. the capability document's route, by 3.3

test('a document where NO row says route is REFUSED, not read as every class local', async () => {
  answers(200, preRouteRecord());
  await assert.rejects(client().capability(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError);
    assert.match(error.message, new RegExp(CAPABILITY_ROUTE_MISSING));
    assert.match(error.message, /neither does any other row/);
    return true;
  });
});

/** The same eleven classes, each stating where it runs. What a server sends. */
function routedRecord(): Record<string, unknown> {
  const document = preRouteRecord();
  for (const row of document['classes'] as Record<string, unknown>[]) row['route'] = 'local';
  return document;
}

test('reading a pre-phase-15 document is a STATEMENT, not a filled-in default', async () => {
  // The difference shows in the other direction: the same document with ONE
  // row routed is not "ten defaults and one fact", it is a defect.
  const document = preRouteRecord();
  const classes = document['classes'] as Record<string, unknown>[];
  classes[3]!['route'] = 'upstream';
  answers(200, document);
  await assert.rejects(client().capability(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError);
    assert.match(error.message, new RegExp(CAPABILITY_ROUTE_MISSING));
    // It names the ROW, so a person reading the log knows which one.
    assert.match(error.message, /classes\[0\]/);
    return true;
  });
});

test('a route outside the vocabulary is capability_route_unknown', async () => {
  const document = preRouteRecord();
  const classes = document['classes'] as Record<string, unknown>[];
  for (const row of classes) row['route'] = 'local';
  classes[5]!['route'] = 'cloud';
  answers(200, document);
  await assert.rejects(client().capability(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError);
    assert.match(error.message, new RegExp(CAPABILITY_ROUTE_UNKNOWN));
    assert.match(error.message, /classes\[5\]\.route is "cloud"/);
    return true;
  });
});

test('a route that is not a string at all is refused by the same name', async () => {
  const document = preRouteRecord();
  const classes = document['classes'] as Record<string, unknown>[];
  for (const row of classes) row['route'] = 'local';
  classes[1]!['route'] = 7;
  answers(200, document);
  await assert.rejects(client().capability(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError);
    assert.match(error.message, new RegExp(CAPABILITY_ROUTE_UNKNOWN));
    return true;
  });
});

test('a phase-15 document is read exactly as before', async () => {
  const document = preRouteRecord();
  const classes = document['classes'] as Record<string, unknown>[];
  for (const row of classes) row['route'] = 'local';
  classes[2]!['route'] = 'upstream';
  classes[2]!['selected'] = 'anthropic/claude-sonnet-5';
  answers(200, document);
  const record = await client().capability();
  const translate = record.classes[2]!;
  assert.equal(translate.route, 'upstream');
  assert.equal(translate.selected, 'anthropic/claude-sonnet-5');
  assert.equal(record.classes[0]!.route, 'local');
});

// ------------------------------------------- 2. a clock on the three probes

test('capability() takes a timeoutMs and abandons a server that is asleep', async () => {
  answers(200, routedRecord());
  stall = 5000;
  const started = Date.now();
  await assert.rejects(client().capability({ timeoutMs: 120 }), (error: unknown) => {
    // The PLATFORM's own error, not one this client invented: a caller that
    // wants to tell a timeout from a cancel reads `name`.
    assert.equal((error as Error).name, 'TimeoutError');
    return true;
  });
  assert.ok(Date.now() - started < 4000, 'it must not wait for the server');
  stall = 0;
});

test('ping() and info() take the same clock', async () => {
  answers(200, { crucible: true, name: 'x', api_version: 1 });
  stall = 5000;
  await assert.rejects(client().ping({ timeoutMs: 120 }), (error: unknown) => {
    assert.equal((error as Error).name, 'TimeoutError');
    return true;
  });
  await assert.rejects(client().info({ timeoutMs: 120 }), (error: unknown) => {
    assert.equal((error as Error).name, 'TimeoutError');
    return true;
  });
  stall = 0;
});

test("a caller's own signal still aborts, and says so with its own name", async () => {
  answers(200, routedRecord());
  stall = 5000;
  const control = new AbortController();
  const pending = client().capability({ signal: control.signal, timeoutMs: 5000 });
  control.abort();
  await assert.rejects(pending, (error: unknown) => {
    assert.equal((error as Error).name, 'AbortError');
    return true;
  });
  stall = 0;
});

test('no clock means no clock — a probe with neither option still answers', async () => {
  answers(200, routedRecord());
  const record = await client().capability();
  assert.equal(record.classes.length, 11);
});

test('a timeoutMs that is not a positive number is refused, not rounded', async () => {
  answers(200, routedRecord());
  for (const bad of [0, -1, Number.NaN]) {
    await assert.rejects(client().capability({ timeoutMs: bad }), /timeoutMs/);
  }
});

// -------------------------------------------- 3. where the pairing file is

/**
 * `process.platform` is read-only-ish but configurable, which is how every
 * cross-platform path test in node does it. Restored in a finally.
 *
 * AWAITED INSIDE THE TRY, and that is the whole point: `cruciblePairingPath`
 * is async (it dynamic-imports node's builtins), so a synchronous helper
 * would restore the platform before the function under test ever read it and
 * every one of these would quietly test this machine instead.
 */
async function asPlatform<T>(platform: string, body: () => Promise<T>): Promise<T> {
  const original = Object.getOwnPropertyDescriptor(process, 'platform')!;
  Object.defineProperty(process, 'platform', { value: platform, configurable: true });
  try {
    return await body();
  } finally {
    Object.defineProperty(process, 'platform', original);
  }
}

async function withEnv<T>(
  values: Record<string, string | undefined>,
  body: () => Promise<T>,
): Promise<T> {
  const before: Record<string, string | undefined> = {};
  for (const [name, value] of Object.entries(values)) {
    before[name] = process.env[name];
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
  try {
    return await body();
  } finally {
    for (const [name, value] of Object.entries(before)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  }
}

test('on win32 the pairing file is %LOCALAPPDATA%\\Crucible\\pairing', async () => {
  const path = await asPlatform('win32', () =>
    withEnv({ CRUCIBLE_HOME: undefined, LOCALAPPDATA: 'C:\\Users\\tellt\\AppData\\Local' }, () =>
      cruciblePairingPath(),
    ),
  );
  assert.ok(path.endsWith(join(WINDOWS_HOME_DIRNAME, 'pairing')), path);
  assert.ok(path.startsWith('C:\\Users\\tellt\\AppData\\Local'), path);
  // NOT the POSIX home, which on Windows is a directory nothing writes to.
  assert.ok(!path.includes('.crucible'), path);
});

test('an unset LOCALAPPDATA on win32 is REFUSED, never assembled from a username', async () => {
  await assert.rejects(
    asPlatform('win32', () =>
      withEnv({ CRUCIBLE_HOME: undefined, LOCALAPPDATA: undefined }, () =>
        cruciblePairingPath(),
      ),
    ),
    /LOCALAPPDATA/,
  );
});

test('CRUCIBLE_HOME overrides on EVERY platform — the same rule as crucible_home()', async () => {
  for (const platform of ['win32', 'linux', 'darwin']) {
    const path = await asPlatform(platform, () =>
      withEnv({ CRUCIBLE_HOME: join(tmpdir(), 'elsewhere'), LOCALAPPDATA: undefined }, () =>
        cruciblePairingPath(),
      ),
    );
    assert.equal(path, join(tmpdir(), 'elsewhere', 'pairing'), platform);
  }
});

test('on linux and darwin it is ~/.crucible/pairing', async () => {
  for (const platform of ['linux', 'darwin']) {
    const path = await asPlatform(platform, () =>
      withEnv({ CRUCIBLE_HOME: undefined }, () => cruciblePairingPath()),
    );
    assert.ok(path.endsWith(join('.crucible', 'pairing')), `${platform}: ${path}`);
  }
});

test('an explicit home still wins over both, on every platform', async () => {
  const path = await asPlatform('win32', () =>
    withEnv({ CRUCIBLE_HOME: 'C:\\ignored', LOCALAPPDATA: 'C:\\also-ignored' }, () =>
      cruciblePairingPath(join(tmpdir(), 'stated')),
    ),
  );
  assert.equal(path, join(tmpdir(), 'stated', 'pairing'));
});

test('a second non-empty line is pairing_file_malformed, not a guessed token', async () => {
  const home = mkdtempSync(join(tmpdir(), 'crucible-pairing-'));
  writeFileSync(
    join(home, 'pairing'),
    'crucible://crucible%40pc@127.0.0.1:7100/#one\ncrucible://crucible%40pc@127.0.0.1:7100/#two\n',
    'utf8',
  );
  await assert.rejects(readPairingFile(home), (error: unknown) => {
    assert.ok(error instanceof CruciblePairingFileError);
    assert.equal(error.code, PAIRING_FILE_MALFORMED);
    assert.match(error.message, /holds 2 lines/);
    return true;
  });
});

test('the writer writes one line with a trailing newline, and that still reads', async () => {
  const home = mkdtempSync(join(tmpdir(), 'crucible-pairing-'));
  writeFileSync(join(home, 'pairing'), 'crucible://crucible%40pc@127.0.0.1:7100/#tok3n\n', 'utf8');
  const pairing = await readPairingFile(home);
  assert.equal(pairing?.token, 'tok3n');
  assert.equal(pairing?.name, 'crucible@pc');
});

test('CRLF is one line, because a Windows editor is not a second server', async () => {
  const home = mkdtempSync(join(tmpdir(), 'crucible-pairing-'));
  writeFileSync(join(home, 'pairing'), 'crucible://crucible%40pc@127.0.0.1:7100/#tok3n\r\n', 'utf8');
  const pairing = await readPairingFile(home);
  assert.equal(pairing?.token, 'tok3n');
});

// ---------------------------- 4. a clock at CONSTRUCTION, on every call

test('timeoutMs on the constructor puts a deadline on every call', async () => {
  answers(200, routedRecord());
  stall = 5000;
  const impatient = new CrucibleClient({
    url: base,
    token: 'the-token',
    clientName: 'unit-phase15',
    timeoutMs: 120,
  });
  const started = Date.now();
  await assert.rejects(impatient.capability(), (error: unknown) => {
    assert.equal((error as Error).name, 'TimeoutError');
    return true;
  });
  // And it is not only the probes: the same clock is on an authed door that
  // takes no options at all.
  await assert.rejects(impatient.health(), (error: unknown) => {
    assert.equal((error as Error).name, 'TimeoutError');
    return true;
  });
  assert.ok(Date.now() - started < 4000);
  stall = 0;
});

test("a per-call signal REPLACES the constructor's clock rather than racing it", async () => {
  // Two different statements: the constructor's is this app's patience, the
  // call's is "this caller owns this request". A second deadline quietly
  // ANDed onto a caller's cancel would end a stream they were still reading.
  answers(200, routedRecord());
  stall = 400;
  const impatient = new CrucibleClient({
    url: base,
    token: 'the-token',
    clientName: 'unit-phase15',
    timeoutMs: 50,
  });
  const patient = new AbortController();
  const record = await impatient.capability({ signal: patient.signal });
  assert.equal(record.classes.length, 11);
  stall = 0;
});

test('no timeoutMs means no clock, and a bad one is refused at construction', async () => {
  answers(200, routedRecord());
  stall = 200;
  const unhurried = new CrucibleClient({
    url: base,
    token: 'the-token',
    clientName: 'unit-phase15',
  });
  assert.equal((await unhurried.capability()).classes.length, 11);
  stall = 0;
  for (const bad of [0, -1, Number.NaN]) {
    assert.throws(
      () =>
        new CrucibleClient({
          url: base,
          token: 't',
          clientName: 'x',
          timeoutMs: bad,
        }),
      /timeoutMs/,
    );
  }
});

// ------------------- 5. testUpstream says WHO refused, in the server's words

test("testUpstream relays the SERVER's sentence, not this client's wrapper", async () => {
  // The defect: `error.message` is "crucible refused the request (…)", which
  // a settings window renders beside the key field — naming Crucible for a
  // key ANTHROPIC rejected.
  answers(502, {
    error: {
      code: 'upstream_rejected',
      message:
        'anthropic rejected the credential this server sent it, with HTTP 401. ' +
        "anthropic said: invalid x-api-key. This is the upstream's answer " +
        "about the key, not Crucible's about your token",
      details: { upstream: 'anthropic', upstream_status: 401 },
    },
  });
  const result = await client().testUpstream('anthropic', { key: 'sk-ant-wrong' });
  assert.equal(result.ok, false);
  if (result.ok) throw new Error('unreachable');
  assert.equal(result.code, 'upstream_rejected');
  assert.ok(result.message.startsWith('anthropic rejected the credential'));
  assert.ok(!result.message.includes('crucible refused'));
  assert.ok(!result.message.includes('crucible failed'));
});

test('an unreachable upstream names the address that did not answer', async () => {
  answers(502, {
    error: {
      code: 'upstream_unreachable',
      message:
        'nothing answered at http://192.168.68.20:11434/api/tags, which is ' +
        'where this server reaches ollama: ConnectError: [Errno 111]',
      details: { upstream: 'ollama', url: 'http://192.168.68.20:11434/api/tags' },
    },
  });
  const result = await client().testUpstream('ollama', {
    url: 'http://192.168.68.20:11434',
  });
  assert.equal(result.ok, false);
  if (result.ok) throw new Error('unreachable');
  assert.match(result.message, /192\.168\.68\.20:11434/);
});

test('a real auth failure of THIS server still throws, and is not a test refusal', async () => {
  answers(401, {
    error: { code: 'unauthorized', message: "that is not this server's token" },
  });
  await assert.rejects(client().testUpstream('anthropic', { key: 'k' }), (error: unknown) => {
    assert.equal((error as { code?: string }).code, 'unauthorized');
    return true;
  });
});

// --------------------------- 6. removeSubject — PHASE15-HOST.md 3.5a

test('removeSubject DELETEs the subject and resolves on 204', async () => {
  answers(204, null);
  await client().removeSubject('model', 'qwen3.5-9b');
  assert.equal(lastMethod, 'DELETE');
  assert.equal(lastPath, '/v1/catalog/model/qwen3.5-9b');
});

test('a kind or id with a slash in it cannot walk out of the route', async () => {
  answers(204, null);
  await client().removeSubject('model' as never, 'a/b');
  assert.equal(lastPath, '/v1/catalog/model/a%2Fb');
});

test('each of the four refusals arrives by name', async () => {
  const cases: Array<[number, string]> = [
    [404, SUBJECT_UNKNOWN],
    [409, SUBJECT_NOT_INSTALLED],
    [409, SUBJECT_IN_USE],
    [500, SUBJECT_REMOVE_FAILED],
  ];
  for (const [status, code] of cases) {
    answers(status, {
      error: { code, message: 'no', details: { who: 'the card', path: '/x' } },
    });
    await assert.rejects(
      client().removeSubject('model', 'qwen3.5-9b'),
      (error: unknown) => {
        assert.equal((error as { code?: string }).code, code);
        return true;
      },
    );
  }
});

test('the engine subject is removable like any other', async () => {
  answers(204, null);
  await client().removeSubject('engine', 'llama-cpp');
  assert.equal(lastPath, '/v1/catalog/engine/llama-cpp');
});

// ------------------------------ 7. the engine task — PHASE15-HOST.md 4.7

test('submitTask sends an engine move as {type, target}', async () => {
  answers(202, { task_id: 'abc123' });
  const id = await client().submitTask({ type: 'engine', target: 'wsl' });
  assert.equal(id, 'abc123');
  assert.equal(lastMethod, 'POST');
  assert.equal(lastPath, '/v1/tasks');
});

test('the reverse move is refused by this package, before it leaves', async () => {
  // Moving BACK to Windows is an explicit operator act (section 6). A
  // client that could ask for it here would get a server's refusal for a
  // request this package already knew was wrong.
  await assert.rejects(
    client().submitTask({ type: 'engine', target: 'windows' as never }),
    /target/,
  );
});

test('a task type this build does not have names the four that exist', async () => {
  await assert.rejects(
    client().submitTask({ type: 'reboot' } as never),
    /'pull', 'install', 'module' or 'engine'/,
  );
});

// ---------- 8. the voice document's needs_reference — the SAME rule, by 3.3

/**
 * The defect Foundry found against a live server on 2026-09-14: a Crucible
 * built one commit before 743dc1a answers voice rows with no `needs_reference`
 * on them, and this client threw
 * `info.capabilities[6].models[0] has no field "needs_reference"` at its very
 * first read — and the same from `voices()`. Section 3.3's reading rule is the
 * fix, applied exactly as it is to `route`: the vintage is asked ONCE of the
 * whole document, and a document nobody's row states it in is a PRE-FIELD
 * server's, where every voice is a checkpoint.
 */
const PRE_FIELD_VOICE = {
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
  revision: 'a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4',
  fingerprint: 'deathstalker@a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4',
  memory_bytes_estimate: 19 * GIB,
  estimate_basis: 'declared',
  max_chars: 800,
  sample_rate: 24000,
  takes: 2,
  // NO `needs_reference`. That is the whole fixture.
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

function preFieldVoices(): Record<string, unknown>[] {
  return [
    { ...PRE_FIELD_VOICE },
    { ...PRE_FIELD_VOICE, id: 'mistborn', display: 'Mistborn' },
    { ...PRE_FIELD_VOICE, id: 'leah', display: 'Leah' },
  ];
}

/** `GET /v1/info` from the same server: the voice rows sit under `tts`. */
function preFieldInfo(rows: Record<string, unknown>[]): Record<string, unknown> {
  return {
    server: { name: 'crucible@wsl', version: '0.5.0', api_version: 1 },
    host: {
      platform: 'linux',
      arch: 'x86_64',
      backend: 'cuda-linux',
      gpu: { vendor: 'nvidia', name: 'NVIDIA GeForce RTX 3090 Ti', vram_bytes: 24 * GIB },
    },
    job_types: ['tts', 'load-voice', 'unload-voice'],
    capabilities: [
      { job_type: 'echo', models: [] },
      { job_type: 'tts', models: rows },
    ],
  };
}

test('a voices() document where NO row says needs_reference reads every voice as false', async () => {
  answers(200, preFieldVoices());
  const voices = await client().voices();
  assert.equal(voices.length, 3);
  for (const voice of voices) {
    assert.equal(voice.needsReference, false, `${voice.id} should read as false`);
  }
});

test("info()'s tts rows read the same way — the seam Foundry hit first", async () => {
  answers(200, preFieldInfo(preFieldVoices()));
  const info = await client().info();
  const tts = info.capabilities.find(isTtsCapability);
  assert.ok(tts, 'the tts capability should be read as voice rows');
  assert.equal(tts.models.length, 3);
  for (const voice of tts.models) {
    assert.equal(voice.needsReference, false, `${voice.id} should read as false`);
  }
});

test('reading a pre-field document is a STATEMENT, not a filled-in default', async () => {
  // Same proof as the route rule's: the same document with ONE row stating the
  // field is not "two defaults and one fact", it is a defect.
  const rows = preFieldVoices();
  rows[2]!['needs_reference'] = true;
  answers(200, rows);
  await assert.rejects(client().voices(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError);
    assert.match(error.message, new RegExp(VOICES_NEEDS_REFERENCE_MISSING));
    // It names the ROW — both by path and by the voice's own id.
    assert.match(error.message, /voices\[0\]/);
    assert.match(error.message, /deathstalker/);
    return true;
  });
});

test('a half-stated info document is refused the same way', async () => {
  const rows = preFieldVoices();
  rows[0]!['needs_reference'] = false;
  answers(200, preFieldInfo(rows));
  await assert.rejects(client().info(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError);
    assert.match(error.message, new RegExp(VOICES_NEEDS_REFERENCE_MISSING));
    assert.match(error.message, /capabilities\[1\]\.models\[1\]/);
    return true;
  });
});

test('a needs_reference that is not a boolean is voices_needs_reference_unknown', async () => {
  const rows = preFieldVoices();
  for (const row of rows) row['needs_reference'] = false;
  rows[1]!['needs_reference'] = 'yes';
  answers(200, rows);
  await assert.rejects(client().voices(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError);
    assert.match(error.message, new RegExp(VOICES_NEEDS_REFERENCE_UNKNOWN));
    assert.match(error.message, /voices\[1\]\.needs_reference is "yes"/);
    return true;
  });
});

test('a null needs_reference is refused too — null is not false', async () => {
  const rows = preFieldVoices();
  for (const row of rows) row['needs_reference'] = true;
  rows[0]!['needs_reference'] = null;
  answers(200, rows);
  await assert.rejects(client().voices(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError);
    assert.match(error.message, new RegExp(VOICES_NEEDS_REFERENCE_UNKNOWN));
    return true;
  });
});

test('a current document is read exactly as before, on both routes', async () => {
  const rows = preFieldVoices();
  for (const row of rows) row['needs_reference'] = false;
  rows[1]!['kind'] = 'zeroshot';
  rows[1]!['needs_reference'] = true;
  answers(200, rows);
  const voices = await client().voices();
  assert.deepEqual(
    voices.map((voice) => [voice.id, voice.needsReference]),
    [['deathstalker', false], ['mistborn', true], ['leah', false]],
  );
  answers(200, preFieldInfo(rows));
  const info = await client().info();
  const tts = info.capabilities.find(isTtsCapability);
  assert.ok(tts);
  assert.deepEqual(
    tts.models.map((voice) => [voice.id, voice.needsReference]),
    [['deathstalker', false], ['mistborn', true], ['leah', false]],
  );
});
