/**
 * `capability()` — `GET /v1/capability`. PHASE9-CAPABILITY.md.
 *
 * Against an in-process `node:http` fixture, for the reason every unit file
 * here gives: the case that matters most is one a healthy server never
 * produces on a good day — a host that has decided NOTHING, which must not be
 * read as a host that can do nothing.
 *
 * Run: `npm run test:unit` (exits non-zero on any failure).
 */

import assert from 'node:assert/strict';
import { createServer, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import {
  CAPABILITY_UNDECIDED,
  CrucibleCapabilityUndecided,
  CrucibleClient,
  CrucibleProtocolError,
  CrucibleServerError,
} from '../src/index.js';

// ------------------------------------------------------------------ fixture

let reply: { status: number; body: unknown } = { status: 500, body: {} };
let lastPath = '';
let lastMethod = '';

let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-capability' });
}

function answers(status: number, body: unknown): void {
  reply = { status, body };
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

before(async () => {
  server = createServer((request, response) => {
    lastPath = request.url ?? '';
    lastMethod = request.method ?? '';
    json(response, reply.status, reply.body);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

const GIB = 1024 ** 3;

/** The 3090 Ti record from `tests/test_capability.py`, as the server writes it. */
const RECORD = {
  backend_kind: 'cuda-linux',
  total_bytes: 24 * GIB,
  desktop_allowance_bytes: 3 * GIB,
  classes: [
    {
      capability: 'echo',
      enabled: true,
      selected: '',
      reason: 'always available: the test job type; it never touches the accelerator',
      shortfall_bytes: 0,
      route: 'local',
      work: null,
      context_ceilings: null,
    },
    {
      capability: 'clean',
      enabled: true,
      selected: 'qwen3.5-9b',
      reason: 'qwen3.5-9b fits: it needs 19.0 GiB and there is 21.0 GiB available',
      shortfall_bytes: 0,
      route: 'local',
      work: null,
      context_ceilings: null,
    },
    {
      capability: 'translate',
      enabled: true,
      selected: 'qwen3.8-27b-4bit',
      reason: 'qwen3.8-27b-4bit fits: it needs 20.1 GiB and there is 21.0 GiB available',
      shortfall_bytes: 0,
      route: 'local',
      work: null,
      context_ceilings: null,
    },
    {
      capability: 'tts',
      enabled: false,
      selected: '',
      reason:
        'disabled: the smallest of 4 voices is deathstalker at 24.0 GiB and there is only ' +
        '21.0 GiB available — short by 3.0 GiB. Higgs v3 is not quantized and will not be.',
      shortfall_bytes: 3 * GIB,
      route: 'local',
      work: null,
      context_ceilings: null,
    },
  ],
};

// ------------------------------------------------------------------- reading

test('capability() reads the record, and the rows arrive under classes', async () => {
  answers(200, RECORD);
  const record = await client().capability();
  assert.equal(lastMethod, 'GET');
  assert.equal(lastPath, '/v1/capability');
  assert.equal(record.backendKind, 'cuda-linux');
  assert.equal(record.totalBytes, 24 * GIB);
  assert.equal(record.desktopAllowanceBytes, 3 * GIB);
  assert.deepEqual(
    record.classes.map((row) => row.capability),
    ['echo', 'clean', 'translate', 'tts'],
  );
  const translate = record.classes[2]!;
  assert.equal(translate.enabled, true);
  assert.equal(translate.selected, 'qwen3.8-27b-4bit');
  assert.equal(translate.shortfallBytes, 0);
  assert.match(String(translate.reason), /fits/);
});

test('a disabled class is an answer with a number, not an error', async () => {
  answers(200, RECORD);
  const record = await client().capability();
  const tts = record.classes.find((row) => row.capability === 'tts')!;
  assert.equal(tts.enabled, false);
  // The server's TOML has no null: nothing fit, so `selected` is '' and the
  // shortfall is the number that turned the class off, as a number and not only
  // inside the sentence.
  assert.equal(tts.selected, '');
  assert.equal(tts.shortfallBytes, 3 * GIB);
  assert.match(String(tts.reason), /short by 3\.0 GiB/);
  // A class that needs no accelerator is enabled with nothing selected, and
  // that is not the same reading as a class nothing fit for.
  const echo = record.classes.find((row) => row.capability === 'echo')!;
  assert.equal(echo.enabled, true);
  assert.equal(echo.selected, '');
});

// ANY CRUCIBLE THAT ANSWERS WORKS (Owen, 2026-09-24). A row's reason and
// shortfall inform a person; whether the class is enabled and what it selected
// are what a caller asks for work with. The first kind reads null when a
// server does not state it; the second is still refused by name.

test('a row without an informational field reads it as null, not as a refusal', async () => {
  const { shortfall_bytes: _gone, reason: _said, ...bare } = RECORD.classes[3]!;
  answers(200, { ...RECORD, classes: [bare] });
  const [tts] = (await client().capability()).classes;
  assert.ok(tts !== undefined);
  assert.equal(tts.enabled, false);
  assert.equal(tts.shortfallBytes, null);
  assert.equal(tts.reason, null);
});

test('a row missing a load-bearing field is still a protocol error, by name', async () => {
  const { enabled: _gone, ...withoutEnabled } = RECORD.classes[3]!;
  answers(200, { ...RECORD, classes: [withoutEnabled] });
  await assert.rejects(client().capability(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /capability\.classes\[0\] has no field "enabled"/);
    return true;
  });
});

test('an informational field that is PRESENT with the wrong type is a protocol error', async () => {
  // Absent is an older server; `work: "banana"` is a broken one. API v1 adds
  // fields and never retypes them, so the two are never the same news.
  answers(200, { ...RECORD, classes: [{ ...RECORD.classes[1]!, work: 'banana' }] });
  await assert.rejects(client().capability(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /capability\.classes\[0\]\.work is present but is not a JSON object/);
    return true;
  });
});

test('a record without classes is a protocol error', async () => {
  const { classes: _rows, ...withoutClasses } = RECORD;
  answers(200, withoutClasses);
  await assert.rejects(client().capability(), CrucibleProtocolError);
});

// ------------------------------------------------------------------ undecided

test('503 capability_undecided is its own type, and is still a server error', async () => {
  answers(503, {
    error: {
      code: CAPABILITY_UNDECIDED,
      message:
        'this server has no capability record; nothing has probed the card on this host ' +
        'yet. Run `crucible capability --write` (or reinstall) to decide',
    },
  });
  await assert.rejects(client().capability(), (error: unknown) => {
    assert.ok(error instanceof CrucibleCapabilityUndecided, `got ${String(error)}`);
    // Still a 5xx, so every handler that catches one still catches this.
    assert.ok(error instanceof CrucibleServerError);
    assert.equal(error.status, 503);
    assert.equal(error.code, 'capability_undecided');
    assert.match(error.serverMessage, /crucible capability --write/);
    return true;
  });
});

test('an undecided host throws rather than answering with empty rows', async () => {
  // Absent is its own answer. Empty rows would read as "probed, and nothing
  // fit", which is the opposite news — and a client that rendered them would
  // tell the user a machine can do nothing when nobody has asked it yet.
  answers(503, {
    error: { code: 'capability_undecided', message: 'this server has no capability record' },
  });
  let returned: unknown = 'nothing';
  try {
    returned = await client().capability();
  } catch (error) {
    assert.ok(error instanceof CrucibleCapabilityUndecided);
  }
  assert.equal(returned, 'nothing', 'an undecided host must not resolve to a record');
});

test('a 5xx that is not the undecided refusal stays a plain server error', async () => {
  answers(503, { error: { code: 'overloaded', message: 'the lane is wedged' } });
  await assert.rejects(client().capability(), (error: unknown) => {
    assert.ok(error instanceof CrucibleServerError, `got ${String(error)}`);
    assert.ok(
      !(error instanceof CrucibleCapabilityUndecided),
      'only capability_undecided gets the narrower type',
    );
    return true;
  });
});

// ------------------------------------------------------ a client-sized class

const SIZED = {
  ...RECORD,
  classes: [
    {
      capability: 'generate',
      enabled: true,
      selected: 'qwen3.5-9b',
      reason: 'qwen3.5-9b fits',
      shortfall_bytes: 0,
      route: 'local',
      work: { tokens: 16384, concurrency: 2, source: 'stated by the client', from: 'request' },
      context_ceilings: [
        {
          model: 'qwen3.8-27b-4bit',
          tokens: 11630,
          bound_by: 'memory',
          served_context: 16384,
          served_context_source: 'the manifest',
          memory_context: 11630,
          memory_context_source: 'this host',
          concurrency: 2,
        },
      ],
    },
  ],
};

test('capability() sizes a client-sized class on the query and reads the echo', async () => {
  answers(200, SIZED);
  const record = await client().capability(
    {},
    { class: 'generate', contextTokens: 16384, concurrency: 2 },
  );
  assert.equal(lastPath, '/v1/capability?class=generate&context_tokens=16384&concurrency=2');
  const row = record.classes[0]!;
  assert.deepEqual(row.work, {
    tokens: 16384,
    concurrency: 2,
    source: 'stated by the client',
    from: 'request',
  });
  assert.deepEqual(row.contextCeilings, [
    {
      model: 'qwen3.8-27b-4bit',
      tokens: 11630,
      boundBy: 'memory',
      servedContext: 16384,
      memoryContext: 11630,
      concurrency: 2,
    },
  ]);
});

test('without sizing no query is sent, and a fractional size never leaves the client', async () => {
  answers(200, RECORD);
  await client().capability();
  assert.equal(lastPath, '/v1/capability');
  await assert.rejects(
    client().capability({}, { class: 'generate', contextTokens: 2.5 }),
    /context_tokens is 2.5/,
  );
});

test('a 1.0.23-shaped row, with no work or ceilings key, reads both as null', async () => {
  // The exact failure the ruling was made over: after the 1.0.24 repin every
  // BookForge fake without `work` threw `capability.classes[0] has no field
  // "work"`. A server that predates the fields has not stated them.
  const { work: _work, context_ceilings: _ceilings, ...bare } = SIZED.classes[0]!;
  answers(200, { ...RECORD, classes: [bare] });
  const [row] = (await client().capability()).classes;
  assert.ok(row !== undefined);
  assert.equal(row.selected, SIZED.classes[0]!.selected);
  assert.equal(row.work, null);
  assert.equal(row.contextCeilings, null);
});

test('a record with no sizing figures at all still reads', async () => {
  const { backend_kind: _k, total_bytes: _t, desktop_allowance_bytes: _d, ...bare } = RECORD;
  answers(200, bare);
  const record = await client().capability();
  assert.equal(record.backendKind, null);
  assert.equal(record.totalBytes, null);
  assert.equal(record.desktopAllowanceBytes, null);
  assert.equal(record.classes.length, 4);
});
