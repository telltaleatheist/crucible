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
    },
    {
      capability: 'clean',
      enabled: true,
      selected: 'qwen3.5-9b',
      reason: 'qwen3.5-9b fits: it needs 19.0 GiB and there is 21.0 GiB available',
      shortfall_bytes: 0,
      route: 'local',
    },
    {
      capability: 'translate',
      enabled: true,
      selected: 'qwen3.8-27b-4bit',
      reason: 'qwen3.8-27b-4bit fits: it needs 20.1 GiB and there is 21.0 GiB available',
      shortfall_bytes: 0,
      route: 'local',
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
  assert.match(translate.reason, /fits/);
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
  assert.match(tts.reason, /short by 3\.0 GiB/);
  // A class that needs no accelerator is enabled with nothing selected, and
  // that is not the same reading as a class nothing fit for.
  const echo = record.classes.find((row) => row.capability === 'echo')!;
  assert.equal(echo.enabled, true);
  assert.equal(echo.selected, '');
});

test('a row missing a promised field is a protocol error, not a quiet default', async () => {
  const { shortfall_bytes: _gone, ...withoutShortfall } = RECORD.classes[3]!;
  answers(200, { ...RECORD, classes: [withoutShortfall] });
  await assert.rejects(client().capability(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /capability\.classes\[0\] has no field "shortfall_bytes"/);
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
