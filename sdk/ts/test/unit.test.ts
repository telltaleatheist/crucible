/**
 * Unit tests: the error map, and nothing else.
 *
 * The e2e test is what proves the client against the real server. These few
 * cases exist because the error map is the one part of the client the real
 * server will not exercise on a good day — a crucible under test does not
 * return 500, and a crucible is not a non-crucible.
 *
 * Run: `npm run test:unit` (exits non-zero on any failure).
 */

import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import { readFileSync } from 'node:fs';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import {
  CrucibleAuthError,
  CrucibleBusy,
  CrucibleClient,
  CrucibleConfigError,
  CrucibleNotACrucible,
  CrucibleProtocolError,
  CrucibleRefused,
  CrucibleServerError,
  CrucibleUnreachable,
  CrucibleVersionError,
  SDK_VERSION,
  isServerSpecificRefusal,
} from '../src/index.js';

/** The body `crucible/jobs/queue.py` actually sends with a 409 server_busy. */
const BUSY_DETAILS = {
  holder: 'foundry/0.9.0',
  job_id: 'a1b2c3',
  type: 'tts',
  model: 'higgs-v3',
  status: 'running',
  since: '2026-09-13T18:04:11Z',
  progress: 0.62,
  message: 'rendering 118 of 280',
};

/** What the fixture should answer next, set by each test before it calls. */
let reply: { status: number; body: string; contentType: string } = {
  status: 200,
  body: '{}',
  contentType: 'application/json',
};

/** The User-Agent of the last request the fixture saw. */
let lastUserAgent: string | undefined;
/** The headers of the last request the fixture saw. */
let lastHeaders: Record<string, string | string[] | undefined> = {};

let server: Server;
let base = '';
/** A port nothing is listening on, for the unreachable case. */
let deadBase = '';

function client(url: string = base): CrucibleClient {
  return new CrucibleClient({ url, token: 'the-token', clientName: 'unit-test' });
}

function answer(status: number, body: unknown, contentType = 'application/json'): void {
  reply = {
    status,
    body: typeof body === 'string' ? body : JSON.stringify(body),
    contentType,
  };
}

before(async () => {
  server = createServer((request, response) => {
    lastUserAgent = request.headers['user-agent'];
    lastHeaders = request.headers;
    response.writeHead(reply.status, { 'Content-Type': reply.contentType });
    response.end(reply.body);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;

  // Take a port, learn its number, give it back: connecting to it now refuses.
  const spare = createServer();
  await new Promise<void>((resolve) => spare.listen(0, '127.0.0.1', resolve));
  deadBase = `http://127.0.0.1:${(spare.address() as AddressInfo).port}`;
  await new Promise<void>((resolve) => spare.close(() => resolve()));
});

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

// ------------------------------------------------------------------ mapping

test('401 becomes CrucibleAuthError carrying the server code and message', async () => {
  answer(401, { error: { code: 'unauthorized', message: 'bearer token is not this server\'s token' } });
  await assert.rejects(client().info(), (error: unknown) => {
    assert.ok(error instanceof CrucibleAuthError, `got ${String(error)}`);
    assert.equal(error.status, 401);
    assert.equal(error.code, 'unauthorized');
    assert.equal(error.serverMessage, "bearer token is not this server's token");
    return true;
  });
});

test('426 becomes CrucibleVersionError naming both versions', async () => {
  answer(426, {
    error: {
      code: 'api_version_mismatch',
      message: 'client speaks API version 1, this server speaks 2',
      details: { server_api_version: 2, client_api_version: 1 },
    },
  });
  await assert.rejects(client().health(), (error: unknown) => {
    assert.ok(error instanceof CrucibleVersionError, `got ${String(error)}`);
    assert.equal(error.status, 426);
    assert.equal(error.code, 'api_version_mismatch');
    assert.equal(error.serverApiVersion, 2);
    assert.equal(error.clientApiVersion, 1);
    return true;
  });
});

test('a 4xx refusal becomes CrucibleRefused with the named reason and details', async () => {
  answer(400, {
    error: {
      code: 'unknown_job_type',
      message: "unknown job type 'summon'; this server offers ['echo']",
      details: { offered: ['echo'] },
    },
  });
  await assert.rejects(
    client().submit({ type: 'summon', params: {}, inputs: {} }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
      assert.equal(error.status, 400);
      assert.equal(error.code, 'unknown_job_type');
      assert.deepEqual(error.details, { offered: ['echo'] });
      return true;
    },
  );
});

test('5xx becomes CrucibleServerError and is never retried', async () => {
  let calls = 0;
  const counting = createServer((request, response) => {
    calls += 1;
    response.writeHead(503, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify({ error: { code: 'overloaded', message: 'the lane is wedged' } }));
  });
  await new Promise<void>((resolve) => counting.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${(counting.address() as AddressInfo).port}`;
  try {
    await assert.rejects(client(url).health(), (error: unknown) => {
      assert.ok(error instanceof CrucibleServerError, `got ${String(error)}`);
      assert.equal(error.status, 503);
      assert.equal(error.code, 'overloaded');
      return true;
    });
    assert.equal(calls, 1, 'the client must not retry a 5xx');
  } finally {
    await new Promise<void>((resolve) => counting.close(() => resolve()));
  }
});

test('a responder that is not a crucible is CrucibleNotACrucible, not an auth error', async () => {
  answer(200, { hello: 'world' });
  await assert.rejects(client().ping(), (error: unknown) => {
    assert.ok(error instanceof CrucibleNotACrucible, `got ${String(error)}`);
    assert.match(error.message, /did not identify as a crucible/);
    return true;
  });

  answer(200, '<html>nginx</html>', 'text/html');
  await assert.rejects(client().ping(), (error: unknown) => {
    assert.ok(error instanceof CrucibleNotACrucible, `got ${String(error)}`);
    return true;
  });
});

test('nothing listening is CrucibleUnreachable', async () => {
  await assert.rejects(client(deadBase).ping(), (error: unknown) => {
    assert.ok(error instanceof CrucibleUnreachable, `got ${String(error)}`);
    assert.equal(error.url, deadBase);
    return true;
  });
});

test('an error body that is not the crucible envelope is a protocol error', async () => {
  answer(502, '<html>bad gateway</html>', 'text/html');
  await assert.rejects(client().info(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /502/);
    return true;
  });
});

// ------------------------------------------------------------------- config

test('every constructor option is required and is named when missing', () => {
  const cases: Array<[Record<string, unknown>, string]> = [
    [{ token: 't', clientName: 'c' }, 'url'],
    [{ url: 'http://127.0.0.1:7100', clientName: 'c' }, 'token'],
    [{ url: 'http://127.0.0.1:7100', token: 't' }, 'clientName'],
    [{ url: '', token: 't', clientName: 'c' }, 'url'],
  ];
  for (const [options, option] of cases) {
    assert.throws(
      () => new CrucibleClient(options as never),
      (error: unknown) => {
        assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
        assert.equal(error.option, option);
        return true;
      },
      `expected ${option} to be refused by name`,
    );
  }
});

test('a url that already carries /v1 is refused rather than doubled', () => {
  assert.throws(
    () => new CrucibleClient({ url: 'http://127.0.0.1:7100/v1', token: 't', clientName: 'c' }),
    CrucibleConfigError,
  );
});

// ------------------------------------------------------------------ headers

test('authed calls carry the token, the api header and the client name', async () => {
  answer(200, { status: 'ok', queue_depth: 0, resident_models: [], resident_kind: null, stopping: null });
  await client().health();
  assert.equal(lastHeaders['authorization'], 'Bearer the-token');
  assert.equal(lastHeaders['x-crucible-api'], '1');
  assert.equal(lastUserAgent, `unit-test crucible-client/${SDK_VERSION}`);
});

test('ping carries neither the token nor the api header', async () => {
  answer(200, { crucible: true, name: 'crucible@fixture', api_version: 1 });
  const ping = await client().ping();
  assert.equal(ping.name, 'crucible@fixture');
  assert.equal(lastHeaders['authorization'], undefined);
  assert.equal(lastHeaders['x-crucible-api'], undefined);
});

// ------------------------------------------------------------------ version

test('SDK_VERSION matches package.json', () => {
  const manifest = JSON.parse(
    readFileSync(new URL('../../package.json', import.meta.url), 'utf8'),
  ) as { version: string };
  assert.equal(SDK_VERSION, manifest.version);
});


// ------------------------------------------- 409 server_busy is its own type

test('a 409 server_busy is read into CrucibleBusy, fields and all', async () => {
  answer(409, {
    error: {
      code: 'server_busy',
      message: 'this server is busy with job a1b2c3 (tts), running since ...',
      details: BUSY_DETAILS,
    },
  });
  await assert.rejects(
    client().submit({ type: 'tts', params: {}, inputs: {} }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleBusy, `got ${String(error)}`);
      // Still a CrucibleRefused: nothing that already handles 4xx changes.
      assert.ok(error instanceof CrucibleRefused);
      assert.equal(error.status, 409);
      assert.equal(error.holder, 'foundry/0.9.0');
      assert.equal(error.jobId, 'a1b2c3');
      assert.equal(error.jobType, 'tts');
      assert.equal(error.model, 'higgs-v3');
      assert.equal(error.jobStatus, 'running');
      assert.equal(error.since, '2026-09-13T18:04:11Z');
      assert.equal(error.progress, 0.62);
      assert.equal(error.jobMessage, 'rendering 118 of 280');
      // The one line a bench puts in front of a human.
      assert.equal(error.busyLine, 'busy: foundry/0.9.0, tts higgs-v3, 62% done — rendering 118 of 280');
      return true;
    },
  );
});

test('an unnamed holder is reported as unnamed, never guessed', async () => {
  // The server refuses to invent a name when a client sent no User-Agent, so a
  // bench must never be confidently wrong about whose render is on the card
  // (PHASE7-LANES.md section 5). null means "it did not say".
  answer(409, {
    error: {
      code: 'server_busy',
      message: 'busy',
      details: { ...BUSY_DETAILS, holder: null, model: null, message: null },
    },
  });
  await assert.rejects(
    client().submit({ type: 'tts', params: {}, inputs: {} }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleBusy, `got ${String(error)}`);
      assert.equal(error.holder, null);
      assert.equal(error.model, null);
      assert.equal(error.jobMessage, null);
      assert.equal(error.busyLine, 'busy: an unnamed client, tts, 62% done');
      return true;
    },
  );
});

test('a server_busy body missing a promised field is a protocol error, not a quiet downgrade', async () => {
  // These fields are API v1's promise. A silent fall back to a plain
  // CrucibleRefused would hide a broken wire behind an error that still looks
  // normal: the caller would see "busy" and never learn the holder and progress
  // it was about to display had gone missing.
  const { progress: _dropped, ...withoutProgress } = BUSY_DETAILS;
  answer(409, {
    error: { code: 'server_busy', message: 'busy', details: withoutProgress },
  });
  await assert.rejects(
    client().submit({ type: 'tts', params: {}, inputs: {} }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
      assert.match(error.message, /progress/);
      return true;
    },
  );
});

test('only refusals about a SERVER travel to the next one', () => {
  // The fact a `waitFor: "any"` walk needs, and the only honest owner of it is
  // the server that emits the code (PHASE7-LANES.md section 4.2.1).
  for (const code of [
    'server_busy',
    // Not the lane — narrator's wire. This is what a render gets while the
    // browser extension is streaming from that machine, and it travels just as
    // well: another server's card may be free.
    'engine_in_use',
    'stream_session_open',
    'job_type_disabled',
    'model_not_resident',
    'unknown_model',
    'env_missing',
  ]) {
    assert.equal(isServerSpecificRefusal(code), true, code);
  }
  // These are about the REQUEST. They are refused identically everywhere, so a
  // walk that retried them would report the fourth machine's error after three
  // pointless round trips.
  for (const code of [
    'invalid_request',
    'unknown_job_type',
    'unknown_blob',
    'unknown_job',
    'job_not_cancellable',
  ]) {
    assert.equal(isServerSpecificRefusal(code), false, code);
  }
  // An unknown code answers false on purpose: a refusal this build has never
  // seen is surfaced to the caller rather than swallowed by a walk.
  assert.equal(isServerSpecificRefusal('something_invented_next_year'), false);
});


// ------------------------------------------ the bench read, and what it cannot say

/** A `/v1/activity` body with a session open and the lane free. */
const ACTIVITY_WITH_SESSION = {
  server: { name: 'crucible@mac', version: '0.4.0', api_version: 1, backend: 'mlx-darwin', uptime_s: 12.5 },
  resident: { kind: 'tts', id: 'deathstalker', since: '2026-09-13T18:00:00Z', memory_bytes_estimate: 19000000000 },
  // Nothing was told to go: present and null, like `claim` and `lease` below.
  stopping: null,
  warming: null,
  claim: { held_by: 'tts stream 3f2a' },
  streaming: {
    session_id: '3f2a',
    voice: 'deathstalker',
    language: 'en',
    narrator_engine: 'higgs-v3',
    since: '2026-09-13T18:02:00Z',
    client: 'bookforge-extension crucible-client/0.4.0',
    progress: null,
    said: 7,
    finished: 6,
    in_flight: 1,
    seconds: 41.2,
    chars: 903,
  },
  chat: { in_flight: 0, rows: [] },
  // Nobody has said they are mid-run. Present and null, like `claim` above:
  // an absent key would mean a build that does not speak the field.
  lease: null,
  slots: { accelerated: { busy: 0, of: 1, queue_depth: 0, accepts_work: false } },
  running: [],
  queued: [],
};

test('a machine with a session open is not reported as idle', async () => {
  // The lane really is free and still says so; what changed is that the CARD's
  // holder is now on the same read, and `acceptsWork` composes the two.
  answer(200, ACTIVITY_WITH_SESSION);
  const seen = await client().activity();
  assert.equal(seen.slots.accelerated.busy, 0);
  assert.deepEqual(seen.running, []);
  assert.equal(seen.slots.accelerated.acceptsWork, false);
  assert.deepEqual(seen.claim, { heldBy: 'tts stream 3f2a' });
  assert.equal(seen.streaming?.sessionId, '3f2a');
  assert.equal(seen.streaming?.client, 'bookforge-extension crucible-client/0.4.0');
  assert.equal(seen.streaming?.said, 7);
  assert.equal(seen.streaming?.inFlight, 1);
  // No percentage, and the type says so: `progress: null` is the declared type,
  // not a value it happens to hold.
  assert.equal(seen.streaming?.progress, null);
});

test('a session that claims a percentage is a protocol error', async () => {
  // A session has no denominator. A number here would be a fraction of the work
  // that happens to have arrived — one that falls as more arrives — and a bench
  // drawing it would show a reader's progress bar going backwards. Caught rather
  // than rendered.
  answer(200, {
    ...ACTIVITY_WITH_SESSION,
    streaming: { ...ACTIVITY_WITH_SESSION.streaming, progress: 0.85 },
  });
  await assert.rejects(client().activity(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /no total to be a fraction of/);
    return true;
  });
});

test('an idle machine reports both nulls, and both keys are required', async () => {
  answer(200, {
    ...ACTIVITY_WITH_SESSION,
    resident: null,
    claim: null,
    streaming: null,
    slots: { accelerated: { busy: 0, of: 1, queue_depth: 0, accepts_work: true } },
  });
  const seen = await client().activity();
  assert.equal(seen.claim, null);
  assert.equal(seen.streaming, null);
  assert.equal(seen.resident, null);
  assert.equal(seen.slots.accelerated.acceptsWork, true);

  // An ABSENT key is not the same news as a present null: it means the server
  // does not speak the field, which this client will not silently read as "and
  // therefore nothing is happening".
  const { claim: _gone, ...withoutClaim } = ACTIVITY_WITH_SESSION;
  answer(200, withoutClaim);
  await assert.rejects(client().activity(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /claim/);
    return true;
  });
});


test('a chat in flight is named as the act it IS, and does not gate work', async () => {
  // Before Crucible everything ran under "translate". A simplify must never be
  // reported as one — Owen, 2026-09-13 — and the server refuses an unknown act
  // at the door rather than recording a name nothing knows.
  answer(200, {
    ...ACTIVITY_WITH_SESSION,
    claim: null,
    streaming: null,
    chat: {
      in_flight: 2,
      rows: [
        { id: 1, act: 'simplify', model: 'qwen3.8-27b-4bit', client: 'foundry/0.9.0', since: '2026-09-13T19:00:00Z' },
        { id: 2, act: null, model: 'qwen3.8-27b-4bit', client: null, since: '2026-09-13T19:00:04Z' },
      ],
    },
    slots: { accelerated: { busy: 0, of: 1, queue_depth: 0, accepts_work: true } },
  });
  const seen = await client().activity();
  assert.equal(seen.chat.inFlight, 2);
  const [named, anonymous] = seen.chat.rows;
  assert.ok(named !== undefined && anonymous !== undefined, 'both rows are read');
  assert.equal(named.act, 'simplify');
  assert.equal(named.client, 'foundry/0.9.0');
  // Null is "the client did not say", never an inferred act.
  assert.equal(anonymous.act, null);
  assert.equal(anonymous.client, null);
  // Counted, never gating: the engine batches, so the server really will take
  // more. The lane is what `accepts_work` is about, and it is free.
  assert.equal(seen.slots.accelerated.acceptsWork, true);
  assert.equal(seen.slots.accelerated.busy, 0);
});

// ------------------------------------- what was told to go and has not gone

/**
 * `GET /v1/health` off a server whose narrator ignored its SIGTERM, captured
 * from `crucible/api.py`'s route — the shape `DyingResident.to_dict` writes.
 *
 * The state matters because nothing in Crucible ends it: `engines/base.py`
 * never SIGKILLs, since a killed CUDA process wedges WSL2 until Windows
 * reboots. So `status` reads `ok`, `resident_models` is empty, and every load
 * is nonetheless refused `engine_still_stopping` until somebody stops pid
 * 41288 by hand — which is why the pids are on the wire.
 */
const HEALTH_WHILE_STOPPING = {
  status: 'ok',
  queue_depth: 0,
  resident_models: [],
  resident_kind: null,
  stopping: {
    kind: 'tts',
    id: 'deathstalker',
    since: '2026-09-18T04:12:07+00:00',
    pids: [41288, 41301],
  },
};

test('health reads what is stopping, pids and all', async () => {
  answer(200, HEALTH_WHILE_STOPPING);
  const seen = await client().health();
  assert.equal(seen.status, 'ok');
  assert.deepEqual(seen.residentModels, []);
  assert.deepEqual(seen.stopping, {
    kind: 'tts',
    id: 'deathstalker',
    since: '2026-09-18T04:12:07+00:00',
    pids: [41288, 41301],
  });
});

test('activity carries the same object, and an absent key is a protocol error', async () => {
  answer(200, { ...ACTIVITY_WITH_SESSION, stopping: HEALTH_WHILE_STOPPING.stopping });
  const seen = await client().activity();
  assert.equal(seen.stopping?.id, 'deathstalker');
  assert.deepEqual(seen.stopping?.pids, [41288, 41301]);

  // An absent key is not the news "nothing is stopping": it is a build that
  // does not speak the field, and a client that read it as a null would draw
  // a free card on a server that refuses everything.
  const { stopping: _gone, ...withoutStopping } = ACTIVITY_WITH_SESSION;
  answer(200, withoutStopping);
  await assert.rejects(client().activity(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /stopping/);
    return true;
  });
});

test('a pid that is not an integer is refused rather than rounded', async () => {
  // It is what an operator types into a kill command. A float or a string
  // there is unusable, and quietly coercing one would hand him a number no
  // process has.
  answer(200, {
    ...HEALTH_WHILE_STOPPING,
    stopping: { ...HEALTH_WHILE_STOPPING.stopping, pids: ['41288'] },
  });
  await assert.rejects(client().health(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /pids\[0\] is not an integer pid/);
    return true;
  });
});
