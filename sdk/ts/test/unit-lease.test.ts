/**
 * Unit tests for the lease: `lease`, `heartbeat`, `release`, the typed `leased`
 * refusal, and the `lease` field on the bench read.
 *
 * WHAT A LEASE IS FOR, because these tests read oddly without it. A chat
 * completion holds nothing on a Crucible — no lane, no job, no claim — so a
 * server translating a book block by block reports itself idle between any two
 * blocks, and a `load-voice` submitted in one of those gaps takes the
 * translator off the card mid-run. The client is the only thing that knows the
 * run exists, so it says so, heartbeats while it lives, and releases at the end.
 *
 * These cover what a healthy real server will not produce on a good day: a
 * second lease refused while one is open, a heartbeat against a lease that
 * expired, a refusal body missing the fields a bench is about to display. The
 * live proof is the python suite's `tests/test_leases.py`.
 *
 * Run: `npm run test:unit` (exits non-zero on any failure).
 */

import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleConfigError,
  CrucibleLeased,
  CrucibleProtocolError,
  CrucibleRefused,
  LEASED,
  isServerSpecificRefusal,
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

let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-lease' });
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

/** Answer the next request with this, whatever it asks for. */
function answer(status: number, body: unknown): void {
  handle = (_request, response) => json(response, status, body);
}

before(async () => {
  server = createServer((request, response) => {
    let body = '';
    request.on('data', (chunk) => {
      body += chunk;
    });
    request.on('end', () => {
      lastPath = request.url ?? '';
      lastMethod = request.method ?? '';
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

// ------------------------------------------------------------------- the wire

const LEASE = {
  lease_id: '9c1f',
  kind: 'llm',
  subject: 'qwen3.8-27b-4bit',
  client: 'foundry/owens-pc crucible-client/0.5.0',
  act: 'translate',
  since: '2026-09-14T03:00:00+00:00',
  expires_at: '2026-09-14T03:02:00+00:00',
};

test('lease() names the subject in the path and sends the act and ttl the server reads', async () => {
  answer(201, LEASE);
  const lease = await client().lease('qwen3.8-27b-4bit', { act: 'translate', ttlSeconds: 120 });

  assert.equal(lastMethod, 'POST');
  assert.equal(lastPath, '/v1/models/qwen3.8-27b-4bit/lease');
  // `ttl_seconds`, not `ttlSeconds`: the wire is snake_case and the caller's
  // language is not.
  assert.deepEqual(JSON.parse(lastBody), { act: 'translate', ttl_seconds: 120 });

  assert.equal(lease.leaseId, '9c1f');
  // The subject and the kind: the id is what was asked for, the kind is the
  // server's own reading of what is on the card. A caller states neither, and
  // that is what lets one route lease a model, a voice or an aligner.
  assert.equal(lease.subject, 'qwen3.8-27b-4bit');
  assert.equal(lease.kind, 'llm');
  assert.equal(lease.act, 'translate');
  assert.equal(lease.client, 'foundry/owens-pc crucible-client/0.5.0');
  assert.equal(lease.expiresAt, '2026-09-14T03:02:00+00:00');
});

test('a voice and an aligner lease through the same door, and say which they are', async () => {
  // The regression this closes: the card is cleared the moment nothing holds
  // it, so a book rendered CHAPTER BY CHAPTER paid a narrator load per chapter
  // and a book aligned chapter by chapter paid an aligner load per chapter —
  // the lease was the one thing that could have held them and it could only
  // name a model. One route, one shape, three kinds.
  answer(201, { ...LEASE, kind: 'tts', subject: 'mistborn', act: 'tts' });
  const voice = await client().lease('mistborn', { act: 'tts', ttlSeconds: 300 });
  assert.equal(lastPath, '/v1/models/mistborn/lease');
  assert.equal(voice.kind, 'tts');
  assert.equal(voice.subject, 'mistborn');
  // The caller never states a kind: the card holds one thing, so the id alone
  // identifies it and the server supplies the rest.
  assert.deepEqual(JSON.parse(lastBody), { act: 'tts', ttl_seconds: 300 });

  answer(201, { ...LEASE, kind: 'align', subject: 'qwen3-aligner', act: 'align' });
  const aligner = await client().lease('qwen3-aligner', { act: 'align', ttlSeconds: 300 });
  assert.equal(aligner.kind, 'align');
  assert.equal(aligner.subject, 'qwen3-aligner');
});

test('a refusal on a voice lease says so, so a bench does not report the wrong card', async () => {
  answer(409, {
    error: {
      ...LEASED_BODY.error,
      details: { ...LEASED_BODY.error.details, kind: 'tts', act: 'tts' },
    },
  });
  await assert.rejects(
    client().submit({ type: 'load-model', model: 'qwen3.5-9b', params: {}, inputs: {} }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleLeased, `got ${String(error)}`);
      assert.equal(error.kind, 'tts');
      assert.equal(error.act, 'tts');
      return true;
    },
  );
});

test('an id with a slash or a space is escaped rather than pasted into the path', async () => {
  answer(201, { ...LEASE, subject: 'owen/model one' });
  await client().lease('owen/model one', { act: 'clean', ttlSeconds: 60 });
  assert.equal(lastPath, '/v1/models/owen%2Fmodel%20one/lease');
});

test('lease() insists on a subject, an act and a whole number of seconds', async () => {
  const c = client();
  await assert.rejects(c.lease('', { act: 'clean', ttlSeconds: 60 }), CrucibleConfigError);
  await assert.rejects(c.lease('m', { act: '', ttlSeconds: 60 }), CrucibleConfigError);
  await assert.rejects(
    c.lease('m', { act: 'clean', ttlSeconds: 1.5 }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleConfigError, `got ${String(error)}`);
      assert.equal(error.option, 'ttlSeconds');
      return true;
    },
  );
  // The RANGE is deliberately not checked here. The server states its own in
  // the refusal, and a second copy of two numbers in this file is a second
  // thing to drift (ARCHITECTURE.md R1).
});

test('heartbeat() returns the new deadline, and release() sends a DELETE with no body to read', async () => {
  answer(200, { expires_at: '2026-09-14T03:04:00+00:00' });
  const expiresAt = await client().heartbeat('9c1f');
  assert.equal(lastMethod, 'POST');
  assert.equal(lastPath, '/v1/leases/9c1f/heartbeat');
  assert.equal(expiresAt, '2026-09-14T03:04:00+00:00');

  handle = (_request, response) => {
    response.writeHead(204);
    response.end();
  };
  await client().release('9c1f');
  assert.equal(lastMethod, 'DELETE');
  assert.equal(lastPath, '/v1/leases/9c1f');
});

test('a lease that is gone is a refusal, not a shrug', async () => {
  // A client that thinks it still holds one has to be told it does not: its run
  // is no longer protected and the card may move under it.
  answer(404, {
    error: {
      code: 'unknown_lease',
      message: 'lease 9c1f is no longer open: it expired at 2026-09-14T03:02:00+00:00',
      details: { lease_id: '9c1f', reason: 'it expired at 2026-09-14T03:02:00+00:00' },
    },
  });
  await assert.rejects(client().heartbeat('9c1f'), (error: unknown) => {
    assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
    assert.equal(error.code, 'unknown_lease');
    return true;
  });

  answer(404, {
    error: { code: 'unknown_lease', message: 'this server has no lease 9c1f' },
  });
  await assert.rejects(client().release('9c1f'), CrucibleRefused);
});

// ---------------------------------------------------------- the typed refusal

const LEASED_BODY = {
  error: {
    code: 'leased',
    message:
      "'qwen3.8-27b-4bit' is leased by 'foundry/owens-pc crucible-client/0.5.0' for " +
      "'translate' since 2026-09-14T03:00:00+00:00",
    details: {
      lease_id: '9c1f',
      kind: 'llm',
      client: 'foundry/owens-pc crucible-client/0.5.0',
      act: 'translate',
      since: '2026-09-14T03:00:00+00:00',
      expires_at: '2026-09-14T03:02:00+00:00',
    },
  },
};

test('leased arrives typed, with the one line a bench puts in front of a human', async () => {
  answer(409, LEASED_BODY);
  await assert.rejects(
    client().submit({ type: 'load-voice', model: 'deathstalker', params: {}, inputs: {} }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleLeased, `got ${String(error)}`);
      assert.equal(error.code, LEASED);
      assert.equal(error.status, 409);
      assert.equal(error.leaseId, '9c1f');
      assert.equal(error.kind, 'llm');
      assert.equal(error.holder, 'foundry/owens-pc crucible-client/0.5.0');
      assert.equal(error.act, 'translate');
      assert.equal(error.since, '2026-09-14T03:00:00+00:00');
      assert.equal(error.expiresAt, '2026-09-14T03:02:00+00:00');
      assert.equal(
        error.leasedLine,
        'leased: foundry/owens-pc crucible-client/0.5.0, translate, until 2026-09-14T03:02:00+00:00',
      );
      return true;
    },
  );
});

test('a lease taken by a client that did not name itself renders as an unnamed client', async () => {
  // Null means "it did not say" and is never a guess, for the reason a bench
  // must never be confidently wrong about whose run is on the card.
  answer(409, {
    error: { ...LEASED_BODY.error, details: { ...LEASED_BODY.error.details, client: null } },
  });
  await assert.rejects(client().lease('qwen3.8-27b-4bit', { act: 'clean', ttlSeconds: 60 }), (error: unknown) => {
    assert.ok(error instanceof CrucibleLeased, `got ${String(error)}`);
    assert.equal(error.holder, null);
    // The act is the HOLDER's, read off the refusal — not the act this caller
    // asked for, which is the one thing it did not get.
    assert.equal(
      error.leasedLine,
      'leased: an unnamed client, translate, until 2026-09-14T03:02:00+00:00',
    );
    return true;
  });
});

test('a leased body missing what a bench displays is a protocol error, not a quiet downgrade', async () => {
  const { expires_at: _gone, ...withoutDeadline } = LEASED_BODY.error.details;
  answer(409, { error: { ...LEASED_BODY.error, details: withoutDeadline } });
  await assert.rejects(client().heartbeat('9c1f'), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /expires_at/);
    return true;
  });
});

test('leased is about THIS machine, so a walk should try the next one', async () => {
  assert.equal(isServerSpecificRefusal(LEASED), true);
  // And so is the lease door's own refusal, which names no kind because the
  // door takes an id of any kind.
  assert.equal(isServerSpecificRefusal('not_resident'), true);
});

// -------------------------------------------------------------- the bench read

const ACTIVITY = {
  server: { name: 'crucible@pc', version: '0.5.0', api_version: 1, backend: 'cuda-linux', uptime_s: 900 },
  resident: {
    kind: 'llm',
    id: 'qwen3.8-27b-4bit',
    since: '2026-09-14T02:40:00+00:00',
    memory_bytes_estimate: 19000000000,
  },
  warming: null,
  claim: null,
  streaming: null,
  chat: { in_flight: 1, rows: [{ id: 4, act: 'translate', model: 'qwen3.8-27b-4bit', client: 'foundry', since: '2026-09-14T03:01:00+00:00' }] },
  lease: {
    lease_id: '9c1f',
    kind: 'llm',
    client: 'foundry/owens-pc crucible-client/0.5.0',
    act: 'translate',
    since: '2026-09-14T03:00:00+00:00',
    expires_at: '2026-09-14T03:02:00+00:00',
  },
  slots: { accelerated: { busy: 0, of: 1, queue_depth: 0, accepts_work: true } },
  running: [],
  queued: [],
};

test('the bench reads the lease, and the lease does not pretend to be a busy lane', async () => {
  answer(200, ACTIVITY);
  const seen = await client().activity();
  assert.equal(seen.lease?.leaseId, '9c1f');
  assert.equal(seen.lease?.kind, 'llm');
  assert.equal(seen.lease?.act, 'translate');
  assert.equal(seen.lease?.client, 'foundry/owens-pc crucible-client/0.5.0');
  assert.equal(seen.lease?.expiresAt, '2026-09-14T03:02:00+00:00');
  // A lease is a refusal, not a reservation: the lane is free and this server
  // will still take work that leaves the card's contents alone.
  assert.equal(seen.slots.accelerated.acceptsWork, true);
  // WHICH thing it is has one owner on this read, and it is `resident`. The
  // KIND is carried anyway, because the same fields are a `leased` refusal's
  // details, where there is no `resident` beside them.
  assert.equal(seen.resident?.id, 'qwen3.8-27b-4bit');
  assert.equal('subject' in (seen.lease as object), false);
  assert.equal('model' in (seen.lease as object), false);
});

test('a server with nobody mid-run says null, and an absent key is not that statement', async () => {
  answer(200, { ...ACTIVITY, lease: null });
  assert.equal((await client().activity()).lease, null);

  const { lease: _gone, ...withoutLease } = ACTIVITY;
  answer(200, withoutLease);
  await assert.rejects(client().activity(), (error: unknown) => {
    assert.ok(error instanceof CrucibleProtocolError, `got ${String(error)}`);
    assert.match(error.message, /lease/);
    return true;
  });
});
