/**
 * The stale-pooled-socket retry, and its three boundaries.
 *
 * WHAT THIS IS A KEEPER FOR. align's first `GET /v1/info` after a render's
 * last artifact fetch failed `read ECONNRESET` four times — 2026-09-18 and
 * 2026-09-19 against the PC, 2026-09-20 at 00:57 against the Mac on
 * 127.0.0.1 — with the server up before and after each time. Node's `fetch`
 * (undici) keeps an idle pooled connection about 4 s and uvicorn's default
 * `timeout_keep_alive` is 5 s, so the next request was written onto a socket
 * the server was already closing. Crucible now states `KEEP_ALIVE_SECONDS`
 * on both of its uvicorn doors; this is the other half, because a restart or
 * a proxy can close a pooled connection whenever it likes.
 *
 * `globalThis.fetch` is replaced rather than a server stood up: the whole of
 * what is under test is WHEN the client calls fetch a second time, and a real
 * socket cannot be made to go stale on demand.
 *
 * Run: `npm run test:unit` (this file is found by `scripts/unit.mjs`).
 */

import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';

import { CrucibleClient, CrucibleUnreachable } from '../src/index.js';

const realFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = realFetch;
});

/** What undici raises when a pooled connection was closed under a request. */
function reset(): Error {
  const cause = new Error('read ECONNRESET') as Error & { code: string };
  cause.code = 'ECONNRESET';
  return new TypeError('fetch failed', { cause });
}

/** A `GET /v1/ping` answer the client accepts. */
function pong(): Response {
  return new Response(JSON.stringify({ crucible: true, name: 'crucible@test', api_version: 1 }), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

/** What undici raises when nothing is listening on the port at all. */
function refused(): Error {
  const cause = new Error('connect ECONNREFUSED 127.0.0.1:7100') as Error & { code: string };
  cause.code = 'ECONNREFUSED';
  return new TypeError('fetch failed', { cause });
}

/** Replace `fetch` with one that plays `outcomes` in order. Returns the call log. */
function plan(outcomes: Array<'reset' | 'refused' | Response>): { calls: string[] } {
  const calls: string[] = [];
  let index = 0;
  globalThis.fetch = (async (input: unknown, init?: RequestInit) => {
    calls.push(`${init?.method ?? 'GET'} ${String(input)}`);
    const outcome = outcomes[index++];
    assert.notEqual(outcome, undefined, `fetch was called ${index} times; ${outcomes.length} planned`);
    if (outcome === 'reset') throw reset();
    if (outcome === 'refused') throw refused();
    return outcome as Response;
  }) as typeof fetch;
  return { calls };
}

function client(): CrucibleClient {
  return new CrucibleClient({ url: 'http://127.0.0.1:7100', token: 't', clientName: 'keeper' });
}

test('a GET that meets a stale socket is retried exactly once and succeeds', async () => {
  const log = plan(['reset', pong()]);
  const answer = await client().ping();
  assert.equal(answer.name, 'crucible@test');
  assert.deepEqual(log.calls, [
    'GET http://127.0.0.1:7100/v1/ping',
    'GET http://127.0.0.1:7100/v1/ping',
  ]);
});

test('a POST is never retried, because a reset cannot say whether the server read it', async () => {
  const log = plan(['reset']);
  await assert.rejects(
    () => client().heartbeat('lease-1'),
    (error: unknown) => error instanceof CrucibleUnreachable,
  );
  assert.deepEqual(log.calls, ['POST http://127.0.0.1:7100/v1/leases/lease-1/heartbeat']);
});

test('a GET that resets twice surfaces the error rather than looping', async () => {
  const log = plan(['reset', 'reset']);
  await assert.rejects(
    () => client().ping(),
    (error: unknown) => error instanceof CrucibleUnreachable,
  );
  assert.equal(log.calls.length, 2, log.calls.join(' | '));
});

test('a transport failure that is not a stale socket is not retried at all', async () => {
  // The boundary that keeps this a reconnect and not a general retry: a
  // refused connection means nothing is listening, and asking twice is just a
  // client that takes twice as long to say so.
  const log = plan(['refused']);
  await assert.rejects(
    () => client().ping(),
    (error: unknown) => error instanceof CrucibleUnreachable,
  );
  assert.equal(log.calls.length, 1, log.calls.join(' | '));
});
