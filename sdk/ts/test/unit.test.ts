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
  CrucibleClient,
  CrucibleConfigError,
  CrucibleNotACrucible,
  CrucibleProtocolError,
  CrucibleRefused,
  CrucibleServerError,
  CrucibleUnreachable,
  CrucibleVersionError,
  SDK_VERSION,
} from '../src/index.js';

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
  answer(200, { status: 'ok', queue_depth: 0, resident_models: [] });
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
