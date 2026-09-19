import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { API_VERSION, crucibleAddress, startPairing, pollPairing, CrucibleConnectionError, CrucibleClient, CrucibleProtocolError } from '../src/index.js';

test('IP and hostname discovery uses the canonical port, preserving explicit ports and HTTPS', () => {
  assert.equal(crucibleAddress('192.168.1.9'), 'http://192.168.1.9:7100');
  assert.equal(crucibleAddress('mac.local'), 'http://mac.local:7100');
  assert.equal(crucibleAddress('http://mac.local:80'), 'http://mac.local');
  assert.equal(crucibleAddress('mac.local:7200'), 'http://mac.local:7200');
  assert.equal(crucibleAddress('https://engine.example'), 'https://engine.example');
  assert.equal(crucibleAddress('::1'), 'http://[::1]:7100');
  assert.equal(crucibleAddress('[::1]:7300'), 'http://[::1]:7300');
  for (const raw of ['', 'host/path', 'user@host', 'host#token', 'host?secret', 'ftp://host', 'host:0', 'host:65536']) {
    assert.throws(() => crucibleAddress(raw), CrucibleConnectionError);
  }
});

test('real HTTP pairing keeps the device secret in the body and returns credentials only after approval', async () => {
  let approved = false;
  const server = createServer(async (request, response) => {
    assert.equal(request.headers['x-crucible-api'], '1');
    assert.equal(request.headers.authorization, undefined);
    response.setHeader('Content-Type', 'application/json');
    if (request.url === '/v1/ping') {
      response.end(JSON.stringify({ crucible: true, name: 'fixture', api_version: 1, pairing_version: 1 }));
      return;
    }
    let raw = '';
    for await (const part of request) raw += part;
    const body = JSON.parse(raw);
    if (request.url === '/v1/pairing/start') {
      assert.deepEqual(body, { client_name: 'BookForge' });
      response.end(JSON.stringify({ name: 'fixture', id: 'fixture-id', device_code: 'private-device-secret', user_code: 'ABCD-EFGH', expires_in: 300, interval: 2 }));
      return;
    }
    assert.equal(request.url, '/v1/pairing/poll');
    assert.deepEqual(body, { id: 'fixture-id', device_code: 'private-device-secret' });
    response.end(JSON.stringify(approved ? { status: 'approved', name: 'fixture', token: 'approved-token' } : { status: 'pending' }));
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  try {
    const address = server.address() as { port: number };
    const request = await startPairing(`127.0.0.1:${address.port}`, 'BookForge');
    assert.equal(request.userCode, 'ABCD-EFGH');
    // This fixture sends NO `approval_required`, which is an engine older than
    // the field — and every one of those asked for approval. Asserted here
    // rather than left incidental, because it is the safe half: a client that
    // read `false` from an old engine would skip a step that engine requires.
    assert.equal(request.approvalRequired, true);
    assert.deepEqual(await pollPairing(request), { status: 'pending' });
    approved = true;
    assert.deepEqual(await pollPairing(request), { status: 'approved', pairing: { name: 'fixture', url: request.url, token: 'approved-token' } });
  } finally { server.close(); server.closeAllConnections(); }
});

test('approval_required crosses the seam, both ways and when absent', async () => {
  // THE FIELD THE SERVER SENDS AND NO CLIENT COULD READ. `startPairing` builds
  // its result explicitly, so until 2026-09-18 `approval_required` arrived on
  // the wire and was dropped here — an engine with open pairing looked exactly
  // like one demanding approval, and every app had to word its connect screen
  // for both at once.
  //
  // Three cases, because absence is a third answer and not a missing one.
  for (const [sent, expected] of [
    [{ approval_required: false }, false],
    [{ approval_required: true }, true],
    [{}, true],
  ] as const) {
    const fake = (async (input: unknown) => {
      const url = String(input);
      if (url.endsWith('/v1/ping')) {
        // `pairing_version: 1` is the precondition startPairing checks; `pairing: true`
        // is not a field it reads, and a fixture that guessed got pairing_unavailable.
        return new Response(JSON.stringify({ crucible: true, name: 'fixture', api_version: 1, pairing_version: 1 }));
      }
      return new Response(JSON.stringify({
        name: 'fixture', id: 'fixture-id', device_code: 'private-device-secret',
        user_code: 'ABCD-EFGH', expires_in: 300, interval: 2, ...sent,
      }));
    }) as typeof fetch;
    const request = await startPairing('fixture', 'app', { fetch: fake });
    assert.equal(request.approvalRequired, expected,
      `approval_required ${JSON.stringify(sent)} should read as ${expected}`);
  }
});

/**
 * INSTALL-UNINSTALL.md §6.5.5: compatibility is the API VERSION, and a Crucible
 * that speaks another one is refused BY NAME rather than folded into
 * `not_crucible`, which said "this address is not a compatible Crucible engine"
 * about a Crucible and told nobody which half was wrong. The SDK's own
 * `SDK_VERSION` never enters it — a client and a server at different builds of
 * one API version talk to each other, which is the whole point of the header.
 */
test('a Crucible speaking another api_version is refused by name, saying both versions', async () => {
  let calls = 0;
  const fake = (async () => {
    calls++;
    return new Response(JSON.stringify({ crucible: true, name: 'future', api_version: 2, pairing_version: 1 }));
  }) as typeof fetch;
  await assert.rejects(startPairing('fixture', 'app', { fetch: fake }), (error: unknown) => {
    assert.ok(error instanceof CrucibleConnectionError);
    assert.equal(error.code, 'api_version_mismatch');
    assert.match(error.message, /API version 2/);
    assert.match(error.message, new RegExp(`speaks ${API_VERSION}`));
    return true;
  });
  assert.equal(calls, 1, 'nothing is posted to a server this client cannot speak to');
});

test('old engines and wrong services refuse pairing without probing secrets', async () => {
  for (const [ping, code] of [
    [{ crucible: true, name: 'old', api_version: 1 }, 'pairing_unavailable'],
    [{ service: 'unrelated' }, 'not_crucible'],
    [{ crucible: true, name: 'nameless' }, 'not_crucible'],
  ] as const) {
    let calls = 0;
    const fake = (async () => { calls++; return new Response(JSON.stringify(ping)); }) as typeof fetch;
    await assert.rejects(startPairing('fixture', 'app', { fetch: fake }), (error: unknown) => error instanceof CrucibleConnectionError && error.code === code);
    assert.equal(calls, 1);
  }
});

test('denied and expired requests never manufacture a connection', async () => {
  const request = { url: 'http://fixture:7100', name: 'fixture', id: 'id', deviceCode: 'secret', userCode: 'ABCD-EFGH', expiresIn: 300, interval: 2, approvalRequired: true };
  for (const status of ['denied', 'expired'] as const) {
    const fake = (async () => new Response(JSON.stringify({ status }))) as typeof fetch;
    assert.deepEqual(await pollPairing(request, { fetch: fake }), { status });
  }
  const wrong = (async () => new Response(JSON.stringify({ status: 'approved', name: 'changed', token: 'secret' }))) as typeof fetch;
  await assert.rejects(pollPairing(request, { fetch: wrong }), CrucibleConnectionError);
});

test('trusted apps list and approve pairing through authenticated API calls', async () => {
  let malformed = false;
  const server = createServer(async (request, response) => {
    assert.equal(request.headers.authorization, 'Bearer trusted-token');
    assert.equal(request.headers['x-crucible-api'], '1');
    response.setHeader('Content-Type', 'application/json');
    if (request.url === '/v1/pairing/requests') {
      assert.equal(request.method, 'GET');
      response.end(JSON.stringify({ requests: [{ id: 'id', user_code: 'ABCD-EFGH', client_name: 'BookForge', address: '192.0.2.1', ...(malformed ? {} : { expires_in: 42 }) }] }));
      return;
    }
    assert.equal(request.url, '/v1/pairing/decision');
    assert.equal(request.method, 'POST');
    let raw = '';
    for await (const part of request) raw += part;
    assert.deepEqual(JSON.parse(raw), { id: 'id', user_code: 'ABCD-EFGH', allow: true });
    response.end(JSON.stringify({ status: 'approved' }));
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  try {
    const port = (server.address() as { port: number }).port;
    const client = new CrucibleClient({ url: `http://127.0.0.1:${port}`, token: 'trusted-token', clientName: 'Foundry' });
    assert.deepEqual(await client.listPairingRequests(), [{ id: 'id', userCode: 'ABCD-EFGH', clientName: 'BookForge', address: '192.0.2.1', expiresIn: 42 }]);
    assert.deepEqual(await client.decidePairing('id', 'ABCD-EFGH', true), { status: 'approved' });
    malformed = true;
    await assert.rejects(client.listPairingRequests(), CrucibleProtocolError);
  } finally { server.close(); server.closeAllConnections(); }
});
