import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { API_VERSION, crucibleAddress, looksLikeLanAddress, startPairing, pollPairing, CrucibleConnectionError, CrucibleClient, CrucibleProtocolError } from '../src/index.js';

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
      response.end(JSON.stringify({ name: 'fixture', id: 'fixture-id', device_code: 'private-device-secret', user_code: 'ABCD-EFGH', expires_in: 300, interval: 2, approval_required: true }));
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
    assert.equal(request.approvalRequired, true);
    assert.deepEqual(await pollPairing(request), { status: 'pending' });
    approved = true;
    assert.deepEqual(await pollPairing(request), { status: 'approved', pairing: { name: 'fixture', url: request.url, token: 'approved-token' } });
  } finally { server.close(); server.closeAllConnections(); }
});

test('approval_required crosses the seam, both ways', async () => {
  // THE FIELD THE SERVER SENDS AND NO CLIENT COULD READ. `startPairing` builds
  // its result explicitly, so until 2026-09-18 `approval_required` arrived on
  // the wire and was dropped here — an engine with open pairing looked exactly
  // like one demanding approval, and every app had to word its connect screen
  // for both at once.
  for (const [sent, expected] of [
    [{ approval_required: false }, false],
    [{ approval_required: true }, true],
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
  let shape: 'whole' | 'no-code' = 'whole';
  const server = createServer(async (request, response) => {
    assert.equal(request.headers.authorization, 'Bearer trusted-token');
    assert.equal(request.headers['x-crucible-api'], '1');
    response.setHeader('Content-Type', 'application/json');
    if (request.url === '/v1/pairing/requests') {
      assert.equal(request.method, 'GET');
      response.end(JSON.stringify({ requests: [{ id: 'id', ...(shape === 'no-code' ? {} : { user_code: 'ABCD-EFGH' }), client_name: 'BookForge', address: '192.0.2.1', expires_in: 42 }] }));
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
    // The code IS the approval's subject: a request without one is refused.
    shape = 'no-code';
    await assert.rejects(client.listPairingRequests(), /has no field "user_code"/);
  } finally { server.close(); server.closeAllConnections(); }
});


// A friend's laptop, 2026-10-09: Crucible in its WSL2 guest, bound to loopback, and B-Side
// on another device "refused with no hint". From that device nothing answers at all (no
// listener on the LAN address, and Windows Firewall drops the SYN unanswered), so the server
// never sees the attempt and cannot name the cause. The client is the one place it can be.
const refusedFetch = (async () => { throw new TypeError('fetch failed'); }) as unknown as typeof fetch;

const silentFetch = ((_input: unknown, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
  init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), { once: true });
})) as typeof fetch;

async function refusal(run: () => Promise<unknown>): Promise<CrucibleConnectionError> {
  try {
    await run();
  } catch (error) {
    assert.ok(error instanceof CrucibleConnectionError, `got ${String(error)}`);
    return error;
  }
  assert.fail('it answered');
}

test('a LAN address that refuses names `crucible lan enable`, the Public network and Local Network access', async () => {
  const error = await refusal(() => startPairing('192.168.68.40', 'B-Side', { fetch: refusedFetch }));
  assert.equal(error.code, 'connection_unreachable');
  assert.match(error.message, /^Nothing answered at 192\.168\.68\.40:7100\./);
  assert.match(error.message, /`crucible lan enable` in PowerShell/);
  assert.match(error.message, /Share on your network/);
  assert.match(error.message, /marked Public/);
  assert.match(error.message, /Local Network/);
});

test('a LAN address that never answers is a timeout, named as one, with the same hint', async () => {
  const error = await refusal(() => startPairing('kylies-pc', 'B-Side', { fetch: silentFetch, timeoutMs: 30 }));
  assert.equal(error.code, 'connection_timed_out', 'a firewall drop is not a cancel');
  assert.match(error.message, /^Nothing answered within 0 s at kylies-pc:7100\./);
  assert.match(error.message, /crucible lan enable/);
});

test('a cancel by the caller is a cancel, not a network fault', async () => {
  const controller = new AbortController();
  const pending = refusal(() => startPairing('192.168.1.9', 'B-Side', { fetch: silentFetch, signal: controller.signal }));
  controller.abort();
  const error = await pending;
  assert.equal(error.code, 'connection_cancelled');
  assert.doesNotMatch(error.message, /lan enable/);
});

test('an address beyond the local network is not told about the Windows LAN door', async () => {
  const error = await refusal(() => startPairing('engine.example.com', 'B-Side', { fetch: refusedFetch }));
  assert.equal(error.code, 'connection_unreachable');
  assert.doesNotMatch(error.message, /lan enable/);
  assert.match(error.message, /^Nothing answered at engine\.example\.com:7100\. Check the address/);
});

test('something that answers but not with JSON is not a Crucible, not an unreachable one', async () => {
  const html = (async () => new Response('<html>router login</html>', { status: 200 })) as unknown as typeof fetch;
  const error = await refusal(() => startPairing('192.168.1.1', 'B-Side', { fetch: html }));
  assert.equal(error.code, 'not_crucible');
  assert.match(error.message, /192\.168\.1\.1:7100 answered HTTP 200, but not as a Crucible/);
});

test('what counts as an address on this network', () => {
  for (const url of ['http://192.168.68.40:7100', 'http://10.0.0.5:7100', 'http://172.20.1.2:7100',
    'http://169.254.3.4:7100', 'http://mac-studio.local:7100', 'http://kylies-pc:7100',
    'http://[fd00::1]:7100', 'http://[fe80::1]:7100']) {
    assert.equal(looksLikeLanAddress(url), true, url);
  }
  for (const url of ['http://100.64.0.3:7100', 'http://8.8.8.8:7100', 'http://172.32.0.1:7100',
    'http://engine.example.com', 'http://localhost:7100', 'http://[2001:db8::1]:7100']) {
    assert.equal(looksLikeLanAddress(url), false, url);
  }
});
