/**
 * A Crucible that asks for no token (`[auth] mode = "none"`, docs/DESIGN.md section 9).
 *
 * Owen, 2026-10-11: "lets make it so b sides and any other claude agent is capable of making
 * calls to crucible without requiring the access token ... ollama doesnt require access tokens
 * for calling its api". So a client made with no `token` sends no `Authorization` header at all
 * (never an empty `Bearer `), `ping` says which kind of server it reached before anybody is
 * asked for anything, and a pairing or setup from such a server carries a null token. A client
 * WITH a token still sends it, so one that stored a token keeps working against both kinds.
 *
 * Run: `npm run test:unit`.
 */

import assert from 'node:assert/strict';
import { createServer, type IncomingHttpHeaders, type Server } from 'node:http';
import { once } from 'node:events';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import { CrucibleClient, CrucibleConfigError, pollPairing, type PairingRequest } from '../src/index.js';

let server: Server;
let base = '';
let seen: IncomingHttpHeaders = {};
let body: unknown = {};

before(async () => {
  server = createServer((request, response) => {
    seen = request.headers;
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify(body));
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

const HEALTH = { status: 'ok', queue_depth: 0, resident_models: [], resident_kind: null, stopping: null };

test('a client with no token sends no Authorization header, and still the api header', async () => {
  body = HEALTH;
  await new CrucibleClient({ url: base, clientName: 'b-sides' }).health();
  assert.equal(seen['authorization'], undefined);
  assert.equal(seen['x-crucible-api'], '1');
});

test('a null token is the same as none', async () => {
  body = HEALTH;
  await new CrucibleClient({ url: base, token: null, clientName: 'b-sides' }).health();
  assert.equal(seen['authorization'], undefined);
});

test('a client with a token still sends it', async () => {
  body = HEALTH;
  await new CrucibleClient({ url: base, token: 'kept', clientName: 'b-sides' }).health();
  assert.equal(seen['authorization'], 'Bearer kept');
});

test('an empty token is refused by name rather than sent as a blank bearer', () => {
  assert.throws(() => new CrucibleClient({ url: base, token: '', clientName: 'c' }), CrucibleConfigError);
});

test('ping says whether the server asks for a token, and refuses a word it does not know', async () => {
  const client = new CrucibleClient({ url: base, clientName: 'b-sides' });
  body = { crucible: true, name: 'crucible@pc', api_version: 1, pairing_version: 1, auth: 'none' };
  assert.equal((await client.ping()).auth, 'none');
  body = { ...(body as object), auth: 'token' };
  assert.equal((await client.ping()).auth, 'token');
  body = { ...(body as object), auth: 'maybe' };
  await assert.rejects(() => client.ping(), /ping\.auth/);
});

test('setup from a server that asks for none carries a null token and token-less lines', async () => {
  body = {
    name: 'crucible@pc', version: '2.0.2', backend: 'cuda-linux', bind: 'http://0.0.0.0:7100',
    urls: ['http://192.168.68.20:7100'], auth: 'none', token: null,
    pairing: ['crucible://crucible%40pc@192.168.68.20:7100/'], job_types: ['llm'],
    config_path: '/home/x/.crucible/config.toml',
  };
  const setup = await new CrucibleClient({ url: base, clientName: 'b-sides' }).setup();
  assert.equal(setup.auth, 'none');
  assert.equal(setup.token, null);
  assert.deepEqual(setup.pairing, ['crucible://crucible%40pc@192.168.68.20:7100/']);
});

test('an approved pairing with a null token is a pairing with no token', async () => {
  const request: PairingRequest = {
    url: base, name: 'crucible@pc', id: 'i', deviceCode: 'd', userCode: 'ABCD-EFGH',
    expiresIn: 300, interval: 2, approvalRequired: false,
  };
  body = { status: 'approved', name: 'crucible@pc', token: null };
  assert.deepEqual(await pollPairing(request), {
    status: 'approved', pairing: { name: 'crucible@pc', url: base, token: null },
  });
  body = { status: 'approved', name: 'crucible@pc', token: '' };
  await assert.rejects(() => pollPairing(request), /incompatible pairing decision/);
});
