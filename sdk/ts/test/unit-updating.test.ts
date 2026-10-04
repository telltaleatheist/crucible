import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, beforeEach, test } from 'node:test';

import { CrucibleClient, CrucibleUpdating } from '../src/index.js';

// A deploy holds the server before restarting it (crucible/updating.py): every door that creates
// work answers `503 server_updating` and admitted nothing. The client waits that out and sends
// again once the restarted server answers - B-Side's phone song should not die of a deploy.

let server: Server;
let base = '';
let log: string[] = [];
let updatingFor = 0;
let pingsDown = 0;

before(async () => {
  server = createServer((request, response) => {
    const url = request.url ?? '';
    if (url === '/v1/ping') {
      log.push('ping');
      if (pingsDown > 0) {
        pingsDown -= 1;
        request.socket.destroy(); // the old server has stopped; the new one is not up yet
        return;
      }
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end('{"ok":true}');
      return;
    }
    log.push(`${request.method} ${url}`);
    if (updatingFor > 0) {
      updatingFor -= 1;
      response.writeHead(503, { 'Content-Type': 'application/json', 'Retry-After': '1' });
      response.end(
        JSON.stringify({
          error: {
            code: 'server_updating',
            message: 'this server is about to restart for an update to 9.9.9',
            details: { release: '9.9.9', until: '2026-10-04T20:00:00+00:00', retry_after: 1 },
          },
        }),
      );
      return;
    }
    response.writeHead(202, { 'Content-Type': 'application/json' });
    response.end('{"job_id":"j-1","resume_id":null}');
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

beforeEach(() => {
  log = [];
  updatingFor = 0;
  pingsDown = 0;
});

after(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
});

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-updating' });
}

const job = { type: 'echo', params: {}, inputs: {} };

test('a submit refused for an update is sent again once the restarted server answers', async () => {
  updatingFor = 1;
  pingsDown = 2;
  assert.equal(await client().submit(job), 'j-1');
  // Refused, then pings until the new server answers, and only then the submit again.
  assert.deepEqual(log, ['POST /v1/jobs', 'ping', 'ping', 'ping', 'POST /v1/jobs']);
});

test('a caller that gives up while waiting sees the update by name', async () => {
  updatingFor = 5;
  const signal = AbortSignal.timeout(300);
  await assert.rejects(
    client().submit(job, { signal }),
    (error: unknown) => error instanceof CrucibleUpdating && error.release === '9.9.9' && error.status === 503,
  );
  assert.deepEqual(log, ['POST /v1/jobs']);
});
