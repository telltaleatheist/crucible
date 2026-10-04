import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import { CrucibleClient, CrucibleUnreachable } from '../src/index.js';

// A stream that breaks mid-read is the same weather as a connect that fails: it must come
// out as CrucibleUnreachable, not as the platform's raw error (undici's "terminated",
// WebKit's "Load failed") - B-Side on an iPhone, 2026-10-04: a locked phone gave up on a
// job the server finished, because the client retried only on CrucibleUnreachable.

let server: Server;
let base = '';

before(async () => {
  server = createServer((request, response) => {
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    if ((request.url ?? '').startsWith('/v1/openai/chat/completions')) {
      response.write('data: {"choices":[{"delta":{"content":"half an "}}]}\n\n');
    } else {
      response.write('id: 1\nevent: queued\ndata: {"position":1,"of":1}\n\n');
    }
    // Then the connection dies with no terminal event and no clean end.
    setTimeout(() => request.socket.destroy(), 50);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  server.closeAllConnections();
  await new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
});

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-stream-drop' });
}

test('a job stream cut off mid-read is CrucibleUnreachable, naming the event to resume after', async () => {
  const seen: number[] = [];
  await assert.rejects(
    (async () => {
      for await (const event of client().events('job-1')) seen.push(event.id);
    })(),
    (error: unknown) => error instanceof CrucibleUnreachable && /broke after event 1/.test(error.message),
  );
  assert.deepEqual(seen, [1]);
});

test('a chat stream cut off mid-answer is CrucibleUnreachable that says the answer is truncated', async () => {
  const parts: string[] = [];
  await assert.rejects(
    (async () => {
      for await (const part of client().chatStream({ model: 'm', messages: [{ role: 'user', content: 'hi' }] })) {
        parts.push(part);
      }
    })(),
    (error: unknown) => error instanceof CrucibleUnreachable && /truncated/.test(error.message),
  );
  assert.deepEqual(parts, ['half an ']);
});
