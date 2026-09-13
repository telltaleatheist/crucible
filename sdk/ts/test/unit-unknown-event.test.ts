/**
 * An event kind this build does not know must not end the stream.
 *
 * The server's event vocabulary grows without moving `api_version`. `chunk`
 * arrived for `tts` on exactly that argument, written into PHASE3-TTS.md section
 * 6: "a client that does not know the kind still sees every `progress`,
 * `artifact` and `done` it saw before. The SDK yields unknown event kinds
 * through rather than dropping them, so this stays true for the next one too."
 *
 * It was not true. `readEvent` narrowed the name against a closed list and threw,
 * so a client built before a kind existed lost the WHOLE stream at the first
 * frame carrying it — an 0.2.0 client watching any job at all on a server that
 * had learned `chunk`, and the same again for phase 4's kinds. The contract
 * asserted a property the code did not have, which is the worst kind of contract
 * sentence: it reads as a guarantee and nobody checks it.
 *
 * These tests are that check. They deliberately use a name no version of this
 * server has ever sent, because a test written against `chunk` would pass the
 * day `chunk` was added to the client and stop testing anything.
 *
 * Run: `npm run test:unit` (exits non-zero on any failure).
 */

import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import { CrucibleClient, CrucibleProtocolError, type JobEvent } from '../src/index.js';

const TOKEN = 'test-token';

/** An SSE stream carrying one kind from the future, between two ordinary ones. */
const STREAM = [
  'id: 1\nevent: queued\ndata: {"position": 1}\n\n',
  'id: 2\nevent: aurora\ndata: {"colour": "green", "lumens": 4}\n\n',
  'id: 3\nevent: progress\ndata: {"fraction": 0.5, "message": "halfway"}\n\n',
  'id: 4\nevent: done\ndata: {"artifacts": ["41.flac"]}\n\n',
].join('');

let server: Server;
let url: string;

before(async () => {
  server = createServer((request, response) => {
    if (request.url === '/v1/jobs/j1/events') {
      response.writeHead(200, { 'Content-Type': 'text/event-stream' });
      response.end(STREAM);
      return;
    }
    response.writeHead(404).end('{}');
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

function client(): CrucibleClient {
  return new CrucibleClient({ url, token: TOKEN, clientName: 'test' });
}

test('an unknown event kind is carried through, not thrown', async () => {
  const seen: JobEvent[] = [];
  for await (const event of client().events('j1')) seen.push(event);

  assert.deepEqual(
    seen.map((e) => e.event),
    ['queued', 'unknown', 'progress', 'done'],
    'the stream must reach its real ending past a kind it does not know',
  );
});

test('the unknown event keeps the name and the data the server sent', async () => {
  const seen: JobEvent[] = [];
  for await (const event of client().events('j1')) seen.push(event);

  const frame = seen[1];
  assert.ok(frame, 'the stream had a second event');
  // Narrowing on `event` rather than casting: if the union ever loses its
  // `unknown` arm this stops compiling, which is the point.
  if (frame.event !== 'unknown') {
    throw new Error(`expected an unknown event, got ${frame.event}`);
  }
  assert.equal(frame.id, 2);
  assert.equal(frame.kind, 'aurora', 'the server said `aurora`; say so');
  assert.deepEqual(frame.data, { colour: 'green', lumens: 4 });
});

test('an unknown kind is never terminal', async () => {
  const seen: JobEvent[] = [];
  for await (const event of client().events('j1')) seen.push(event);

  assert.equal(
    seen.at(-1)?.event,
    'done',
    'ending the stream on an unknown kind would lose every event after it, ' +
      'which is the failure this whole file exists to stop',
  );
});

test('strictness about a kind the client DOES claim is untouched', async () => {
  // The line this draws: tolerant of kinds it makes no claim about, strict about
  // every field of every kind it does. A `progress` frame with no `fraction` is
  // still a protocol error, and must stay one.
  const strict = createServer((request, response) => {
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    response.end('id: 1\nevent: progress\ndata: {"message": "no fraction here"}\n\n');
  });
  await new Promise<void>((resolve) => strict.listen(0, '127.0.0.1', resolve));
  const strictUrl = `http://127.0.0.1:${(strict.address() as AddressInfo).port}`;

  try {
    const c = new CrucibleClient({ url: strictUrl, token: TOKEN, clientName: 'test' });
    await assert.rejects(async () => {
      for await (const _ of c.events('j1')) void _;
    }, CrucibleProtocolError);
  } finally {
    await new Promise<void>((resolve) => strict.close(() => resolve()));
  }
});
