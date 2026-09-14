/**
 * `stream()` — the live TTS session. PHASE3-TTS.md section 7.
 *
 * Against an in-process `node:http` fixture rather than a live Crucible, for
 * the reason `unit-voices.test.ts` gives about its own: a fixture answers with
 * whatever bytes a test wants, which is how the cases a healthy server never
 * produces on a good day get proved. Here that is most of the file — a
 * connection destroyed mid-row, a reattach that must carry `Last-Event-ID`, a
 * `restart` frame, an event kind the client does not know, and an id that does
 * not follow the last one.
 *
 * Run: `npm run test:unit` (exits non-zero on any failure).
 */

import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleError,
  CrucibleProtocolError,
  CrucibleRefused,
  CrucibleUnreachable,
  type StreamEvent,
  type TtsStreamSession,
} from '../src/index.js';
import { decodeBase64, encodeBase64 } from '../src/base64.js';

// ------------------------------------------------------------------ fixture

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => {
  response.writeHead(500, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: { code: 'no_handler', message: 'the test set none' } }));
};

/** Every request the fixture has seen, so a test can assert on what was sent. */
let seen: { method: string; path: string; body: string; lastEventId?: string }[] = [];

let server: Server;
let base = '';

const SESSION = {
  session_id: 'cafebabe',
  voice: 'deathstalker',
  fingerprint: 'deathstalker@0123456789abcdef0123456789abcdef01234567',
  sample_rate: 24000,
  backend: 'cuda-linux',
};

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-stream' });
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

function openSse(response: ServerResponse): void {
  response.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-store',
  });
}

function frame(response: ServerResponse, id: number, event: string, data: unknown): void {
  response.write(`id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

/**
 * Write one frame and then kill the socket **once it has actually gone out**.
 *
 * `socket.destroy()` straight after a `write()` throws away whatever is still
 * in the buffer, so the client never sees the frame it is supposed to resume
 * from and asks to reattach from zero — which looks exactly like a reattach
 * that works and proves nothing. Measured 2026-09-13: the first version of the
 * test below reattached with no `Last-Event-ID` at all. The write callback
 * fires when the bytes have reached the OS, and killing it there is a tunnel
 * collapsing mid-row.
 */
function frameThenDie(
  request: IncomingMessage,
  response: ServerResponse,
  id: number,
  event: string,
  data: unknown,
): void {
  response.write(
    `id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`,
    () => request.socket.destroy(),
  );
}

/** A tone, as the fake narrator makes one: predictable bytes a test can check. */
function pcmBytes(samples: number[]): string {
  const bytes = new Uint8Array(samples.length * 2);
  const view = new DataView(bytes.buffer);
  samples.forEach((value, index) => view.setInt16(index * 2, value, true));
  return encodeBase64(bytes);
}

const READY = { voice: SESSION.voice, fingerprint: SESSION.fingerprint, sample_rate: 24000, backend: 'cuda-linux' };

before(async () => {
  server = createServer((request, response) => {
    const chunks: Buffer[] = [];
    request.on('data', (chunk: Buffer) => chunks.push(chunk));
    request.on('end', () => {
      const body = Buffer.concat(chunks).toString('utf8');
      const entry: { method: string; path: string; body: string; lastEventId?: string } = {
        method: request.method ?? '',
        path: request.url ?? '',
        body,
      };
      const resume = request.headers['last-event-id'];
      if (typeof resume === 'string') entry.lastEventId = resume;
      seen.push(entry);
      handle(request, response, body);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

function reset(): void {
  seen = [];
}

/**
 * The commonest fixture: open answers the session, ops answer `answer`, and the
 * event stream is whatever `events` writes.
 */
function serving(
  events: (response: ServerResponse) => void,
  answer: (body: string) => unknown = () => ({}),
): void {
  handle = (request, response, body) => {
    const path = request.url ?? '';
    if (path === '/v1/tts/stream' && request.method === 'POST') {
      json(response, 201, SESSION);
      return;
    }
    if (path.endsWith('/events')) {
      openSse(response);
      events(response);
      return;
    }
    if (request.method === 'DELETE') {
      json(response, 200, { session_id: SESSION.session_id, closed: true });
      return;
    }
    json(response, 202, answer(body));
  };
}

async function collect(session: TtsStreamSession): Promise<StreamEvent[]> {
  const events: StreamEvent[] = [];
  for await (const event of session) events.push(event);
  return events;
}

/** The ops posted on the session, in order — `say`, `cancel`, `cancel_all`. */
function ops(): typeof seen {
  return seen.filter(
    (entry) => entry.method === 'POST' && entry.path === `/v1/tts/stream/${SESSION.session_id}`,
  );
}

// ------------------------------------------------------------------ opening


test('stream() opens a session and answers the identity of what will speak', async () => {
  reset();
  serving((response) => {
    frame(response, 1, 'ready', READY);
    frame(response, 2, 'closed', { reason: 'done' });
    response.end();
  });
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  assert.equal(session.sessionId, SESSION.session_id);
  assert.equal(session.voice, 'deathstalker');
  // The fingerprint binds what is heard to a merge rather than to a name.
  assert.equal(session.fingerprint, SESSION.fingerprint);
  assert.equal(session.sampleRate, 24000);
  assert.equal(session.backend, 'cuda-linux');
  assert.deepEqual(JSON.parse(seen[0]!.body), { voice: 'deathstalker', language: 'en' });
  await collect(session);
});

test('stream() needs both a voice and a language, and neither has a default', async () => {
  reset();
  serving(() => undefined);
  const crucible = client();
  await assert.rejects(
    () => crucible.stream({ language: 'en' } as never),
    (error: unknown) => error instanceof CrucibleError && /voice/.test((error as Error).message),
  );
  await assert.rejects(
    () => crucible.stream({ voice: 'deathstalker' } as never),
    (error: unknown) => error instanceof CrucibleError && /language/.test((error as Error).message),
  );
});

test('a refusal to open travels back by name', async () => {
  reset();
  handle = (_request, response) => {
    json(response, 409, {
      error: {
        code: 'voice_not_resident',
        message: "'deathstalker' is not resident on this server; no voice is",
      },
    });
  };
  await assert.rejects(
    () => client().stream({ voice: 'deathstalker', language: 'en' }),
    (error: unknown) =>
      error instanceof CrucibleRefused && error.code === 'voice_not_resident',
  );
});

// -------------------------------------------------------------------- ops

test('say posts the op and answers the row id, never the audio', async () => {
  reset();
  serving(
    (response) => {
      frame(response, 1, 'ready', READY);
      frame(response, 2, 'closed', { reason: 'done' });
      response.end();
    },
    () => ({ id: 'r1' }),
  );
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  assert.equal(await session.say('r1', 'He had been walking for some time.'), 'r1');
  const op = JSON.parse(ops()[0]!.body);
  assert.deepEqual(op, {
    op: 'say',
    id: 'r1',
    text: 'He had been walking for some time.',
    take: 0,
  });
  await collect(session);
});

test('say sends the take it is given and refuses one that is not a rung', async () => {
  reset();
  serving(
    (response) => {
      frame(response, 1, 'ready', READY);
      frame(response, 2, 'closed', { reason: 'done' });
      response.end();
    },
    () => ({ id: 'r1' }),
  );
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  await session.say('r1', 'Rain.', 2);
  assert.equal(JSON.parse(ops()[0]!.body).take, 2);
  await assert.rejects(() => session.say('r2', 'Rain.', -1));
  await assert.rejects(() => session.say('r3', 'Rain.', 1.5));
  await collect(session);
});

test('cancel answers what it cost, and an outcome this client does not know is a protocol error', async () => {
  reset();
  let outcome = 'aborting_batch';
  serving(
    (response) => {
      frame(response, 1, 'ready', READY);
      frame(response, 2, 'closed', { reason: 'done' });
      response.end();
    },
    (body) => (JSON.parse(body).op === 'cancel' ? { id: 'r1', outcome } : { cancelled: 3 }),
  );
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  // The three outcomes name three different costs: nothing, a whole batch, and
  // a race that was already over.
  assert.equal(await session.cancel('r1'), 'aborting_batch');
  outcome = 'dropped';
  assert.equal(await session.cancel('r1'), 'dropped');
  outcome = 'already_finished';
  assert.equal(await session.cancel('r1'), 'already_finished');
  outcome = 'sort_of';
  await assert.rejects(() => session.cancel('r1'), CrucibleProtocolError);
  assert.equal(await session.cancelAll(), 3);
  await collect(session);
});

test('close tolerates a session the grace window already closed', async () => {
  reset();
  handle = (request, response) => {
    if ((request.url ?? '') === '/v1/tts/stream' && request.method === 'POST') {
      json(response, 201, SESSION);
      return;
    }
    if ((request.url ?? '').endsWith('/events')) {
      openSse(response);
      frame(response, 1, 'ready', READY);
      frame(response, 2, 'closed', { reason: 'done' });
      response.end();
      return;
    }
    json(response, 404, {
      error: { code: 'unknown_session', message: 'there is no streaming session' },
    });
  };
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  // Nothing to close is not a failure to close: a client that raced the
  // watchdog has still ended up where it wanted to be.
  await session.close();
});

// ------------------------------------------------------------------ frames

test('iterating yields decoded audio, the row retiring, and then ends on closed', async () => {
  reset();
  const samples = [0, 8000, -8000, 32767, -32768];
  serving((response) => {
    frame(response, 1, 'ready', READY);
    frame(response, 2, 'audio', {
      id: 'r1',
      seq: 0,
      pcm_base64: pcmBytes(samples),
      seconds: samples.length / 24000,
    });
    frame(response, 3, 'done', {
      id: 'r1',
      seconds: 0.2,
      chars: 34,
      chars_per_sec: 170,
      capped: null,
      cancelled: false,
    });
    frame(response, 4, 'closed', { reason: 'the client closed the session' });
    response.end();
  });
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  const events = await collect(session);

  assert.equal(events.length, 2, JSON.stringify(events));
  const audio = events[0]!;
  assert.equal(audio.kind, 'audio');
  if (audio.kind !== 'audio') throw new Error('unreachable');
  assert.equal(audio.id, 'r1');
  assert.equal(audio.seq, 0);
  // Read as little-endian regardless of the host's byte order, and the extremes
  // are in the sample set on purpose: a sign error only shows at -32768.
  assert.deepEqual(Array.from(audio.pcm), samples);

  const done = events[1]!;
  if (done.kind !== 'done') throw new Error('expected a done');
  assert.equal(done.cancelled, false);
  assert.equal(done.chars, 34);
  // `null` means narrator did not say, and must never be read as `false`.
  assert.equal(done.capped, null);
});

test('a restart frame says which of a row s audio is void', async () => {
  reset();
  serving((response) => {
    frame(response, 1, 'ready', READY);
    frame(response, 2, 'audio', { id: 'r1', seq: 0, pcm_base64: pcmBytes([1, 2]), seconds: 0.1 });
    frame(response, 3, 'restart', {
      id: 'r1',
      from_seq: 1,
      reason: 'the batch this row was in was aborted to cancel another row',
    });
    frame(response, 4, 'audio', { id: 'r1', seq: 1, pcm_base64: pcmBytes([3, 4]), seconds: 0.1 });
    frame(response, 5, 'closed', { reason: 'done' });
    response.end();
  });
  const events = await collect(
    await client().stream({ voice: 'deathstalker', language: 'en' }),
  );
  const restart = events[1]!;
  if (restart.kind !== 'restart') throw new Error('expected a restart');
  assert.equal(restart.fromSeq, 1);
  // seq never restarts, so the void is a prefix and everything at or above
  // `fromSeq` is the audio to keep.
  const kept = events.filter(
    (event) => event.kind === 'audio' && event.seq >= restart.fromSeq,
  );
  assert.equal(kept.length, 1);
});

test('an error naming a row is yielded; one naming none ends the session', async () => {
  reset();
  serving((response) => {
    frame(response, 1, 'ready', READY);
    frame(response, 2, 'error', { id: 'r1', code: 'row_failed', message: 'no audio generated' });
    frame(response, 3, 'closed', { reason: 'done' });
    response.end();
  });
  const events = await collect(
    await client().stream({ voice: 'deathstalker', language: 'en' }),
  );
  assert.equal(events.length, 1);
  assert.equal(events[0]!.kind, 'error');

  reset();
  serving((response) => {
    frame(response, 1, 'ready', READY);
    frame(response, 2, 'error', {
      id: null,
      code: 'engine_failed',
      message: 'narrator exited 1 in the middle of a request',
    });
    response.end();
  });
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  await assert.rejects(() => collect(session), CrucibleProtocolError);
});

test('an event kind this client does not know is a refusal, never a skip', async () => {
  reset();
  serving((response) => {
    frame(response, 1, 'ready', READY);
    frame(response, 2, 'phoneme', { id: 'r1' });
    frame(response, 3, 'closed', { reason: 'done' });
    response.end();
  });
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  await assert.rejects(() => collect(session), CrucibleProtocolError);
});

test('a ready that disagrees with the open reply is not this session s stream', async () => {
  // One fact, two copies, compared (ARCHITECTURE.md R1): the identity came back
  // on the open reply and comes again on `ready`, and every field is checked.
  // Refused from `stream()` itself now that the frame is read before it
  // resolves — nobody is handed a session whose stream says something else.
  for (const disagreeing of [
    { ...READY, voice: 'mistborn' },
    { ...READY, fingerprint: 'deathstalker@ffffffffffffffffffffffffffffffffffffffff' },
    { ...READY, sample_rate: 22050 },
    { ...READY, backend: 'mlx-darwin' },
  ]) {
    reset();
    serving((response) => {
      frame(response, 1, 'ready', disagreeing);
      response.end();
    });
    await assert.rejects(
      () => client().stream({ voice: 'deathstalker', language: 'en' }),
      CrucibleProtocolError,
    );
    // The session the server is holding was put down, so the next open is not
    // refused `stream_session_open` for a session this caller never had.
    assert.equal(seen.filter((entry) => entry.method === 'DELETE').length, 1);
  }
});

// ----------------------------------------------------------- resolves attached

test('stream() resolves only after the event stream is attached and ready has arrived', async () => {
  // The server refuses a `say` on a session whose stream has never been opened
  // (`stream_not_attached`). Until 2026-09-14 the SDK attached lazily inside
  // the iterator, so a caller could not know when its first `say` was allowed
  // and BookForge polled the refusal away. Now the order on the wire is the
  // proof: the events GET, then ready, then — and only then — the first op.
  reset();
  let readySent = false;
  let sessionResponse: ServerResponse | null = null;
  serving(
    (response) => {
      sessionResponse = response;
      // `ready` arrives late, as it does from a real narrator that is still
      // settling. A client that resolved on the open reply would say into
      // nothing here.
      setTimeout(() => {
        frame(response, 1, 'ready', READY);
        readySent = true;
      }, 120);
    },
    () => ({ id: 'r1' }),
  );
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  assert.equal(readySent, true, 'stream() resolved before the server said ready');
  await session.say('r1', 'Rain.');
  const order = seen.map((entry) => `${entry.method} ${entry.path}`);
  assert.deepEqual(order, [
    'POST /v1/tts/stream',
    `GET /v1/tts/stream/${SESSION.session_id}/events`,
    `POST /v1/tts/stream/${SESSION.session_id}`,
  ]);
  await session.close();
  // The fixture's DELETE does not end the stream the way the real server's
  // does, so the `closed` frame is written by hand — a response left open here
  // would hold the fixture's server up for Node's whole request timeout.
  frame(sessionResponse!, 2, 'closed', { reason: 'the client closed the session' });
  sessionResponse!.end();
  assert.deepEqual(await collect(session), []);
});

test('a row said before the loop begins is not lost, and a broken loop resumes', async () => {
  reset();
  let sessionResponse: ServerResponse | null = null;
  serving(
    (response) => {
      sessionResponse = response;
      frame(response, 1, 'ready', READY);
    },
    (body) => {
      // The audio for a `say` is emitted the moment the op lands — before the
      // caller has iterated anything. It must wait in the stream, not vanish.
      const op = JSON.parse(body);
      frame(sessionResponse!, 2, 'audio', {
        id: op.id, seq: 0, pcm_base64: pcmBytes([7, 8]), seconds: 0.1,
      });
      frame(sessionResponse!, 3, 'audio', {
        id: op.id, seq: 1, pcm_base64: pcmBytes([9]), seconds: 0.05,
      });
      return { id: op.id };
    },
  );
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  await session.say('r1', 'Rain.');

  // First loop takes one chunk and breaks. Breaking detaches nothing.
  const first: StreamEvent[] = [];
  for await (const event of session) {
    first.push(event);
    break;
  }
  assert.equal(first.length, 1);
  assert.equal(first[0]!.kind, 'audio');
  assert.deepEqual(Array.from((first[0] as { pcm: Int16Array }).pcm), [7, 8]);

  // The second loop picks up the chunk the first one left, from the same
  // stream: no reattach, no replay, and `seq` carries straight on.
  frame(sessionResponse!, 4, 'closed', { reason: 'done' });
  sessionResponse!.end();
  const rest = await collect(session);
  assert.equal(rest.length, 1);
  assert.equal((rest[0] as { seq: number }).seq, 1);
  assert.equal(seen.filter((entry) => entry.path.endsWith('/events')).length, 1);
});

test('a stream that closes before it is ready is refused, with the server s reason', async () => {
  reset();
  serving((response) => {
    frame(response, 1, 'closed', { reason: 'narrator failed before the first row' });
    response.end();
  });
  await assert.rejects(
    () => client().stream({ voice: 'deathstalker', language: 'en' }),
    (error: unknown) =>
      error instanceof CrucibleProtocolError &&
      /closed before it was ready: narrator failed/.test(error.message),
  );
  // The server closed it, so there is nothing left to put down: a DELETE here
  // would only be refused `unknown_session`.
  assert.equal(seen.filter((entry) => entry.method === 'DELETE').length, 0);
});

test('a stream that says anything else before ready is not one this client knows', async () => {
  reset();
  serving((response) => {
    frame(response, 1, 'audio', { id: 'r1', seq: 0, pcm_base64: pcmBytes([1]), seconds: 0.1 });
    response.end();
  });
  await assert.rejects(
    () => client().stream({ voice: 'deathstalker', language: 'en' }),
    (error: unknown) =>
      error instanceof CrucibleProtocolError && /began with a audio frame/.test(error.message),
  );
});

test('a second ready is a fault, not a re-announcement', async () => {
  reset();
  serving((response) => {
    frame(response, 1, 'ready', READY);
    frame(response, 2, 'ready', READY);
    response.end();
  });
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  await assert.rejects(
    () => collect(session),
    (error: unknown) =>
      error instanceof CrucibleProtocolError && /ready twice/.test(error.message),
  );
});

test('a refusal to attach travels back from stream() by name', async () => {
  reset();
  handle = (request, response) => {
    if ((request.url ?? '') === '/v1/tts/stream' && request.method === 'POST') {
      json(response, 201, SESSION);
      return;
    }
    if ((request.url ?? '').endsWith('/events')) {
      json(response, 404, {
        error: { code: 'unknown_session', message: 'there is no streaming session' },
      });
      return;
    }
    json(response, 404, {
      error: { code: 'unknown_session', message: 'there is no streaming session' },
    });
  };
  await assert.rejects(
    () => client().stream({ voice: 'deathstalker', language: 'en' }),
    (error: unknown) => error instanceof CrucibleRefused && error.code === 'unknown_session',
  );
});

test('an id that does not follow the last one is a protocol error', async () => {
  reset();
  serving((response) => {
    frame(response, 5, 'ready', READY);
    frame(response, 3, 'done', {
      id: 'r1', seconds: 1, chars: 5, chars_per_sec: 5, capped: null, cancelled: false,
    });
    response.end();
  });
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  await assert.rejects(() => collect(session), CrucibleProtocolError);
});

// ---------------------------------------------------------- the reattach

test('a dropped stream reattaches with Last-Event-ID and loses nothing', async () => {
  reset();
  let attempt = 0;
  handle = (request, response) => {
    const path = request.url ?? '';
    if (path === '/v1/tts/stream' && request.method === 'POST') {
      json(response, 201, SESSION);
      return;
    }
    if (!path.endsWith('/events')) {
      json(response, 202, { id: 'r1' });
      return;
    }
    attempt += 1;
    openSse(response);
    if (attempt === 1) {
      frame(response, 1, 'ready', READY);
      // The tunnel collapses mid-row: the socket dies, no `closed` frame, and
      // the row goes on generating on the server.
      frameThenDie(request, response, 2, 'audio', {
        id: 'r1', seq: 0, pcm_base64: pcmBytes([1]), seconds: 0.1,
      });
      return;
    }
    frame(response, 3, 'audio', { id: 'r1', seq: 1, pcm_base64: pcmBytes([2]), seconds: 0.1 });
    frame(response, 4, 'done', {
      id: 'r1', seconds: 0.2, chars: 5, chars_per_sec: 25, capped: null, cancelled: false,
    });
    frame(response, 5, 'closed', { reason: 'done' });
    response.end();
  };

  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  const events = await collect(session);

  // The client came back on its own, and told the server exactly where it got
  // to — which is the whole reason this door is SSE rather than a socket.
  const resumes = seen.filter((entry) => entry.path.endsWith('/events'));
  assert.equal(resumes.length, 2);
  assert.equal(resumes[0]!.lastEventId, undefined);
  assert.equal(resumes[1]!.lastEventId, '2');

  // No gap and no repeat: the two halves are one row.
  const seqs = events.filter((event) => event.kind === 'audio').map((event) =>
    event.kind === 'audio' ? event.seq : -1,
  );
  assert.deepEqual(seqs, [0, 1]);
  assert.equal(events.at(-1)!.kind, 'done');
});

test('a refusal on the reattach is reported, never retried away', async () => {
  reset();
  let attempt = 0;
  handle = (request, response) => {
    const path = request.url ?? '';
    if (path === '/v1/tts/stream' && request.method === 'POST') {
      json(response, 201, SESSION);
      return;
    }
    attempt += 1;
    if (attempt === 1) {
      openSse(response);
      frameThenDie(request, response, 1, 'ready', READY);
      return;
    }
    // The grace window ran out while the tunnel was down. Trying again quietly
    // would turn a closed session into a hang; skipping the gap would hand the
    // caller audio with a hole in it.
    json(response, 409, {
      error: {
        code: 'replay_unavailable',
        message: 'session cafebabe can no longer replay from event 1',
      },
    });
  };
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  await assert.rejects(
    () => collect(session),
    (error: unknown) =>
      error instanceof CrucibleRefused && error.code === 'replay_unavailable',
  );
  assert.equal(attempt, 2);
});

test('a reattach that replays a frame already delivered is a protocol error', async () => {
  reset();
  let attempt = 0;
  handle = (request, response) => {
    const path = request.url ?? '';
    if (path === '/v1/tts/stream' && request.method === 'POST') {
      json(response, 201, SESSION);
      return;
    }
    attempt += 1;
    openSse(response);
    if (attempt === 1) {
      frameThenDie(request, response, 1, 'ready', READY);
      return;
    }
    // The resume ignored `Last-Event-ID` and started again from the top. Ids
    // are how a client knows it has missed nothing, so one that does not
    // increase is the thing that must not be tolerated across a reattach —
    // tolerating it is how a row's first chunk plays twice.
    frame(response, 1, 'ready', READY);
    frame(response, 2, 'closed', { reason: 'done' });
    response.end();
  };
  const session = await client().stream({ voice: 'deathstalker', language: 'en' });
  await assert.rejects(() => collect(session), CrucibleProtocolError);
  assert.equal(attempt, 2);
});

test('a server that never comes back gives up rather than retrying for ever', async () => {
  // **Slow on purpose.** It waits out the same window the server holds a
  // dropped session open for (`crucible/ttsstream.py`'s `GRACE_SECONDS`, 15 s).
  // After it the session is closed and every row in flight is cancelled, so a
  // reattach later than this cannot succeed — and a client that went on trying
  // would turn a dead session into a silent hang, which is the one outcome this
  // door must not have. The only way to prove it stops is to let the clock run.
  reset();
  const lonely = createServer((request, response) => {
    if ((request.url ?? '') === '/v1/tts/stream' && request.method === 'POST') {
      response.writeHead(201, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify(SESSION));
      return;
    }
    openSse(response);
    frameThenDie(request, response, 1, 'ready', READY);
  });
  await new Promise<void>((resolve) => lonely.listen(0, '127.0.0.1', resolve));
  const crucible = new CrucibleClient({
    url: `http://127.0.0.1:${(lonely.address() as AddressInfo).port}`,
    token: 'the-token',
    clientName: 'unit-stream',
  });
  const session = await crucible.stream({ voice: 'deathstalker', language: 'en' });
  const iterator = (session as AsyncIterable<StreamEvent>)[Symbol.asyncIterator]();
  const first = iterator.next();
  // The machine goes away entirely: every reattach from here is a refused
  // connection rather than a refusal with a body.
  await new Promise<void>((resolve) => lonely.close(() => resolve()));
  await assert.rejects(async () => {
    await first;
    for (;;) {
      const step = await iterator.next();
      if (step.done === true) return;
    }
  }, CrucibleUnreachable);
});

// -------------------------------------------------------------- base64

test('decodeBase64 round-trips and refuses what it cannot read', () => {
  for (let length = 0; length < 12; length += 1) {
    const bytes = new Uint8Array(length);
    for (let index = 0; index < length; index += 1) bytes[index] = (index * 37) % 256;
    assert.deepEqual(Array.from(decodeBase64(encodeBase64(bytes))), Array.from(bytes));
  }
  // Not skipped past. One bad character on the wire would otherwise shift every
  // sample after it, and the click in the middle of the sentence would have no
  // explanation anywhere in the pipeline.
  assert.throws(() => decodeBase64('AA*A'), /alphabet/);
  assert.throws(() => decodeBase64('AAA'), /groups/);
});
