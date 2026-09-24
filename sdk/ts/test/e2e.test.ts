/**
 * End to end against a real Crucible server. This is the test that matters.
 *
 * It needs two environment variables and refuses to run without them — a
 * skipped e2e is a green run that proved nothing:
 *
 *   CRUCIBLE_URL    e.g. http://127.0.0.1:7100  (no /v1)
 *   CRUCIBLE_TOKEN  the token `crucible token --show` prints
 *
 * `scripts/e2e.sh` starts a throwaway server with echo enabled, exports both,
 * runs this, and stops the server with SIGTERM.
 *
 * The server must have been initialised with `--enable-echo`.
 */

import assert from 'node:assert/strict';
import { createHash, randomBytes } from 'node:crypto';
import { test } from 'node:test';

import {
  CrucibleAcceleratorUnreadable,
  CrucibleAuthError,
  CrucibleClient,
  CrucibleRefused,
  type JobEvent,
} from '../src/index.js';

function required(name: string): string {
  const value = process.env[name];
  if (value === undefined || value.trim() === '') {
    throw new Error(
      `${name} is not set. The crucible e2e runs against a real server and will ` +
        `not pretend otherwise: set CRUCIBLE_URL and CRUCIBLE_TOKEN, or run ` +
        `scripts/e2e.sh which does it for you.`,
    );
  }
  return value;
}

const URL_ = required('CRUCIBLE_URL');
const TOKEN = required('CRUCIBLE_TOKEN');

const crucible = new CrucibleClient({
  url: URL_,
  token: TOKEN,
  clientName: 'crucible-e2e',
});

const sha256 = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');

/** Drain a job's whole event stream. */
async function collect(jobId: string, lastEventId?: number): Promise<JobEvent[]> {
  const events: JobEvent[] = [];
  const options = lastEventId === undefined ? {} : { lastEventId };
  for await (const event of crucible.events(jobId, options)) events.push(event);
  return events;
}

const names = (events: readonly JobEvent[]): string[] => events.map((event) => event.event);

// ---------------------------------------------------------------- handshake

test('ping identifies the server without a token', async () => {
  const ping = await crucible.ping();
  assert.equal(ping.crucible, true);
  assert.equal(ping.apiVersion, 1);
  assert.ok(ping.name.length > 0, 'the server must name itself');
});

test('info reports a real backend and offers echo', async () => {
  const info = await crucible.info();
  assert.equal(info.server.apiVersion, 1);
  // Informational on the wire, and a current server states them all.
  const backend = info.host.backend;
  assert.ok(
    backend !== null && ['cuda-linux', 'mlx-darwin', 'llama-windows'].includes(backend),
    `unexpected backend ${String(backend)}`,
  );
  const gpuName = info.host.gpu?.name;
  assert.ok(typeof gpuName === 'string' && gpuName.length > 0, 'the backend must name its gpu');
  const echo = info.capabilities.find((capability) => capability.jobType === 'echo');
  assert.ok(echo !== undefined, 'the e2e server must be initialised with --enable-echo');
  assert.deepEqual(echo.models, [], 'echo serves no models');
});

test('health reports the lane, and says what kind of thing holds the card', async () => {
  const health = await crucible.health();
  assert.ok(health.status !== null && ['ok', 'warming', 'busy'].includes(health.status));
  assert.ok(Number.isInteger(health.queueDepth));
  assert.deepEqual(health.residentModels, []);
  // Nothing is loaded on an echo-only server, so the kind is the null it reports
  // when nothing holds the card.
  assert.equal(health.residentKind, null);
});

test('the accelerator probe answers a state, or refuses by name — never zeroes', async () => {
  // The one route whose failure mode is the point: a host with no readable card
  // (no nvidia-smi, a CI box, a Mac where the query cannot be asked) must
  // REFUSE, because a client polling for a free GPU would read zeroes as "it is
  // free". Both outcomes are contract; a resolved value full of zeroes is not.
  let state: Awaited<ReturnType<typeof crucible.accelerator>> | null = null;
  try {
    state = await crucible.accelerator();
  } catch (error) {
    assert.ok(
      error instanceof CrucibleAcceleratorUnreadable,
      `an unreadable probe must be accelerator_unreadable, got ${String(error)}`,
    );
    assert.equal(error.code, 'accelerator_unreadable');
    return;
  }
  // Informational on the wire, and a current server states them all.
  assert.ok(
    state.backend !== null && ['cuda-linux', 'mlx-darwin', 'llama-windows'].includes(state.backend),
  );
  const totalBytes = state.gpu?.totalBytes;
  assert.ok(typeof totalBytes === 'number' && totalBytes > 0, 'a readable probe names a real card');
  assert.ok(state.freeBytes !== null && state.freeBytes >= 0);
  // Never negative and never a substitute for one: the server clamps at zero.
  assert.ok(state.unattributedBytes === null || state.unattributedBytes >= 0);
  assert.equal(state.resident, null, 'an echo-only server holds nothing');
  assert.ok(state.holders !== null, 'a current server lists the holders');
  for (const holder of state.holders) {
    assert.ok(Number.isInteger(holder.pid));
    // Null is the driver declining to say and is a legal answer; a string or a
    // missing key is not, and the reader would already have thrown.
    assert.ok(holder.bytes === null || holder.bytes >= 0);
  }
});

test('voices is refused by name on a server with tts disabled', async () => {
  // `scripts/e2e.sh` initialises with `--enable-echo` and nothing else, so this
  // exercises the route and the refusal that guards it rather than needing a
  // voice on disk.
  await assert.rejects(crucible.voices(), (error: unknown) => {
    assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
    assert.equal(error.code, 'job_type_disabled');
    return true;
  });
});

test('a wrong token is an auth error, not a refusal', async () => {
  const wrong = new CrucibleClient({ url: URL_, token: 'not-the-token', clientName: 'crucible-e2e' });
  await assert.rejects(wrong.info(), CrucibleAuthError);
});

test('an unknown job type is refused by name', async () => {
  await assert.rejects(
    crucible.submit({ type: 'summon', params: {}, inputs: {} }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
      assert.equal(error.code, 'unknown_job_type');
      return true;
    },
  );
});

// -------------------------------------------------------------- echo inline

test('an inline echo job runs end to end, in order, with identical bytes', async () => {
  const payload = new Uint8Array(randomBytes(64 * 1024));
  const jobId = await crucible.submit({
    type: 'echo',
    params: { delay_ms: 40 },
    inputs: { 'payload.bin': { inline: payload } },
  });

  const events = await collect(jobId);

  // The exact stream the phase-1 server emits for one input:
  //   queued, progress(started), progress(copying), artifact, progress(done), done
  assert.deepEqual(names(events), [
    'queued',
    'progress',
    'progress',
    'artifact',
    'progress',
    'done',
  ]);

  // Ids are the server's counter: contiguous from 1, strictly increasing.
  assert.deepEqual(
    events.map((event) => event.id),
    [1, 2, 3, 4, 5, 6],
  );

  const artifactEvent = events[3];
  assert.ok(artifactEvent !== undefined);
  assert.equal(artifactEvent.event, 'artifact');
  assert.equal(artifactEvent.event === 'artifact' ? artifactEvent.data.name : null, 'payload.bin');

  const last = events.at(-1);
  assert.ok(last !== undefined);
  assert.equal(last.event, 'done');
  assert.deepEqual(last.event === 'done' ? last.data.artifacts : null, ['payload.bin']);

  const fetched = await crucible.artifact(jobId, 'payload.bin');
  assert.equal(fetched.byteLength, payload.byteLength);
  assert.equal(sha256(fetched), sha256(payload), 'the echoed bytes must be identical');

  const status = await crucible.job(jobId);
  assert.equal(status.jobId, jobId);
  assert.equal(status.status, 'done');
  assert.equal(status.progress, 1);
  assert.equal(status.position, null);
  assert.equal(status.error, null);
  assert.deepEqual(status.artifacts, ['payload.bin']);
  assert.ok(status.started !== null && status.finished !== null);

  const provenance = await crucible.provenance(jobId, 'payload.bin');
  assert.equal(provenance.job_type, 'echo');
  assert.equal(provenance.model, null);
  // Informational on the wire, and a current server states them all.
  const server = provenance.server;
  assert.ok(server !== null && server.name !== null && server.name.length > 0);
  assert.ok(server.version !== null && server.version.length > 0);
  assert.ok(
    provenance.backend !== null &&
      ['cuda-linux', 'mlx-darwin', 'llama-windows'].includes(provenance.backend),
  );
  assert.deepEqual(provenance.params, { delay_ms: 40 });
  assert.ok(provenance.started !== null, 'provenance records when the job started');
  assert.ok(
    provenance.finished !== null && provenance.finished.length > 0,
    'provenance records when the job finished',
  );
});

// -------------------------------------------------------------- echo upload

test('an uploaded blob can be named as a job input', async () => {
  const payload = new Uint8Array(randomBytes(128 * 1024));
  const uploaded = await crucible.upload(payload, { filename: 'page.bin' });
  assert.equal(uploaded.bytes, payload.byteLength);
  assert.equal(uploaded.sha256, sha256(payload), 'the server must hash what it stored');
  assert.ok(uploaded.blobId.length > 0);

  const jobId = await crucible.submit({
    type: 'echo',
    params: { delay_ms: 0 },
    inputs: { 'page.bin': { blobId: uploaded.blobId } },
  });
  const events = await collect(jobId);
  assert.equal(names(events).at(-1), 'done');

  const fetched = await crucible.artifact(jobId, 'page.bin');
  assert.equal(sha256(fetched), sha256(payload));
});

test('an unknown blob id is refused by name', async () => {
  await assert.rejects(
    crucible.submit({
      type: 'echo',
      params: {},
      inputs: { 'x.bin': { blobId: '0'.repeat(32) } },
    }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
      assert.equal(error.code, 'unknown_blob');
      return true;
    },
  );
});

// ------------------------------------------------------------------- resume

test('a stream resumes from Last-Event-ID without losing or repeating an event', async () => {
  const jobId = await crucible.submit({
    type: 'echo',
    params: { delay_ms: 400 },
    inputs: {
      'one.bin': { inline: new Uint8Array(randomBytes(1024)) },
      'two.bin': { inline: new Uint8Array(randomBytes(1024)) },
    },
  });

  // Read the first two events live, then walk away mid-stream.
  const first: JobEvent[] = [];
  for await (const event of crucible.events(jobId)) {
    first.push(event);
    if (first.length === 2) break;
  }
  assert.deepEqual(
    first.map((event) => event.id),
    [1, 2],
  );

  const resumed = await collect(jobId, first[1]!.id);
  assert.equal(resumed[0]?.id, 3, 'the resumed stream starts at the next id');
  assert.equal(names(resumed).at(-1), 'done');

  const whole = [...first, ...resumed];
  assert.deepEqual(
    whole.map((event) => event.id),
    whole.map((_, index) => index + 1),
    'ids across the seam are contiguous and monotonic',
  );
  assert.deepEqual(names(whole), [
    'queued',
    'progress',
    'progress',
    'artifact',
    'progress',
    'artifact',
    'progress',
    'done',
  ]);

  for (const name of ['one.bin', 'two.bin']) {
    const provenance = await crucible.provenance(jobId, name);
    assert.equal(provenance.job_type, 'echo');
  }
});

// ------------------------------------------------------------------- cancel

test('cancelling a running job ends the stream with a cancelled event', async () => {
  const jobId = await crucible.submit({
    type: 'echo',
    params: { delay_ms: 5000 },
    inputs: { 'slow.bin': { inline: new Uint8Array(randomBytes(1024)) } },
  });

  const seen: JobEvent[] = [];
  let cancelStatus: string | null = null;
  for await (const event of crucible.events(jobId)) {
    seen.push(event);
    // The second progress event means the job is inside run(); cancel there.
    if (cancelStatus === null && event.event === 'progress' && event.data.fraction === 0) {
      const result = await crucible.cancel(jobId);
      assert.equal(result.jobId, jobId);
      cancelStatus = result.status;
    }
  }

  assert.ok(
    cancelStatus === 'cancelling' || cancelStatus === 'cancelled',
    `cancel answered ${String(cancelStatus)}`,
  );
  const last = seen.at(-1);
  assert.ok(last !== undefined);
  assert.equal(last.event, 'cancelled');
  assert.equal(last.event === 'cancelled' ? last.data.status : null, 'cancelled');

  const status = await crucible.job(jobId);
  assert.equal(status.status, 'cancelled');
  assert.equal(status.position, null);

  await assert.rejects(crucible.cancel(jobId), (error: unknown) => {
    assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
    assert.equal(error.code, 'job_not_cancellable');
    return true;
  });
});
