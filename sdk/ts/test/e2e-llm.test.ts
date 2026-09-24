/**
 * End to end against a real Crucible server with the `llm` job type, a real
 * engine, and a real card. This is the test that proves phase 2.
 *
 * It needs three environment variables and refuses to run without them — a
 * skipped e2e is a green run that proved nothing:
 *
 *   CRUCIBLE_URL        e.g. http://127.0.0.1:7100  (no /v1)
 *   CRUCIBLE_TOKEN      the token `crucible token --show` prints
 *   CRUCIBLE_LLM_MODEL  the model id to exercise, e.g. qwen3.5-9b
 *
 * The tests run in file order and depend on each other: the model is loaded
 * once at the top and leased for the chat series. The server settles an
 * unleased model after each chat; a sequence of chats must explicitly retain
 * residency. The lease is heartbeated and released before testing unload,
 * with an after hook cleaning up even when an earlier assertion fails.
 *
 * It touches the GPU. Before running it, `nvidia-smi` must show only the
 * desktop: Crucible never evicts anyone else's work, so a busy card makes the
 * load refuse with `accelerator_busy`, which is the contract working, not a
 * test failure.
 */

import assert from 'node:assert/strict';
import { after, test } from 'node:test';

import {
  CrucibleClient,
  CrucibleRefused,
  isLlmCapability,
  type JobEvent,
  type ModelInfo,
} from '../src/index.js';

function required(name: string): string {
  const value = process.env[name];
  if (value === undefined || value.trim() === '') {
    throw new Error(
      `${name} is not set. The crucible llm e2e runs against a real server with a ` +
        `real engine and will not pretend otherwise: set CRUCIBLE_URL, ` +
        `CRUCIBLE_TOKEN and CRUCIBLE_LLM_MODEL.`,
    );
  }
  return value;
}

const URL_ = required('CRUCIBLE_URL');
const TOKEN = required('CRUCIBLE_TOKEN');
const MODEL = required('CRUCIBLE_LLM_MODEL');

const crucible = new CrucibleClient({
  url: URL_,
  token: TOKEN,
  clientName: 'crucible-e2e-llm',
});

let loadedBySuite = false;
let leaseId: string | null = null;
let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
let heartbeatInFlight: Promise<void> | null = null;
let heartbeatFailure: unknown = null;

async function retainModel(): Promise<void> {
  // Lease acts are capability names; the test client's identity is User-Agent.
  leaseId = (await crucible.lease(MODEL, { act: 'clean', ttlSeconds: 120 })).leaseId;
  heartbeatTimer = setInterval(() => {
    if (heartbeatInFlight !== null || leaseId === null) return;
    heartbeatInFlight = crucible.heartbeat(leaseId)
      .then(() => undefined)
      .catch((error: unknown) => { heartbeatFailure = error; })
      .finally(() => { heartbeatInFlight = null; });
  }, 30_000);
  heartbeatTimer.unref();
}

function assertRetained(): void {
  assert.notEqual(leaseId, null, 'the chat series must hold a model lease');
  assert.equal(heartbeatFailure, null, `the model lease heartbeat failed: ${String(heartbeatFailure)}`);
}

async function releaseModel(): Promise<void> {
  if (heartbeatTimer !== null) clearInterval(heartbeatTimer);
  heartbeatTimer = null;
  // Do not release while a previous heartbeat is still using the receipt.
  await heartbeatInFlight;
  const id = leaseId;
  leaseId = null;
  if (id !== null) await crucible.release(id);
}

after(async () => {
  const failures: unknown[] = [];
  try { await releaseModel(); } catch (error) { failures.push(error); }
  try {
    // Only unload the model this suite loaded, never another client's model.
    // Lease release may already have settled it; then there is nothing to do.
    if (loadedBySuite && (await row()).resident) {
      const events = await collect(await crucible.unloadModel(MODEL));
      assert.equal(events.at(-1)?.event, 'done', `cleanup unload failed: ${JSON.stringify(events.at(-1))}`);
    }
  } catch (error) { failures.push(error); }
  if (heartbeatFailure !== null) failures.push(heartbeatFailure);
  if (failures.length > 0) throw new AggregateError(failures, 'LLM e2e cleanup or lease renewal failed');
});

/** The row for the model under test, or a failure naming what the server does offer. */
async function row(): Promise<ModelInfo> {
  const models = await crucible.models();
  const found = models.find((model) => model.id === MODEL);
  assert.ok(
    found !== undefined,
    `CRUCIBLE_LLM_MODEL=${MODEL} is not one of the server's models ` +
      `(${models.map((model) => model.id).join(', ') || 'none'})`,
  );
  return found;
}

/** Drain a job's whole event stream. */
async function collect(jobId: string): Promise<JobEvent[]> {
  const events: JobEvent[] = [];
  for await (const event of crucible.events(jobId)) events.push(event);
  return events;
}

// ------------------------------------------------------------------- models

test('the server offers the llm capability and lists the model', async () => {
  const info = await crucible.info();
  const llm = info.capabilities.find((capability) => capability.jobType === 'llm');
  assert.ok(llm !== undefined, 'the phase-2 server must offer the llm job type');
  assert.ok(isLlmCapability(llm), 'the llm capability carries /v1/models rows');

  const model = await row();
  // Informational fields are nullable on the wire (any Crucible that answers
  // works, Owen 2026-09-24); THIS server is current and must state them all.
  assert.ok(model.family !== null && model.family.length > 0, 'the manifest must name a family');
  assert.ok(model.paramsB !== null && model.paramsB > 0, 'the manifest must give a parameter count');
  assert.ok(model.memoryBytesEstimate !== null && model.memoryBytesEstimate > 0,
    'the manifest must give a measured memory estimate');
  assert.ok(
    model.contextDefault !== null && model.contextDefault > 0,
    'the manifest must give a default context',
  );
  assert.equal(model.backendSupported, true, `${MODEL} has no block for this host's backend`);
  assert.equal(model.installed, true, `${MODEL} is not installed: crucible models pull ${MODEL}`);
  assert.ok(
    model.revision !== null && /^[0-9a-f]{40}$/.test(model.revision),
    `${MODEL} must name the commit it is pinned to, got ${String(model.revision)}`,
  );
  // What a client writes into a book's record, and what it sizes a request
  // against: the id alone identifies neither the weights nor the context.
  assert.equal(model.fingerprint, `${MODEL}@${model.revision}`);
  assert.ok(
    model.maxModelLen !== null && model.maxModelLen > 0,
    `${MODEL} must report the context it is served at, got ${String(model.maxModelLen)}`,
  );
  if (!model.loadable) {
    assert.fail(`${MODEL} is not loadable: ${String(model.reason)}`);
  }
  assert.equal(model.reason, null, 'a loadable model carries no reason');

  // One model, one description: what /info says and what /models says are the
  // same rows, not two accounts a client would have to reconcile.
  assert.deepEqual(llm.models, await crucible.models());
  console.log(
    `    ${MODEL} @ ${model.revision} — ${String(info.host.backend)} on ` +
      `${String(info.host.gpu?.name)}`,
  );
});

// --------------------------------------------------------------------- load

test('load-model warms the engine and finishes naming the resident model', async () => {
  const jobId = await crucible.loadModel(MODEL);
  loadedBySuite = true;
  const events = await collect(jobId);

  const names = events.map((event) => event.event);
  assert.equal(names[0], 'queued', `the stream began with ${String(names[0])}`);
  assert.equal(names.at(-1), 'done', `the load ended ${String(names.at(-1))}: ${JSON.stringify(events.at(-1))}`);
  assert.ok(
    names.includes('warming'),
    'loading an engine must stream its readiness as warming events',
  );
  for (const event of events) {
    if (event.event === 'warming') {
      const message = event.data.message;
      assert.equal(typeof message, 'string');
      assert.ok(message !== null && message.length > 0, 'a warming event must say something');
    }
  }

  const done = events.at(-1)!;
  assert.equal(done.event, 'done');
  assert.equal(done.event === 'done' ? done.data.resident : null, MODEL);
  await retainModel();

  const health = await crucible.health();
  assert.deepEqual(health.residentModels, [MODEL]);

  const model = await row();
  assert.equal(model.resident, true, `${MODEL} finished loading but does not read as resident`);
});

// --------------------------------------------------------------------- chat

test('chat returns a non-empty completion from the resident engine', async () => {
  assertRetained();
  // `thinking: false` is what makes a 64-token ceiling honest against a
  // reasoning model: without it Qwen3.5 spends the whole budget in `reasoning`
  // and the message comes back with no `content` at all.
  const answer = await crucible.chat({
    model: MODEL,
    messages: [
      { role: 'system', content: 'Answer in one short sentence.' },
      { role: 'user', content: 'Name one colour.' },
    ],
    temperature: 0,
    maxTokens: 64,
    thinking: false,
  });
  console.log(`    content: ${JSON.stringify(answer.content)}`);
  console.log(`    finish_reason: ${answer.finishReason}, usage: ${JSON.stringify(answer.usage)}`);

  assert.ok(answer.id !== null && answer.id.length > 0, 'the completion must carry an id');
  assert.equal(answer.model, MODEL);
  assert.ok(answer.content.trim().length > 0, 'the completion must carry text');
  assert.ok(answer.finishReason.length > 0, 'the engine must say why it stopped');
  const usage = answer.usage;
  assert.ok(usage !== null, 'the engine must report usage');
  const { promptTokens, completionTokens, totalTokens } = usage;
  assert.ok(promptTokens !== null && promptTokens > 0, 'the engine must count the prompt');
  assert.ok(
    completionTokens !== null && completionTokens > 0,
    'the engine must count the completion',
  );
  assert.equal(totalTokens, promptTokens + completionTokens, 'the totals must add up');
});

test('chatStream yields deltas that concatenate to the answer', async () => {
  assertRetained();
  const deltas: string[] = [];
  for await (const delta of crucible.chatStream({
    model: MODEL,
    messages: [
      { role: 'system', content: 'Answer in one short sentence.' },
      { role: 'user', content: 'Name one colour.' },
    ],
    temperature: 0,
    maxTokens: 64,
    thinking: false,
  })) {
    deltas.push(delta);
  }

  assert.ok(deltas.length > 0, 'a streamed completion must yield at least one delta');
  assert.ok(deltas.join('').trim().length > 0, 'the deltas must concatenate to text');
  console.log(`    ${deltas.length} deltas: ${JSON.stringify(deltas.join(''))}`);
});

test('chat on a model that is not resident is refused by name, never loaded implicitly', async () => {
  assertRetained();
  const wrong = `${MODEL}-not-a-real-model`;
  await assert.rejects(
    crucible.chat({ model: wrong, messages: [{ role: 'user', content: 'hello' }] }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
      assert.equal(error.status, 409);
      assert.equal(error.code, 'model_not_resident');
      assert.match(
        error.serverMessage,
        new RegExp(MODEL.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')),
        'the refusal must name the model that IS resident',
      );
      return true;
    },
  );

  // The refusal must not have changed what is resident.
  assert.deepEqual((await crucible.health()).residentModels, [MODEL]);
});

// ------------------------------------------------------------------- unload

test('unload-model frees the card and the model stops reading as resident', async () => {
  assertRetained();
  // A lease deliberately refuses unload-model. Releasing it also asks the
  // settlement layer to clear the model; the explicit unload remains valid
  // when that clearance has already completed or is still finishing.
  await releaseModel();
  const jobId = await crucible.unloadModel(MODEL);
  const events = await collect(jobId);
  const done = events.at(-1)!;
  assert.equal(
    done.event,
    'done',
    `the unload ended ${String(done.event)}: ${JSON.stringify(done)}`,
  );
  // An unload reports what is resident now, and after an unload that is
  // nothing: the field is answered with null, never left out (PHASE2 section 5).
  assert.equal(done.event === 'done' ? done.data.resident : 'x', null);

  const model = await row();
  assert.equal(model.resident, false, `${MODEL} unloaded but still reads as resident`);
  loadedBySuite = false;
  assert.deepEqual((await crucible.health()).residentModels, []);

  await assert.rejects(
    crucible.chat({ model: MODEL, messages: [{ role: 'user', content: 'hello' }] }),
    (error: unknown) => {
      assert.ok(error instanceof CrucibleRefused, `got ${String(error)}`);
      assert.equal(error.code, 'model_not_resident');
      return true;
    },
  );
});
