/**
 * The operator door in the client: `setup`, `catalog`, `submitTask`, `task`,
 * `tasks`, `taskEvents`, `cancelTask`.
 *
 * PHASE13-OPERATOR.md section 3.6, against a fake server like every other
 * method in this directory. What is proved here is the half a live server
 * cannot prove on a good day: that the snake_case wire becomes the camelCase
 * types exactly, that a field the contract promises and the server omits is a
 * protocol error rather than an `undefined` handed to a page, that a `pull`'s
 * progress and an `install`'s are told apart on their keys, and that a task
 * event kind this build never heard of does not cost the caller the rest of
 * the stream.
 *
 * Run: `npm run test:unit`.
 */

import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';

import {
  CrucibleBusy,
  CrucibleCardHeld,
  CrucibleClient,
  CrucibleProtocolError,
  CrucibleRefused,
  isServerSpecificRefusal,
  isTaskBytesProgress,
  isTaskLineProgress,
  parsePairing,
  type TaskEvent,
} from '../src/index.js';

// ------------------------------------------------------------------ fixture

type Handler = (request: IncomingMessage, response: ServerResponse, body: string) => void;

let handle: Handler = (_request, response) => {
  response.writeHead(500, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify({ error: { code: 'no_handler', message: 'the test set none' } }));
};

let lastPath = '';
let lastMethod = '';
let lastBody = '';

let server: Server;
let base = '';

function client(): CrucibleClient {
  return new CrucibleClient({ url: base, token: 'the-token', clientName: 'unit-operator' });
}

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

function answer(status: number, body: unknown): void {
  handle = (_request, response) => json(response, status, body);
}

/** Serve these SSE frames and close, which is what a finished task looks like. */
function sse(frames: { id: number; event: string; data: unknown }[]): void {
  handle = (_request, response) => {
    response.writeHead(200, { 'Content-Type': 'text/event-stream' });
    for (const frame of frames) {
      response.write(`id: ${frame.id}\nevent: ${frame.event}\ndata: ${JSON.stringify(frame.data)}\n\n`);
    }
    response.end();
  };
}

before(async () => {
  server = createServer((request, response) => {
    let body = '';
    request.on('data', (chunk) => {
      body += chunk;
    });
    request.on('end', () => {
      lastPath = request.url ?? '';
      lastMethod = request.method ?? '';
      lastBody = body;
      handle(request, response, body);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
  await new Promise<void>((resolve, reject) =>
    server.close((error) => (error ? reject(error) : resolve())),
  );
});

// ------------------------------------------------------------------- setup

const SETUP = {
  name: 'crucible@mac-studio',
  version: '0.6.0',
  backend: 'mlx-darwin',
  bind: 'http://0.0.0.0:7100',
  urls: ['http://192.168.68.20:7100', 'http://100.64.0.3:7100'],
  token: 's3cret-t0ken_x',
  pairing: [
    'crucible://crucible%40mac-studio@192.168.68.20:7100/#s3cret-t0ken_x',
    'crucible://crucible%40mac-studio@100.64.0.3:7100/#s3cret-t0ken_x',
  ],
  job_types: ['llm', 'tts'],
  config_path: '/Users/telltale/.crucible/config.toml',
};

test('setup reads every field of section 3.1', async () => {
  answer(200, SETUP);
  const setup = await client().setup();
  assert.equal(lastPath, '/v1/setup');
  assert.deepEqual(setup, {
    name: 'crucible@mac-studio',
    version: '0.6.0',
    backend: 'mlx-darwin',
    bind: 'http://0.0.0.0:7100',
    urls: SETUP.urls,
    token: 's3cret-t0ken_x',
    pairing: SETUP.pairing,
    jobTypes: ['llm', 'tts'],
    configPath: '/Users/telltale/.crucible/config.toml',
  });
});

test('a pairing line off setup opens a client with no typing', async () => {
  answer(200, SETUP);
  const setup = await client().setup();
  const parsed = parsePairing(setup.pairing[0]!);
  assert.equal(parsed.name, setup.name);
  assert.equal(parsed.token, setup.token);
  assert.equal(parsed.url, setup.urls[0]);
});

test('a setup missing a field the contract promises is a protocol error', async () => {
  const { config_path: _dropped, ...without } = SETUP;
  answer(200, without);
  await assert.rejects(() => client().setup(), CrucibleProtocolError);
});

// ----------------------------------------------------------------- catalog

const ROW = {
  kind: 'model',
  id: 'qwen3.5-9b',
  name: 'Qwen3.5 9B',
  job_type: 'llm',
  installed: true,
  installed_bytes: 20718362624,
  expected_bytes: null,
  floors: ['clean'],
  license: null,
  source: 'hf:Qwen/Qwen3.5-9B',
  resident: false,
};

test('catalog reads its rows and camelCases nothing else', async () => {
  answer(200, { rows: [ROW, { ...ROW, kind: 'rvc-base', id: 'base', name: null, expected_bytes: 600, installed: false, installed_bytes: null, floors: [] }] });
  const rows = await client().catalog();
  assert.equal(lastPath, '/v1/catalog');
  assert.deepEqual(rows[0], {
    kind: 'model',
    id: 'qwen3.5-9b',
    name: 'Qwen3.5 9B',
    jobType: 'llm',
    installed: true,
    installedBytes: 20718362624,
    expectedBytes: null,
    floors: ['clean'],
    license: null,
    source: 'hf:Qwen/Qwen3.5-9B',
    resident: false,
  });
  assert.equal(rows[1]!.kind, 'rvc-base');
  assert.equal(rows[1]!.name, null);
  assert.equal(rows[1]!.installedBytes, null);
});

test('a kind outside the five is a protocol error, not a passed-through string', async () => {
  answer(200, { rows: [{ ...ROW, kind: 'sorcery' }] });
  await assert.rejects(() => client().catalog(), CrucibleProtocolError);
});

test('a row whose installed_bytes is absent is refused rather than read as null', async () => {
  const { installed_bytes: _dropped, ...without } = ROW;
  answer(200, { rows: [without] });
  await assert.rejects(() => client().catalog(), CrucibleProtocolError);
});

// ------------------------------------------------------------ submitting

test('a pull is sent as the server spells it', async () => {
  answer(202, { task_id: 't1' });
  const id = await client().submitTask({ type: 'pull', kind: 'voice', id: 'higgs-default' });
  assert.equal(id, 't1');
  assert.equal(lastMethod, 'POST');
  assert.equal(lastPath, '/v1/tasks');
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'pull',
    kind: 'voice',
    id: 'higgs-default',
  });
});

test('an install omits narrator_engine rather than sending null', async () => {
  answer(202, { task_id: 't2' });
  await client().submitTask({ type: 'install', jobType: 'llm' });
  assert.deepEqual(JSON.parse(lastBody), { type: 'install', job_type: 'llm' });
});

test('an install for tts states its engine', async () => {
  answer(202, { task_id: 't3' });
  await client().submitTask({
    type: 'install',
    jobType: 'tts',
    narratorEngine: 'higgs-v3',
  });
  assert.deepEqual(JSON.parse(lastBody), {
    type: 'install',
    job_type: 'tts',
    narrator_engine: 'higgs-v3',
  });
});

test('a module is posted exactly as the app vendored it', async () => {
  const vendored = {
    name: 'bookforge',
    version: '0.6.0+1f2e3d4c5b6a',
    job_types: [{ type: 'llm' }, { type: 'tts', narrator_engine: 'higgs-v3' }],
    // CLASSES since PHASE15-HOST.md 5.3a, and the ids that remain are the
    // app's own choices. Posted byte for byte either way: the document is a
    // file the app vendors and this client is not a second author of it.
    needs: [{ class: 'clean' }],
    subjects: [{ kind: 'model' as const, id: 'faster-whisper-large-v3' }],
  };
  answer(202, { task_id: 't4' });
  await client().submitTask({ type: 'module', module: vendored });
  assert.deepEqual(JSON.parse(lastBody), { type: 'module', module: vendored });
});

test('task_busy is a refusal the caller can act on, and it travels', async () => {
  answer(409, {
    error: {
      code: 'task_busy',
      message: 'this server is already running task t9 (install)',
      details: { task_id: 't9', type: 'install', since: '2026-09-14T03:00:00+00:00' },
    },
  });
  await assert.rejects(
    () => client().submitTask({ type: 'pull', kind: 'model', id: 'qwen3.5-9b' }),
    CrucibleRefused,
  );
  assert.equal(isServerSpecificRefusal('task_busy'), true);
  assert.equal(isServerSpecificRefusal('already_installed'), true);
  assert.equal(isServerSpecificRefusal('job_type_installed'), true);
  // A misspelled id is misspelled on every machine, so a walk must not retry.
  assert.equal(isServerSpecificRefusal('unknown_subject'), false);
  assert.equal(isServerSpecificRefusal('invalid_module'), false);
});

test('an install refused server_busy for a lease names the holder verbatim', async () => {
  answer(409, {
    error: {
      code: 'server_busy',
      message: 'this server cannot install anything right now',
      details: {
        fact: 'a lease',
        who: "'foundry/owens-pc' for 'translate'",
        lease_id: 'l1',
        kind: 'llm',
        client: 'foundry/owens-pc',
        act: 'translate',
        subject: 'qwen3.8-27b-4bit',
        since: '2026-09-14T03:00:00+00:00',
        expires_at: '2026-09-14T03:12:00+00:00',
      },
    },
  });
  try {
    await client().submitTask({ type: 'install', jobType: 'llm' });
    assert.fail('the client accepted a 409');
  } catch (error) {
    // NOT a CrucibleBusy: that type reads a JOB's eight fields and the holder
    // here is a lease, so a client that read it as one would report a protocol
    // error about a body that is exactly the contract. `details.fact` is the
    // discriminator and CrucibleCardHeld is what it produces.
    assert.ok(error instanceof CrucibleCardHeld);
    assert.equal(error instanceof CrucibleBusy, false);
    assert.equal(error.fact, 'a lease');
    assert.match(error.heldLine, /^held by a lease: /);
    const details = error.details as Record<string, unknown>;
    assert.equal(details['fact'], 'a lease');
    assert.equal(details['client'], 'foundry/owens-pc');
    assert.equal(details['act'], 'translate');
    assert.equal(details['subject'], 'qwen3.8-27b-4bit');
  }
});

// -------------------------------------------------------------- reading one

const TASK = {
  task_id: 't1',
  type: 'pull',
  request: { type: 'pull', kind: 'model', id: 'qwen3.5-9b' },
  state: 'done',
  error: null,
  created: '2026-09-14T03:00:00+00:00',
  started: '2026-09-14T03:00:00+00:00',
  finished: '2026-09-14T03:04:00+00:00',
};

test('task reads the record and keeps the echoed request verbatim', async () => {
  answer(200, TASK);
  const task = await client().task('t1');
  assert.equal(lastPath, '/v1/tasks/t1');
  assert.deepEqual(task, {
    taskId: 't1',
    type: 'pull',
    request: { type: 'pull', kind: 'model', id: 'qwen3.5-9b' },
    state: 'done',
    error: null,
    created: '2026-09-14T03:00:00+00:00',
    started: '2026-09-14T03:00:00+00:00',
    finished: '2026-09-14T03:04:00+00:00',
    // ABSENT reads as EMPTY (PHASE15-HOST.md 5.3a): a server that predates
    // the field ran a module in which every class resolved, or named none,
    // because a server that could leave one unmet is one that carries it.
    unmet: [],
  });
});

test("unmet travels with the class and the capability row's own reason", async () => {
  answer(200, {
    ...TASK,
    type: 'module',
    unmet: [{ class: 'pages', reason: 'disabled: mlx-vlm serves no image here' }],
  });
  const task = await client().task('t1');
  assert.deepEqual(task.unmet, [
    { class: 'pages', reason: 'disabled: mlx-vlm serves no image here' },
  ]);
});

test('an unmet row missing its reason is a protocol error, not a blank', async () => {
  // A window drawing "not on this engine" needs the reason, and there is
  // nothing to fall back to.
  answer(200, { ...TASK, unmet: [{ class: 'pages' }] });
  await assert.rejects(() => client().task('t1'), CrucibleProtocolError);
});

test('a task has no queued state, and one claiming it is a protocol error', async () => {
  answer(200, { ...TASK, state: 'queued' });
  await assert.rejects(() => client().task('t1'), CrucibleProtocolError);
});

test('a failed task carries the code and message a page prints', async () => {
  answer(200, {
    ...TASK,
    state: 'failed',
    error: { code: 'install_failed', message: 'exited 3' },
  });
  const task = await client().task('t1');
  assert.equal(task.state, 'failed');
  assert.equal(task.error?.code, 'install_failed');
});

test('tasks lists the last few, newest first, in the same shape', async () => {
  answer(200, { tasks: [TASK, { ...TASK, task_id: 't0' }] });
  const listed = await client().tasks();
  assert.equal(lastPath, '/v1/tasks');
  assert.deepEqual(listed.map((row) => row.taskId), ['t1', 't0']);
});

// ------------------------------------------------------------------ events

async function collect(id: string): Promise<TaskEvent[]> {
  const seen: TaskEvent[] = [];
  for await (const event of client().taskEvents(id)) seen.push(event);
  return seen;
}

test("a pull's events come back typed, bytes and all", async () => {
  sse([
    { id: 1, event: 'started', data: { type: 'pull' } },
    { id: 2, event: 'step', data: { name: 'pull model qwen3.5-9b', index: 1, total: 1 } },
    { id: 3, event: 'progress', data: { bytes_done: 1024, bytes_total: 4096, file: 'model.safetensors' } },
    { id: 4, event: 'progress', data: { bytes_done: 4096, bytes_total: null, file: 'model.safetensors' } },
    { id: 5, event: 'done', data: {} },
  ]);
  const seen = await collect('t1');
  assert.equal(lastPath, '/v1/tasks/t1/events');
  assert.deepEqual(seen.map((event) => event.event), [
    'started',
    'step',
    'progress',
    'progress',
    'done',
  ]);
  const third = seen[2]!;
  assert.equal(third.event, 'progress');
  if (third.event !== 'progress') throw new Error('unreachable');
  assert.equal(isTaskBytesProgress(third.data), true);
  assert.equal(isTaskLineProgress(third.data), false);
  if (!isTaskBytesProgress(third.data)) throw new Error('unreachable');
  assert.equal(third.data.bytesDone, 1024);
  assert.equal(third.data.bytesTotal, 4096);
  assert.equal(third.data.file, 'model.safetensors');
  const fourth = seen[3]!;
  if (fourth.event !== 'progress' || !isTaskBytesProgress(fourth.data)) {
    throw new Error('unreachable');
  }
  assert.equal(fourth.data.bytesTotal, null);
});

test("an install's events carry lines, and the reload step says what became live", async () => {
  sse([
    { id: 1, event: 'started', data: { type: 'install' } },
    { id: 2, event: 'step', data: { name: 'install llm', index: 1, total: 2 } },
    { id: 3, event: 'progress', data: { line: 'Collecting torch==2.5.1' } },
    { id: 4, event: 'step', data: { name: 'reload', index: 2, total: 2, job_types: ['echo', 'load-model', 'unload-model'] } },
    { id: 5, event: 'done', data: {} },
  ]);
  const seen = await collect('t2');
  const line = seen[2]!;
  if (line.event !== 'progress' || !isTaskLineProgress(line.data)) {
    throw new Error('unreachable');
  }
  assert.equal(line.data.line, 'Collecting torch==2.5.1');
  const reload = seen[3]!;
  if (reload.event !== 'step') throw new Error('unreachable');
  assert.equal(reload.data.name, 'reload');
  assert.deepEqual(reload.data.jobTypes, ['echo', 'load-model', 'unload-model']);
});

test('a step that made nothing new reachable has no jobTypes, and that is not a gap', async () => {
  sse([
    { id: 1, event: 'started', data: { type: 'pull' } },
    { id: 2, event: 'step', data: { name: 'pull model x', index: 1, total: 1 } },
    { id: 3, event: 'cancelled', data: {} },
  ]);
  const seen = await collect('t3');
  const step = seen[1]!;
  if (step.event !== 'step') throw new Error('unreachable');
  assert.equal(step.data.jobTypes, undefined);
  assert.equal(seen.at(-1)!.event, 'cancelled');
});

test("a module's skipped entries say why", async () => {
  sse([
    { id: 1, event: 'started', data: { type: 'module' } },
    { id: 2, event: 'step', data: { name: 'pull model x', index: 1, total: 1 } },
    { id: 3, event: 'skipped', data: { reason: 'pull model x: already installed' } },
    { id: 4, event: 'done', data: {} },
  ]);
  const seen = await collect('t4');
  const skipped = seen[2]!;
  if (skipped.event !== 'skipped') throw new Error('unreachable');
  assert.match(skipped.data.reason, /already installed/);
});

test('a failed task event carries the code and stops the stream', async () => {
  sse([
    { id: 1, event: 'started', data: { type: 'install' } },
    { id: 2, event: 'failed', data: { code: 'reload_refused', message: 'a chat holds it' } },
  ]);
  const seen = await collect('t5');
  const failed = seen[1]!;
  if (failed.event !== 'failed') throw new Error('unreachable');
  assert.equal(failed.data.code, 'reload_refused');
  assert.equal(seen.length, 2);
});

test('a progress frame that is neither shape is a protocol error', async () => {
  sse([
    { id: 1, event: 'started', data: { type: 'pull' } },
    { id: 2, event: 'progress', data: { fraction: 0.5 } },
  ]);
  await assert.rejects(() => collect('t6'), CrucibleProtocolError);
});

test('an event kind this build never heard of costs one frame, not the stream', async () => {
  sse([
    { id: 1, event: 'started', data: { type: 'pull' } },
    { id: 2, event: 'verifying', data: { sha256: 'abc' } },
    { id: 3, event: 'done', data: {} },
  ]);
  const seen = await collect('t7');
  assert.deepEqual(seen.map((event) => event.event), ['started', 'unknown', 'done']);
  const unknown = seen[1]!;
  if (unknown.event !== 'unknown') throw new Error('unreachable');
  assert.equal(unknown.kind, 'verifying');
  assert.deepEqual(unknown.data, { sha256: 'abc' });
});

test('a stream that ends without a terminal event is not a finished task', async () => {
  sse([{ id: 1, event: 'started', data: { type: 'pull' } }]);
  await assert.rejects(() => collect('t8'), /ended after event 1 without a terminal event/);
});

// ------------------------------------------------------------------ cancel

test('cancelTask answers cancelling, which is not cancelled', async () => {
  answer(200, { task_id: 't1', status: 'cancelling' });
  const result = await client().cancelTask('t1');
  assert.equal(lastMethod, 'DELETE');
  assert.equal(lastPath, '/v1/tasks/t1');
  assert.deepEqual(result, { taskId: 't1', status: 'cancelling' });
});

test('a server answering cancelled to a DELETE is a protocol error', async () => {
  answer(200, { task_id: 't1', status: 'cancelled' });
  await assert.rejects(() => client().cancelTask('t1'), CrucibleProtocolError);
});

test('cancelling a finished task is refused by name', async () => {
  answer(409, {
    error: { code: 'not_running', message: 'task t1 is already done' },
  });
  await assert.rejects(() => client().cancelTask('t1'), CrucibleRefused);
});
