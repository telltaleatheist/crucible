/** `repair()` — the server and service first, then the server's own repair task, one result. */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { CrucibleRefused, type RepairResult, type TaskEvent, type TaskStatus } from '@crucible/client';

import { backendFor, interpreterFor, repair, type RepairClient, type RepairStep } from '../src/index.js';
import { FakeRunner, refusal } from './fake.js';

const HOME = '/home/owen/.crucible';
const CONFIG = `[server]\nname = "crucible@test"\nhost = "127.0.0.1"\nport = 7100\n[auth]\ntoken = "test-token-not-a-secret"\n`;

function probe(version: string | null, release: string | null): string {
  return [
    `home=${HOME}`,
    'user=owen',
    'free_kib=999999999',
    ...(version === null && release === null ? [] : [`crucible=${HOME}/server/bin/crucible`]),
    ...(version === null ? [] : [`version=${version}`]),
    ...(release === null ? [] : [`release=${release}`, 'python_sha256=abc']),
  ].join('\n') + '\n';
}

const SERVICE = 'ActiveState=active\nLoadState=loaded\n';

const HEALTHY: RepairResult = {
  healthy: true,
  verifyOnly: false,
  checked: [{ kind: 'env', id: 'rvc' }],
  broken: [],
  repaired: [],
  failed: [],
  unverified: [],
  notHashChecked: [],
  bytesHashed: 10,
  bytesSkippedVerified: 0,
};

function status(result: RepairResult | null, state: TaskStatus['state'] = 'done'): TaskStatus {
  return {
    taskId: 't1',
    type: 'repair',
    request: {},
    state,
    error: state === 'failed' ? { code: 'task_failed', message: 'it broke' } : null,
    created: 'c',
    started: 's',
    finished: 'f',
    unmet: [],
    result,
    message: null,
  } as unknown as TaskStatus;
}

function client(result: TaskStatus, events: TaskEvent[] = [], busy = 0): RepairClient & { asked: unknown[] } {
  let refusals = busy;
  const asked: unknown[] = [];
  return {
    asked,
    submitTask: async (request) => {
      asked.push(request);
      if (refusals > 0) {
        refusals -= 1;
        throw new CrucibleRefused(409, 'task_busy', 'a module task is running', {});
      }
      return 't1';
    },
    taskEvents: async function* () { yield* events; },
    task: async () => result,
  };
}

const SERVICE_UP = JSON.stringify({
  mechanism: 'launchd',
  definition: '/home/owen/Library/LaunchAgents/crucible.plist',
  installed: true,
  running: true,
  pid: 1,
  detail: 'running',
  linger: null,
});

const HOST = { platform: 'darwin' as const, homedir: '/home/owen', files: { [`${HOME}/config.toml`]: CONFIG } };

/** A Mac whose runtime answers `version`: the probe, ensureRunning's probe, the service status. */
function machine(version: string | null, release: string | null): FakeRunner {
  const said = { argv: () => true, stdout: probe(version, release) };
  return new FakeRunner(HOST, [said, said, { argv: () => true, stdout: SERVICE_UP }]);
}

test('a server that answers is left alone, and the task\'s result is the answer', async () => {
  const fake = client(status(HEALTHY));
  const steps: RepairStep[] = [];
  const answer = await repairWith(fake, machine('crucible 2.0.34', '2.0.34'), { onStep: (s) => steps.push(s) });
  assert.equal(answer.server, 'ok');
  assert.equal(answer.healthy, true);
  assert.deepEqual(answer.task, HEALTHY);
  assert.deepEqual(fake.asked, [{ type: 'repair' }]);
  assert.deepEqual([...new Set(steps.map((s) => s.name))], ['server', 'service', 'repair']);
});

test('the scope goes to the task as it was given', async () => {
  const fake = client(status(HEALTHY));
  await repairWith(fake, machine('crucible 2.0.34', '2.0.34'), { scope: { verifyOnly: true } });
  assert.deepEqual(fake.asked, [{ type: 'repair', scope: { verifyOnly: true } }]);
});

test('what could not be put back is an unhealthy answer, not a refusal', async () => {
  const failed = { ...HEALTHY, healthy: false, failed: [{ subject: { kind: 'env', id: 'rvc' }, code: 'env_build_failed', message: 'no' }] };
  const answer = await repairWith(client(status(failed)), machine('crucible 2.0.34', '2.0.34'));
  assert.equal(answer.healthy, false);
  assert.equal(answer.task.failed[0]?.code, 'env_build_failed');
});

test('a task that did not finish is repair_failed, with what it said', async () => {
  const r = await refusal(repairWith(client(status(null, 'failed')), machine('crucible 2.0.34', '2.0.34')));
  assert.equal(r.code, 'repair_failed');
  assert.match(r.message, /task_failed: it broke/);
});

test('another task running is waited on, and then the repair is taken', async () => {
  const fake = client(status(HEALTHY), [], 2);
  const slept: number[] = [];
  const answer = await repairWith(fake, machine('crucible 2.0.34', '2.0.34'), {
    sleep: async (ms) => { slept.push(ms); },
  });
  assert.equal(answer.healthy, true);
  assert.equal(fake.asked.length, 3);
  assert.equal(slept.length, 2);
});

test('it does not wait forever', async () => {
  let clock = 0;
  const fake = client(status(HEALTHY), [], 1000);
  await assert.rejects(
    repairWith(fake, machine('crucible 2.0.34', '2.0.34'), {
      sleep: async () => { clock += 60_000; },
      now: () => clock,
    }),
    /task_busy/,
  );
});

test('the task\'s frames reach the caller by kind', async () => {
  const events = [
    { id: 1, event: 'repair', data: { step: 'plan', state: 'running', subject: null, detail: null, bytesToCheck: 9 } },
    { id: 2, event: 'progress', data: { bytesDone: 3, bytesTotal: 9, file: 'a' } },
    { id: 3, event: 'progress', data: { line: 'a pip line' } },
  ] as unknown as TaskEvent[];
  const repairs: unknown[] = [];
  const progress: unknown[] = [];
  const lines: string[] = [];
  await repairWith(client(status(HEALTHY), events), machine('crucible 2.0.34', '2.0.34'), {
    onRepair: (d) => repairs.push(d),
    onProgress: (p) => progress.push(p),
    onLine: (l) => lines.push(l),
  });
  assert.equal(repairs.length, 1);
  assert.deepEqual(progress, [{ bytesDone: 3, bytesTotal: 9, file: 'a' }]);
  assert.deepEqual(lines, ['a pip line']);
});

test('a server that will not answer has the release it names put back, and then the task runs', async () => {
  const pin = interpreterFor(backendFor('darwin'));
  const wheelSha = 'b'.repeat(64);
  const up = { argv: () => true, stdout: probe('crucible 2.0.34', '2.0.34') };
  const runner = new FakeRunner(HOST, [
    { argv: () => true, stdout: probe(null, '2.0.34') },
    { argv: (argv) => (argv[2] ?? '').includes(pin.asset) && (argv[2] ?? '').includes('curl') },
    { argv: (argv) => argv[0] === 'shasum', stdout: `${pin.sha256}  x
` },
    { argv: (argv) => (argv[2] ?? '').includes('tar -xzf') },
    { argv: (argv) => (argv[2] ?? '').includes('curl') },
    { argv: (argv) => argv[0] === 'curl', stdout: `${wheelSha}  w
` },
    { argv: (argv) => argv[0] === 'shasum', stdout: `${wheelSha}  w
` },
    { argv: (argv) => (argv[2] ?? '').includes('pip install') },
    up,
    { argv: () => true, stdout: SERVICE_UP },
  ]);
  const answer = await repairWith(client(status(HEALTHY)), runner);
  assert.equal(answer.server, 'reinstalled');
  assert.equal(answer.healthy, true);
  runner.assertDrained();
});

test('a machine with no server and no release to put back is server_unrepairable', async () => {
  const r = await refusal(repairWith(client(status(HEALTHY)), machine(null, null)));
  assert.equal(r.code, 'server_unrepairable');
  assert.match(r.message, /install\(\) puts a server there/);
});

function repairWith(
  fake: RepairClient,
  runner: FakeRunner,
  extra: Parameters<typeof repair>[0] = {},
): ReturnType<typeof repair> {
  return repair({ clientFactory: () => fake, ...extra }, runner);
}
