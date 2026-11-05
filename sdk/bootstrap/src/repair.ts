import {
  CrucibleClient,
  CrucibleRefused,
  type RepairResult,
  type RepairScope,
  type TaskEvent,
  type TaskRepairData,
  type TaskStatus,
} from '@crucible/client';

import { readLocalConfig, type LocalConfigOptions } from './config.js';
import { resolveDistro } from './distro.js';
import { BootstrapRefusal } from './errors.js';
import { backendFor } from './release.js';
import { installRuntime, probeGuest, type GuestFacts } from './runtime.js';
import { processRunner, type Runner } from './runner.js';
import { ensureRunning, readStatus } from './service.js';
import { describeTarget, resolveTarget, type Target } from './target.js';
import { BOOTSTRAP_VERSION } from './version.js';

/** Long enough to fetch the pinned interpreter and the wheel and install them. */
const SERVER_REINSTALL_TIMEOUT_MS = 30 * 60_000;
/** One operator task at a time: another one is waited on, for this long at most. */
const TASK_BUSY_WAIT_MS = 30 * 60_000;
const TASK_BUSY_POLL_MS = 15_000;

export type RepairStepName = 'server' | 'service' | 'repair';

export interface RepairStep {
  name: RepairStepName;
  status: 'running' | 'ok' | 'failed';
  detail: string;
}

/** What `repair()` needs of a client: the three calls it makes. */
export interface RepairClient {
  submitTask(request: { type: 'repair'; scope?: RepairScope }): Promise<string>;
  taskEvents(taskId: string): AsyncIterable<TaskEvent>;
  task(taskId: string): Promise<TaskStatus>;
}

export interface RepairOptions extends LocalConfigOptions {
  /** What the server's task looks at. Left out: everything installed, repaired. */
  scope?: RepairScope;
  clientName?: string;
  onLine?: (line: string, stream: 'stdout' | 'stderr') => void;
  onStep?: (step: RepairStep) => void;
  /** Each thing the task judges or replaces. */
  onRepair?: (data: TaskRepairData) => void;
  /** Bytes read so far of `bytesToCheck`, or bytes of a download. */
  onProgress?: (progress: { bytesDone: number; bytesTotal: number | null; file: string }) => void;
  /** Injectable so a test can script the answers without a server, or a wait. */
  clientFactory?: (options: { url: string; token: string; clientName: string }) => RepairClient;
  sleep?: (ms: number) => Promise<void>;
  now?: () => number;
}

export interface LocalRepairResult {
  /**
   * `reinstalled` when the server itself would not answer, and the release it names went back.
   * `broken`, on a verify-only run alone: it would not answer, and nothing was done about it.
   */
  server: 'ok' | 'reinstalled' | 'broken';
  /** `stopped`, on a verify-only run alone: it is down, and it was not started. */
  service: 'running' | 'started' | 'stopped';
  /**
   * What the server's own repair task found and did: the one result both doors answer. Null only on
   * a verify-only run whose server or service was not up to ask, which `detail` then says.
   */
  task: RepairResult | null;
  healthy: boolean;
  /** Why a verify-only run could not ask the server; null otherwise. */
  detail: string | null;
}

/** Why the installed server cannot be trusted as it stands, or null when it can. */
function serverProblem(facts: GuestFacts): string | null {
  const server = facts.server;
  if (server === null) return `there is no server at ${facts.home}`;
  if (server.version === null || !/\d+\.\d+/.test(server.version)) {
    return `${server.crucible} would not say its version (${server.version ?? 'no answer'})`;
  }
  if (server.release !== null && !server.version.includes(server.release)) {
    return `${server.crucible} says ${server.version} and its stamp says ${server.release}`;
  }
  return null;
}

/**
 * A `scope.verifyOnly` run changes nothing, on this machine as on the server: a broken runtime or a
 * stopped service is reported, not cured. Null when both are up, and the task can be asked.
 */
async function verifyOnlyAnswer(
  runner: Runner,
  target: Target,
  facts: GuestFacts,
  options: RepairOptions,
  say: (name: RepairStepName) => (status: RepairStep['status'], detail: string) => void,
): Promise<LocalRepairResult | null> {
  const problem = serverProblem(facts);
  if (problem !== null || facts.server === null) {
    say('server')('failed', problem ?? 'there is no server');
    return { server: 'broken', service: 'stopped', task: null, healthy: false, detail: problem };
  }
  say('server')('ok', facts.server.version ?? '');
  const env = options.home === undefined ? undefined : { CRUCIBLE_HOME: options.home };
  const status = await readStatus(runner, target, facts.server.crucible, env, describeTarget(target));
  if (!status.running) {
    const detail = `the service is not running (${status.detail}); a verify-only run does not start it`;
    say('service')('failed', detail);
    return { server: 'ok', service: 'stopped', task: null, healthy: false, detail };
  }
  say('service')('ok', 'running');
  return null;
}

async function repairServer(
  runner: Runner,
  target: Target,
  facts: GuestFacts,
  options: RepairOptions,
  step: (status: RepairStep['status'], detail: string) => void,
): Promise<'ok' | 'reinstalled'> {
  const problem = serverProblem(facts);
  if (problem === null) {
    step('ok', facts.server?.version ?? '');
    return 'ok';
  }
  const release = facts.server?.release ?? null;
  if (release === null) {
    step('failed', problem);
    throw new BootstrapRefusal(
      'server_unrepairable',
      `${problem} inside ${describeTarget(target)}, and nothing there names the release it was, so there is nothing `
        + 'to put back. install() puts a server there.',
      { detail: problem },
    );
  }
  step('running', `${problem}; putting ${release} back`);
  await installRuntime(runner, target, {
    release,
    backend: backendFor(target.kind === 'wsl' ? 'linux' : runner.platform),
    home: facts.home,
    // Nothing that is there is believed: the interpreter comes back as well as the wheel.
    installed: null,
    rollbackTo: null,
    timeoutMs: SERVER_REINSTALL_TIMEOUT_MS,
    onLine: (line, stream) => options.onLine?.(line, stream),
  });
  step('ok', `${release} was put back`);
  return 'reinstalled';
}

async function submitWhenFree(
  client: RepairClient,
  scope: RepairScope | undefined,
  options: RepairOptions,
  step: (status: RepairStep['status'], detail: string) => void,
): Promise<string> {
  const sleep = options.sleep ?? ((ms: number) => new Promise<void>((resolve) => { setTimeout(resolve, ms); }));
  const now = options.now ?? Date.now;
  const deadline = now() + TASK_BUSY_WAIT_MS;
  for (;;) {
    try {
      return await client.submitTask(scope === undefined ? { type: 'repair' } : { type: 'repair', scope });
    } catch (error) {
      if (!(error instanceof CrucibleRefused) || error.code !== 'task_busy' || now() >= deadline) throw error;
      step('running', `waiting its turn: ${error.serverMessage}`);
      await sleep(TASK_BUSY_POLL_MS);
    }
  }
}

/**
 * Repair this machine's Crucible: the server and its service, which cannot be repaired from inside
 * themselves, and then the envs and weights, which the server's own `repair` task verifies by hash
 * and puts back. A server on another machine has only that task
 * (`client.submitTask({type: 'repair'})`), and it answers the same `RepairResult`.
 *
 * What the task could not put back is in the answer's `task.failed` with `healthy: false`, so an
 * app shows one view for both doors; what refuses is a machine with nothing to repair, or a task
 * that did not finish.
 */
export async function repair(options: RepairOptions = {}, runner: Runner = processRunner()): Promise<LocalRepairResult> {
  const say = (name: RepairStepName) => (status: RepairStep['status'], detail: string): void => {
    options.onStep?.({ name, status, detail });
  };
  const where = {
    ...(options.distro === undefined ? {} : { distro: options.distro }),
    ...(options.exact === undefined ? {} : { exact: options.exact }),
  };
  const home = options.home === undefined ? {} : { home: options.home };

  const distro = runner.platform === 'win32' ? await resolveDistro(runner, where) : options.distro;
  const target = resolveTarget(runner, distro);
  say('server')('running', 'asking the server what it is');
  const facts = await probeGuest(runner, target, options.home);
  let server: LocalRepairResult['server'] = 'ok';
  let service: LocalRepairResult['service'] = 'running';
  if (options.scope?.verifyOnly === true) {
    const answer = await verifyOnlyAnswer(runner, target, facts, options, say);
    if (answer !== null) return answer;
  } else {
    server = await repairServer(runner, target, facts, options, say('server'));
    say('service')('running', '');
    const running = await ensureRunning({ ...where, ...home }, runner);
    service = running.started ? 'started' : 'running';
    say('service')('ok', service);
  }

  const config = await readLocalConfig({ ...where, ...home }, runner);
  const clientName = options.clientName ?? `crucible-bootstrap/${BOOTSTRAP_VERSION}`;
  const factory = options.clientFactory ?? ((given) => new CrucibleClient(given) as unknown as RepairClient);
  const client = factory({ url: config.url, token: config.token, clientName });

  say('repair')('running', 'asking the server to check what it has');
  const taskId = await submitWhenFree(client, options.scope, options, say('repair'));
  for await (const event of client.taskEvents(taskId)) {
    if (event.event === 'repair') options.onRepair?.(event.data);
    if (event.event === 'progress' && 'bytesDone' in event.data) options.onProgress?.(event.data);
    if (event.event === 'progress' && 'line' in event.data) options.onLine?.(event.data.line, 'stdout');
  }
  const finished = await client.task(taskId);
  if (finished.state !== 'done' || finished.result === null) {
    const said = finished.error === null ? finished.state : `${finished.error.code}: ${finished.error.message}`;
    say('repair')('failed', said);
    throw new BootstrapRefusal(
      'repair_failed',
      `the repair task ${taskId} on ${config.name} ended ${finished.state}: ${said}`,
      { detail: said },
    );
  }
  say('repair')(finished.result.healthy ? 'ok' : 'failed', finished.result.healthy ? 'healthy' : 'unhealthy');
  return { server, service, task: finished.result, healthy: finished.result.healthy, detail: null };
}
