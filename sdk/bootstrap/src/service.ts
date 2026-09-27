import { resolveDistro } from './distro.js';
import { BootstrapRefusal } from './errors.js';
import { probeGuest, requireRuntime } from './runtime.js';
import { processRunner, type Runner } from './runner.js';
import { describeTarget, resolveTarget, runOn, type Target } from './target.js';

const STATUS_TIMEOUT_MS = 60_000;
const START_TIMEOUT_MS = 2 * 60_000;

export interface EnsureRunningOptions {
  /** win32: the app's WSL distro setting; the `crucible` distro wins when it exists. */
  distro?: string;
  /** win32: use `distro` verbatim, resolving nothing. */
  exact?: boolean;
  /** `CRUCIBLE_HOME`, as the target spells it. */
  home?: string;
}

export interface RunningService {
  running: true;
  /** The service's main pid, or null when the supervisor reported none. */
  pid: number | null;
  mechanism: 'systemd' | 'launchd';
  /** The unit or the plist. */
  definition: string;
  /** Native-Linux systemd linger; null where the question does not apply. */
  linger: boolean | null;
  /** The command that grants linger, when it is off on native Linux. */
  enableLinger: string | null;
  /** Whether this call started it, or found it already up. */
  started: boolean;
}

/** What `crucible service status --json` prints (`crucible/service.py` `Status.to_dict`). */
export interface ServiceStatus {
  mechanism: 'systemd' | 'launchd';
  definition: string;
  installed: boolean;
  running: boolean;
  pid: number | null;
  detail: string;
  linger: boolean | null;
}

export function parseServiceStatus(stdout: string): ServiceStatus | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(stdout);
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== 'object') return null;
  const table = parsed as Record<string, unknown>;
  const mechanism = table['mechanism'];
  const definition = table['definition'];
  const installed = table['installed'];
  const running = table['running'];
  const pid = table['pid'];
  const detail = table['detail'];
  const linger = table['linger'];
  if ((mechanism !== 'systemd' && mechanism !== 'launchd')
    || typeof definition !== 'string'
    || typeof installed !== 'boolean'
    || typeof running !== 'boolean'
    || !(pid === null || (typeof pid === 'number' && Number.isInteger(pid)))
    || typeof detail !== 'string'
    || !(linger === null || typeof linger === 'boolean')) {
    return null;
  }
  return { mechanism, definition, installed, running, pid, detail, linger };
}

export async function ensureRunning(options: EnsureRunningOptions = {}, runner: Runner = processRunner()): Promise<RunningService> {
  const distro = runner.platform === 'win32'
    ? await resolveDistro(runner, {
      ...(options.distro === undefined ? {} : { distro: options.distro }),
      ...(options.exact === undefined ? {} : { exact: options.exact }),
    })
    : options.distro;
  const target = resolveTarget(runner, distro);
  const crucible = requireRuntime(target, await probeGuest(runner, target, options.home)).crucible;
  const env = options.home === undefined ? undefined : { CRUCIBLE_HOME: options.home };
  const where = describeTarget(target);

  const before = await readStatus(runner, target, crucible, env, where);
  if (!before.installed) {
    throw new BootstrapRefusal(
      'service_not_installed',
      `there is no Crucible service inside ${where}: ${before.definition} does not exist (${before.detail}). install() writes it.`,
      { command: `${crucible} service install`, detail: before.detail },
    );
  }
  if (before.running) return running(before, false, target);

  const start = await runOn(runner, target, [crucible, 'service', 'start'], { timeoutMs: START_TIMEOUT_MS, ...(env === undefined ? {} : { env }) });
  if (start.failure !== null || start.code !== 0) {
    const said = (start.stderr.trim() || start.stdout.trim()) || 'no output';
    throw new BootstrapRefusal(
      'service_failed',
      `crucible service start ${start.failure ?? `exited ${start.code}`} inside ${where}: ${said}`,
      { command: logsCommand(before), detail: said },
    );
  }
  const after = await readStatus(runner, target, crucible, env, where);
  if (!after.running) {
    throw new BootstrapRefusal(
      'service_failed',
      `the Crucible service inside ${where} was started and is not running: ${after.detail}`,
      { command: logsCommand(after), detail: after.detail },
    );
  }
  return running(after, true, target);
}

async function readStatus(
  runner: Runner,
  target: Target,
  crucible: string,
  env: Readonly<Record<string, string>> | undefined,
  where: string,
): Promise<ServiceStatus> {
  const result = await runOn(runner, target, [crucible, 'service', 'status', '--json'], { timeoutMs: STATUS_TIMEOUT_MS, ...(env === undefined ? {} : { env }) });
  if (result.failure !== null) {
    throw new BootstrapRefusal('service_failed', `crucible service status could not be asked inside ${where}: ${result.failure}`, { detail: (result.stderr || result.stdout).trim() });
  }
  const status = parseServiceStatus(result.stdout);
  if (status !== null) return status;
  const said = (result.stderr.trim() || result.stdout.trim()) || 'no output';
  if (/no config at/.test(said)) {
    throw new BootstrapRefusal('no_local_config', `no local Crucible inside ${where}: ${said}. install() puts one there.`, { detail: said });
  }
  throw new BootstrapRefusal(
    'service_failed',
    `crucible service status --json inside ${where} answered something that is not its status (exit ${result.code}): ${said}`,
    { detail: said },
  );
}

function running(
  status: ServiceStatus,
  started: boolean,
  target: Target,
): RunningService {
  const linger = target.kind === 'wsl' ? null : status.linger;
  const handOver = status.mechanism === 'systemd' && linger === false;
  return {
    running: true,
    pid: status.pid,
    mechanism: status.mechanism,
    definition: status.definition,
    linger,
    enableLinger: handOver ? 'sudo loginctl enable-linger "$USER"' : null,
    started,
  };
}

function logsCommand(status: ServiceStatus): string {
  return status.mechanism === 'systemd'
    ? 'journalctl --user -u crucible.service -n 100 --no-pager'
    : `crucible service status; tail -n 100 "$(dirname "${status.definition}")/../../.crucible/logs/serve.log"`;
}
