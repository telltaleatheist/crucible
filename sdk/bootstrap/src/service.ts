/**
 * `ensureRunning()` — the service is up; a no-op when it already is.
 *
 * Nothing is spawned as a child that should be a service (PHASE5-APPS.md
 * section 6.0). Starting is `systemctl --user start crucible.service` on
 * cuda-linux and `launchctl kickstart` on mlx-darwin — and both are reached
 * through `crucible service start`, because the unit name, the launchd label
 * and the domain are facts `crucible/service.py` owns (ARCHITECTURE.md R1) and
 * a second spelling of them here would be the drift this package exists to
 * remove. `crucible service status --json` is the question; its exit code is
 * not read, because it is 1 for "installed and stopped" and the JSON says why.
 *
 * Linger is REPORTED, never assumed and never granted: `loginctl enable-linger`
 * is a change to somebody's machine that needs root, so it comes back as a
 * fact beside success, with the command, and the host decides.
 */
import { BootstrapRefusal } from './errors.js';
import { consoleScriptBeside, DEFAULT_CONDA_ROOTS, probeInterpreter } from './host.js';
import { processRunner, type Runner } from './runner.js';
import { describeTarget, resolveTarget, runOn, type Target } from './target.js';

const STATUS_TIMEOUT_MS = 60_000;
const START_TIMEOUT_MS = 2 * 60_000;

export interface EnsureRunningOptions {
  /** Required on win32. Ignored elsewhere. */
  distro?: string;
  /** `CRUCIBLE_HOME`, as the target spells it. Omit for the server's default. */
  home?: string;
  /** Where to look for conda. Defaults to {@link DEFAULT_CONDA_ROOTS}. */
  condaRoots?: readonly string[];
}

export interface RunningService {
  running: true;
  /** The service's main pid, or null when the supervisor reported none. */
  pid: number | null;
  mechanism: 'systemd' | 'launchd';
  /** The unit or the plist. */
  definition: string;
  /**
   * systemd only. `true`/`false` when loginctl answered; `null` when it could
   * not be asked, or on launchd where the question does not exist. `false`
   * means this server stops when the user's last session ends and does not
   * start at boot — a fact, reported beside success.
   */
  linger: boolean | null;
  /** The command that grants linger, when it is off. Root's to run, never this package's. */
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
  const target = resolveTarget(runner, options.distro);
  const interpreter = await probeInterpreter(runner, target, options.condaRoots ?? DEFAULT_CONDA_ROOTS);
  const refusal = interpreter.refusals[0];
  if (refusal !== undefined || interpreter.python === null) throw refusal ?? new Error('unreachable: no python and no refusal');
  const crucible = consoleScriptBeside(interpreter.python.path);
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
  if (before.running) return running(before, false);

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
  return running(after, true);
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

function running(status: ServiceStatus, started: boolean): RunningService {
  return {
    running: true,
    pid: status.pid,
    mechanism: status.mechanism,
    definition: status.definition,
    linger: status.linger,
    enableLinger: status.mechanism === 'systemd' && status.linger === false ? 'sudo loginctl enable-linger "$USER"' : null,
    started,
  };
}

/** Where the service's own words are, when it will not come up. */
function logsCommand(status: ServiceStatus): string {
  return status.mechanism === 'systemd'
    ? 'journalctl --user -u crucible.service -n 100 --no-pager'
    : `crucible service status; tail -n 100 "$(dirname "${status.definition}")/../../.crucible/logs/serve.log"`;
}
