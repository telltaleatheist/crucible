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
 * WHICH CRUCIBLE, AND WHERE. On win32 the distro is resolved by `distro.ts`'s
 * one rule — the `crucible` distro when there is one, else the app's setting —
 * and the binary is the SERVER PACK's, `<CRUCIBLE_HOME>/server/bin/crucible`.
 * A host with no pack is `no_server_pack`: there is no interpreter to go
 * looking for, because the interpreter arrives in the pack.
 *
 * LINGER IS GRANTED ON WIN32 AND REPORTED EVERYWHERE ELSE, and this is where it
 * matters most: without it a systemd user unit dies with the user's last
 * session, so `ensureRunning()` would answer `running: true` about a server
 * that is about to disappear. Inside WSL there is no elevation to hand over —
 * `wsl.exe -u root` is how the guest is entered, measured 2026-09-14 — so the
 * sudo line this used to return is replaced by the command itself, run
 * idempotently (`linger.ts`). On native Linux `sudo` really is elevation and
 * the host app really is the one that can obtain it, so there the fact is
 * still reported with the command and nothing is attempted.
 */
import { resolveDistro } from './distro.js';
import { BootstrapRefusal } from './errors.js';
import { ensureLinger, type LingerOutcome } from './linger.js';
import { probeGuest, requirePack } from './pack.js';
import { processRunner, type Runner } from './runner.js';
import { describeTarget, resolveTarget, runOn, type Target } from './target.js';

const STATUS_TIMEOUT_MS = 60_000;
const START_TIMEOUT_MS = 2 * 60_000;

export interface EnsureRunningOptions {
  /** win32: the app's WSL distro setting. The `crucible` distro wins when it exists. */
  distro?: string;
  /** win32: use `distro` verbatim, resolving nothing. */
  exact?: boolean;
  /** `CRUCIBLE_HOME`, as the target spells it. Omit for the server's default. */
  home?: string;
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
   * not be asked, or on launchd where the question does not exist.
   *
   * **On win32 this is `true` or the call threw.** There is nothing to report:
   * linger was granted if it was off, and a guest that would not say is a
   * named refusal rather than a `false` beside a cheerful `running: true`.
   */
  linger: boolean | null;
  /**
   * The command that grants linger, when it is off and this package may not
   * run it. **Always null on win32** — there the command was run. On native
   * Linux `sudo` is real elevation and this is the line for the host app.
   */
  enableLinger: string | null;
  /**
   * win32 only: what the linger step did, or null on macOS and native Linux
   * where this package does not touch it. `granted: false` with
   * `linger: true` means it was already on.
   */
  lingerStep: LingerOutcome | null;
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
  // THE SERVER PACK, not an interpreter found on the machine: there is nothing
  // to find on a fresh host, and a host with no pack has no service either, so
  // that is one named refusal rather than a hunt (PHASE14 section 0).
  const crucible = requirePack(target, await probeGuest(runner, target, options.home)).crucible;
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
  // ASKED EVEN WHEN IT IS ALREADY UP, because "running" and "will still be
  // running after this person logs out" are different facts and only the
  // second is what `ensureRunning` is for.
  if (before.running) return running(before, false, await ensureLinger(runner, target, env));

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
  return running(after, true, await ensureLinger(runner, target, env));
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
  linger: LingerOutcome | null,
): RunningService {
  // `linger !== null` is exactly "this is win32/WSL", because `ensureLinger`
  // answers null for every other target. Where it ran, its answer wins over
  // the status read — the status was taken before the grant.
  const handOver = linger === null
    && status.mechanism === 'systemd'
    && status.linger === false;
  return {
    running: true,
    pid: status.pid,
    mechanism: status.mechanism,
    definition: status.definition,
    linger: linger === null ? status.linger : linger.linger,
    enableLinger: handOver ? 'sudo loginctl enable-linger "$USER"' : null,
    lingerStep: linger,
    started,
  };
}

/** Where the service's own words are, when it will not come up. */
function logsCommand(status: ServiceStatus): string {
  return status.mechanism === 'systemd'
    ? 'journalctl --user -u crucible.service -n 100 --no-pager'
    : `crucible service status; tail -n 100 "$(dirname "${status.definition}")/../../.crucible/logs/serve.log"`;
}
