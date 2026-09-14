/**
 * `install()` — this host has a Crucible, at the version the host app handed
 * over, with the job types it asked for, running as this machine's service.
 *
 * The sequence, every step by name, every step idempotent:
 *
 *   interpreter        the server's Python 3.11 exists (refused by name if not)
 *   pip-install        `<python> -m pip install <wheel>` — the wheel the HOST
 *                      names, a release path or URL; never guessed
 *   init               `crucible init --token <minted here> --enable-<type>…`
 *                      SKIPPED when a config already exists (its token is kept)
 *   install-<type>     `crucible install <type> [--narrator-engine e] --verbose`
 *   service-install    `crucible service install`
 *   linger             win32 only: `loginctl enable-linger <guest user>` as
 *                      root, SKIPPED when it is already on. See `linger.ts`:
 *                      `wsl.exe -u root` is how the guest is entered, not an
 *                      escalation, so there is no elevation to hand over
 *   capability-write   `crucible capability --write`
 *
 * Pulls are NOT part of this: weights are the app's, later, per model.
 *
 * The token is minted on the client — `crucible init --token` — so the app
 * that installed the server already holds what `readLocalConfig()` would read
 * back. It is never logged: the argv reported for the init step spells it
 * `<redacted>`, and `onLine` only ever sees what the step printed.
 *
 * Every step's stdout and stderr stream to `onLine` as they arrive. A failing
 * step stops the sequence with a {@link BootstrapStepFailed} naming the step,
 * carrying the tail, and listing the steps that finished — what they did is on
 * disk and stays there (ARCHITECTURE.md R6).
 */
import { randomBytes } from 'node:crypto';

import { readLocalConfig, type LocalConfig } from './config.js';
import { BootstrapRefusal, BootstrapStepFailed } from './errors.js';
import { consoleScriptBeside, DEFAULT_CONDA_ROOTS, probeInterpreter } from './host.js';
import { ensureLinger } from './linger.js';
import { processRunner, type OutputStream, type Runner } from './runner.js';
import { describeTarget, resolveTarget, streamOn, type Target } from './target.js';
import { guestPathFor } from './wsl.js';

/** The job types `crucible init --enable-<type>` knows, in the order the CLI lists them. */
export const JOB_TYPES = ['echo', 'llm', 'asr', 'tts', 'align', 'rvc', 'denoise'] as const;
export type JobType = (typeof JOB_TYPES)[number];

/** The job types `crucible install <type>` has an installer for (`cli.INSTALLABLE_JOB_TYPES`). */
export const INSTALLABLE_JOB_TYPES: readonly JobType[] = ['llm', 'tts', 'asr', 'align', 'rvc'];

/**
 * One job type to enable. `tts` must say which narrator engine, because
 * cuda-linux has one env per engine and `crucible install tts` refuses without
 * it; a bare `'tts'` is refused here by name for the same reason.
 */
export type JobTypeRequest = Exclude<JobType, 'tts'> | { type: 'tts'; narratorEngine: string };

export interface InstallTimeouts {
  /** The interpreter probe and every `crucible` verb that builds nothing. */
  quickMs: number;
  /** `pip install <wheel>`. */
  pipMs: number;
  /** `crucible install <type>`, which builds a multi-gigabyte venv. */
  envMs: number;
}

export const DEFAULT_INSTALL_TIMEOUTS: InstallTimeouts = {
  quickMs: 5 * 60_000,
  pipMs: 30 * 60_000,
  envMs: 90 * 60_000,
};

export interface InstallOptions {
  /** Required on win32. Ignored elsewhere. */
  distro?: string;
  jobTypes: readonly JobTypeRequest[];
  /** `CRUCIBLE_HOME` for every `crucible` verb, as the target spells it. Omit for the server's default. */
  home?: string;
  /** The release wheel: an absolute path on this machine, or an `http(s)://` URL. */
  wheel: string;
  /** Every line a step prints, as it prints it. */
  onLine: (line: string, stream: OutputStream, step: string) => void;
  /** Optional: a step beginning, finishing, or being skipped. */
  onStep?: (step: InstallStep) => void;
  /** What `crucible init` should bind. Omit for the server's own defaults (127.0.0.1:7100). */
  bind?: { host?: string; port?: number };
  /** Where to look for conda. Defaults to {@link DEFAULT_CONDA_ROOTS}. */
  condaRoots?: readonly string[];
  timeouts?: Partial<InstallTimeouts>;
}

export interface InstallStep {
  name: string;
  /** What ran, with the token spelled `<redacted>`. Empty for a skipped step. */
  argv: readonly string[];
  status: 'running' | 'ok' | 'skipped';
  detail: string;
}

export interface InstallResult {
  steps: InstallStep[];
  /** The server as its config now describes it. The token is not here; `readLocalConfig()` is. */
  server: { name: string; url: string; configPath: string };
}

/** A 32-byte urlsafe token — the same shape `crucible.config.mint_token` produces. */
export function mintToken(): string {
  return randomBytes(32).toString('base64url');
}

const TAIL_LINES = 40;

interface Plan {
  enableFlags: string[];
  installs: { type: JobType; argv: string[] }[];
}

/** Turn the request into `--enable-*` flags and `crucible install` argv, or refuse by name. */
export function planJobTypes(requests: readonly JobTypeRequest[]): Plan {
  if (requests.length === 0) {
    throw new BootstrapRefusal('bad_job_type', 'jobTypes is empty: a Crucible with no job types serves nothing. Name at least one.');
  }
  const seen = new Set<JobType>();
  const enableFlags: string[] = [];
  const installs: Plan['installs'] = [];
  for (const request of requests) {
    const type = typeof request === 'string' ? request : (request as { type: unknown }).type;
    if (typeof type !== 'string' || !(JOB_TYPES as readonly string[]).includes(type)) {
      throw new BootstrapRefusal('bad_job_type', `unknown job type ${JSON.stringify(type)}; this build knows ${JOB_TYPES.join(', ')}`);
    }
    const jobType = type as JobType;
    if (jobType === 'tts') {
      const engine = typeof request === 'object' ? request.narratorEngine : undefined;
      if (typeof engine !== 'string' || engine.trim() === '') {
        throw new BootstrapRefusal(
          'bad_job_type',
          'tts must name its narrator engine — { type: "tts", narratorEngine: "higgs-v3" } — because cuda-linux has one env per engine',
        );
      }
      if (seen.has('tts')) throw new BootstrapRefusal('bad_job_type', 'tts is listed twice; one narrator engine per install');
      seen.add('tts');
      enableFlags.push('--enable-tts');
      installs.push({ type: 'tts', argv: ['install', 'tts', '--narrator-engine', engine, '--verbose'] });
      continue;
    }
    if (seen.has(jobType)) continue;
    seen.add(jobType);
    enableFlags.push(`--enable-${jobType}`);
    if (INSTALLABLE_JOB_TYPES.includes(jobType)) installs.push({ type: jobType, argv: ['install', jobType, '--verbose'] });
  }
  if (seen.has('denoise') && !seen.has('rvc')) {
    throw new BootstrapRefusal(
      'bad_job_type',
      'denoise shares the rvc env and has no installer of its own (`crucible install rvc` enables both); ask for rvc as well',
    );
  }
  return { enableFlags, installs };
}

export async function install(options: InstallOptions, runner: Runner = processRunner()): Promise<InstallResult> {
  const target = resolveTarget(runner, options.distro);
  const plan = planJobTypes(options.jobTypes);
  const timeouts = { ...DEFAULT_INSTALL_TIMEOUTS, ...options.timeouts };
  const steps: InstallStep[] = [];
  const done: string[] = [];
  const env = options.home === undefined ? undefined : { CRUCIBLE_HOME: options.home };

  const report = (step: InstallStep): InstallStep => {
    steps.push(step);
    options.onStep?.(step);
    return step;
  };

  const runStep = async (name: string, argv: readonly string[], timeoutMs: number, redacted?: readonly string[]): Promise<void> => {
    const shown = redacted ?? argv;
    const step = report({ name, argv: shown, status: 'running', detail: '' });
    const tail: string[] = [];
    const result = await streamOn(runner, target, argv, {
      timeoutMs,
      ...(env === undefined ? {} : { env }),
      onLine: (line, stream) => {
        tail.push(`${stream === 'stderr' ? '! ' : ''}${line}`);
        if (tail.length > TAIL_LINES) tail.shift();
        options.onLine(line, stream, name);
      },
    });
    if (result.failure !== null || result.code !== 0) {
      const why = result.failure ?? `exited ${result.code}`;
      throw new BootstrapStepFailed(
        name,
        result.code,
        tail,
        [...done],
        `install step "${name}" ${why} inside ${describeTarget(target)}: ${shown.join(' ')}`,
        { detail: tail.join('\n') },
      );
    }
    step.status = 'ok';
    step.detail = `exit 0`;
    done.push(name);
    options.onStep?.(step);
  };

  // 1. interpreter
  const interpreter = await probeInterpreter(runner, target, options.condaRoots ?? DEFAULT_CONDA_ROOTS);
  const first = interpreter.refusals[0];
  if (first !== undefined || interpreter.python === null) throw first ?? new Error('unreachable: no python and no refusal');
  const python = interpreter.python.path;
  report({ name: 'interpreter', argv: [], status: 'ok', detail: `Python ${interpreter.python.version} at ${python}` });
  done.push('interpreter');

  // 2. pip install <wheel>
  const wheel = wheelArgument(runner, target, options.wheel);
  await runStep('pip-install', [python, '-m', 'pip', 'install', wheel], timeouts.pipMs);

  // 3. init — skipped when a config is already there
  const crucible = consoleScriptBeside(python);
  const configOptions = { ...(options.distro === undefined ? {} : { distro: options.distro }), ...(options.home === undefined ? {} : { home: options.home }) };
  let existing: LocalConfig | null = null;
  try {
    existing = await readLocalConfig(configOptions, runner);
  } catch (err) {
    if (!(err instanceof BootstrapRefusal) || err.code !== 'no_local_config') throw err;
  }
  if (existing !== null) {
    report({ name: 'init', argv: [], status: 'skipped', detail: `config exists at ${existing.configPath}; its token is kept` });
  } else {
    const token = mintToken();
    const bind: string[] = [];
    if (options.bind?.host !== undefined) bind.push('--host', options.bind.host);
    if (options.bind?.port !== undefined) bind.push('--port', String(options.bind.port));
    const argv = [crucible, 'init', '--token', token, ...bind, ...plan.enableFlags];
    const redacted = [crucible, 'init', '--token', '<redacted>', ...bind, ...plan.enableFlags];
    await runStep('init', argv, timeouts.quickMs, redacted);
  }

  // 4. install <type>, one per type
  for (const entry of plan.installs) {
    await runStep(`install-${entry.type}`, [crucible, ...entry.argv], timeouts.envMs);
  }

  // 5. service install
  await runStep('service-install', [crucible, 'service', 'install'], timeouts.quickMs);

  // 6. linger — win32 only, and DONE rather than reported. A systemd user unit
  // dies with the user's last session without it, so a Crucible installed here
  // and not lingering is a server that disappears the first time somebody logs
  // out. `wsl.exe -u root` needs no password (measured 2026-09-14), so the
  // thing this package used to hand over is one idempotent command it can
  // simply run — and a guest that will not give root is a named refusal, which
  // is the one hand-over that remains.
  const linger = await ensureLinger(runner, target, env);
  if (linger !== null) {
    report({
      name: 'linger',
      argv: linger.argv,
      status: linger.granted ? 'ok' : 'skipped',
      detail: linger.detail,
    });
    if (linger.granted) done.push('linger');
  }

  // 7. capability --write
  await runStep('capability-write', [crucible, 'capability', '--write'], timeouts.quickMs);

  const config = await readLocalConfig(configOptions, runner);
  return { steps, server: { name: config.name, url: config.url, configPath: config.configPath } };
}

/** The wheel as the target spells it: a URL verbatim, a path checked and (on win32) mapped into the guest. */
export function wheelArgument(runner: Runner, target: Target, wheel: string): string {
  if (/^https?:\/\//i.test(wheel)) return wheel;
  if (!runner.fileExists(wheel)) {
    throw new BootstrapRefusal('wheel_missing', `the wheel ${wheel} is not on this machine. The host names the release wheel; nothing here guesses one.`);
  }
  return target.kind === 'wsl' ? guestPathFor(runner, wheel) : wheel;
}
