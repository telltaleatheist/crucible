import { randomBytes } from 'node:crypto';

import { readLocalConfig, type LocalConfig } from './config.js';
import { backendFor, type ServerBackend } from './release.js';
import { BootstrapRefusal, BootstrapStepFailed } from './errors.js';
import {
  doneResult,
  hostInstalled,
  hostRuntimeDir,
  watchInstall,
  HOST_DOOR_URL,
  HOST_ENTRY_POINT,
  HOST_INSTALL_EVENTS_PATH,
  type HostEvent,
  type HostFetch,
  type InstallStatus,
} from './hostdoor.js';
import { releaseAssetUrl } from './release.js';
import { installRuntime, probeGuest, refuseMissingTools, SERVER_SUBDIR } from './runtime.js';
import { processRunner, type OutputStream, type Runner } from './runner.js';
import { INIT_TOKEN_ENV, installSteps, renderArgv, type RefName, type StepPlan } from './steps.js';
import { describeTarget, resolveTarget, streamOn, type Target } from './target.js';
import { BOOTSTRAP_VERSION } from './version.js';

/** The job types `crucible init --enable-<type>` knows, in the order the CLI lists them. */
export const JOB_TYPES = [
  'echo', 'llm', 'asr', 'tts', 'align', 'rvc', 'denoise', 'image', 'audio', 'segment', 'video',
] as const;
export type JobType = (typeof JOB_TYPES)[number];

/** The job types `crucible install <type>` has an installer for (`cli.INSTALLABLE_JOB_TYPES`). */
export const INSTALLABLE_JOB_TYPES: readonly JobType[] = [
  'llm', 'tts', 'asr', 'align', 'rvc', 'image', 'audio', 'segment', 'video',
];

/** One job type to enable; `tts` must name its narrator engine. */
export type JobTypeRequest = Exclude<JobType, 'tts'> | { type: 'tts'; narratorEngine: string };

export interface InstallTimeouts {
  /** Every `crucible` verb that builds nothing, and the host probe. */
  quickMs: number;
  /** The server runtime download: the interpreter and the wheel. */
  runtimeMs: number;
  /** `crucible install <type>`, which pip-installs a recipe. */
  envMs: number;
}

export const DEFAULT_INSTALL_TIMEOUTS: InstallTimeouts = {
  quickMs: 5 * 60_000,
  runtimeMs: 30 * 60_000,
  envMs: 120 * 60_000,
};

export interface InstallOptions {
  /**
   * Ignored by `install()`; accepted so one options object also serves `detectHost()` and
   * `readLocalConfig()`.
   */
  distro?: string;
  /** Ignored by `install()`; see {@link InstallOptions.distro}. */
  exact?: boolean;
  jobTypes: readonly JobTypeRequest[];
  /** `CRUCIBLE_HOME` for every `crucible` verb, as the target spells it. */
  home?: string;
  /** Which release to install; an app passes the channel's answer from `latestRelease()`. */
  release?: string;
  /** An operator rollback: the exact older release to put back, equal to `release`. */
  rollbackTo?: string;
  /** Every line a step prints, as it prints it. */
  onLine: (line: string, stream: OutputStream, step: string) => void;
  /** Optional: a step beginning, finishing, or being skipped. */
  onStep?: (step: InstallStep) => void;
  /** What `crucible init` should bind; omit for the server's defaults. */
  bind?: { host?: string; port?: number };
  timeouts?: Partial<InstallTimeouts>;
  /** win32 only: every event the host's door sent, verbatim. */
  onHostEvent?: (event: HostEvent) => void;
  /** win32 only: the `fetch` the host's door is asked with. */
  fetchImpl?: HostFetch;
}

export interface InstallStep {
  name: string;
  /** What ran, with the token spelled `<redacted>`. */
  argv: readonly string[];
  status: 'running' | 'ok' | 'skipped';
  detail: string;
}

export interface InstallResult {
  steps: InstallStep[];
  /** The server as its config now describes it. */
  server: { name: string; url: string; configPath: string };
  /** Which release is on this host, and which backend it serves. */
  release: string;
  backend: ServerBackend;
  /** `<CRUCIBLE_HOME>/server/bin/crucible`, as the target spells it. */
  crucible: string;
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

async function installThroughHost(options: InstallOptions, release: string, runner: Runner): Promise<InstallResult> {
  planJobTypes(options.jobTypes);

  if (options.rollbackTo !== undefined) {
    throw new BootstrapRefusal(
      'host_rollback_unsupported',
      `rollbackTo is a POSIX-side option: on Windows the host owns the install sequence (PHASE15 4.3) and its door `
        + 'takes no rollback. Roll the host back by hand with the line below, from an ordinary PowerShell.',
      { command: `.\\install.ps1 -Release ${options.rollbackTo} -RollbackTo ${options.rollbackTo}` },
    );
  }

  if (!hostInstalled(runner)) {
    await runInstallPs1(options, release, runner);
  }

  let witnessed: InstallResult | null = null;
  const status = await watchInstall(
    {
      ...(options.fetchImpl === undefined ? {} : { fetchImpl: options.fetchImpl }),
      ...(options.onStep === undefined ? {} : { onStep: options.onStep }),
      onLine: options.onLine,
      onEvent: (event) => {
        if (event.event === 'done') witnessed = doneResult(event.data, `${HOST_DOOR_URL}${HOST_INSTALL_EVENTS_PATH}`);
        options.onHostEvent?.(event);
      },
    },
    runner,
  );
  return installedResult(status, witnessed, runner);
}

async function runInstallPs1(options: InstallOptions, release: string, runner: Runner): Promise<void> {
  const url = releaseAssetUrl(release, 'install.ps1');
  const quoted = (value: string): string => `'${value.replace(/'/g, "''")}'`;
  const script = `$ErrorActionPreference = 'Stop'; `
    + `$p = Join-Path $env:TEMP 'crucible-install.ps1'; `
    + `Invoke-RestMethod ${quoted(url)} -OutFile $p; `
    + `& $p -Release ${quoted(release)}`;
  const argv = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', script];
  options.onStep?.({ name: 'host', argv, status: 'running', detail: `install.ps1 ${release}` });
  const result = await runner.stream(argv, {
    timeoutMs: { ...DEFAULT_INSTALL_TIMEOUTS, ...options.timeouts }.runtimeMs,
    onLine: (line, stream) => options.onLine(line, stream, 'host'),
  });
  if (result.failure !== null || result.code !== 0) {
    throw new BootstrapRefusal(
      'host_not_installed',
      `install.ps1 ${result.failure ?? `exited ${result.code}`} on this machine, so there is still no Crucible host at `
        + `${hostRuntimeDir(runner)}. It is idempotent: the lines above say what it got to, and running install() again `
        + 'resumes from whatever is on disk.',
      { detail: result.stderr.trim() || result.stdout.trim() },
    );
  }
  if (!hostInstalled(runner)) {
    throw new BootstrapRefusal(
      'host_not_installed',
      `install.ps1 reported success and there is still no ${HOST_ENTRY_POINT} at ${hostRuntimeDir(runner)}. `
        + 'Nothing here can say what it installed instead.',
    );
  }
}

function installedResult(
  status: InstallStatus,
  witnessed: InstallResult | null,
  runner: Runner,
): InstallResult {
  const recorded = status.outcome;
  if (recorded === null) {
    throw new BootstrapRefusal(
      'host_install_failed',
      'the Crucible host on this machine recorded no outcome for its engine move, and none is running. '
        + `Its own log (${hostRuntimeDir(runner)}) says what it decided; there is nothing here to report as an install.`,
    );
  }
  if (recorded.state !== 'done') {
    throw new BootstrapRefusal(
      (recorded.code ?? 'host_install_failed') as BootstrapRefusal['code'],
      recorded.sentence ?? `the engine move on this machine ended as ${recorded.state}.`,
      { detail: `${recorded.state} at ${recorded.at}, attempt ${recorded.attempts}, release ${recorded.release}` },
    );
  }
  if (witnessed === null) {
    throw new BootstrapRefusal(
      'host_install_unwitnessed',
      `this machine's engine move finished at ${recorded.at} (release ${recorded.release}) and this call did not see it: `
        + 'the tray keeps the last 200 events of a move and its `done` is no longer among them, which is what a tray '
        + 'restart leaves behind. The machine is on the Linux engine and there is nothing here to describe it with — '
        + "the server's name, url and config path are the guest's, and `readLocalConfig()` is what reads them now.",
      { detail: `${recorded.state} at ${recorded.at}, release ${recorded.release}` },
    );
  }
  return witnessed;
}

export async function install(options: InstallOptions, runner: Runner = processRunner()): Promise<InstallResult> {
  const release = options.release ?? BOOTSTRAP_VERSION;
  const rollback = options.rollbackTo === undefined ? null : options.rollbackTo;
  if (rollback !== null && rollback !== release) {
    throw new BootstrapRefusal(
      'rollback_version_mismatch',
      `rollbackTo names ${rollback} and the release being installed is ${release}. A rollback is an `
        + 'operator naming the exact Crucible they want back, so the two are the same version or this '
        + 'is a downgrade nobody asked for.',
    );
  }
  if (runner.platform === 'win32') return await installThroughHost(options, release, runner);
  const backend = backendFor(runner.platform);
  const target = resolveTarget(runner, undefined);
  const jobs = planJobTypes(options.jobTypes);
  const timeouts = { ...DEFAULT_INSTALL_TIMEOUTS, ...options.timeouts };
  const steps: InstallStep[] = [];
  const done: string[] = [];
  const env = options.home === undefined ? undefined : { CRUCIBLE_HOME: options.home };

  const bind: string[] = [];
  if (options.bind?.host !== undefined) bind.push('--host', options.bind.host);
  if (options.bind?.port !== undefined) bind.push('--port', String(options.bind.port));
  const plan: StepPlan = { enableFlags: jobs.enableFlags, installs: jobs.installs, bind, linger: false };

  const report = (step: InstallStep): InstallStep => {
    steps.push(step);
    options.onStep?.(step);
    return step;
  };

  const runStep = async (
    name: string,
    argv: readonly string[],
    timeoutMs: number,
    secretEnv?: Readonly<Record<string, string>>,
  ): Promise<void> => {
    const step = report({ name, argv, status: 'running', detail: '' });
    const tail: string[] = [];
    const stepEnv = secretEnv === undefined ? env : { ...env, ...secretEnv };
    const result = await streamOn(runner, target, argv, {
      timeoutMs,
      ...(stepEnv === undefined ? {} : { env: stepEnv }),
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
        `install step "${name}" ${why} inside ${describeTarget(target)}: ${argv.join(' ')}`,
        { detail: tail.join('\n') },
      );
    }
    step.status = 'ok';
    step.detail = `exit 0`;
    done.push(name);
    options.onStep?.(step);
  };

  const configOptions = options.home === undefined ? {} : { home: options.home };
  const values: Partial<Record<RefName, string>> = { release, backend };
  let crucible: string | null = null;
  let guest: Awaited<ReturnType<typeof probeGuest>> | null = null;

  for (const step of installSteps(plan)) {
    switch (step.name) {
      case 'host-facts': {
        guest = await probeGuest(runner, target, options.home);
        refuseMissingTools(target, runner.platform, guest.missingTools);
        values.home = guest.home;
        values.user = guest.user;
        report({
          name: step.name,
          argv: [],
          status: 'ok',
          detail: `${describeTarget(target)}: CRUCIBLE_HOME ${guest.home}, user ${guest.user}, `
            + `${(guest.freeBytes / 1024 ** 3).toFixed(1)} GiB free`
            + `${guest.server === null ? '' : `, server ${guest.server.release ?? 'unstamped'}`}`,
        });
        done.push(step.name);
        break;
      }
      case 'server': {
        if (guest === null) throw new Error('unreachable: host-facts did not run before server');
        const serverStep = report({ name: step.name, argv: [], status: 'running', detail: `${SERVER_SUBDIR} for ${backend}, release ${release}` });
        const result = await installRuntime(runner, target, {
          release,
          backend,
          home: guest.home,
          installed: guest.server,
          rollbackTo: rollback,
          timeoutMs: timeouts.runtimeMs,
          onLine: (line, stream) => options.onLine(line, stream, step.name),
        });
        crucible = result.paths.crucible;
        values.crucible = crucible;
        serverStep.status = 'ok';
        serverStep.detail = result.interpreterSkipped
          ? `python ${result.pin.version} was already at ${result.paths.dest}; installed the ${release} wheel into it`
          : `python ${result.pin.version} from python-build-standalone into ${result.paths.dest}, then the ${release} wheel`;
        serverStep.argv = [result.paths.crucible];
        options.onStep?.(serverStep);
        done.push(step.name);
        break;
      }
      case 'init': {
        let existing: LocalConfig | null = null;
        try {
          existing = await readLocalConfig(configOptions, runner);
        } catch (err) {
          if (!(err instanceof BootstrapRefusal) || err.code !== 'no_local_config') throw err;
        }
        if (existing !== null) {
          report({ name: 'init', argv: [], status: 'skipped', detail: `config exists at ${existing.configPath}; its token is kept` });
          break;
        }
        if (step.words === null) throw new Error('unreachable: the init step has no argv');
        await runStep('init', renderArgv(step.words, values), timeouts[step.timeout], { [INIT_TOKEN_ENV]: mintToken() });
        break;
      }
      default: {
        if (step.words === null) throw new Error(`unreachable: step ${step.name} has neither argv nor a handler`);
        await runStep(step.name, renderArgv(step.words, values), timeouts[step.timeout]);
      }
    }
  }

  if (crucible === null) throw new Error('unreachable: no server runtime after the sequence');
  const config = await readLocalConfig(configOptions, runner);
  return {
    steps,
    server: { name: config.name, url: config.url, configPath: config.configPath },
    release,
    backend,
    crucible,
  };
}
