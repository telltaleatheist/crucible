/**
 * `install()` — this host has a Crucible, at the release the app named, with
 * the job types it asked for, running as this machine's service.
 *
 * The sequence is `steps.ts`'s (PHASE14-ENVPACKS.md section 4), and this file
 * WALKS that list rather than restating it, so the generated `install.sh`
 * cannot describe a different install from the one an app performs:
 *
 *   host-facts       CRUCIBLE_HOME, the guest user, free disk, curl/tar/zstd
 *   server-pack      download + verify + unpack the release's server pack into
 *                    <CRUCIBLE_HOME>/server — the interpreter comes WITH it
 *   init             <server>/bin/crucible init --token <minted here> --enable-<type>…
 *                    SKIPPED when a config already exists (its token is kept)
 *   install-<type>   <server>/bin/crucible install <type>
 *   service-install  <server>/bin/crucible service install
 *   linger           win32 only: `loginctl enable-linger <guest user>` as root
 *   capability-write <server>/bin/crucible capability --write
 *
 * **There is no conda step and no pip step.** A fresh machine has no Python at
 * all and does not need one: the server pack is a relocatable CPython with the
 * crucible wheel already installed in it (PHASE14 section 1). `release` — a
 * version string — replaced `wheel` and the conda options, and it defaults to
 * this package's own version, which is the one legitimate default in here: the
 * bootstrapper ships AT the server's version, so "which release" is not a
 * question anybody has to answer.
 *
 * Pulls are NOT part of this: weights are the app's, later, per model.
 *
 * The token is minted on the client — `crucible init --token` — so the app that
 * installed the server already holds what `readLocalConfig()` would read back.
 * It is never logged: the argv reported for the init step spells it
 * `<redacted>`, and `onLine` only ever sees what the step printed.
 *
 * Every step's stdout and stderr stream to `onLine` as they arrive. A failing
 * step stops the sequence with a {@link BootstrapStepFailed} naming the step,
 * carrying the tail, and listing the steps that finished — what they did is on
 * disk and stays there (ARCHITECTURE.md R6).
 */
import { randomBytes } from 'node:crypto';

import { readLocalConfig, type LocalConfig } from './config.js';
import { resolveDistro } from './distro.js';
import { backendFor, envpacksUrl, type PackBackend } from './envpacks.js';
import { BootstrapRefusal, BootstrapStepFailed } from './errors.js';
import { ensureLinger } from './linger.js';
import { fetchManifest, installPack, probeGuest, refuseMissingTools, SERVER_SUBDIR } from './pack.js';
import { processRunner, type OutputStream, type Runner } from './runner.js';
import { installSteps, renderArgv, type RefName, type StepPlan } from './steps.js';
import { describeTarget, resolveTarget, streamOn, type Target } from './target.js';
import { BOOTSTRAP_VERSION } from './version.js';

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
  /** Every `crucible` verb that builds nothing, and the host probe. */
  quickMs: number;
  /** The server pack: a multi-gigabyte download over somebody's home line. */
  packMs: number;
  /** `crucible install <type>`, which downloads a job env pack. */
  envMs: number;
}

export const DEFAULT_INSTALL_TIMEOUTS: InstallTimeouts = {
  quickMs: 5 * 60_000,
  packMs: 120 * 60_000,
  envMs: 120 * 60_000,
};

export interface InstallOptions {
  /** win32: the app's WSL distro setting. The `crucible` distro wins when it exists. */
  distro?: string;
  /** win32: use `distro` verbatim, resolving nothing. The way out of `two_local_crucibles`. */
  exact?: boolean;
  jobTypes: readonly JobTypeRequest[];
  /** `CRUCIBLE_HOME` for every `crucible` verb, as the target spells it. Omit for the server's default. */
  home?: string;
  /**
   * Which release's packs to install. Defaults to {@link BOOTSTRAP_VERSION} —
   * the bootstrapper ships at the server's version, so the default IS the
   * answer rather than a guess at one.
   */
  release?: string;
  /** Every line a step prints, as it prints it. */
  onLine: (line: string, stream: OutputStream, step: string) => void;
  /** Optional: a step beginning, finishing, or being skipped. */
  onStep?: (step: InstallStep) => void;
  /** What `crucible init` should bind. Omit for the server's own defaults (127.0.0.1:7100). */
  bind?: { host?: string; port?: number };
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
  /** Which release's packs are on this host, and which backend they are for. */
  release: string;
  backend: PackBackend;
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

export async function install(options: InstallOptions, runner: Runner = processRunner()): Promise<InstallResult> {
  const release = options.release ?? BOOTSTRAP_VERSION;
  const backend = backendFor(runner.platform);
  const distro = runner.platform === 'win32'
    ? await resolveDistro(runner, {
      ...(options.distro === undefined ? {} : { distro: options.distro }),
      ...(options.exact === undefined ? {} : { exact: options.exact }),
    })
    : undefined;
  const target = resolveTarget(runner, distro);
  const jobs = planJobTypes(options.jobTypes);
  const timeouts = { ...DEFAULT_INSTALL_TIMEOUTS, ...options.timeouts };
  const steps: InstallStep[] = [];
  const done: string[] = [];
  const env = options.home === undefined ? undefined : { CRUCIBLE_HOME: options.home };

  const bind: string[] = [];
  if (options.bind?.host !== undefined) bind.push('--host', options.bind.host);
  if (options.bind?.port !== undefined) bind.push('--port', String(options.bind.port));
  const plan: StepPlan = { enableFlags: jobs.enableFlags, installs: jobs.installs, bind, linger: target.kind === 'wsl' };

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

  const configOptions = {
    ...(distro === undefined ? {} : { distro, exact: true }),
    ...(options.home === undefined ? {} : { home: options.home }),
  };
  const values: Partial<Record<RefName, string>> = { release, backend };
  let crucible: string | null = null;
  // Measured by `host-facts`, consumed by `server-pack`. Locals rather than a
  // state object threaded through the walk: the sequence is a sequence, and
  // the one step that reads them is the next one.
  let manifest: Awaited<ReturnType<typeof fetchManifest>> | null = null;
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
            + `${guest.server === null ? '' : `, server pack ${guest.server.release ?? 'unstamped'}`}`,
        });
        done.push(step.name);

        // The pack fetch needs the manifest, and the manifest is fetched with
        // the GUEST's curl — nothing here reaches the network itself, so a
        // proxy or a VPN inside the distro is the guest's own answer.
        manifest = await fetchManifest(runner, target, release, envpacksUrl(release), timeouts.quickMs);
        break;
      }
      case 'server-pack': {
        if (manifest === null || guest === null) throw new Error('unreachable: host-facts did not run before server-pack');
        const packStep = report({ name: step.name, argv: [], status: 'running', detail: `${SERVER_SUBDIR} pack for ${backend}, release ${release}` });
        const result = await installPack(runner, target, manifest, 'server', {
          release,
          backend,
          home: guest.home,
          freeBytes: guest.freeBytes,
          installed: guest.server,
          timeoutMs: timeouts.packMs,
          onLine: (line, stream) => options.onLine(line, stream, step.name),
        });
        crucible = result.paths.crucible;
        values.crucible = crucible;
        packStep.status = result.skipped ? 'skipped' : 'ok';
        packStep.detail = result.skipped
          ? `${result.paths.dest} is already the ${release} pack (sha ${result.entry.sha256.slice(0, 12)}…)`
          : `unpacked ${result.entry.parts.length} part(s) into ${result.paths.dest} (Python ${result.entry.python})`;
        packStep.argv = result.skipped ? [] : [result.paths.crucible];
        options.onStep?.(packStep);
        if (!result.skipped) done.push(step.name);
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
        const token = mintToken();
        const argv = renderArgv(step.words, { ...values, token });
        const redacted = renderArgv(step.words, { ...values, token: '<redacted>' });
        await runStep('init', argv, timeouts[step.timeout], redacted);
        break;
      }
      case 'linger': {
        const linger = await ensureLinger(runner, target, env);
        if (linger !== null) {
          report({ name: 'linger', argv: linger.argv, status: linger.granted ? 'ok' : 'skipped', detail: linger.detail });
          if (linger.granted) done.push('linger');
        }
        break;
      }
      default: {
        if (step.words === null) throw new Error(`unreachable: step ${step.name} has neither argv nor a handler`);
        await runStep(step.name, renderArgv(step.words, values), timeouts[step.timeout]);
      }
    }
  }

  if (crucible === null) throw new Error('unreachable: no server pack after the sequence');
  const config = await readLocalConfig(configOptions, runner);
  return {
    steps,
    server: { name: config.name, url: config.url, configPath: config.configPath },
    release,
    backend,
    crucible,
  };
}
