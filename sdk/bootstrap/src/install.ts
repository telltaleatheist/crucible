/**
 * `install()` — this host has a Crucible, at the release the app named, with
 * the job types it asked for, running as this machine's service.
 *
 * The sequence is `steps.ts`'s (PHASE14-ENVPACKS.md section 4), and this file
 * WALKS that list rather than restating it, so the generated `install.sh`
 * cannot describe a different install from the one an app performs:
 *
 *   host-facts       CRUCIBLE_HOME, the guest user, free disk, curl/tar
 *   server           the pinned interpreter into <CRUCIBLE_HOME>/server (ONCE,
 *                    skipped when its digest is the one stamped there) and this
 *                    release's wheel pip-installed into it
 *   init             <server>/bin/crucible init --token <minted here> --enable-<type>…
 *                    SKIPPED when a config already exists (its token is kept)
 *   install-<type>   <server>/bin/crucible install <type>
 *   service-install  <server>/bin/crucible service install
 *   linger           WSL only: `loginctl enable-linger <guest user>` as root —
 *                    which since PHASE15 means the HOST runs it, never this file
 *   capability-write <server>/bin/crucible capability --write
 *
 * **There is no conda step.** A fresh machine has no Python at all and does not
 * need one: the `server` step downloads a relocatable CPython from
 * python-build-standalone at a pinned digest (PHASE20 section 2) and pip-installs
 * the release's wheel into it. `release` — a version string — replaced `wheel`
 * and the conda options, and it defaults to this package's own version, which is
 * the one legitimate default in here: the bootstrapper ships AT the server's
 * version, so "which release" is not a question anybody has to answer.
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
 *
 * **ON WINDOWS NONE OF THAT HAPPENS HERE** (PHASE15-HOST.md section 4.3). The
 * sequence has ONE implementation and it is the host's: `crucible host`, the
 * Windows-native tray process, walks the state table and the step list and
 * raises the UAC prompts a WSL install needs. (It has no window of its own —
 * 4.7: its UI is the tray menu and the operator page.) So on win32 this
 * function is two steps and nothing else (PHASE19-AUTOMATIC-WSL.md 2.6) —
 *
 *   host absent?   run `install.ps1`, which is per-user and elevates nothing
 *   then           `watchInstall()`: attach to the move the TRAY started
 *
 * — and the walk below is reached only on linux and darwin, where the machine
 * IS the server and there is no host to ask.
 *
 * **IT NEVER POSTS THE MOVE.** PHASE19 2.3 put that decision in the tray, which
 * takes it at every start from facts on disk: a fresh install, a tray coming
 * back after the reboot `wsl --install` demanded, and an old native install
 * being upgraded are the same decision seen three times, and only a process
 * that is there at login can make all three. A `POST /install` from here would
 * be a second caller racing the first for a claim it would lose.
 *
 * **AND NOBODY IS SHOWN A COMMAND.** This used to refuse `host_not_installed`
 * with the `irm … | iex` line for a person to type. PHASE19: "a command a
 * person could run is a step the app should be running."
 */
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
import { ensureLinger } from './linger.js';
import { installRuntime, probeGuest, refuseMissingTools, SERVER_SUBDIR } from './runtime.js';
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
  /** The server runtime: a ~30 MB interpreter and a 1 MB wheel, over somebody's home line. */
  runtimeMs: number;
  /** `crucible install <type>`, which pips a recipe — gigabytes, from the mirrors. */
  envMs: number;
}

export const DEFAULT_INSTALL_TIMEOUTS: InstallTimeouts = {
  quickMs: 5 * 60_000,
  runtimeMs: 30 * 60_000,
  envMs: 120 * 60_000,
};

export interface InstallOptions {
  /**
   * **No longer read by `install()`.** Which distro a Windows Crucible goes
   * into is the HOST's answer now (PHASE15 4.3): it owns the `crucible` distro
   * and imports it itself, so there is nothing for an app to choose and
   * nothing to send over the door. Kept on the interface because
   * `readLocalConfig()` and `detectHost()` still take it and callers pass one
   * object to all three.
   */
  distro?: string;
  /** Same: the host resolves the distro. See {@link InstallOptions.distro}. */
  exact?: boolean;
  jobTypes: readonly JobTypeRequest[];
  /** `CRUCIBLE_HOME` for every `crucible` verb, as the target spells it. Omit for the server's default. */
  home?: string;
  /**
   * Which release's wheel to install.
   *
   * AN APP PASSES THE RELEASE CHANNEL'S ANSWER (INSTALL-UNINSTALL.md §6.5.1):
   * `latestRelease()` reads `releases/latest`, and the app's own never-older
   * gate decides whether to install it at all. It used to say the default
   * "IS the answer rather than a guess at one" — that sentence was written when
   * the bootstrapper's version and the release a machine should have were one
   * fact, and §6.5.2 separated them: a vendored 1.0.1 installing 1.0.1 over a
   * running 1.0.2 is the defect.
   *
   * Omitted, it is {@link BOOTSTRAP_VERSION} — the release this library was cut
   * with, which is the hand-install case (`npm i @crucible/bootstrap@<v>` and
   * call this) and nothing else. It is never reached FOR a channel that would
   * not answer: that is `release_channel_unreadable` at the caller, before this.
   */
  release?: string;
  /**
   * AN OPERATOR ROLLBACK, and the only way an install goes backwards.
   *
   * `installRuntime` refuses `install_would_downgrade` when
   * `<home>/server/.crucible` names a release newer than the one being
   * installed (INSTALL-UNINSTALL.md
   * §6.5.4). This is how somebody says "yes, put 1.0.1 back" — and it must be
   * the SAME version as {@link InstallOptions.release}, because a rollback is an
   * operator naming the Crucible they want rather than a flag that means
   * "downgrade to whatever". A different version is `rollback_version_mismatch`.
   */
  rollbackTo?: string;
  /** Every line a step prints, as it prints it. */
  onLine: (line: string, stream: OutputStream, step: string) => void;
  /** Optional: a step beginning, finishing, or being skipped. */
  onStep?: (step: InstallStep) => void;
  /** What `crucible init` should bind. Omit for the server's own defaults (127.0.0.1:7100). */
  bind?: { host?: string; port?: number };
  timeouts?: Partial<InstallTimeouts>;
  /**
   * win32 only: every event the host's door sent, verbatim, including the 4c
   * `state` rows that have no place in `onLine`/`onStep`. Ignored off win32,
   * where there is no host and no door.
   */
  onHostEvent?: (event: HostEvent) => void;
  /**
   * win32 only: the `fetch` the host's door is asked with. Defaults to
   * `globalThis.fetch` — the platform's own, on the node this package declares
   * (`engines: node >=20`). This exists so the tests can script a host without
   * opening a socket; an app has no reason to pass it.
   */
  fetchImpl?: HostFetch;
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

/**
 * win32: PHASE15 4.3's two branches, and nothing else.
 *
 * There is no third branch that walks the steps here. A Windows machine
 * installs its Crucible exactly one way — through `crucible host` — so that
 * `install.ps1`, the operator page's engine switch (4.7) and an app's
 * `install()` cannot describe three different installs (PHASE14 4a, "cannot
 * differ").
 *
 * **Why a library posts to a loopback port when there is a whole tasks API.**
 * 4.3 names two callers of the door. The page's `POST /v1/tasks
 * {"type":"engine","target":"wsl"}` is the primary one and goes through the
 * Windows SERVER, which relays the door's events under a task id. This is the
 * other one, and it runs on a machine that has no server yet — the very first
 * install, before there is a page to open or a `/v1/tasks` to post to.
 */
async function installThroughHost(options: InstallOptions, release: string, runner: Runner): Promise<InstallResult> {
  // The one thing this side still does with the job list: refuse a malformed
  // one BY NAME. `{type: 'tts'}` with no narrator engine is the caller's bug,
  // and being told so must not require a host to be installed and answering.
  // The plan itself is thrown away — the host builds its own from the same list.
  planJobTypes(options.jobTypes);

  // A ROLLBACK HAS NO WAY THROUGH THE HOST'S DOOR, so it is refused rather than
  // dropped. The door's request body (`hostdoor.ts`) carries release, job types,
  // home and bind and nothing else, and the host would walk an ordinary install
  // — which `install.ps1`'s own never-older gate then refuses as
  // `install_would_downgrade`, from inside a process this caller cannot see.
  // Naming it here puts the refusal where the option was set, with the line an
  // operator actually runs to go back.
  if (options.rollbackTo !== undefined) {
    throw new BootstrapRefusal(
      'host_rollback_unsupported',
      `rollbackTo is a POSIX-side option: on Windows the host owns the install sequence (PHASE15 4.3) and its door `
        + 'takes no rollback. Roll the host back by hand with the line below, from an ordinary PowerShell.',
      { command: `.\\install.ps1 -Release ${options.rollbackTo} -RollbackTo ${options.rollbackTo}` },
    );
  }

  if (!hostInstalled(runner)) {
    // PHASE19: NOBODY IS EVER SHOWN A COMMAND. This used to refuse with the
    // `irm … | iex` line, on the grounds that a library must not download and
    // elevate an installer from a background call. Half of that still holds
    // and half never applied: `install.ps1` is per-user and elevates nothing
    // (its own header: "No admin. Everything here is per-user and
    // idempotent"), and the UAC prompt a WSL install needs is raised later, by
    // the tray, which is a process a person can see. So the script is run, and
    // the only thing still refused here is elevation.
    await runInstallPs1(options, release, runner);
  }

  // NEVER A POST (PHASE19 2.6). The tray decides at every start whether this
  // machine should be moving (2.3) and has already started if it should, so a
  // POST from here would be a second caller racing the first for a claim it
  // would lose. This attaches to what is already happening.
  //
  // The `done` event is kept as it goes past, because it is the one place the
  // GUEST's facts — its server name, its url, its config path, its console
  // script — cross to this side. The outcome file records that the move
  // finished; it does not describe what it finished into.
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

/**
 * `install.ps1`, run as this user, with no console for anybody to read.
 *
 * Downloaded to a file and run with `-Release` rather than piped through
 * `iex`: a piped script cannot take a switch (the script says so itself), and
 * the release this call was given is the release the host must be, not
 * whatever the channel calls latest at this second.
 */
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

/**
 * The move's status → the {@link InstallResult} `install()` promises, or the
 * named refusal that says why there is none.
 *
 * Every ending of PHASE19 2.2 has an answer and none of them is a shrug:
 * `done` is the result, `cannot`, `failed`, `reboot-pending` and `declined`
 * are refusals carrying the OUTCOME's own code and sentence — which is the 4c
 * code an app switches on.
 */
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
    // THE OUTCOME'S OWN CODE AND SENTENCE, verbatim. The 4c codes cross the
    // wire unwrapped for `hostRefusal`'s reason: the app's next move is chosen
    // from the code, and `virtualization_disabled` wrapped in a generic name
    // would delete the only thing the message was carrying.
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
  // The rollback is checked HERE rather than in `installRuntime`, so that a caller who
  // named two different versions is told so before a single guest command runs.
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
  // ---------------------------------------------------------------------
  // linux and darwin only, from here down: win32 returned above. Every step,
  // every argv and every refusal below is what it was before PHASE15 — the
  // only thing that went is the distro resolution, which had no answer to give
  // on a machine that IS the server and whose one caller was the win32 arm.
  // ---------------------------------------------------------------------
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

  const configOptions = options.home === undefined ? {} : { home: options.home };
  const values: Partial<Record<RefName, string>> = { release, backend };
  let crucible: string | null = null;
  // Measured by `host-facts`, consumed by `server`. A local rather than a
  // state object threaded through the walk: the sequence is a sequence, and
  // the one step that reads them is the next one.
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
