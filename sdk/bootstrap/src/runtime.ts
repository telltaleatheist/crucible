/**
 * The server runtime: a pinned CPython at `<CRUCIBLE_HOME>/server/` with the
 * release's wheel installed into it.
 *
 * PHASE20-CODE-NOT-ENVIRONMENTS.md section 3. This file was `pack.ts`, which
 * downloaded a `crucible-env-server-<backend>-<version>.tar.zst` from our own
 * release — an archive we rebuilt and re-uploaded on every tag, ~55 MB of
 * somebody else's interpreter per backend, for a 1 MB change to our code. The
 * interpreter now comes from its publisher once (`interpreter.ts`) and the
 * deploy is the wheel.
 *
 * Four rules, each with a reason and a test:
 *
 * 1. **Everything happens INSIDE the guest, with the guest's `curl` and `pip`.**
 *    Never onto `/mnt/c` and across the 9p mount: that is slow and it loses the
 *    permission bits a Python tree needs. On macOS and Linux "the guest" is the
 *    machine.
 *
 * 2. **The interpreter is fetched only when its digest is not the one stamped.**
 *    An upgrade is therefore a wheel install and nothing else, which is the
 *    whole ruling: a deploy ships code.
 *
 * 3. **Every download is checked against a digest before it is used.** The
 *    interpreter against `interpreter.ts`'s pin; the wheel against the
 *    `<wheel>.sha256` the release publishes beside it. A mismatch deletes the
 *    file and refuses by name — `runtime_sha_mismatch` — because the next run
 *    must start clean rather than pip-install bytes nobody vouched for.
 *
 * 4. **Never over a newer release.** `<home>/server/.crucible` records
 *    `release=`, so this knows what is on the disk before it writes to it, and
 *    an older release over a newer one is `install_would_downgrade` before
 *    anything downloads (INSTALL-UNINSTALL.md §6.5.4). The one legitimate
 *    downgrade is an operator rollback and it names its exact version.
 *
 * The interpreter unpack is `tar -xzf … -C <dest>.partial` followed by a swap,
 * so a half-unpacked tree is never at the path everything else runs from.
 */
import { compareReleases } from './channel.js';
import { BootstrapRefusal } from './errors.js';
import { DESKTOP_PACKAGES, interpreterFor, interpreterUrl, type StandalonePython } from './interpreter.js';
import { wheelAssetName, wheelShaUrl, wheelUrl, type ServerBackend } from './release.js';
import type { RunResult, Runner } from './runner.js';
import { describeTarget, refuseIfUnrun, runOn, streamOn, type Target } from './target.js';
import { shellQuote } from './wsl.js';

/** `<CRUCIBLE_HOME>/server` — where the interpreter and the wheel live. */
export const SERVER_SUBDIR = 'server';

/**
 * Where the WINDOWS host's runtime goes, under `%LOCALAPPDATA%\Crucible\`
 * (PHASE15-HOST.md 4.4). Beside `wsl\` and `downloads\`, which `distro.ts`
 * already puts there. `crucible/host/paths.py` spells the same word on the
 * Python side and `install.ps1` is generated from this one.
 */
export const HOST_SUBDIR = 'host';
/** `<CRUCIBLE_HOME>/downloads` — an archive lands here and is deleted after use. */
export const DOWNLOADS_SUBDIR = 'downloads';
/** The half-unpacked tree's suffix. Swapped onto the real path only when tar exits 0. */
export const PARTIAL_SUFFIX = '.partial';

/**
 * `<CRUCIBLE_HOME>/server/.crucible` — three `key=value` lines saying what this
 * tree IS: which interpreter digest, which python version, which release.
 *
 * It was `.pack`, and the rename is not tidying: the file's SUBJECT changed.
 * `.pack` recorded the sha256 of an archive of the whole tree, which no longer
 * exists; this records the digest of the INTERPRETER inside it, which is the
 * only thing a re-download could replace. A tree carrying the old file is read
 * as having no interpreter digest at all, so the first install after this
 * fetches 30 MB once and stamps the new shape — loud, cheap and correct.
 */
export const STAMP_NAME = '.crucible';

/** Shared activation transaction used by app installs and the generated install.sh.
 * Arguments are shell expressions (already quoted), never untrusted raw paths.
 *
 * `$partial` is the unpacked `python/` directory, so the tree that lands at
 * `$dest` is the interpreter itself and `$dest/bin/crucible` is what the wheel
 * install then puts inside it.
 */
export function activateRuntimeSh(dest: string, partial: string): string {
  return `activate_crucible_runtime() {\n`
    + `  _crucible_dest=${dest}; _crucible_partial=${partial}; _crucible_previous="$_crucible_dest.previous"\n`
    + `  if [ -e "$_crucible_previous" ]; then echo "upgrade_recovery_required: $_crucible_previous was preserved from an interrupted upgrade" >&2; return 1; fi\n`
    + `  if [ -e "$_crucible_dest" ]; then\n`
    + `    if [ -x "$_crucible_dest/bin/crucible" ]; then "$_crucible_dest/bin/crucible" local shutdown || return 1; fi\n`
    + `    mv "$_crucible_dest" "$_crucible_previous" || return 1\n`
    + `  fi\n`
    + `  if mv "$_crucible_partial" "$_crucible_dest" && "$_crucible_dest/bin/python3" --version; then\n`
    + `    rm -rf "$_crucible_previous" || return 1\n`
    + `  else\n`
    + `    if [ -e "$_crucible_dest" ] && [ ! -e "$_crucible_partial" ]; then mv "$_crucible_dest" "$_crucible_partial" || return 1; fi\n`
    + `    if [ -e "$_crucible_previous" ]; then mv "$_crucible_previous" "$_crucible_dest" || return 1; fi\n`
    + `    echo "upgrade_activation_failed: the previous runtime was preserved" >&2; return 1\n`
    + `  fi\n`
    + `}\nactivate_crucible_runtime`;
}

/**
 * curl's arguments, one owner for both the TypeScript and the generated
 * `install.sh`. `-f` so an HTML error page is never saved as an archive, `-L`
 * for the release redirect, `--retry` for a flaky minute, `--create-dirs` so
 * nothing has to mkdir first.
 *
 * `--continue-at -` is GONE with the parts it was for. Everything fetched here
 * is tens of megabytes and digest-checked the moment it lands, so a truncated
 * file is deleted and fetched whole rather than appended to — which is also the
 * only way to be sure a proxy that ignored the range did not build a corrupt
 * archive.
 */
export const CURL_ARGS: readonly string[] = ['-fL', '--retry', '3', '--retry-delay', '2', '--create-dirs'];

/** tar's, likewise. `-z`, because python-build-standalone publishes `.tar.gz`. */
export const TAR_ARGS: readonly string[] = ['-xzf'];

/** The sha256 tool, per platform. macOS has no `sha256sum`; every Linux does. */
export function shaArgv(platform: NodeJS.Platform, file: string): string[] {
  return platform === 'darwin' ? ['shasum', '-a', '256', file] : ['sha256sum', file];
}

/**
 * The tools an install needs, probed before anything is fetched.
 *
 * `zstd` is not among them any more, and its absence is the measure of what
 * changed: it was here for the 8 GB environment archives, and nothing this
 * installer touches is compressed with it. python-build-standalone publishes
 * gzip, which every `tar` reads.
 */
export const REQUIRED_TOOLS: readonly string[] = ['curl', 'tar'];

const PROBE_TIMEOUT_MS = 60_000;

/** What the one guest probe answers. Every field measured; nothing inferred from a missing one. */
export interface GuestFacts {
  /** `$CRUCIBLE_HOME`, or `$HOME/.crucible`, as the guest resolved it. */
  home: string;
  /** The guest user — `loginctl enable-linger` is about this one, not the Windows one. */
  user: string;
  /** Free bytes on the filesystem `home` is (or would be) on. */
  freeBytes: number;
  /** Tools from {@link REQUIRED_TOOLS} that are not there. Empty on a healthy guest. */
  missingTools: string[];
  /** The server runtime already installed, or null. */
  server: InstalledRuntime | null;
}

export interface InstalledRuntime {
  /** `<home>/server/bin/crucible`. */
  crucible: string;
  /** `<home>/server/bin/python3`. */
  python: string;
  /** What `crucible --version` printed, or null when the tree is there and the binary would not answer. */
  version: string | null;
  /** The INTERPRETER's sha256 the stamp records, or null when there is no stamp. */
  pythonSha256: string | null;
  /** The release the stamp records, or null. */
  release: string | null;
}

/**
 * ONE script, exit 0 always, `key=value` lines out — the same shape the conda
 * probe used, for the same reason: a guest that answers nothing and a guest
 * that answers "no" must be told apart, and an exit code cannot carry five
 * facts. No backslash and no host-side `$`; with `--exec` the guest's bash is
 * the first thing that reads it.
 */
export function guestProbeScript(home: string | undefined): string {
  const where = home === undefined ? 'h="${CRUCIBLE_HOME:-$HOME/.crucible}"; ' : `h=${shellQuote(home)}; `;
  return `${where}`
    + 'echo "home=$h"; echo "user=$(id -un)"; '
    // df of the nearest EXISTING ancestor: ~/.crucible may not be there yet.
    + 'd="$h"; while [ ! -d "$d" ] && [ "$d" != "/" ]; do d=$(dirname "$d"); done; '
    + 'echo "free_kib=$(df -Pk "$d" | awk \'NR==2 {print $4}\')"; '
    + `for t in ${REQUIRED_TOOLS.join(' ')}; do command -v "$t" >/dev/null 2>&1 || echo "missing=$t"; done; `
    + `c="$h/${SERVER_SUBDIR}/bin/crucible"; `
    + 'if test -x "$c"; then echo "crucible=$c"; echo "version=$("$c" --version 2>&1 | head -1)"; fi; '
    + `s="$h/${SERVER_SUBDIR}/${STAMP_NAME}"; `
    + 'if test -f "$s"; then cat "$s"; fi; '
    + 'exit 0';
}

function parseKeyValues(stdout: string): { single: Map<string, string>; missing: string[] } {
  const single = new Map<string, string>();
  const missing: string[] = [];
  for (const raw of stdout.split(/\r?\n/)) {
    const eq = raw.indexOf('=');
    if (eq <= 0) continue;
    const key = raw.slice(0, eq).trim();
    const value = raw.slice(eq + 1).trim();
    if (key === 'missing') missing.push(value);
    else single.set(key, value);
  }
  return { single, missing };
}

/** Ask the target what it has. Refuses only when the probe could not run at all. */
export async function probeGuest(runner: Runner, target: Target, home: string | undefined): Promise<GuestFacts> {
  const result = await runOn(runner, target, ['bash', '-c', guestProbeScript(home)], { timeoutMs: PROBE_TIMEOUT_MS });
  refuseIfUnrun(target, result, 'read this host');
  const { single, missing } = parseKeyValues(result.stdout);
  const guestHome = single.get('home');
  const user = single.get('user');
  const freeKib = single.get('free_kib');
  if (guestHome === undefined || user === undefined || freeKib === undefined || !/^\d+$/.test(freeKib)) {
    throw new BootstrapRefusal(
      'host_unresponsive',
      `the probe inside ${describeTarget(target)} did not answer home, user and free space: `
        + `${JSON.stringify(result.stdout.trim().slice(0, 200))}`,
      { detail: (result.stderr || result.stdout).trim() },
    );
  }
  const crucible = single.get('crucible');
  const server: InstalledRuntime | null = crucible === undefined
    ? null
    : {
      crucible,
      python: `${guestHome}/${SERVER_SUBDIR}/bin/python3`,
      version: single.get('version') ?? null,
      pythonSha256: single.get('python_sha256') ?? null,
      release: single.get('release') ?? null,
    };
  return { home: guestHome, user, freeBytes: Number(freeKib) * 1024, missingTools: missing, server };
}

/** The tools refusal, with the command for the platform that is missing them. */
export function refuseMissingTools(target: Target, platform: NodeJS.Platform, missing: readonly string[]): void {
  if (missing.length === 0) return;
  const command = platform === 'darwin'
    ? `brew install ${missing.join(' ')}`
    : `sudo apt-get update && sudo apt-get install -y ${missing.join(' ')}`;
  throw new BootstrapRefusal(
    'guest_missing_tool',
    `${describeTarget(target)} has no ${missing.join(', ')}. The interpreter is fetched with curl and unpacked with `
      + 'tar, so the install stops here rather than downloading something it cannot open.',
    { command },
  );
}

/** Where everything about the runtime lives, from one home. */
export interface RuntimePaths {
  /** `<home>/server`. */
  dest: string;
  /** `<home>/server.partial`. */
  partial: string;
  /** `<home>/downloads`. */
  downloads: string;
  /** `<home>/downloads/<cpython asset>`. */
  archive: string;
  /** `<home>/downloads/<wheel>`. */
  wheel: string;
  /** `<home>/server/.crucible`. */
  stamp: string;
  /** `<home>/server/bin/crucible`. */
  crucible: string;
  /** `<home>/server/bin/python3`. */
  python: string;
}

export function runtimePaths(home: string, pin: StandalonePython, release: string): RuntimePaths {
  const dest = `${home}/${SERVER_SUBDIR}`;
  return {
    dest,
    partial: `${dest}${PARTIAL_SUFFIX}`,
    downloads: `${home}/${DOWNLOADS_SUBDIR}`,
    archive: `${home}/${DOWNLOADS_SUBDIR}/${pin.asset}`,
    wheel: `${home}/${DOWNLOADS_SUBDIR}/${wheelAssetName(release)}`,
    stamp: `${dest}/${STAMP_NAME}`,
    crucible: `${dest}/bin/crucible`,
    python: `${dest}/bin/python3`,
  };
}

export interface RuntimeInstallOptions {
  /** The release whose wheel is installed. */
  release: string;
  backend: ServerBackend;
  /** `<CRUCIBLE_HOME>`, as the target spells it. */
  home: string;
  /** The runtime already there, from {@link probeGuest} — a matching digest skips the interpreter. */
  installed: InstalledRuntime | null;
  /**
   * An operator rollback: the EXACT older release being put back, or null.
   *
   * REQUIRED AND EXPLICITLY NULL rather than optional, because it is the one
   * thing that opens the never-older gate and a caller that forgot to think
   * about it must say so in the type. See §6.5.4.
   */
  rollbackTo: string | null;
  timeoutMs: number;
  onLine: (line: string, stream: 'stdout' | 'stderr') => void;
}

export interface RuntimeInstallResult {
  paths: RuntimePaths;
  pin: StandalonePython;
  /** True when the stamped interpreter digest already matched and none was fetched. */
  interpreterSkipped: boolean;
  /** Every guest command this ran, in order, as the guest saw it. */
  ran: string[][];
}

/**
 * Put the pinned interpreter and this release's wheel at `<home>/server`.
 *
 * Idempotent in the half that costs anything: a `.crucible` stamp whose
 * `python_sha256` is the pin's, beside a `bin/python3` that exists, means the
 * interpreter step does nothing. The wheel install always runs — it IS the
 * deploy, it is one megabyte, and re-running it is how a half-finished install
 * is repaired.
 */
export async function installRuntime(
  runner: Runner,
  target: Target,
  options: RuntimeInstallOptions,
): Promise<RuntimeInstallResult> {
  const pin = interpreterFor(options.backend);
  const paths = runtimePaths(options.home, pin, options.release);
  const ran: string[][] = [];

  // NEVER OVER A NEWER RELEASE (INSTALL-UNINSTALL.md §6.5.4). The stamp's
  // `release=` is what this disk already holds, and writing an older tree over
  // it is how one app's set-up button silently took another app's engine back a
  // version. Checked BEFORE any guest command, so a refused downgrade downloads
  // nothing and leaves nothing half-written.
  //
  // An unstamped runtime is not read as "older": `installed.release` is null on
  // a tree that predates the stamp, and a version nobody recorded cannot be
  // compared, so there is nothing here to refuse.
  if (options.installed !== null && options.installed.release !== null
    && compareReleases(options.installed.release, options.release) > 0) {
    if (options.rollbackTo === null) {
      throw new BootstrapRefusal(
        'install_would_downgrade',
        `${paths.dest} inside ${describeTarget(target)} is the ${options.installed.release} release and this would `
          + `install ${options.release} over it. Nothing was downloaded. An operator who means to go back names `
          + `the version: rollbackTo: "${options.release}".`,
      );
    }
    if (options.rollbackTo !== options.release) {
      throw new BootstrapRefusal(
        'rollback_version_mismatch',
        `rollbackTo names ${options.rollbackTo} and the release being installed is ${options.release}. A rollback is `
          + 'an operator naming the exact Crucible they want back, so the two are the same version.',
      );
    }
  }

  const run = async (step: string, script: string, code: RuntimeFailureCode, what: string): Promise<void> => {
    const argv = ['bash', '-c', script];
    ran.push(argv);
    const result = await streamOn(runner, target, argv, { timeoutMs: options.timeoutMs, onLine: options.onLine });
    refuseIfUnrun(target, result, what);
    if (result.code !== 0) throw runtimeFailure(code, target, step, what, result);
  };

  const digestOf = async (file: string, what: string): Promise<string> => {
    const argv = shaArgv(runner.platform, file);
    ran.push(argv);
    const digest = await runOn(runner, target, argv, { timeoutMs: options.timeoutMs });
    refuseIfUnrun(target, digest, what);
    const said = digest.stdout.trim().split(/\s+/)[0] ?? '';
    if (digest.code !== 0 || !/^[0-9a-f]{64}$/.test(said)) {
      throw runtimeFailure('runtime_download_failed', target, 'sha256', what, digest);
    }
    return said;
  };

  const interpreterSkipped = options.installed !== null && options.installed.pythonSha256 === pin.sha256;
  if (!interpreterSkipped) {
    await run(
      'fetch-python',
      `rm -f ${shellQuote(paths.archive)}`
      + ` && curl ${CURL_ARGS.join(' ')} -o ${shellQuote(paths.archive)} ${shellQuote(interpreterUrl(pin))}`,
      'runtime_download_failed',
      `download ${interpreterUrl(pin)}`,
    );
    const got = await digestOf(paths.archive, `checksum ${paths.archive}`);
    if (got !== pin.sha256) {
      const remove = `rm -f ${shellQuote(paths.archive)}`;
      ran.push(['bash', '-c', remove]);
      await runOn(runner, target, ['bash', '-c', remove], { timeoutMs: options.timeoutMs });
      throw new BootstrapRefusal(
        'runtime_sha_mismatch',
        `${paths.archive} inside ${describeTarget(target)} hashes ${got} and @crucible/bootstrap pins ${pin.sha256} `
          + `for ${pin.asset}. The archive was deleted; run the install again.`,
      );
    }
    // `install_only` archives carry ONE top-level `python/` directory and that
    // directory IS the interpreter, so the SWAP moves `python/` rather than the
    // archive's root — everything one level deeper would put `bin/python3`
    // where nothing looks for it.
    await run(
      'unpack-python',
      `rm -rf ${shellQuote(paths.partial)} && mkdir -p ${shellQuote(paths.partial)}`
        + ` && tar ${TAR_ARGS.join(' ')} ${shellQuote(paths.archive)} -C ${shellQuote(paths.partial)}`
        + ` && test -x ${shellQuote(`${paths.partial}/python/bin/python3`)}`
        + ` && ${activateRuntimeSh(shellQuote(paths.dest), shellQuote(`${paths.partial}/python`))}`
        + ` && rm -rf ${shellQuote(paths.partial)} ${shellQuote(paths.archive)}`,
      'runtime_unpack_failed',
      `unpack ${pin.asset} into ${paths.dest}`,
    );
  }

  // THE DEPLOY. Its digest comes from the release, beside the bytes, because
  // this side pins an interpreter and never a version of our own code.
  await run(
    'fetch-wheel',
    `rm -f ${shellQuote(paths.wheel)}`
      + ` && curl ${CURL_ARGS.join(' ')} -o ${shellQuote(paths.wheel)} ${shellQuote(wheelUrl(options.release))}`,
    'runtime_download_failed',
    `download ${wheelUrl(options.release)}`,
  );
  const wantResult = await runOn(
    runner, target, ['curl', '-fsSL', '--retry', '3', wheelShaUrl(options.release)],
    { timeoutMs: options.timeoutMs },
  );
  ran.push(['curl', '-fsSL', '--retry', '3', wheelShaUrl(options.release)]);
  refuseIfUnrun(target, wantResult, `fetch ${wheelShaUrl(options.release)}`);
  const want = (wantResult.stdout.trim().split(/\s+/)[0] ?? '').toLowerCase();
  if (wantResult.code !== 0 || !/^[0-9a-f]{64}$/.test(want)) {
    throw new BootstrapRefusal(
      'runtime_download_failed',
      `${wheelShaUrl(options.release)} is not a sha256 (curl exit ${wantResult.code}): `
        + `${JSON.stringify(wantResult.stdout.trim().slice(0, 80))}`,
    );
  }
  const gotWheel = await digestOf(paths.wheel, `checksum ${paths.wheel}`);
  if (gotWheel !== want) {
    const remove = `rm -f ${shellQuote(paths.wheel)}`;
    ran.push(['bash', '-c', remove]);
    await runOn(runner, target, ['bash', '-c', remove], { timeoutMs: options.timeoutMs });
    throw new BootstrapRefusal(
      'runtime_sha_mismatch',
      `${paths.wheel} inside ${describeTarget(target)} hashes ${gotWheel} and ${wheelShaUrl(options.release)} says `
        + `${want}. The wheel was deleted; run the install again.`,
    );
  }

  // The server is shut down before its own `site-packages` is rewritten under
  // it. `local shutdown` is a no-op on a machine where nothing is running, and
  // `service-install` later in the sequence is what starts it again.
  const desktop = runner.platform === 'darwin'
    ? ` && ${shellQuote(paths.python)} -m pip install ${DESKTOP_PACKAGES.map(shellQuote).join(' ')}`
    : '';
  await run(
    'install-wheel',
    `if [ -x ${shellQuote(paths.crucible)} ]; then ${shellQuote(paths.crucible)} local shutdown || true; fi`
      + ` && ${shellQuote(paths.python)} -m pip install --upgrade --no-input ${shellQuote(paths.wheel)}`
      + desktop
      + ` && rm -f ${shellQuote(paths.wheel)}`
      + ` && printf 'python_sha256=%s\\npython_version=%s\\nrelease=%s\\n' ${shellQuote(pin.sha256)} `
      + `${shellQuote(pin.version)} ${shellQuote(options.release)} > ${shellQuote(paths.stamp)}`
      + ` && ${shellQuote(paths.crucible)} --version`,
    'runtime_install_failed',
    `install ${wheelAssetName(options.release)} into ${paths.dest}`,
  );

  return { paths, pin, interpreterSkipped, ran };
}

type RuntimeFailureCode = 'runtime_download_failed' | 'runtime_unpack_failed' | 'runtime_install_failed';

function runtimeFailure(
  code: RuntimeFailureCode,
  target: Target,
  step: string,
  what: string,
  result: RunResult,
): BootstrapRefusal {
  const said = (result.stderr.trim() || result.stdout.trim()) || 'no output';
  return new BootstrapRefusal(
    code,
    `could not ${what} inside ${describeTarget(target)} (${step} exited ${result.code}): ${said}`,
    { detail: said },
  );
}

/** The installed runtime, or the named refusal that says nothing has installed one. */
export function requireRuntime(target: Target, facts: GuestFacts): InstalledRuntime {
  if (facts.server !== null) return facts.server;
  throw new BootstrapRefusal(
    'no_server_runtime',
    `there is no Crucible server at ${facts.home}/${SERVER_SUBDIR}/bin/crucible inside ${describeTarget(target)}. `
      + 'install() downloads the pinned interpreter and this release\'s wheel and puts one there.',
  );
}
