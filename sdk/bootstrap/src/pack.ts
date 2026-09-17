/**
 * The server pack: a relocatable CPython with the crucible wheel already in it,
 * downloaded from the release and unpacked into `<CRUCIBLE_HOME>/server/`.
 *
 * This replaces the conda walk entirely (PHASE14-ENVPACKS.md section 0). There
 * is no interpreter to find on a fresh machine, because there is no interpreter
 * on a fresh machine: the pack carries one. Everything `install()` runs after
 * this step is `<CRUCIBLE_HOME>/server/bin/crucible`.
 *
 * Three rules, each with a reason and a test:
 *
 * 1. **The download happens INSIDE the guest, with the guest's `curl`, into the
 *    guest's filesystem.** Never onto `/mnt/c` and across the 9p mount: that is
 *    slow, it loses the permission bits a Python tree needs, and it puts an 8 GB
 *    file on the wrong disk. On macOS and Linux "the guest" is the machine.
 *
 * 2. **Peak extra disk is one part.** Each part is fetched, appended to the
 *    archive, and deleted. The archive is rebuilt from zero on every run, so a
 *    half-appended archive from a killed run is never continued into.
 *
 * 3. **The sha256 is computed in the guest and compared here.** A mismatch
 *    deletes the archive and refuses by name — `pack_sha_mismatch` — because the
 *    next run must start clean rather than resume into a corrupt file.
 *
 * The unpack is `tar --zstd -xf … -C <dest>.partial` followed by a rename, so a
 * half-unpacked tree is never at the path everything else runs from.
 */
import { BootstrapRefusal } from './errors.js';
import { findPack, packAssetName, parseEnvpacks, releaseAssetUrl, type EnvPacks, type PackBackend, type PackEntry } from './envpacks.js';
import type { RunResult, Runner } from './runner.js';
import { describeTarget, refuseIfUnrun, runOn, streamOn, type Target } from './target.js';
import { shellQuote } from './wsl.js';

/** `<CRUCIBLE_HOME>/server` — where the server pack is unpacked. */
export const SERVER_SUBDIR = 'server';

/**
 * Where the WINDOWS host pack unpacks, under `%LOCALAPPDATA%\Crucible\`
 * (PHASE15-HOST.md 4.4). Beside `wsl\` and `downloads\`, which
 * `distro.ts` already puts there. `crucible/host/paths.py` spells the same
 * word on the Python side and `install.ps1` is generated from this one.
 */
export const HOST_SUBDIR = 'host';
/** `<CRUCIBLE_HOME>/downloads` — parts land here and are deleted as they are appended. */
export const DOWNLOADS_SUBDIR = 'downloads';
/** The half-unpacked tree's suffix. Renamed onto the real path only when tar exits 0. */
export const PARTIAL_SUFFIX = '.partial';

/** Shared activation transaction used by app installs and generated install.sh.
 * Arguments are shell expressions (already quoted), never untrusted raw paths.
 */
export function activatePackSh(dest: string, partial: string): string {
  return `activate_crucible_pack() {\n`
    + `  _crucible_dest=${dest}; _crucible_partial=${partial}; _crucible_previous="$_crucible_dest.previous"\n`
    + `  if [ -e "$_crucible_previous" ]; then echo "upgrade_recovery_required: $_crucible_previous was preserved from an interrupted upgrade" >&2; return 1; fi\n`
    + `  if [ -e "$_crucible_dest" ]; then\n`
    + `    "$_crucible_partial/bin/crucible" local shutdown || return 1\n`
    + `    mv "$_crucible_dest" "$_crucible_previous" || return 1\n`
    + `  fi\n`
    + `  if mv "$_crucible_partial" "$_crucible_dest" && "$_crucible_dest/bin/crucible" --version; then\n`
    + `    rm -rf "$_crucible_previous" || return 1\n`
    + `  else\n`
    + `    if [ -e "$_crucible_dest" ] && [ ! -e "$_crucible_partial" ]; then mv "$_crucible_dest" "$_crucible_partial" || return 1; fi\n`
    + `    if [ -e "$_crucible_previous" ]; then mv "$_crucible_previous" "$_crucible_dest" || return 1; fi\n`
    + `    echo "upgrade_activation_failed: the previous runtime was preserved" >&2; return 1\n`
    + `  fi\n`
    + `}\nactivate_crucible_pack`;
}
/** `<CRUCIBLE_HOME>/server/.pack` — `sha256=` and `release=`, the two facts that say what this tree is. */
export const STAMP_NAME = '.pack';

/**
 * curl's arguments, one owner for both the TypeScript and the generated
 * `install.sh`. `-f` so an HTML error page is never saved as a part, `-L` for
 * the release redirect, `--retry` for a flaky minute, `--continue-at -` so a
 * killed run resumes the part it was on, `--create-dirs` so nothing has to
 * mkdir first.
 */
export const CURL_ARGS: readonly string[] = ['-fL', '--retry', '3', '--retry-delay', '2', '--continue-at', '-', '--create-dirs'];

/** tar's, likewise. `--zstd` is why zstd is probed for before anything downloads. */
export const TAR_ARGS: readonly string[] = ['--zstd', '-xf'];

/** The sha256 tool, per platform. macOS has no `sha256sum`; every Linux does. */
export function shaArgv(platform: NodeJS.Platform, file: string): string[] {
  return platform === 'darwin' ? ['shasum', '-a', '256', file] : ['sha256sum', file];
}

/** The tools a pack install needs, probed before anything is fetched. */
export const REQUIRED_TOOLS: readonly string[] = ['curl', 'tar', 'zstd'];

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
  /** The server pack already unpacked, or null. */
  server: InstalledPack | null;
}

export interface InstalledPack {
  /** `<home>/server/bin/crucible`. */
  crucible: string;
  /** `<home>/server/bin/python3`. */
  python: string;
  /** What `crucible --version` printed, or null when the stamp is there and the binary would not answer. */
  version: string | null;
  /** The archive sha256 the stamp records, or null when there is no stamp. */
  sha256: string | null;
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
  const server: InstalledPack | null = crucible === undefined
    ? null
    : {
      crucible,
      python: `${guestHome}/${SERVER_SUBDIR}/bin/python3`,
      version: single.get('version') ?? null,
      sha256: single.get('sha256') ?? null,
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
    `${describeTarget(target)} has no ${missing.join(', ')}. A pack is fetched with curl and unpacked with `
      + 'tar --zstd, so the install stops here rather than downloading gigabytes it cannot open.',
    { command },
  );
}

/** Where everything about a pack lives, from one home. */
export interface PackPaths {
  /** `<home>/server`. */
  dest: string;
  /** `<home>/server.partial`. */
  partial: string;
  /** `<home>/downloads`. */
  downloads: string;
  /** `<home>/downloads/<asset>.tar.zst` — the reassembled archive. */
  archive: string;
  /** `<home>/server/.pack`. */
  stamp: string;
  /** `<home>/server/bin/crucible`. */
  crucible: string;
}

export function packPaths(home: string, entry: PackEntry, release: string): PackPaths {
  const dir = entry.name === 'server' ? SERVER_SUBDIR : `envs/${entry.name}`;
  const dest = `${home}/${dir}`;
  return {
    dest,
    partial: `${dest}${PARTIAL_SUFFIX}`,
    downloads: `${home}/${DOWNLOADS_SUBDIR}`,
    // THE ROW'S release again: this path is what the parts are concatenated
    // into, and a carried pack's parts are still named for the release that
    // built them. Naming the joined file after a version that appears in none
    // of its parts leaves a scratch file whose name contradicts its contents.
    archive: `${home}/${DOWNLOADS_SUBDIR}/${packAssetName(entry.name, entry.backend, entry.release)}`,
    stamp: `${dest}/${STAMP_NAME}`,
    crucible: `${dest}/bin/crucible`,
  };
}

/**
 * `unpacked_bytes` + the whole archive + one part. The part size is the
 * archive's size over the part count: the splitter cuts equal parts under
 * 1900 MiB and only the last is short, so this over-states by less than a part
 * and never under-states.
 */
export function requiredBytes(entry: PackEntry): number {
  const part = Math.ceil(entry.bytes / entry.parts.length);
  return entry.unpackedBytes + entry.bytes + part;
}

export function gib(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

/** Fetch and validate the release's manifest, with the guest's curl. */
export async function fetchManifest(
  runner: Runner,
  target: Target,
  release: string,
  url: string,
  timeoutMs: number,
): Promise<EnvPacks> {
  const result = await runOn(runner, target, ['curl', '-fsSL', '--retry', '3', url], { timeoutMs });
  refuseIfUnrun(target, result, `fetch ${url}`);
  if (result.code !== 0) {
    const said = (result.stderr.trim() || result.stdout.trim()) || `exit ${result.code}`;
    // curl's exit 6/7/28 are "the network did not work", told apart from "the
    // release has no such file" (22, with a 404 in the message).
    const networkish = [6, 7, 28, 35].includes(result.code ?? -1);
    throw new BootstrapRefusal(
      networkish ? 'guest_no_network' : 'pack_manifest_unreadable',
      `${describeTarget(target)} could not fetch ${url} (curl exit ${result.code}): ${said}`,
      { detail: said },
    );
  }
  return parseEnvpacks(result.stdout, url, release);
}

export interface PackInstallOptions {
  /** The release whose assets are fetched. */
  release: string;
  backend: PackBackend;
  /** `<CRUCIBLE_HOME>`, as the target spells it. */
  home: string;
  /** What the guest reported free, from {@link probeGuest}. */
  freeBytes: number;
  /** The pack already there, from {@link probeGuest} — a matching sha is a skip. */
  installed: InstalledPack | null;
  timeoutMs: number;
  onLine: (line: string, stream: 'stdout' | 'stderr') => void;
}

export interface PackInstallResult {
  entry: PackEntry;
  paths: PackPaths;
  /** True when the stamp already matched and nothing was downloaded. */
  skipped: boolean;
  /** Every guest command this ran, in order, as the guest saw it. */
  ran: string[][];
}

/**
 * Download, verify and unpack one pack. Idempotent: a `.pack` stamp whose
 * sha256 is the manifest's, beside a `bin/crucible` that exists, is a skip.
 */
export async function installPack(
  runner: Runner,
  target: Target,
  manifest: EnvPacks,
  name: string,
  options: PackInstallOptions,
): Promise<PackInstallResult> {
  const entry = findPack(manifest, name, options.backend);
  const paths = packPaths(options.home, entry, options.release);
  const ran: string[][] = [];

  if (options.installed !== null && options.installed.sha256 === entry.sha256) {
    return { entry, paths, skipped: true, ran };
  }

  const needed = requiredBytes(entry);
  if (options.freeBytes < needed) {
    throw new BootstrapRefusal(
      'pack_disk',
      `the ${entry.name} pack needs ${gib(needed)} free on ${describeTarget(target)} `
        + `(${gib(entry.unpackedBytes)} unpacked + ${gib(entry.bytes)} archive + one part) and there is ${gib(options.freeBytes)}. `
        + 'Nothing was downloaded.',
    );
  }

  const run = async (step: string, script: string, code: BootstrapRefusalCodeForPack, what: string): Promise<void> => {
    const argv = ['bash', '-c', script];
    ran.push(argv);
    const result = await streamOn(runner, target, argv, { timeoutMs: options.timeoutMs, onLine: options.onLine });
    refuseIfUnrun(target, result, what);
    if (result.code !== 0) throw packFailure(code, target, step, what, result);
  };

  // The archive is rebuilt from zero: a half-appended one from a killed run is
  // never continued into, because nothing on disk records how far it got.
  await run('start', `rm -f ${shellQuote(paths.archive)} && mkdir -p ${shellQuote(paths.downloads)}`, 'pack_download_failed', 'clear the download directory');

  for (const part of entry.parts) {
    // THE ROW'S RELEASE, NOT THE ONE BEING INSTALLED. An unchanged pack is
    // carried by reference rather than rebuilt or copied, so its parts stay in
    // the release that built them — see PackEntry.release.
    const url = releaseAssetUrl(entry.release, part);
    const file = `${paths.downloads}/${part}`;
    const script = `curl ${CURL_ARGS.join(' ')} -o ${shellQuote(file)} ${shellQuote(url)}`
      + ` && cat ${shellQuote(file)} >> ${shellQuote(paths.archive)}`
      + ` && rm -f ${shellQuote(file)}`;
    await run(part, script, 'pack_download_failed', `download ${url}`);
  }

  const digest = await runOn(runner, target, shaArgv(runner.platform, paths.archive), { timeoutMs: options.timeoutMs });
  ran.push(shaArgv(runner.platform, paths.archive));
  refuseIfUnrun(target, digest, `checksum ${paths.archive}`);
  const said = digest.stdout.trim().split(/\s+/)[0] ?? '';
  if (digest.code !== 0 || !/^[0-9a-f]{64}$/.test(said)) {
    throw packFailure('pack_download_failed', target, 'sha256', `checksum ${paths.archive}`, digest);
  }
  if (said !== entry.sha256) {
    const remove = `rm -f ${shellQuote(paths.archive)}`;
    ran.push(['bash', '-c', remove]);
    await runOn(runner, target, ['bash', '-c', remove], { timeoutMs: options.timeoutMs });
    throw new BootstrapRefusal(
      'pack_sha_mismatch',
      `${paths.archive} inside ${describeTarget(target)} hashes ${said} and the ${options.release} manifest says ${entry.sha256}. `
        + 'The archive was deleted; run the install again.',
    );
  }

  await run(
    'unpack',
    `rm -rf ${shellQuote(paths.partial)} && mkdir -p ${shellQuote(paths.partial)}`
      + ` && tar ${TAR_ARGS.join(' ')} ${shellQuote(paths.archive)} -C ${shellQuote(paths.partial)}`,
    'pack_unpack_failed',
    `unpack ${paths.archive}`,
  );

  // The pack is what it says it is BEFORE it is moved into place: a tree that
  // cannot run `crucible --version` must not become the thing every later step
  // runs, and `.partial` is where a person can look at it.
  await run(
    'verify',
    `test -x ${shellQuote(`${paths.partial}/bin/crucible`)} && ${shellQuote(`${paths.partial}/bin/crucible`)} --version`,
    'pack_unpack_failed',
    `run ${paths.partial}/bin/crucible --version`,
  );

  await run(
    'install',
    activatePackSh(shellQuote(paths.dest), shellQuote(paths.partial))
      + ` && printf 'sha256=%s\\nrelease=%s\\n' ${shellQuote(entry.sha256)} ${shellQuote(options.release)} > ${shellQuote(paths.stamp)}`
      + ` && rm -f ${shellQuote(paths.archive)}`,
    'pack_unpack_failed',
    `put the pack at ${paths.dest}`,
  );

  return { entry, paths, skipped: false, ran };
}

type BootstrapRefusalCodeForPack = 'pack_download_failed' | 'pack_unpack_failed';

function packFailure(
  code: BootstrapRefusalCodeForPack,
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

/** The installed pack, or the named refusal that says nothing has installed one. */
export function requirePack(target: Target, facts: GuestFacts): InstalledPack {
  if (facts.server !== null) return facts.server;
  throw new BootstrapRefusal(
    'no_server_pack',
    `there is no Crucible server at ${facts.home}/${SERVER_SUBDIR}/bin/crucible inside ${describeTarget(target)}. `
      + 'install() downloads the server pack from the release and puts one there.',
  );
}
