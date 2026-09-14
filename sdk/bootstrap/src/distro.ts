/**
 * The Crucible-owned WSL distro (PHASE14-ENVPACKS.md section 4b).
 *
 * **Owen, 2026-09-14: "yes, lets do it. make it idiot proof."**
 *
 * On Windows a Crucible does not live in the person's Ubuntu. It lives in a
 * distro named `crucible`, imported from a rootfs asset on the same release,
 * under `%LOCALAPPDATA%\Crucible\wsl\`. Four reasons, each of which is why some
 * other approach was rejected:
 *
 * - **No first-run prompt.** `wsl --install -d Ubuntu` opens an interactive
 *   username/password dialog that an app cannot answer. `wsl --import` asks
 *   nothing.
 * - **No collision.** Owen's Ubuntu — with a training run, a live server and
 *   five years of state in it — is never written to. A machine that has BOTH a
 *   `crucible` distro and a config in another distro is `two_local_crucibles`,
 *   refused by name, because which one is `local` is not a thing to guess.
 * - **systemd is a fact, not a probe.** The image ships `[boot] systemd=true`.
 *   The repair path in 4c exists for distros a person chose by hand.
 * - **The GPU is the host's driver.** WSL exposes the Windows NVIDIA driver
 *   into every distro; there is nothing to install in the guest.
 *
 * The rootfs is downloaded on the WINDOWS side — `wsl --import` reads a Windows
 * path, so there is no guest to download it into yet — with `curl.exe` and
 * verified with `certutil -hashfile`, both of which ship with Windows 10 and
 * 11. Its digest is a sibling asset, `<rootfs>.sha256`, one line.
 *
 * Nothing here ever runs `wsl --shutdown`: that stops EVERY distro, including
 * the one with somebody's training run in it. `wsl --terminate crucible` is the
 * only stop this file performs, and only on the distro it created.
 */
import { BootstrapRefusal } from './errors.js';
import { rootfsAssetName, releaseAssetUrl } from './envpacks.js';
import type { RunResult, Runner, StreamOptions } from './runner.js';
import { parseWslList, wslArgv, wslListArgv, wslRootArgv, type WslDistro } from './wsl.js';

/** The distro Crucible owns. One name, everywhere. */
export const CRUCIBLE_DISTRO = 'crucible';

/**
 * The line `/etc/wsl.conf` carries in the Crucible rootfs and nowhere else.
 * A `crucible` distro without it was not imported from our image — either a
 * half-finished import, or somebody's own distro wearing the name.
 */
export const WSL_CONF_MARKER = '# crucible-rootfs';

/** `/etc/wsl.conf`, exactly as the rootfs ships it and as the repair writes it. */
export const WSL_CONF_TEXT = `${WSL_CONF_MARKER}\n[boot]\nsystemd=true\n[user]\ndefault=crucible\n`;

const LIST_TIMEOUT_MS = 15_000;
const PROBE_TIMEOUT_MS = 60_000;
/** An import writes a multi-gigabyte ext4 file. It is slow and it is not a hang. */
const IMPORT_TIMEOUT_MS = 30 * 60_000;
const DOWNLOAD_TIMEOUT_MS = 60 * 60_000;

// ------------------------------------------------------------------ listing

export async function listDistros(runner: Runner): Promise<{ distros: WslDistro[]; default: string | null }> {
  const result = await runner.run(wslListArgv(), { timeoutMs: LIST_TIMEOUT_MS });
  if (result.failure !== null) {
    if (/ENOENT/.test(result.failure)) {
      throw new BootstrapRefusal('wsl_missing', `wsl.exe is not on this machine: ${result.failure}.`, { command: 'wsl --install --no-distribution' });
    }
    throw new BootstrapRefusal('wsl_read_failed', `wsl.exe -l -v did not answer: ${result.failure}`, { detail: (result.stderr || result.stdout).trim() });
  }
  if (result.code !== 0) {
    const said = (result.stderr.trim() || result.stdout.trim()) || `exit ${result.code}`;
    throw new BootstrapRefusal('wsl_missing', `WSL is not usable here (wsl.exe -l -v said: ${said}).`, { command: 'wsl --install --no-distribution', detail: said });
  }
  return parseWslList(result.stdout);
}

// --------------------------------------------------------------- which one

export interface DistroChoiceOptions {
  /** The app's own WSL distro setting, when it has one. */
  distro?: string;
  /** Use `distro` verbatim and resolve nothing. The way out of `two_local_crucibles`. */
  exact?: boolean;
}

/**
 * WHICH DISTRO IS `local` — the rule, in one place.
 *
 * 1. `{exact: true}` with a name: that one, no questions. This exists so a
 *    person who deliberately runs two can say which.
 * 2. A `crucible` distro exists: that one. It is ours, and an app that imported
 *    it is not then supposed to read a server out of somebody else's guest.
 * 3. Otherwise the app's setting.
 * 4. Otherwise `no_wsl_distro` — there is no "the default distro" here, for the
 *    reason `target.ts` gives: a server read from the wrong guest is the wrong
 *    server.
 *
 * Between 2 and 3 there is one refusal: when a `crucible` distro exists AND the
 * app's distro has a config of its own, both are local Crucibles and picking
 * silently would move the app from the server it has been using to an empty one.
 */
export async function resolveDistro(runner: Runner, options: DistroChoiceOptions = {}): Promise<string> {
  const named = options.distro !== undefined && options.distro.trim() !== '' ? options.distro : null;
  if (options.exact === true) {
    if (named === null) throw new BootstrapRefusal('no_wsl_distro', '{exact: true} needs a distro name to be exact about.');
    return named;
  }
  const listed = await listDistros(runner);
  const ours = listed.distros.find((entry) => entry.name === CRUCIBLE_DISTRO);
  if (ours === undefined) {
    if (named !== null) return named;
    throw new BootstrapRefusal(
      'no_wsl_distro',
      `there is no "${CRUCIBLE_DISTRO}" distro on this machine and no distro was named. `
        + `WSL lists ${listed.distros.map((entry) => entry.name).join(', ') || 'nothing'}. `
        + 'ensureDistro() imports ours; or name one.',
      { command: 'crucible install (or pass {distro})' },
    );
  }
  if (named !== null && named !== CRUCIBLE_DISTRO) {
    const theirs = await hasConfig(runner, named);
    if (theirs) {
      throw new BootstrapRefusal(
        'two_local_crucibles',
        `this machine has a "${CRUCIBLE_DISTRO}" distro AND a Crucible config inside "${named}". `
          + 'Two local servers is a thing an operator does on purpose, so which one is `local` is not guessed here: '
          + `pass {distro: "${named}", exact: true} for the old one, {distro: "${CRUCIBLE_DISTRO}", exact: true} for ours, `
          + 'or remove one.',
        { command: `wsl.exe -d ${named} --exec bash -c 'ls -l ~/.crucible/config.toml'` },
      );
    }
  }
  return CRUCIBLE_DISTRO;
}

/** Does this distro hold a Crucible config? A read, never a write. */
async function hasConfig(runner: Runner, distro: string): Promise<boolean> {
  const result = await runner.run(
    wslArgv(distro, ['bash', '-c', 'test -f "${CRUCIBLE_HOME:-$HOME/.crucible}/config.toml"']),
    { timeoutMs: PROBE_TIMEOUT_MS },
  );
  // A distro that will not answer is not a distro with a config: `two_local_crucibles`
  // must not fire because a stopped guest took too long to boot. It is reported
  // as itself.
  if (result.failure !== null) {
    throw new BootstrapRefusal(
      'wsl_read_failed',
      `could not ask "${distro}" whether it holds a Crucible config: ${result.failure}`,
      { detail: (result.stderr || result.stdout).trim() },
    );
  }
  return result.code === 0;
}

// ------------------------------------------------------------------ import

export interface EnsureDistroOptions {
  /** The release whose rootfs asset is imported. */
  release: string;
  /** Where the distro's ext4 file goes. Defaults to `%LOCALAPPDATA%\\Crucible\\wsl`. */
  installDir?: string;
  /** Where the rootfs is downloaded to. Defaults to `%LOCALAPPDATA%\\Crucible\\downloads`. */
  downloadDir?: string;
  /**
   * A rootfs URL to import INSTEAD of the release asset. For a live check
   * against a stock image before the release carries one; never a fallback —
   * the caller names it or the release's asset is used.
   */
  rootfsUrl?: string;
  /** Every line the import prints. */
  onLine?: StreamOptions['onLine'];
}

export interface DistroOutcome {
  name: string;
  /** Did THIS call import it? */
  imported: boolean;
  /** Where its ext4 file is, when this call imported it. */
  installDir: string | null;
  /** Was `/etc/wsl.conf` written by this call (the 4c repair)? */
  repaired: boolean;
  detail: string;
}

/** `%LOCALAPPDATA%\Crucible\wsl` — from the environment, never assembled from a username. */
export function crucibleAppData(runner: Runner, leaf: string): string {
  const local = runner.env['LOCALAPPDATA'];
  if (local === undefined || local.trim() === '') {
    throw new BootstrapRefusal(
      'host_unresponsive',
      'LOCALAPPDATA is not set, so there is no per-user place to put the Crucible distro. '
        + 'Pass {installDir} to say where it goes.',
    );
  }
  return `${local}\\Crucible\\${leaf}`;
}

export function importArgv(installDir: string, rootfs: string): string[] {
  return ['wsl.exe', '--import', CRUCIBLE_DISTRO, installDir, rootfs, '--version', '2'];
}

/** `wsl --terminate crucible`. Only ever this distro — never `--shutdown`, which stops every VM. */
export function terminateArgv(distro: string = CRUCIBLE_DISTRO): string[] {
  return ['wsl.exe', '--terminate', distro];
}

/** `wsl --unregister crucible`. Destroys the distro; called only for a partial import of ours. */
export function unregisterArgv(distro: string = CRUCIBLE_DISTRO): string[] {
  return ['wsl.exe', '--unregister', distro];
}

/**
 * The `crucible` distro exists, runs systemd, and was imported from our image.
 * Idempotent: present and marked is a no-op; present and unmarked with nothing
 * in it is re-imported; present, unmarked and holding a config is refused.
 */
export async function ensureDistro(options: EnsureDistroOptions, runner: Runner): Promise<DistroOutcome> {
  if (runner.platform !== 'win32') {
    throw new BootstrapRefusal(
      'unsupported_platform',
      `the Crucible WSL distro is a Windows thing and this is ${runner.platform}; on Linux and macOS the server runs on the machine.`,
    );
  }
  const onLine: StreamOptions['onLine'] = options.onLine ?? ((): void => undefined);
  const listed = await listDistros(runner);
  const present = listed.distros.find((entry) => entry.name === CRUCIBLE_DISTRO);

  if (present !== undefined) {
    if (present.version !== 2) {
      throw new BootstrapRefusal(
        'wsl1_only',
        `the "${CRUCIBLE_DISTRO}" distro is WSL${present.version}. Only WSL2 passes the GPU through.`,
        { command: `wsl --set-version ${CRUCIBLE_DISTRO} 2` },
      );
    }
    const conf = await readWslConf(runner, CRUCIBLE_DISTRO);
    if (conf.includes(WSL_CONF_MARKER)) {
      return { name: CRUCIBLE_DISTRO, imported: false, installDir: null, repaired: false, detail: `"${CRUCIBLE_DISTRO}" is already imported and marked` };
    }
    if (await hasConfig(runner, CRUCIBLE_DISTRO)) {
      throw new BootstrapRefusal(
        'distro_unmarked',
        `a distro named "${CRUCIBLE_DISTRO}" exists, holds a Crucible config, and has no ${WSL_CONF_MARKER} in /etc/wsl.conf, `
          + 'so it was not imported from our rootfs. It is not re-imported over — that would delete a server somebody is using.',
        { command: `wsl.exe -d ${CRUCIBLE_DISTRO} --exec cat /etc/wsl.conf` },
      );
    }
    // A partial import: our name, our place, nothing in it. Unregistered and redone.
    const unregister = await runner.run(unregisterArgv(), { timeoutMs: IMPORT_TIMEOUT_MS });
    if (unregister.failure !== null || unregister.code !== 0) {
      throw new BootstrapRefusal(
        'distro_import_failed',
        `"${CRUCIBLE_DISTRO}" is half-imported (no ${WSL_CONF_MARKER}) and would not unregister: `
          + `${unregister.failure ?? (unregister.stderr.trim() || `exit ${unregister.code}`)}`,
      );
    }
    onLine(`unregistered a half-imported "${CRUCIBLE_DISTRO}"`, 'stdout');
  }

  const installDir = options.installDir ?? crucibleAppData(runner, 'wsl');
  const downloadDir = options.downloadDir ?? crucibleAppData(runner, 'downloads');
  const asset = rootfsAssetName(options.release);
  const url = options.rootfsUrl ?? releaseAssetUrl(options.release, asset);
  const rootfs = `${downloadDir}\\${asset}`;

  const fetch = await runner.stream(['curl.exe', '-fL', '--retry', '3', '--create-dirs', '-o', rootfs, url], { timeoutMs: DOWNLOAD_TIMEOUT_MS, onLine });
  if (fetch.failure !== null || fetch.code !== 0) {
    throw new BootstrapRefusal(
      'pack_download_failed',
      `the Crucible rootfs would not download from ${url} (${fetch.failure ?? `curl exit ${fetch.code}`}).`,
      { detail: (fetch.stderr || fetch.stdout).trim() },
    );
  }

  // The digest, when the release publishes one beside the image. `rootfsUrl`
  // names an image that is not ours (a live check against a stock Ubuntu), and
  // there is no digest asset for it — so the caller that supplied the URL is
  // the one vouching for it, and that is said out loud rather than checked
  // against a file that does not exist.
  if (options.rootfsUrl === undefined) {
    await verifyRootfs(runner, options.release, asset, rootfs, onLine);
  } else {
    onLine(`rootfs came from {rootfsUrl} ${url}; its digest is the caller's to vouch for`, 'stderr');
  }

  const imported = await runner.stream(importArgv(installDir, rootfs), { timeoutMs: IMPORT_TIMEOUT_MS, onLine });
  if (imported.failure !== null || imported.code !== 0) {
    throw new BootstrapRefusal(
      'distro_import_failed',
      `wsl --import ${CRUCIBLE_DISTRO} ${installDir} failed (${imported.failure ?? `exit ${imported.code}`}): `
        + `${(imported.stderr.trim() || imported.stdout.trim()) || 'no output'}`,
      { detail: (imported.stderr || imported.stdout).trim() },
    );
  }

  const conf = await readWslConf(runner, CRUCIBLE_DISTRO);
  let repaired = false;
  if (!conf.includes(WSL_CONF_MARKER)) {
    // A stock image (or an older rootfs) has no marker. It is OURS — we just
    // imported it under our name into our directory — so writing wsl.conf is
    // not a change to somebody's machine, it is finishing the import.
    await writeWslConf(runner, CRUCIBLE_DISTRO);
    repaired = true;
  }

  // Once, so systemd and the default user take. `--terminate`, never
  // `--shutdown`: the other distros on this machine are not ours to stop.
  const stop = await runner.run(terminateArgv(), { timeoutMs: LIST_TIMEOUT_MS });
  if (stop.failure !== null || stop.code !== 0) {
    throw new BootstrapRefusal(
      'distro_import_failed',
      `"${CRUCIBLE_DISTRO}" was imported and would not terminate, so systemd has not started in it: `
        + `${stop.failure ?? (stop.stderr.trim() || `exit ${stop.code}`)}`,
    );
  }

  return {
    name: CRUCIBLE_DISTRO,
    imported: true,
    installDir,
    repaired,
    detail: `imported ${asset} into ${installDir} as "${CRUCIBLE_DISTRO}" (WSL2)${repaired ? ', wrote /etc/wsl.conf' : ''}`,
  };
}

/** `certutil -hashfile <file> SHA256` against the `<asset>.sha256` the release publishes. */
async function verifyRootfs(
  runner: Runner,
  release: string,
  asset: string,
  rootfs: string,
  onLine: StreamOptions['onLine'],
): Promise<void> {
  const url = releaseAssetUrl(release, `${asset}.sha256`);
  const expected = await runner.run(['curl.exe', '-fsSL', '--retry', '3', url], { timeoutMs: DOWNLOAD_TIMEOUT_MS });
  if (expected.failure !== null || expected.code !== 0) {
    throw new BootstrapRefusal(
      'pack_download_failed',
      `the rootfs digest ${url} would not download (${expected.failure ?? `curl exit ${expected.code}`}), so the image cannot be verified.`,
      { detail: (expected.stderr || expected.stdout).trim() },
    );
  }
  const want = (expected.stdout.trim().split(/\s+/)[0] ?? '').toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(want)) {
    throw new BootstrapRefusal('pack_sha_mismatch', `${url} is not a sha256: ${JSON.stringify(expected.stdout.trim().slice(0, 80))}`);
  }
  const got = await runner.run(['certutil', '-hashfile', rootfs, 'SHA256'], { timeoutMs: DOWNLOAD_TIMEOUT_MS });
  const hex = /^[0-9a-f ]{64,}$/im.exec(got.stdout.replace(/\r/g, ''))?.[0]?.replace(/ /g, '').toLowerCase() ?? '';
  if (got.failure !== null || got.code !== 0 || hex.length !== 64) {
    throw new BootstrapRefusal(
      'pack_download_failed',
      `certutil would not hash ${rootfs} (${got.failure ?? `exit ${got.code}`}): ${(got.stderr.trim() || got.stdout.trim()) || 'no output'}`,
    );
  }
  if (hex !== want) {
    throw new BootstrapRefusal(
      'pack_sha_mismatch',
      `the downloaded rootfs hashes ${hex} and ${url} says ${want}. Delete ${rootfs} and run this again.`,
    );
  }
  onLine(`rootfs verified: ${hex}`, 'stdout');
}

/** `/etc/wsl.conf` as the distro has it, or '' when there is none. */
export async function readWslConf(runner: Runner, distro: string): Promise<string> {
  const result: RunResult = await runner.run(
    wslRootArgv(distro, ['bash', '-c', 'test -f /etc/wsl.conf && cat /etc/wsl.conf || true']),
    { timeoutMs: PROBE_TIMEOUT_MS },
  );
  if (result.failure !== null) {
    throw new BootstrapRefusal(
      'wsl_read_failed',
      `could not read /etc/wsl.conf in "${distro}": ${result.failure}`,
      { detail: (result.stderr || result.stdout).trim() },
    );
  }
  return result.stdout;
}

/** Write `/etc/wsl.conf` as root. The caller decides whether the distro is ours to write to. */
export async function writeWslConf(runner: Runner, distro: string): Promise<void> {
  const script = `cat > /etc/wsl.conf <<'EOF'\n${WSL_CONF_TEXT}EOF\n`;
  const result = await runner.run(wslRootArgv(distro, ['bash', '-c', script]), { timeoutMs: PROBE_TIMEOUT_MS });
  if (result.failure !== null || result.code !== 0) {
    throw new BootstrapRefusal(
      'distro_not_systemd',
      `could not write /etc/wsl.conf in "${distro}" (${result.failure ?? `exit ${result.code}`}): `
        + `${(result.stderr.trim() || result.stdout.trim()) || 'no output'}`,
      { command: `wsl.exe -d ${distro} -u root --exec bash -c "printf '[boot]\\nsystemd=true\\n' > /etc/wsl.conf" && wsl.exe --terminate ${distro}` },
    );
  }
}
