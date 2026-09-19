/**
 * The Crucible-owned WSL distro (PHASE14-ENVPACKS.md 4b, image by PHASE20 §2).
 *
 * **Owen, 2026-09-14: "yes, lets do it. make it idiot proof."**
 *
 * On Windows a Crucible does not live in the person's Ubuntu. It lives in a
 * distro named `crucible`, imported from CANONICAL'S OWN WSL image under
 * `%LOCALAPPDATA%\Crucible\wsl\`. Four reasons, each of which is why some
 * other approach was rejected:
 *
 * - **No first-run prompt.** `wsl --install -d Ubuntu` opens an interactive
 *   username/password dialog that an app cannot answer. `wsl --import` asks
 *   nothing.
 * - **No collision.** Owen's Ubuntu — with a training run, a live server and
 *   five years of state in it — is never written to. A machine that has BOTH a
 *   `crucible` distro and a config in another distro is `two_local_crucibles`,
 *   refused by name, because which one is `local` is not a thing to guess.
 * - **systemd is a fact, not a probe.** `finishImport` writes `[boot]
 *   systemd=true` into the distro it just created. The repair path in 4c exists
 *   for distros a person chose by hand.
 * - **The GPU is the host's driver.** WSL exposes the Windows NVIDIA driver
 *   into every distro; there is nothing to install in the guest.
 *
 * WHERE THE IMAGE COMES FROM, AND WHY IT STOPPED BEING OURS
 * ---------------------------------------------------------
 * It used to be `crucible-rootfs-<version>.tar.zst`, built by a Docker job on
 * every tag and uploaded to every release: ~30 MB of Ubuntu, rebuilt because
 * OUR code changed, for an image whose contents had not moved in a month
 * (PHASE20's measurement — 190 MB uploaded per release for a 1 MB wheel).
 * Canonical publishes exactly this image, for WSL, with a `SHA256SUMS` beside
 * it, so the release carries neither any more.
 *
 * The four things `build-rootfs.sh` baked in are now done INSIDE the distro
 * right after the import ({@link finishImport}) — the `crucible` user,
 * passwordless sudo, the `# crucible-rootfs` marker and `[boot] systemd=true`.
 * Canonical's image already carries curl and ca-certificates.
 *
 * WE STORE NO DIGEST OF THEIRS. The mirror's own `SHA256SUMS`, fetched from the
 * same directory, is the digest's owner; a pin of ours would be a second copy
 * of somebody else's fact, stale the day they rebuild. What `current/` moving
 * looks like is therefore a sums file that no longer names our download — which
 * is detected, by name — rather than a check that quietly passes.
 *
 * The image is downloaded on the WINDOWS side — `wsl --import` reads a Windows
 * path, so there is no guest to download it into yet — with `curl.exe` and
 * verified with `certutil -hashfile`, both of which ship with Windows 10 and 11.
 *
 * Nothing here ever runs `wsl --shutdown`: that stops EVERY distro, including
 * the one with somebody's training run in it. `wsl --terminate crucible` is the
 * only stop this file performs, and only on the distro it created.
 */
import { BootstrapRefusal } from './errors.js';

import type { RunResult, Runner, StreamOptions } from './runner.js';
import { parseWslList, wslArgv, wslListArgv, wslRootArgv, type WslDistro } from './wsl.js';

/** The distro Crucible owns. One name, everywhere. */
export const CRUCIBLE_DISTRO = 'crucible';

/**
 * CANONICAL'S WSL IMAGES, and the release series we import. ONE PLACE.
 *
 * `current/` is a moving pointer by design — Canonical republishes the point
 * release there — and that is what we want: an image a month old is a longer
 * `apt-get upgrade` for no benefit, and nothing of ours depends on its bytes
 * beyond "Ubuntu 24.04 with systemd". What pins it is the SERIES, `24.04`, and
 * what proves the download is the `SHA256SUMS` in the same directory, which is
 * the digest's one owner (see this file's header).
 *
 * Checked 2026-09-18: 200, 340 MB.
 */
export const UBUNTU_WSL_SERIES = '24.04';
export const UBUNTU_WSL_BASE = `https://cloud-images.ubuntu.com/wsl/releases/${UBUNTU_WSL_SERIES}/current`;
export const UBUNTU_WSL_ROOTFS = 'ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz';
export const UBUNTU_WSL_SUMS = 'SHA256SUMS';

export function rootfsUrl(): string {
  return `${UBUNTU_WSL_BASE}/${UBUNTU_WSL_ROOTFS}`;
}

export function rootfsSumsUrl(): string {
  return `${UBUNTU_WSL_BASE}/${UBUNTU_WSL_SUMS}`;
}

/**
 * The line `/etc/wsl.conf` carries in a Crucible distro and nowhere else.
 *
 * THE MARKER IS STILL OURS. The IMAGE is Canonical's and carries no `wsl.conf`
 * at all, so this is written by {@link finishImport} moments after the import
 * rather than baked in — and it is still exactly what tells our distro from
 * somebody's own wearing the same name. A `crucible` distro without it is
 * either a half-finished import (unregistered and redone) or theirs (refused,
 * `distro_unmarked`), which is the §4c rule unchanged: we write it, then we
 * detect it.
 */
/**
 * `%LOCALAPPDATA%\Crucible\wsl-outcome.json` — what happened to the engine
 * move on this machine (PHASE19-AUTOMATIC-WSL.md 2.2).
 *
 * Spelled HERE because three things read it and none of them can be the owner:
 * `crucible/host/outcome.py` writes and parses it, `install.ps1` reads it for
 * its closing sentence (2.7), and both of those are GENERATED from this file.
 * A name written down in a shell script and again in Python is a file two
 * programs can disagree about the location of.
 */
export const WSL_OUTCOME_NAME = 'wsl-outcome.json';

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
  /**
   * The Crucible release this install is for.
   *
   * **It no longer decides which image is imported** — that is Canonical's
   * `current/` — and it is kept because the refusals and the lines this prints
   * say which install they belong to, and because every other entry point in
   * this package takes one.
   */
  release: string;
  /** Where the distro's ext4 file goes. Defaults to `%LOCALAPPDATA%\\Crucible\\wsl`. */
  installDir?: string;
  /** Where the rootfs is downloaded to. Defaults to `%LOCALAPPDATA%\\Crucible\\downloads`. */
  downloadDir?: string;
  /**
   * A rootfs URL to import INSTEAD of Canonical's. For a live check against a
   * local file or a mirror; never a fallback — the caller names it or
   * Canonical's is used, and a caller that names one is the one vouching for
   * its bytes, out loud, because there is no `SHA256SUMS` beside it to check.
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
  /**
   * Was `/etc/wsl.conf` written by this call?
   *
   * TRUE ON EVERY IMPORT NOW, and that is the change rather than a bug:
   * Canonical's image carries no `wsl.conf`, so `finishImport` always writes
   * one. It stays `false` for a distro that was already there and already
   * marked, which is the question a caller is really asking — did anything
   * change on this machine.
   */
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
  const asset = UBUNTU_WSL_ROOTFS;
  const url = options.rootfsUrl ?? rootfsUrl();
  const rootfs = `${downloadDir}\\${asset}`;

  const fetch = await runner.stream(['curl.exe', '-fL', '--retry', '3', '--create-dirs', '-o', rootfs, url], { timeoutMs: DOWNLOAD_TIMEOUT_MS, onLine });
  if (fetch.failure !== null || fetch.code !== 0) {
    throw new BootstrapRefusal(
      'runtime_download_failed',
      `the Ubuntu WSL image would not download from ${url} (${fetch.failure ?? `curl exit ${fetch.code}`}).`,
      { detail: (fetch.stderr || fetch.stdout).trim() },
    );
  }

  // Canonical's own SHA256SUMS, from the same directory as the image. A
  // `rootfsUrl` names something else, which has no sums file beside it — so the
  // caller that supplied the URL is the one vouching for it, and that is said
  // out loud rather than checked against a file that does not exist.
  if (options.rootfsUrl === undefined) {
    await verifyRootfs(runner, asset, rootfs, onLine);
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

  // WHAT `build-rootfs.sh` USED TO BAKE, done here instead. Canonical's image
  // has no `crucible` user, no sudoers file and no `/etc/wsl.conf`, and it is
  // OURS — just imported under our name into our directory — so writing them is
  // finishing the import rather than touching somebody's machine.
  await finishImport(runner, CRUCIBLE_DISTRO);

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
    repaired: true,
    detail: `imported ${asset} into ${installDir} as "${CRUCIBLE_DISTRO}" (WSL2), `
      + 'created the crucible user and wrote /etc/wsl.conf',
  };
}

/**
 * The four things a Crucible distro needs that Canonical's image does not have,
 * as ONE root script.
 *
 * IT IS A STRING RATHER THAN FOUR CALLS because it has a SECOND reader:
 * `crucible/host/wsl_states.py` carries it as data, emitted by
 * `scripts/gen-install-scripts.ts`, so the Windows host's own importer
 * (`crucible/host/installer.py`) finishes an import exactly the way this
 * package does. That is the seam PHASE15-HOST.md 4.3 names — the table crosses
 * into Python by GENERATION, never by a second hand-written copy.
 *
 * `wsl -u root` is a process launch each time, and these four are lines that
 * have to either all be there or none. Idempotent — `useradd` is guarded, the
 * sudoers file and `wsl.conf` are rewritten — because the 4c repair path can
 * call it on a distro that already has some of it.
 */
export function finishImportScript(): string {
  return [
    "id -u crucible >/dev/null 2>&1 || useradd --create-home --shell /bin/bash crucible",
    'passwd --delete crucible >/dev/null',
    "printf 'crucible ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/crucible",
    'chmod 0440 /etc/sudoers.d/crucible',
    `cat > /etc/wsl.conf <<'EOF'\n${WSL_CONF_TEXT}EOF`,
  ].join('\n');
}

/**
 * Run {@link finishImportScript} in a distro we just imported.
 *
 * The caller decides whether the distro is ours to write to; this does not ask.
 */
export async function finishImport(runner: Runner, distro: string): Promise<void> {
  const result = await runner.run(
    wslRootArgv(distro, ['bash', '-c', finishImportScript()]),
    { timeoutMs: PROBE_TIMEOUT_MS },
  );
  if (result.failure !== null || result.code !== 0) {
    throw new BootstrapRefusal(
      'distro_import_failed',
      `"${distro}" was imported and could not be finished (${result.failure ?? `exit ${result.code}`}): `
        + `${(result.stderr.trim() || result.stdout.trim()) || 'no output'}. It has no crucible user, no `
        + 'passwordless sudo or no /etc/wsl.conf, so nothing can be installed into it.',
      { command: `wsl.exe --unregister ${distro}` },
    );
  }
}

/**
 * `certutil -hashfile <file> SHA256` against Canonical's own `SHA256SUMS`.
 *
 * THE SUMS FILE LISTS EVERY IMAGE IN THAT DIRECTORY, one `<digest>  *<name>`
 * line each, so the row is found BY FILENAME. A sums file that does not name
 * our download is exactly what `current/` moving under us looks like, and it is
 * refused by name rather than passing because no row was compared.
 */
async function verifyRootfs(
  runner: Runner,
  asset: string,
  rootfs: string,
  onLine: StreamOptions['onLine'],
): Promise<void> {
  const url = rootfsSumsUrl();
  const expected = await runner.run(['curl.exe', '-fsSL', '--retry', '3', url], { timeoutMs: DOWNLOAD_TIMEOUT_MS });
  if (expected.failure !== null || expected.code !== 0) {
    throw new BootstrapRefusal(
      'runtime_download_failed',
      `Canonical's ${url} would not download (${expected.failure ?? `curl exit ${expected.code}`}), so the image cannot be verified.`,
      { detail: (expected.stderr || expected.stdout).trim() },
    );
  }
  const row = expected.stdout.split(/\r?\n/).find((line) => line.trim().endsWith(asset));
  const want = (row?.trim().split(/\s+/)[0] ?? '').toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(want)) {
    throw new BootstrapRefusal(
      'runtime_sha_mismatch',
      `${url} names no sha256 for ${asset}. Canonical republishes ${UBUNTU_WSL_SERIES}'s point releases under `
        + `current/, so an image that is not in its own sums file means the file this installer asks for has `
        + `been renamed there: ${JSON.stringify(expected.stdout.trim().slice(0, 120))}`,
    );
  }
  const got = await runner.run(['certutil', '-hashfile', rootfs, 'SHA256'], { timeoutMs: DOWNLOAD_TIMEOUT_MS });
  const hex = /^[0-9a-f ]{64,}$/im.exec(got.stdout.replace(/\r/g, ''))?.[0]?.replace(/ /g, '').toLowerCase() ?? '';
  if (got.failure !== null || got.code !== 0 || hex.length !== 64) {
    throw new BootstrapRefusal(
      'runtime_download_failed',
      `certutil would not hash ${rootfs} (${got.failure ?? `exit ${got.code}`}): ${(got.stderr.trim() || got.stdout.trim()) || 'no output'}`,
    );
  }
  if (hex !== want) {
    throw new BootstrapRefusal(
      'runtime_sha_mismatch',
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
