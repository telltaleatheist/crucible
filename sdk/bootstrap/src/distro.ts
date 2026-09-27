import { BootstrapRefusal } from './errors.js';

import type { RunResult, Runner, StreamOptions } from './runner.js';
import { parseWslList, wslArgv, wslListArgv, wslRootArgv, type WslDistro } from './wsl.js';

/** The distro Crucible owns. */
export const CRUCIBLE_DISTRO = 'crucible';

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

export const WSL_OUTCOME_NAME = 'wsl-outcome.json';

export const WSL_CONF_MARKER = '# crucible-rootfs';

/** `/etc/wsl.conf`, exactly as an import and the repair write it. */
export const WSL_CONF_TEXT = `${WSL_CONF_MARKER}\n[boot]\nsystemd=true\n[user]\ndefault=crucible\n`;

const LIST_TIMEOUT_MS = 15_000;
const PROBE_TIMEOUT_MS = 60_000;
const IMPORT_TIMEOUT_MS = 30 * 60_000;
const DOWNLOAD_TIMEOUT_MS = 60 * 60_000;

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

export interface DistroChoiceOptions {
  /** The app's own WSL distro setting, when it has one. */
  distro?: string;
  /** Use `distro` verbatim and resolve nothing. */
  exact?: boolean;
}

/** Which WSL distro is `local`: `exact`, else the `crucible` distro, else the app's setting. */
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

async function hasConfig(runner: Runner, distro: string): Promise<boolean> {
  const result = await runner.run(
    wslArgv(distro, ['bash', '-c', 'test -f "${CRUCIBLE_HOME:-$HOME/.crucible}/config.toml"']),
    { timeoutMs: PROBE_TIMEOUT_MS },
  );
  if (result.failure !== null) {
    throw new BootstrapRefusal(
      'wsl_read_failed',
      `could not ask "${distro}" whether it holds a Crucible config: ${result.failure}`,
      { detail: (result.stderr || result.stdout).trim() },
    );
  }
  return result.code === 0;
}

export interface EnsureDistroOptions {
  /** The Crucible release this install is for, named in its refusals and output. */
  release: string;
  /** Where the distro's ext4 file goes. */
  installDir?: string;
  /** Where the rootfs is downloaded to. */
  downloadDir?: string;
  /** A rootfs URL to import instead of Canonical's image; the caller vouches for its bytes. */
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
  /** Whether this call wrote `/etc/wsl.conf`. */
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

/** `wsl --terminate crucible`. */
export function terminateArgv(distro: string = CRUCIBLE_DISTRO): string[] {
  return ['wsl.exe', '--terminate', distro];
}

/** `wsl --unregister crucible`. */
export function unregisterArgv(distro: string = CRUCIBLE_DISTRO): string[] {
  return ['wsl.exe', '--unregister', distro];
}

/** Ensure the `crucible` distro exists, runs systemd, and carries Crucible's marker. */
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

  await finishImport(runner, CRUCIBLE_DISTRO);

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

export function finishImportScript(): string {
  return [
    "id -u crucible >/dev/null 2>&1 || useradd --create-home --shell /bin/bash crucible",
    'passwd --delete crucible >/dev/null',
    "printf 'crucible ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/crucible",
    'chmod 0440 /etc/sudoers.d/crucible',
    'mkdir -p /etc/cloud && touch /etc/cloud/cloud-init.disabled',
    "mkdir -p /usr/lib/binfmt.d && printf ':WSLInterop:M::MZ::/init:PF\\n' > /usr/lib/binfmt.d/WSLInterop.conf",
    `cat > /etc/wsl.conf <<'EOF'\n${WSL_CONF_TEXT}EOF`,
  ].join('\n');
}

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

/** Write `/etc/wsl.conf` as root. */
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
