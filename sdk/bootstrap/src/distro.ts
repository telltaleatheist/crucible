import { BootstrapRefusal } from './errors.js';

import type { Runner } from './runner.js';
import { parseWslList, wslArgv, wslListArgv, type WslDistro } from './wsl.js';

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
        + "Crucible's Windows tray imports ours (install() hands the move to it); or name one.",
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

/** `%LOCALAPPDATA%`/Crucible/<leaf> - from the environment, never assembled from a username. */
export function crucibleAppData(runner: Runner, leaf: string): string {
  const local = runner.env['LOCALAPPDATA'];
  if (local === undefined || local.trim() === '') {
    throw new BootstrapRefusal(
      'host_unresponsive',
      `LOCALAPPDATA is not set, so there is no per-user place for Crucible's ${leaf} on this machine.`,
    );
  }
  return `${local}\\Crucible\\${leaf}`;
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
