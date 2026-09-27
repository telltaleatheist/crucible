import { BootstrapRefusal } from './errors.js';
import type { Runner } from './runner.js';

export const WSL_EXE = 'wsl.exe';

function forTheTransport(argv: readonly string[]): string[] {
  return argv.map((arg) => arg.replace(/\\/g, '\\\\'));
}

/** `wsl.exe -d <distro> --exec <argv…>`, backslashes doubled for the transport. */
export function wslArgv(distro: string, argv: readonly string[]): string[] {
  if (argv.length === 0) throw new Error('wslArgv: nothing to exec');
  return [WSL_EXE, '-d', distro, '--exec', ...forTheTransport(argv)];
}

/** `wsl.exe -d <distro> -u root --exec <argv…>`: the guest entered as root, with no elevation. */
export function wslRootArgv(distro: string, argv: readonly string[]): string[] {
  if (argv.length === 0) throw new Error('wslRootArgv: nothing to exec');
  return [WSL_EXE, '-d', distro, '-u', 'root', '--exec', ...forTheTransport(argv)];
}

/** `wsl.exe -l -v`: every distribution, its state, its version, and which is default. */
export function wslListArgv(): string[] {
  return [WSL_EXE, '-l', '-v'];
}

export interface WslDistro {
  name: string;
  /** WSL version, 1 or 2. */
  version: number;
  default: boolean;
  /** `Running` or `Stopped`, as wsl.exe spelled it. */
  state: string;
}

/** Parse `wsl.exe -l -v`'s table, already decoded from UTF-16. */
export function parseWslList(text: string): { distros: WslDistro[]; default: string | null } {
  const distros: WslDistro[] = [];
  let fallbackDefault: string | null = null;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (line.length === 0) continue;
    if (/^NAME\s+STATE\s+VERSION$/i.test(line)) continue;
    const match = /^(\*?)\s*(\S+)\s+(\S+)\s+(\d+)$/.exec(line);
    if (match === null) continue;
    const isDefault = match[1] === '*';
    const entry: WslDistro = {
      name: match[2] as string,
      state: match[3] as string,
      version: Number(match[4]),
      default: isDefault,
    };
    distros.push(entry);
    if (isDefault) fallbackDefault = entry.name;
  }
  return { distros, default: fallbackDefault };
}

/** A Windows path as the distro sees it: `C:\a\b` → `/mnt/c/a/b`. */
export function toWslPath(windowsPath: string): string {
  const normalised = windowsPath.replace(/\\/g, '/');
  if (normalised.startsWith('//')) {
    throw new BootstrapRefusal(
      'network_path',
      `${windowsPath} is a network path, which has no /mnt mapping inside WSL2. Put it on a local drive.`,
    );
  }
  const drive = /^([A-Za-z]):\//.exec(normalised);
  if (drive === null || drive[1] === undefined) {
    throw new BootstrapRefusal(
      'not_a_windows_path',
      `${windowsPath} is not an absolute Windows path, so it has no WSL spelling.`,
    );
  }
  return `/mnt/${drive[1].toLowerCase()}${normalised.slice(2)}`;
}

/** The UNC path a Windows path really lives on, or null if it is on a local disk. */
export function networkPathBehind(runner: Runner, windowsPath: string): string | null {
  let resolved: string;
  try {
    resolved = runner.realpathNative(windowsPath);
  } catch {
    return null;
  }
  return resolved.replace(/\\/g, '/').startsWith('//') ? resolved : null;
}

/** {@link toWslPath}, after asking the filesystem whether the drive is really a share. */
export function guestPathFor(runner: Runner, windowsPath: string): string {
  const share = networkPathBehind(runner, windowsPath);
  if (share !== null) {
    throw new BootstrapRefusal(
      'network_path',
      `${windowsPath} is on ${share}, a mapped network drive. WSL2 auto-mounts fixed drives only, so the `
        + 'guest would be handed a path that does not exist there. Put it on a local drive.',
    );
  }
  return toWslPath(windowsPath);
}

/** Quote one argument for a `bash -c` string. */
export function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}
