/**
 * Crossing from Windows into the guest — the facts, each with a test.
 *
 * docs/FROM-FOUNDRY-WSL-VLLM.md section 2, carried verbatim:
 *
 * - **Always `wsl.exe -d <distro> --exec …`, never the implicit shell.** Without
 *   `--exec`, wsl.exe joins the arguments and runs them through the distro's
 *   default shell, which pre-expands `$var` and `$(...)` before bash ever sees
 *   the script — `f=hi; echo $f` prints an empty line. `--` does NOT help; only
 *   `--exec` means "no shell" (memory `wsl-exe-implicit-shell-trap`).
 *
 * - **wsl.exe halves backslashes once before bash exists**, deterministic and
 *   quote-blind. Anything carrying a backslash is doubled here, in
 *   {@link wslArgv}, so what arrives is what was meant. Nothing this package
 *   sends carries one today (guest paths are forward-slash, the token is
 *   urlsafe base64); the rule is applied anyway, because the first caller that
 *   sends one will not be thinking about it.
 *
 * - **`toWslPath` maps `C:\a\b` → `/mnt/c/a/b` and refuses UNC.** A `\\server\share`
 *   path has no `/mnt` mapping and is refused rather than mangled into something
 *   that resolves to nothing.
 *
 * - **`realpath.native` catches MAPPED drives WSL2 does not automount.** `Z:\x`
 *   passes every string test on its way to a `/mnt/z` that does not exist; only
 *   the filesystem can tell `Z:` (a share wearing a letter) from `C:`
 *   (memory `wsl-cannot-see-network-drives`). {@link guestPathFor} asks it.
 *
 * - **Prebuilt environments are unpacked by the DISTRO's own tar, never through
 *   `\\wsl$`**, whose 9P redirector flattens the symlink thicket a Python
 *   install is. The archive is downloaded on the Windows side, handed across
 *   as `/mnt/c/…`, and {@link guestUnpackArgv} is the one spelling of that.
 */
import { BootstrapRefusal } from './errors.js';
import type { Runner } from './runner.js';

export const WSL_EXE = 'wsl.exe';

/** `wsl.exe -d <distro> --exec <argv…>`, backslashes doubled for the transport. */
export function wslArgv(distro: string, argv: readonly string[]): string[] {
  if (argv.length === 0) throw new Error('wslArgv: nothing to exec');
  return [WSL_EXE, '-d', distro, '--exec', ...argv.map((arg) => arg.replace(/\\/g, '\\\\'))];
}

/** `wsl.exe -l -v`: every distribution, its state, its version, and which is default. */
export function wslListArgv(): string[] {
  return [WSL_EXE, '-l', '-v'];
}

export interface WslDistro {
  name: string;
  /** 1 or 2. Only 2 has the GPU passthrough a Crucible needs. */
  version: number;
  default: boolean;
  /** `Running` or `Stopped`, as wsl.exe spelled it. */
  state: string;
}

/**
 * Parse `wsl.exe -l -v`'s table (already decoded from UTF-16). The header row
 * is skipped by content, not position, because some builds print a blank line
 * first. A row is `[*] NAME STATE VERSION`.
 */
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

/**
 * A Windows path as the distro sees it: `C:\a\b` → `/mnt/c/a/b`. Pure — the
 * mapped-drive question is {@link networkPathBehind}'s, and asking it here would
 * make the conversion untestable without a disk.
 */
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

/**
 * The UNC path a Windows path really lives on, or null if it is on a local disk.
 * A path that cannot be resolved answers null: "I could not tell" must not read
 * as "it is a network drive".
 */
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

/**
 * One argument, safe inside a `bash -c` string: single quotes, with the only
 * character they cannot contain spliced back in the standard way. Guards
 * word-splitting in the guest; the transport's backslash-halving is
 * {@link wslArgv}'s.
 */
export function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

/**
 * Unpack a prebuilt environment archive INSIDE the distro, with its own tar.
 * The archive sits on a local Windows drive (already downloaded and sha256-
 * verified by the host — a null digest is a refusal there, not here) and is
 * read through `/mnt/<drive>`. `guestDest` is a guest path and is created.
 */
export function guestUnpackArgv(runner: Runner, distro: string, windowsArchive: string, guestDest: string): string[] {
  const archive = guestPathFor(runner, windowsArchive);
  return wslArgv(distro, ['bash', '-c', `mkdir -p ${shellQuote(guestDest)} && exec tar -xzf ${shellQuote(archive)} -C ${shellQuote(guestDest)}`]);
}
