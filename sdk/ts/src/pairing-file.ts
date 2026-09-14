/**
 * `readPairingFile` — the pairing line a server on THIS machine left for us.
 *
 * PHASE15-HOST.md sections 3.6 and 5.1. `crucible init` and
 * `crucible service install` write one loopback pairing line to
 * `<CRUCIBLE_HOME>/pairing` at mode 0600, and an app's connect door reads it
 * FIRST — before offering to take a pasted line, before offering to install
 * one. Nobody types a token for a server running on their own computer.
 *
 * NULL IS A FACT HERE, NOT A FALLBACK. There being no file is the normal state
 * of a machine with no local Crucible, and the caller's next line is "paste
 * one" or "install one" — not an error dialog. What is NOT null-ed is a file
 * that exists and is malformed: that throws `CruciblePairingError`, because a
 * line somebody's installer wrote badly is a broken install, and answering
 * "there is no server here" would send the user to install a second one.
 *
 * WHY IT IS ASYNC, AND WHY THE IMPORTS ARE ASSEMBLED AT RUN TIME. This is the
 * second place in the package that touches the filesystem, and it obeys the
 * same rule `writeArtifactsTo` does (README, "the one place this touches
 * node:fs"): a STATIC `import … from 'node:fs'` would put fs into the module
 * graph of `import {CrucibleClient}` itself, and a bundler targeting a
 * browser-ish runtime resolves that at build time and fails on it. Building
 * the specifier at run time keeps it out of static analysis, and a dynamic
 * import is a promise — so this function is async, and a connect door that
 * was going to await a server anyway pays nothing for it.
 */

import { CrucibleError, CruciblePairingFileError } from './errors.js';
import { parsePairing, type Pairing } from './pairing.js';

/** The environment variable a server's home is overridden with. */
export const CRUCIBLE_HOME_ENV = 'CRUCIBLE_HOME';

/** The file's name inside that home. One line, mode 0600. */
export const PAIRING_FILE = 'pairing';

/**
 * The directory `%LOCALAPPDATA%\Crucible` — the Windows home, whose name is
 * `crucible/config.py`'s `WINDOWS_HOME_DIRNAME` and `crucible/host/paths.py`'s
 * `APPDATA_DIRNAME`. One word on both sides of the language boundary.
 */
export const WINDOWS_HOME_DIRNAME = 'Crucible';

/** The three calls this module makes, proven to exist before it makes them. */
interface NodeApis {
  readFile(path: string, encoding: string): Promise<string>;
  join(...parts: string[]): string;
  homedir(): string;
}

let apis: Promise<NodeApis> | null = null;

function loadNodeApis(): Promise<NodeApis> {
  if (apis === null) {
    apis = (async () => {
      // Assembled rather than written, for the reason in the module comment.
      const scheme = 'node:';
      let fs: Record<string, unknown>;
      let path: Record<string, unknown>;
      let os: Record<string, unknown>;
      try {
        fs = (await import(/* webpackIgnore: true */ `${scheme}fs/promises`)) as never;
        path = (await import(/* webpackIgnore: true */ `${scheme}path`)) as never;
        os = (await import(/* webpackIgnore: true */ `${scheme}os`)) as never;
      } catch (cause) {
        throw new CrucibleError(
          'readPairingFile needs node:fs/promises, node:path and node:os, and ' +
            'this runtime has none of them. The pairing file is a file on the ' +
            'machine a server runs on; in a browser there is no such machine, ' +
            'and a pasted pairing line (parsePairing) is the door instead.',
          { cause },
        );
      }
      for (const [name, found] of [
        ['readFile', fs['readFile']],
        ['join', path['join']],
        ['homedir', os['homedir']],
      ] as const) {
        if (typeof found !== 'function') {
          throw new CrucibleError(
            `this runtime's node builtins have no ${name}(); readPairingFile ` +
              'cannot read a file without it',
          );
        }
      }
      return {
        readFile: fs['readFile'] as NodeApis['readFile'],
        join: path['join'] as NodeApis['join'],
        homedir: os['homedir'] as NodeApis['homedir'],
      };
    })();
  }
  return apis;
}

/**
 * Where a server on this machine keeps its pairing line, by the SAME rule the
 * server itself uses — `crucible/config.py`'s `crucible_home()`, one function
 * re-implemented here in the one language that cannot call it:
 *
 * 1. `$CRUCIBLE_HOME`, when it is set and non-empty. Every platform.
 * 2. **win32:** `%LOCALAPPDATA%\Crucible\pairing` (PHASE15-HOST.md 3.6's
 *    table). NOT `~/.crucible`: on Windows the server is the WSL guest's or
 *    the host-mode child's, and the thing that writes a WINDOWS-side pairing
 *    file is `crucible host`, whose per-machine root is that same
 *    directory — `wsl`, `downloads` and `host` are already under it.
 *    `LOCALAPPDATA` is READ and never assembled from a username, and unset is
 *    REFUSED rather than guessed: a Windows session without it is broken in a
 *    way that would make every path here wrong, and `~/.crucible` would be a
 *    directory nothing writes to, so an app would report "no engine" about a
 *    machine that is running one.
 * 3. **linux/darwin:** `~/.crucible/pairing`.
 *
 * Exported because an app that can say WHERE it looked is more useful than one
 * that only says "not found".
 */
export async function cruciblePairingPath(home?: string): Promise<string> {
  const node = await loadNodeApis();
  if (home !== undefined && home !== '') return node.join(home, PAIRING_FILE);
  const override = process.env[CRUCIBLE_HOME_ENV];
  if (override !== undefined && override !== '') {
    return node.join(override, PAIRING_FILE);
  }
  if (process.platform === 'win32') {
    const local = process.env['LOCALAPPDATA'];
    if (local === undefined || local === '') {
      throw new CrucibleError(
        '%LOCALAPPDATA% is not set, so this Windows session cannot say where ' +
          `Crucible's home is. Set ${CRUCIBLE_HOME_ENV} to the directory the ` +
          'server was initialised with. It is never assembled from a username: ' +
          'a roaming profile or a redirected AppData would make the guess wrong ' +
          'and the answer ("no engine on this machine") a lie.',
      );
    }
    return node.join(local, WINDOWS_HOME_DIRNAME, PAIRING_FILE);
  }
  return node.join(node.homedir(), '.crucible', PAIRING_FILE);
}

/**
 * The local server's pairing, or `null` because there is no file.
 *
 * ```ts
 * const local = await readPairingFile();
 * if (local !== null) {
 *   const crucible = new CrucibleClient({ ...local, clientName: 'bookforge' });
 * }
 * ```
 */
export async function readPairingFile(home?: string): Promise<Pairing | null> {
  const node = await loadNodeApis();
  const path = await cruciblePairingPath(home);
  let text: string;
  try {
    text = await node.readFile(path, 'utf8');
  } catch (cause) {
    const code = (cause as { code?: string }).code;
    // ENOENT and ENOTDIR are "there is no server here". Anything else — a
    // permission error, a directory where the file should be — is raised: a
    // caller told `null` would offer to install a second Crucible over the
    // top of one that is already running.
    if (code === 'ENOENT' || code === 'ENOTDIR') return null;
    throw cause;
  }
  const lines = text
    .split(/\r?\n/)
    .map((each) => each.trim())
    .filter((each) => each !== '');
  if (lines.length === 0) {
    // AN EMPTY FILE IS A DEFECT, NOT AN ABSENCE. `null` here means "there is
    // no server on this machine", which sends an app's connect door to
    // "install one" — over a server that is running and whose installer
    // truncated its own pairing file. The absent case is the file not being
    // there, and nothing else.
    throw new CruciblePairingFileError(
      path,
      'is empty. A pairing file holds one line; an empty one means something ' +
        'wrote it and did not finish. Delete it and re-run `crucible token ' +
        '--url`, or restart the server, which rewrites it.',
    );
  }
  if (lines.length > 1) {
    // The WRITER enforces one line (`crucible/pairing.py`), so a second one
    // means something else has been appending to this file — a shell
    // redirect, a second installer, an editor. Which line is the server's is
    // then a guess, and a guessed bearer token is a connect door that fails
    // with an auth error nobody can explain. Named, like every other refusal.
    throw new CruciblePairingFileError(
      path,
      `holds ${lines.length} lines and a pairing file holds exactly one. ` +
        'Something other than `crucible init` has written to it; delete it ' +
        'and re-run `crucible token --url`.',
    );
  }
  return parsePairing(lines[0] as string);
}
