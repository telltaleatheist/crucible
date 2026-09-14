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

import { CrucibleError } from './errors.js';
import { parsePairing, type Pairing } from './pairing.js';

/** The environment variable a server's home is overridden with. */
export const CRUCIBLE_HOME_ENV = 'CRUCIBLE_HOME';

/** The file's name inside that home. One line, mode 0600. */
export const PAIRING_FILE = 'pairing';

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
 * Where a server on this machine keeps its pairing line, by the same rule the
 * server itself uses: `$CRUCIBLE_HOME`, else `~/.crucible`.
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
  const line = text.trim();
  if (line === '') return null;
  return parsePairing(line);
}
