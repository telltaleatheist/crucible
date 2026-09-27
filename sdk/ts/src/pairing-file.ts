import { CrucibleError, CruciblePairingFileError } from './errors.js';
import { loadNodeBuiltins, requireFunctions } from './node-builtins.js';
import { parsePairing, type Pairing } from './pairing.js';

/** The environment variable a server's home is overridden with. */
export const CRUCIBLE_HOME_ENV = 'CRUCIBLE_HOME';

/** The file's name inside that home. */
export const PAIRING_FILE = 'pairing';

/** The Windows home's directory name under `%LOCALAPPDATA%`. */
export const WINDOWS_HOME_DIRNAME = 'Crucible';

interface NodeApis {
  readFile(path: string, encoding: string): Promise<string>;
  join(...parts: string[]): string;
  homedir(): string;
}

async function loadNodeApis(): Promise<NodeApis> {
  const node = await loadNodeBuiltins(
    'readPairingFile needs node:fs/promises, node:path and node:os, and ' +
      'this runtime has none of them. The pairing file is a file on the ' +
      'machine a server runs on; in a browser there is no such machine, ' +
      'and a pasted pairing line (parsePairing) is the door instead.',
  );
  const missing = (name: string): string =>
    `this runtime's node builtins have no ${name}(); readPairingFile ` +
    'cannot read a file without it';
  requireFunctions(node.fs, ['readFile'], missing);
  requireFunctions(node.path, ['join'], missing);
  requireFunctions(node.os, ['homedir'], missing);
  return {
    readFile: node.fs['readFile'] as NodeApis['readFile'],
    join: node.path['join'] as NodeApis['join'],
    homedir: node.os['homedir'] as NodeApis['homedir'],
  };
}

/**
 * Where a server on this machine keeps its pairing line, by the server's own `crucible_home()`
 * rule.
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

/** The local server's pairing, or `null` because there is no file. */
export async function readPairingFile(home?: string): Promise<Pairing | null> {
  const node = await loadNodeApis();
  const path = await cruciblePairingPath(home);
  let text: string;
  try {
    text = await node.readFile(path, 'utf8');
  } catch (cause) {
    const code = (cause as { code?: string }).code;
    if (code === 'ENOENT' || code === 'ENOTDIR') return null;
    throw cause;
  }
  const lines = text
    .split(/\r?\n/)
    .map((each) => each.trim())
    .filter((each) => each !== '');
  if (lines.length === 0) {
    throw new CruciblePairingFileError(
      path,
      'is empty. A pairing file holds one line; an empty one means something ' +
        'wrote it and did not finish. Delete it and re-run `crucible token ' +
        '--url`, or restart the server, which rewrites it.',
    );
  }
  if (lines.length > 1) {
    throw new CruciblePairingFileError(
      path,
      `holds ${lines.length} lines and a pairing file holds exactly one. ` +
        'Something other than `crucible init` has written to it; delete it ' +
        'and re-run `crucible token --url`.',
    );
  }
  return parsePairing(lines[0] as string);
}
