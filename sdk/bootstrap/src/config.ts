/**
 * The local Crucible server — read from its OWN config, never copied.
 *
 * This is the rule BookForge's `electron/crucible/local.ts` states, moved to
 * the package that ships with the server so it has one owner (ARCHITECTURE.md
 * R1); BookForge imports it from here. The rule:
 *
 * The local server has ONE owner: `<CRUCIBLE_HOME>/config.toml`, the file the
 * server itself reads (`crucible/config.py`). This reads that file and nothing
 * else. `[server] name, host, port` and `[auth] token`.
 *
 * The connect address is DERIVED from the bind address: a server bound to
 * `0.0.0.0` or `::` is reached at `127.0.0.1`; one bound to a specific address
 * is reached there. That is a bind→connect mapping, not a fallback — the two
 * are different facts and the file only records the first.
 *
 * On macOS and Linux: `$CRUCIBLE_HOME/config.toml`, default `~/.crucible`,
 * exactly as `crucible_home()` resolves it. On Windows the local server lives
 * in WSL2, so the file is read THROUGH `wsl.exe -d <distro> --exec bash -c …`,
 * which resolves `$CRUCIBLE_HOME` / `$HOME` inside the guest. Always `--exec`.
 *
 * A machine may legitimately have no local Crucible. That is `no_local_config`,
 * a NAMED state the caller shows, not an empty client. Every other way this can
 * fail is its own code.
 */
import { posix as path } from 'node:path';

import { resolveDistro } from './distro.js';
import { BootstrapRefusal } from './errors.js';
import { processRunner, type Runner } from './runner.js';
import { describeTarget, resolveTarget, runOn, type Target } from './target.js';
import { parseToml, TomlError, type TomlTable } from './toml.js';
import { shellQuote } from './wsl.js';

/** The reserved server name that means "the server on this machine". */
export const LOCAL_SERVER_NAME = 'local';

/** How `crucible_home()` names its override, verbatim. */
export const CRUCIBLE_HOME_ENV = 'CRUCIBLE_HOME';

/** How long the guest gets to `cat` one small file. */
const CONFIG_READ_TIMEOUT_MS = 30_000;

/** The local server, as its own config describes it. */
export interface LocalConfig {
  /** `[server] name` — what the server calls itself, e.g. `crucible@owens-pc-wsl`. */
  name: string;
  /** Base URL to connect to, without `/v1`. Derived from `[server] host`/`port`. */
  url: string;
  /** `[auth] token`, verbatim. Never log it. */
  token: string;
  /** The file this was read from, as the reading side names it. */
  configPath: string;
  /** `file` — read directly; `wsl` — read through `wsl.exe -d <distro>`. */
  via: 'file' | 'wsl';
}

export interface LocalConfigOptions {
  /**
   * win32: the app's own WSL distro setting. **The `crucible` distro wins over
   * it when one exists** (PHASE14-ENVPACKS.md 4b): that distro is Crucible's
   * own, and an app that imported it is not then supposed to read a server out
   * of somebody else's guest. A machine with both a `crucible` distro and a
   * config inside this one is `two_local_crucibles`, refused by name.
   * Ignored off win32.
   */
  distro?: string;
  /** win32: use `distro` verbatim and resolve nothing — the way out of `two_local_crucibles`. */
  exact?: boolean;
  /**
   * `CRUCIBLE_HOME`, as the target spells it (a guest path on win32). Omit to
   * resolve it the way the server does: `$CRUCIBLE_HOME`, else `~/.crucible`.
   */
  home?: string;
}

/**
 * `$CRUCIBLE_HOME/config.toml`, resolved exactly as `crucible_home()` does.
 * POSIX joins: the file is only ever read natively on darwin and linux; on win32
 * it lives in the guest and is read through `wsl.exe`.
 */
export function localConfigPath(env: NodeJS.ProcessEnv, homedir: string, home?: string): string {
  if (home !== undefined) return path.join(home, 'config.toml');
  const override = env[CRUCIBLE_HOME_ENV];
  const root = override !== undefined && override !== '' ? override : path.join(homedir, '.crucible');
  return path.join(root, 'config.toml');
}

/**
 * The address a client connects to, from the address the server bound.
 * `0.0.0.0`/`::` mean "every interface", and the interface a local client uses
 * is loopback. Anything else is a specific address and is used as written.
 */
export function connectHost(bindHost: string): string {
  if (bindHost === '0.0.0.0' || bindHost === '::' || bindHost === '') return '127.0.0.1';
  return bindHost;
}

/** What to run when the file is there and wrong. Mints a NEW token; every client needs it. */
const REINIT = 'crucible init --force';

/**
 * Parse a config.toml into a {@link LocalConfig}. Requires the same keys
 * `crucible/config.py`'s `load_config` requires, and refuses by name when one
 * is missing or of the wrong type — the server would refuse to start on that
 * file, so a client must not pretend it describes a server.
 */
export function parseLocalConfig(text: string, configPath: string, via: 'file' | 'wsl'): LocalConfig {
  let table: TomlTable;
  try {
    table = parseToml(text);
  } catch (err) {
    const reason = err instanceof TomlError ? err.message : (err as Error).message;
    throw new BootstrapRefusal(
      'config_unreadable',
      `${configPath} is not TOML this reader accepts (${reason}). The server would refuse it too.`,
      { command: REINIT, cause: err },
    );
  }
  const name = requireKey(table, 'server', 'name', 'string', configPath) as string;
  const host = requireKey(table, 'server', 'host', 'string', configPath) as string;
  const port = requireKey(table, 'server', 'port', 'integer', configPath) as number;
  const token = requireKey(table, 'auth', 'token', 'string', configPath) as string;
  if (token.trim() === '') {
    throw new BootstrapRefusal(
      'config_missing_key',
      `${configPath}: auth.token is empty. A Crucible has no anonymous mode.`,
      { command: REINIT },
    );
  }
  const urlHost = connectHost(host);
  const url = `http://${urlHost.includes(':') ? `[${urlHost}]` : urlHost}:${port}`;
  return { name, url, token, configPath, via };
}

function requireKey(
  table: TomlTable,
  section: string,
  key: string,
  kind: 'string' | 'integer',
  configPath: string,
): unknown {
  const block = table[section];
  if (block === undefined || typeof block !== 'object' || Array.isArray(block)) {
    throw new BootstrapRefusal('config_missing_key', `${configPath} is missing the [${section}] section`, { command: REINIT });
  }
  const value = block[key];
  if (value === undefined) {
    throw new BootstrapRefusal('config_missing_key', `${configPath} is missing ${section}.${key}`, { command: REINIT });
  }
  const ok = kind === 'string' ? typeof value === 'string' : typeof value === 'number' && Number.isInteger(value);
  if (!ok) {
    throw new BootstrapRefusal(
      'config_missing_key',
      `${configPath}: ${section}.${key} must be ${kind === 'string' ? 'a string' : 'an integer'}, got ${typeof value}`,
      { command: REINIT },
    );
  }
  return value;
}

/**
 * The one guest-side script. Exit 3 is "no config there", told apart from every
 * other failure. Carries no backslash and no host-side `$`: with `--exec` the
 * guest's bash is the first thing to see it.
 */
export function guestReadScript(home: string | undefined): string {
  const where = home === undefined
    ? 'p="${CRUCIBLE_HOME:-$HOME/.crucible}/config.toml"; '
    : `p=${shellQuote(home)}/config.toml; `;
  return `${where}if [ ! -f "$p" ]; then echo "$p" >&2; exit 3; fi; echo "$p"; cat "$p"`;
}

/**
 * The local server, from its own config, or a {@link BootstrapRefusal} with one
 * of `no_wsl_distro`, `no_local_config`, `wsl_read_failed`, `config_unreadable`,
 * `config_missing_key`.
 */
export async function readLocalConfig(options: LocalConfigOptions = {}, runner: Runner = processRunner()): Promise<LocalConfig> {
  const distro = runner.platform === 'win32'
    ? await resolveDistro(runner, {
      ...(options.distro === undefined ? {} : { distro: options.distro }),
      ...(options.exact === undefined ? {} : { exact: options.exact }),
    })
    : options.distro;
  const target = resolveTarget(runner, distro);
  if (target.kind === 'wsl') return readThroughWsl(runner, target, options.home);

  const configPath = localConfigPath(runner.env, runner.homedir, options.home);
  if (!runner.fileExists(configPath)) {
    throw new BootstrapRefusal(
      'no_local_config',
      `no local Crucible: ${configPath} does not exist. install() puts one there; or add a remote server.`,
    );
  }
  let text: string;
  try {
    text = runner.readFile(configPath);
  } catch (err) {
    throw new BootstrapRefusal('config_unreadable', `could not read ${configPath}: ${(err as Error).message}`, { cause: err });
  }
  return parseLocalConfig(text, configPath, 'file');
}

async function readThroughWsl(runner: Runner, target: Target & { kind: 'wsl' }, home: string | undefined): Promise<LocalConfig> {
  const result = await runOn(runner, target, ['bash', '-c', guestReadScript(home)], { timeoutMs: CONFIG_READ_TIMEOUT_MS });
  if (result.failure !== null) {
    throw new BootstrapRefusal(
      'wsl_read_failed',
      `could not run wsl.exe -d ${target.distro}: ${result.failure}`,
      { detail: (result.stderr || result.stdout).trim() },
    );
  }
  if (result.code === 3) {
    const missing = result.stderr.trim();
    throw new BootstrapRefusal(
      'no_local_config',
      `no local Crucible: ${missing} does not exist inside ${describeTarget(target)}. install() puts one there; or add a remote server.`,
    );
  }
  if (result.code !== 0) {
    throw new BootstrapRefusal(
      'wsl_read_failed',
      `reading the local Crucible config inside ${describeTarget(target)} failed (exit ${result.code}): `
        + `${result.stderr.trim() || '(no stderr)'}`,
    );
  }
  const newline = result.stdout.indexOf('\n');
  if (newline < 0) {
    throw new BootstrapRefusal(
      'wsl_read_failed',
      `the guest printed no config path before the file (stdout: ${JSON.stringify(result.stdout.slice(0, 80))})`,
    );
  }
  const guestPath = result.stdout.slice(0, newline).trim();
  const text = result.stdout.slice(newline + 1);
  return parseLocalConfig(text, `${target.distro}:${guestPath}`, 'wsl');
}
