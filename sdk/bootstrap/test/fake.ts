/**
 * A scripted {@link Runner}: the tests say which argv they expect, in order, and
 * what each answers. An unexpected call fails the test naming the argv, so the
 * suite proves the exact commands that would have run on a machine — the same
 * discipline `crucible/tests/test_service.py` applies to systemctl and launchctl.
 */
import assert from 'node:assert/strict';

import type { OutputStream, RunOptions, RunResult, Runner, StreamOptions } from '../src/index.js';

export interface Expectation {
  /** The exact argv, or a predicate over it. */
  argv: readonly string[] | ((argv: readonly string[]) => boolean);
  /** The env the call must carry (checked when given). */
  env?: Readonly<Record<string, string>>;
  code?: number | null;
  stdout?: string;
  stderr?: string;
  failure?: string;
  /** For `stream`: the lines to feed `onLine`, in order. */
  lines?: readonly [string, OutputStream][];
}

export interface Call {
  argv: readonly string[];
  env: Readonly<Record<string, string>> | undefined;
  timeoutMs: number;
  streamed: boolean;
}

export interface FakeHost {
  platform?: NodeJS.Platform;
  env?: NodeJS.ProcessEnv;
  homedir?: string;
  /** Files that exist, with their content. */
  files?: Record<string, string>;
  /** What `realpathNative` answers; a path that is a file answers itself. */
  realpaths?: Record<string, string>;
  /** Files that exist but cannot be read. */
  unreadable?: readonly string[];
}

export class FakeRunner implements Runner {
  readonly platform: NodeJS.Platform;
  readonly env: NodeJS.ProcessEnv;
  readonly homedir: string;
  readonly calls: Call[] = [];
  private readonly files: Record<string, string>;
  private readonly realpaths: Record<string, string>;
  private readonly unreadable: readonly string[];
  private readonly queue: Expectation[];

  constructor(host: FakeHost, expectations: readonly Expectation[]) {
    this.platform = host.platform ?? 'win32';
    this.env = host.env ?? {};
    this.homedir = host.homedir ?? (this.platform === 'win32' ? 'C:\\Users\\owen' : '/home/owen');
    this.files = host.files ?? {};
    this.realpaths = host.realpaths ?? {};
    this.unreadable = host.unreadable ?? [];
    this.queue = [...expectations];
  }

  /** Every expectation was consumed. */
  assertDrained(): void {
    assert.equal(this.queue.length, 0, `unconsumed expectations: ${JSON.stringify(this.queue.map((e) => e.argv))}`);
  }

  private next(argv: readonly string[], options: RunOptions, streamed: boolean): { expectation: Expectation; result: RunResult } {
    assert.ok(Number.isFinite(options.timeoutMs) && options.timeoutMs > 0, `every call needs a timeout: ${argv.join(' ')}`);
    const expectation = this.queue.shift();
    assert.ok(expectation !== undefined, `unexpected call: ${JSON.stringify(argv)}`);
    if (typeof expectation.argv === 'function') {
      assert.ok(expectation.argv(argv), `argv did not match predicate: ${JSON.stringify(argv)}`);
    } else {
      assert.deepEqual([...argv], [...expectation.argv]);
    }
    if (expectation.env !== undefined) assert.deepEqual(options.env, expectation.env);
    this.calls.push({ argv, env: options.env, timeoutMs: options.timeoutMs, streamed });
    return {
      expectation,
      result: {
        code: expectation.failure !== undefined ? null : (expectation.code ?? 0),
        stdout: expectation.stdout ?? '',
        stderr: expectation.stderr ?? '',
        failure: expectation.failure ?? null,
      },
    };
  }

  async run(argv: readonly string[], options: RunOptions): Promise<RunResult> {
    return this.next(argv, options, false).result;
  }

  async stream(argv: readonly string[], options: StreamOptions): Promise<RunResult> {
    const { expectation, result } = this.next(argv, options, true);
    for (const [line, stream] of expectation.lines ?? []) options.onLine(line, stream);
    return result;
  }

  fileExists(path: string): boolean {
    return path in this.files || this.unreadable.includes(path);
  }

  readFile(path: string): string {
    if (this.unreadable.includes(path)) throw new Error(`EACCES: permission denied, open '${path}'`);
    const content = this.files[path];
    if (content === undefined) throw new Error(`ENOENT: no such file or directory, open '${path}'`);
    return content;
  }

  realpathNative(path: string): string {
    const mapped = this.realpaths[path];
    if (mapped !== undefined) return mapped;
    if (this.fileExists(path)) return path;
    throw new Error(`ENOENT: no such file or directory, lstat '${path}'`);
  }
}

/** The guest's `config.toml`, as `crucible init` wrote it on Owen's PC (token replaced). */
export const GUEST_CONFIG = `[server]
name = "crucible@owens-pc-wsl"
host = "127.0.0.1"
port = 7100

[auth]
token = "test-token-not-a-secret"

[backend]
kind = "cuda-linux"

[jobs]
enable_echo = true
enable_llm = true

[accelerator]
desktop_allowance_bytes = 3221225472

[capability]
backend_kind = "cuda-linux"
total_bytes = 25757220864
desktop_allowance_bytes = 3221225472

[[capability.classes]]
capability = "tts"
enabled = true
selected = "deathstalker"
reason = "deathstalker fits: it needs 17.7 GiB and there is 21.0 GiB available (24.0 GiB card less a 3.0 GiB desktop allowance); 7 of 7 voices fit"
shortfall_bytes = 0
`;

/** What `wsl.exe -l -v` printed on Owen's PC, decoded. */
export const WSL_LIST = '  NAME      STATE           VERSION\r\n* Ubuntu    Running         2\r\n';

/** What `wsl.exe -l -v` prints on a machine that also has the Crucible distro. */
export const WSL_LIST_WITH_CRUCIBLE = '  NAME        STATE           VERSION\r\n* Ubuntu      Running         2\r\n  crucible    Stopped         2\r\n';

/** The guest probe's answer on a host with nothing installed yet. */
export const GUEST_BARE = 'home=/home/owen/.crucible\nuser=owen\nfree_kib=400000000\n';

/** The archive sha256 the fixture manifest names. 64 hex, like a real one. */
export const PACK_SHA = 'a'.repeat(64);

/** The guest probe's answer on a host whose server pack is already the fixture's. */
export const GUEST_INSTALLED = `home=/home/owen/.crucible\nuser=owen\nfree_kib=400000000\n`
  + `crucible=/home/owen/.crucible/server/bin/crucible\nversion=crucible 0.6.0\nsha256=${PACK_SHA}\nrelease=0.6.0\n`;

/** `envpacks.json` for 0.6.0, as the release publishes it (two parts, both backends). */
export const ENVPACKS_JSON = JSON.stringify({
  schema: 1,
  version: '0.6.0',
  packs: [
    {
      name: 'server',
      backend: 'cuda-linux',
      python: '3.11.13',
      bytes: 120_000_000,
      sha256: PACK_SHA,
      parts: ['crucible-env-server-cuda-linux-0.6.0.tar.zst.part00', 'crucible-env-server-cuda-linux-0.6.0.tar.zst.part01'],
      unpacked_bytes: 400_000_000,
    },
    {
      name: 'server',
      backend: 'mlx-darwin',
      python: '3.11.13',
      bytes: 110_000_000,
      sha256: 'b'.repeat(64),
      parts: ['crucible-env-server-mlx-darwin-0.6.0.tar.zst.part00'],
      unpacked_bytes: 380_000_000,
    },
    {
      name: 'llm',
      backend: 'cuda-linux',
      python: '3.11.13',
      bytes: 8_000_000_000,
      sha256: 'c'.repeat(64),
      parts: ['crucible-env-llm-cuda-linux-0.6.0.tar.zst.part00'],
      recipe_sha256: 'd'.repeat(64),
      unpacked_bytes: 9_000_000_000,
    },
  ],
});

/** The server pack's console script, where every install puts it. */
export const CRUCIBLE_BIN = '/home/owen/.crucible/server/bin/crucible';
/** Where the fixture's parts and reassembled archive land. */
export const DOWNLOADS = '/home/owen/.crucible/downloads';
export const ARCHIVE = `${DOWNLOADS}/crucible-env-server-cuda-linux-0.6.0.tar.zst`;
export const DEST = '/home/owen/.crucible/server';

export async function refusal<T>(promise: Promise<T>): Promise<{ code: string; message: string; command: string | null; detail: string | null; error: unknown }> {
  try {
    await promise;
  } catch (err) {
    const e = err as { code?: string; message: string; command?: string | null; detail?: string | null };
    assert.equal((err as Error).name === 'BootstrapRefusal' || (err as Error).name === 'BootstrapStepFailed', true, `not a refusal: ${(err as Error).stack}`);
    return { code: e.code ?? '', message: e.message, command: e.command ?? null, detail: e.detail ?? null, error: err };
  }
  assert.fail('expected a refusal');
}
