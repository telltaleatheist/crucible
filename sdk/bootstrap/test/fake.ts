/**
 * A scripted {@link Runner}: the tests say which argv they expect, in order, and
 * what each answers. An unexpected call fails the test naming the argv, so the
 * suite proves the exact commands that would have run on a machine — the same
 * discipline `crucible/tests/test_service.py` applies to systemctl and launchctl.
 */
import assert from 'node:assert/strict';

import { interpreterFor } from '../src/interpreter.js';
import { wheelAssetName } from '../src/release.js';

import type { HostFetch, OutputStream, RunOptions, RunResult, Runner, StreamOptions } from '../src/index.js';

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
  /**
   * Files that appear once a command has RUN — an installer's own output.
   *
   * PHASE19 made `install()` run `install.ps1` on a machine with no host, and
   * the thing it then asserts is that `crucible.cmd` is there afterwards. A
   * fake whose files never change could only ever say "it was there all along"
   * or "it never appeared", and neither is the sequence under test.
   */
  appearAfter?: Record<string, string>;
}

export class FakeRunner implements Runner {
  readonly platform: NodeJS.Platform;
  readonly env: NodeJS.ProcessEnv;
  readonly homedir: string;
  readonly calls: Call[] = [];
  private readonly files: Record<string, string>;
  private readonly appearAfter: Record<string, string>;
  private readonly realpaths: Record<string, string>;
  private readonly unreadable: readonly string[];
  private readonly queue: Expectation[];

  constructor(host: FakeHost, expectations: readonly Expectation[]) {
    this.platform = host.platform ?? 'win32';
    this.env = host.env ?? {};
    this.homedir = host.homedir ?? (this.platform === 'win32' ? 'C:\\Users\\owen' : '/home/owen');
    this.files = { ...(host.files ?? {}) };
    this.appearAfter = host.appearAfter ?? {};
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
    if ((expectation.code ?? 0) === 0 && expectation.failure === undefined) {
      Object.assign(this.files, this.appearAfter);
    }
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

/**
 * The pinned interpreter for the fixture's backend, IMPORTED rather than typed.
 *
 * A fixture that spelled its own digest would be a second copy of the pin, and
 * every test below would keep passing about an interpreter this package stopped
 * installing.
 */
export const PIN = interpreterFor('cuda-linux');
export const PY_SHA = PIN.sha256;

/** The guest probe's answer on a host whose runtime is already the fixture's. */
export const GUEST_INSTALLED = `home=/home/owen/.crucible\nuser=owen\nfree_kib=400000000\n`
  + `crucible=/home/owen/.crucible/server/bin/crucible\nversion=crucible 0.6.0\n`
  + `python_sha256=${PY_SHA}\npython_version=${PIN.version}\nrelease=0.6.0\n`;

/** The server's console script, where every install puts it. */
export const CRUCIBLE_BIN = '/home/owen/.crucible/server/bin/crucible';
/** Where the interpreter archive and the wheel land while they are used. */
export const DOWNLOADS = '/home/owen/.crucible/downloads';
export const ARCHIVE = `${DOWNLOADS}/${PIN.asset}`;
export const WHEEL = `${DOWNLOADS}/${wheelAssetName('0.6.0')}`;
export const DEST = '/home/owen/.crucible/server';

// ---------------------------------------------------- the host's loopback door

/** Where `install.ps1` unpacks the host pack on the fake machine, and its `.cmd`. */
export const HOST_DIR = 'C:\\Users\\owen\\AppData\\Local\\Crucible\\host';
export const HOST_CMD = `${HOST_DIR}\\crucible.cmd`;
/** The host-mode server's own config — where {@link hostToken} reads the bearer. */
export const HOST_CONFIG_PATH = 'C:\\Users\\owen\\AppData\\Local\\Crucible\\config.toml';
/** A win32 `FakeHost`'s environment. `LOCALAPPDATA` is READ, never assembled. */
export const WIN_ENV = { LOCALAPPDATA: 'C:\\Users\\owen\\AppData\\Local' };
/**
 * The host-mode `config.toml`, as `crucible init` wrote it before the migrate.
 * `llama-windows` and not `none`: PHASE15 section 0's amendment makes Windows
 * a backend of its own (llama.cpp + GGUF), which WSL then upgrades.
 */
export const HOST_CONFIG = `[server]
name = "crucible@owens-pc"
host = "127.0.0.1"
port = 7100

[auth]
token = "host-token-not-a-secret"

[backend]
kind = "llama-windows"
`;

/** What the fake door was asked. One request per {@link fakeHostDoor}. */
export interface DoorRequest {
  url: string;
  method: string | undefined;
  authorization: string | null;
  contentType: string | null;
  body: unknown;
}

export interface FakeDoorScript {
  /** The HTTP status. 200 unless a test is exercising 401/409. */
  status?: number;
  /**
   * The response body, already chunked. Each entry is one chunk off the wire —
   * so a test that wants an ndjson line split across two reads simply puts half
   * of it in one entry and half in the next.
   */
  chunks?: readonly string[];
  /**
   * The `data` of each event, keyed by kind, in order. The `{"id", "event",
   * "data"}` envelope is added here so no test has to keep the ids in step by
   * hand — and so a test that DOES want a malformed envelope writes `chunks`.
   */
  events?: readonly (readonly [kind: string, data: unknown])[];
  /** `fetch` rejects with this instead of answering: a refused connection, a timeout. */
  rejectWith?: Error;
  /** A non-streaming body, for the statuses that carry a sentence rather than events. */
  text?: string;
}

/** `{"id": n, "event": kind, "data": …}` — `crucible/tasks.py`'s envelope, one line. */
export function doorLine(id: number, kind: string, data: unknown): string {
  return `${JSON.stringify({ id, event: kind, data })}\n`;
}

/**
 * A `fetchImpl` that replays a scripted door. Nothing opens a socket, and the
 * request the client built is kept verbatim so a test can assert the URL, the
 * method, the bearer and the body it would have sent.
 */
export function fakeHostDoor(script: FakeDoorScript): { fetchImpl: HostFetch; requests: DoorRequest[] } {
  const requests: DoorRequest[] = [];
  const fetchImpl = (async (input: unknown, init?: RequestInit): Promise<Response> => {
    const headers = new Headers(init?.headers ?? {});
    requests.push({
      url: String(input),
      method: init?.method,
      authorization: headers.get('authorization'),
      contentType: headers.get('content-type'),
      body: typeof init?.body === 'string' ? JSON.parse(init.body) : init?.body,
    });
    if (script.rejectWith !== undefined) throw script.rejectWith;
    const status = script.status ?? 200;
    if (script.text !== undefined) return new Response(script.text, { status });
    const chunks = script.chunks ?? (script.events ?? []).map(([kind, data], index) => doorLine(index + 1, kind, data));
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller): void {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        controller.close();
      },
    });
    return new Response(stream, { status });
  }) as HostFetch;
  return { fetchImpl, requests };
}

/** PHASE19 2.6's status document, with the fields a test does not care about filled in. */
export interface FakeStatus {
  running?: boolean;
  outcome?: unknown;
  presence?: unknown;
}

export interface FakeWatchScript {
  /**
   * What `GET /install` answers, in order; the LAST one repeats. A test that
   * wants "not decided yet, then running, then done" writes three.
   */
  statuses: readonly FakeStatus[];
  /** The move's events, served once on `GET /install/events`. Absent means 404. */
  events?: readonly (readonly [kind: string, data: unknown])[];
}

export const FAKE_PRESENCE = { distro: 'absent', engine: 'running', owner: 'child', detail: 'the Windows engine' };

/**
 * A `fetchImpl` for the PHASE19 2.6 door: three routes, answered by path.
 *
 * {@link fakeHostDoor} replays one script for one request, which was right
 * while `install()` made exactly one. It now reads the status, attaches, and
 * reads the status again, so the fake has to know which door it is answering.
 */
export function fakeWatchDoor(script: FakeWatchScript): { fetchImpl: HostFetch; requests: DoorRequest[] } {
  const requests: DoorRequest[] = [];
  let asked = 0;
  let streamed = false;
  const fetchImpl = (async (input: unknown, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    const headers = new Headers(init?.headers ?? {});
    requests.push({
      url,
      method: init?.method,
      authorization: headers.get('authorization'),
      contentType: headers.get('content-type'),
      body: typeof init?.body === 'string' ? JSON.parse(init.body) : init?.body,
    });
    if (url.endsWith('/install/events')) {
      if (streamed || script.events === undefined) {
        return new Response(JSON.stringify({ error: { code: 'no_install_running', message: 'nothing to watch' } }), { status: 404 });
      }
      streamed = true;
      const encoder = new TextEncoder();
      const lines = script.events.map(([kind, data], index) => doorLine(index + 1, kind, data));
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller): void {
            for (const line of lines) controller.enqueue(encoder.encode(line));
            controller.close();
          },
        }),
        { status: 200 },
      );
    }
    const index = Math.min(asked, script.statuses.length - 1);
    asked += 1;
    const status = script.statuses[index] ?? {};
    return new Response(
      JSON.stringify({
        running: status.running ?? false,
        outcome: status.outcome ?? null,
        presence: status.presence ?? FAKE_PRESENCE,
      }),
      { status: 200 },
    );
  }) as HostFetch;
  return { fetchImpl, requests };
}

const GUEST_CRUCIBLE = '/home/crucible/.crucible/server/bin/crucible';

/** The `data` of a `done` event describing a finished WSL install, as `crucible/host/door.py` sends it. */
export const HOST_DONE_DATA = {
  server: { name: 'crucible@owens-pc-wsl', url: 'http://127.0.0.1:7100', config_path: 'crucible:/home/crucible/.crucible/config.toml' },
  release: '0.6.0',
  backend: 'cuda-linux',
  crucible: GUEST_CRUCIBLE,
  steps: [
    { name: 'host-facts', argv: [], status: 'ok', detail: 'crucible: CRUCIBLE_HOME /home/crucible/.crucible, user crucible, 380.0 GiB free' },
    { name: 'server', argv: [GUEST_CRUCIBLE], status: 'ok', detail: 'python 3.11.16, then the 0.6.0 wheel' },
    { name: 'init', argv: [GUEST_CRUCIBLE, 'init', '--token', '<redacted>', '--enable-llm'], status: 'ok', detail: 'exit 0' },
    { name: 'service-install', argv: [GUEST_CRUCIBLE, 'service', 'install'], status: 'ok', detail: 'exit 0' },
    { name: 'linger', argv: ['loginctl', 'enable-linger', 'crucible'], status: 'ok', detail: 'granted' },
    { name: 'capability-write', argv: [GUEST_CRUCIBLE, 'capability', '--write'], status: 'ok', detail: 'exit 0' },
  ],
};

/** That `done`, as a scripted event: `['done', HOST_DONE_DATA]`. */
export const HOST_DONE: readonly [string, unknown] = ['done', HOST_DONE_DATA];

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
