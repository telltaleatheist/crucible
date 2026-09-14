/**
 * The one door to the machine: every process this package starts, every file
 * it reads, goes through a {@link Runner}. The tests supply a scripted one and
 * assert on the argv that would have run; {@link processRunner} is the real one.
 *
 * The real runner carries the facts Foundry's WSL launcher learned the hard way
 * (docs/FROM-FOUNDRY-WSL-VLLM.md section 2), each with a test:
 *
 *   1. **wsl.exe's OWN output is UTF-16LE with a BOM; output from inside the
 *      distro is UTF-8, on the same handles.** `wsl -l -v` and every message
 *      wsl.exe writes itself ("there is no distribution with the supplied
 *      name") are UTF-16; a bash pipeline's stdout is UTF-8. The decode is
 *      decided per chunk by looking for interleaved NULs — {@link decodeWslBytes}.
 *
 *   2. **Every one-shot call gets its own timeout.** A booting distro or a
 *      blocking profile makes wsl.exe never return; `timeoutMs` is required on
 *      every call and a timeout is a reported failure, never a hang.
 *
 *   3. **Argument arrays, never shell strings.** `spawn(argv[0], argv.slice(1))`
 *      with `windowsHide`; nothing here is ever joined into a command line.
 *
 * The runner knows nothing about WSL beyond how to decode its bytes. What to
 * run, and how a Windows path or a backslash crosses into the guest, is
 * `wsl.ts`'s.
 */
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import * as fs from 'node:fs';
import * as os from 'node:os';

export interface RunResult {
  /** Null when the process never started or was killed by the timeout. */
  code: number | null;
  stdout: string;
  stderr: string;
  /** Set when there is no exit code to report — a spawn error, or the timeout. */
  failure: string | null;
}

export interface RunOptions {
  /** Required. There is no call here that may block forever. */
  timeoutMs: number;
  /** Extra environment for the child, merged over the process's own. */
  env?: Readonly<Record<string, string>>;
}

export type OutputStream = 'stdout' | 'stderr';

export interface StreamOptions extends RunOptions {
  /** Called once per line, on either handle, as the lines arrive. */
  onLine: (line: string, stream: OutputStream) => void;
}

export interface Runner {
  readonly platform: NodeJS.Platform;
  readonly env: NodeJS.ProcessEnv;
  readonly homedir: string;
  /** Run `argv[0]` with `argv.slice(1)` and collect what it said. Never throws. */
  run(argv: readonly string[], options: RunOptions): Promise<RunResult>;
  /** The same, but `onLine` sees every line as it arrives. Never throws. */
  stream(argv: readonly string[], options: StreamOptions): Promise<RunResult>;
  fileExists(path: string): boolean;
  /** Throws on any failure; the caller names the refusal. */
  readFile(path: string): string;
  /** `fs.realpathSync.native`. Throws when the path cannot be resolved. */
  realpathNative(path: string): string;
}

/**
 * Bytes from a wsl.exe handle → text, whichever of the two encodings it is.
 *
 * The test is structural: UTF-16LE ASCII has a NUL after every character, and
 * UTF-8 text never contains a NUL at all. A BOM is stripped either way, because
 * `wsl -l -v`'s first line would otherwise begin with one and never match.
 */
export function decodeWslBytes(buffer: Buffer): string {
  const probe = buffer.subarray(0, Math.min(buffer.length, 256));
  const utf16 = probe.includes(0);
  return buffer.toString(utf16 ? 'utf16le' : 'utf8').replace(/^﻿/, '');
}

/**
 * Split text into lines on `\n`, `\r\n` AND a bare `\r`, because pip repaints
 * its progress bar with `\r` and no newline; splitting on both turns that into
 * successive lines rather than one line that never ends. Returns the complete
 * lines and the unterminated remainder.
 */
export function splitLines(pending: string): { lines: string[]; rest: string } {
  const parts = pending.split(/\r?\n|\r/);
  const rest = parts.pop() ?? '';
  return { lines: parts, rest };
}

/**
 * Kill a child and everything under it. `child.kill()` reaches wsl.exe alone
 * and leaves the guest-side process it is relaying for; taskkill's tree flag is
 * the only thing on Windows that gets the rest. Used for timed-out calls and
 * nothing else — never against a service, which is stopped by its supervisor.
 */
function killTree(child: ChildProcess, platform: NodeJS.Platform): void {
  if (child.pid === undefined || child.exitCode !== null) return;
  if (platform === 'win32') {
    try {
      spawnSync('taskkill', ['/pid', String(child.pid), '/t', '/f'], { windowsHide: true });
      return;
    } catch {
      // fall through to the signal
    }
  }
  try {
    child.kill('SIGTERM');
  } catch {
    // already gone
  }
}

function launch(
  argv: readonly string[],
  options: RunOptions,
  onChunk: (chunk: Buffer, stream: OutputStream) => void,
  onFinish: (result: RunResult) => void,
): void {
  const program = argv[0];
  if (program === undefined) {
    onFinish({ code: null, stdout: '', stderr: '', failure: 'empty argv' });
    return;
  }
  let child: ChildProcess;
  try {
    child = spawn(program, argv.slice(1), {
      windowsHide: true,
      env: options.env === undefined ? process.env : { ...process.env, ...options.env },
    });
  } catch (err) {
    onFinish({ code: null, stdout: '', stderr: '', failure: (err as Error).message });
    return;
  }

  const out: Buffer[] = [];
  const err: Buffer[] = [];
  let settled = false;
  const finish = (code: number | null, failure: string | null): void => {
    if (settled) return;
    settled = true;
    clearTimeout(timer);
    onFinish({
      code,
      stdout: decodeWslBytes(Buffer.concat(out)),
      stderr: decodeWslBytes(Buffer.concat(err)),
      failure,
    });
  };
  const timer = setTimeout(() => {
    killTree(child, process.platform);
    finish(null, `${program} did not answer within ${Math.round(options.timeoutMs / 1000)}s`);
  }, options.timeoutMs);

  child.stdout?.on('data', (chunk: Buffer) => {
    out.push(chunk);
    onChunk(chunk, 'stdout');
  });
  child.stderr?.on('data', (chunk: Buffer) => {
    err.push(chunk);
    onChunk(chunk, 'stderr');
  });
  child.on('error', (e) => finish(null, e.message));
  child.on('close', (code) => finish(code, null));
}

/** The real machine. */
export function processRunner(): Runner {
  return {
    platform: process.platform,
    env: process.env,
    homedir: os.homedir(),
    run(argv, options) {
      return new Promise<RunResult>((resolve) => {
        launch(argv, options, () => undefined, resolve);
      });
    },
    stream(argv, options) {
      return new Promise<RunResult>((resolve) => {
        const pending: Record<OutputStream, string> = { stdout: '', stderr: '' };
        launch(
          argv,
          options,
          (chunk, stream) => {
            // Decoded PER CHUNK: a UTF-16 message from wsl.exe itself can arrive
            // between two UTF-8 pip lines on the same handle.
            const { lines, rest } = splitLines(pending[stream] + decodeWslBytes(chunk));
            pending[stream] = rest;
            for (const line of lines) if (line.trim().length > 0) options.onLine(line, stream);
          },
          (result) => {
            for (const stream of ['stdout', 'stderr'] as const) {
              if (pending[stream].trim().length > 0) options.onLine(pending[stream], stream);
            }
            resolve(result);
          },
        );
      });
    },
    fileExists: (path) => fs.existsSync(path),
    readFile: (path) => fs.readFileSync(path, 'utf-8'),
    realpathNative: (path) => fs.realpathSync.native(path),
  };
}
