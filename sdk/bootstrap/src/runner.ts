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
 *      name") are UTF-16; a bash pipeline's stdout is UTF-8. **Both can arrive
 *      in ONE chunk**, so the decode is decided per RUN of bytes, by looking
 *      for interleaved NULs — {@link segmentWslBytes}, {@link decodeWslBytes} —
 *      and a chunk that ends inside a character holds those bytes back for the
 *      next one ({@link incompleteTailBytes}).
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
 * One run of bytes in a single encoding, inside a buffer that may hold both.
 */
export interface WslSegment {
  encoding: 'utf16le' | 'utf8';
  start: number;
  /** Exclusive. */
  end: number;
}

/**
 * Split a buffer into runs of UTF-16LE and runs of UTF-8.
 *
 * **Per BUFFER was not enough, and the test that caught it is real.** wsl.exe's
 * own UTF-16 message and the guest's UTF-8 output land on the same handle, and
 * a pipe hands them over in whatever chunking the OS felt like — including BOTH
 * IN ONE CHUNK. Deciding the encoding for a whole chunk then decodes the UTF-8
 * half as UTF-16 and prints `慴汩 …` (the swapped-pair look is the giveaway:
 * `ta` → `慴`). So the decision is per RUN, not per chunk.
 *
 * The structure it keys on: UTF-8 never contains a NUL, and UTF-16LE text in
 * any Latin script has one as the high byte of nearly every code unit.
 *
 * - A run starts at a UTF-16 BOM (`FF FE`), or at a byte whose successor is NUL.
 * - A run CONTINUES while this pair or the NEXT one has a NUL high byte, so a
 *   single `—` (`14 20`) inside an otherwise ASCII message does not end it,
 *   while two consecutive non-NUL-high pairs — which is what UTF-8 text looks
 *   like — do.
 */
export function segmentWslBytes(buffer: Buffer): WslSegment[] {
  const length = buffer.length;
  if (length === 0) return [];
  if (!buffer.includes(0)) return [{ encoding: 'utf8', start: 0, end: length }];

  const segments: WslSegment[] = [];
  const push = (encoding: WslSegment['encoding'], start: number, end: number): void => {
    if (end <= start) return;
    const last = segments[segments.length - 1];
    if (last !== undefined && last.encoding === encoding && last.end === start) last.end = end;
    else segments.push({ encoding, start, end });
  };

  let i = 0;
  let utf8Start = 0;
  const startsUtf16 = (at: number): boolean => {
    if (at + 1 >= length) return false;
    if (buffer[at] === 0xff && buffer[at + 1] === 0xfe) return true;
    return buffer[at + 1] === 0 && buffer[at] !== 0;
  };
  const continuesUtf16 = (at: number): boolean => {
    if (at + 1 >= length) return false;
    if (buffer[at + 1] === 0) return true;
    return at + 3 < length && buffer[at + 3] === 0;
  };
  while (i < length) {
    if (!startsUtf16(i)) {
      i += 1;
      continue;
    }
    push('utf8', utf8Start, i);
    const runStart = i;
    while (continuesUtf16(i)) i += 2;
    push('utf16le', runStart, i);
    utf8Start = i;
  }
  push('utf8', utf8Start, length);
  return segments;
}

/**
 * Bytes from a wsl.exe handle → text, whichever of the two encodings each run
 * of it is. A BOM is stripped from the front of every run, because `wsl -l -v`'s
 * first line would otherwise begin with one and never match.
 */
export function decodeWslBytes(buffer: Buffer): string {
  let text = '';
  for (const segment of segmentWslBytes(buffer)) {
    text += buffer.subarray(segment.start, segment.end).toString(segment.encoding).replace(/^﻿/, '');
  }
  return text;
}

/**
 * How many trailing bytes of a chunk must WAIT for the next one.
 *
 * Two ways a chunk boundary lands inside a character, and both produce a
 * mangled line that no amount of per-run decoding can repair:
 *
 * - an ODD number of bytes at the end of a UTF-16 run: half a code unit;
 * - an incomplete UTF-8 multi-byte sequence: `é` split down the middle.
 *
 * Nothing else is held back. A complete character is handed over the moment it
 * arrives, so a progress line that ends exactly at a chunk boundary is not
 * delayed waiting for output that may be minutes away.
 */
export function incompleteTailBytes(buffer: Buffer): number {
  const segments = segmentWslBytes(buffer);
  const last = segments[segments.length - 1];
  if (last === undefined) return 0;
  const size = last.end - last.start;
  if (last.encoding === 'utf16le') return size % 2;
  // A UTF-16 run whose last code unit is cut in half ends the buffer with ONE
  // orphan byte, which segments as a one-byte UTF-8 run after a UTF-16 one.
  // That byte is the high half of a character, not a character.
  const previous = segments[segments.length - 2];
  if (size === 1 && previous !== undefined && previous.encoding === 'utf16le') return 1;
  // UTF-8: a lead byte in the last three positions that announces more bytes
  // than the buffer has.
  for (let back = 1; back <= Math.min(3, size); back += 1) {
    const byte = buffer[buffer.length - back] as number;
    if ((byte & 0xc0) === 0x80) continue; // a continuation byte; keep looking back
    const needed = (byte & 0xe0) === 0xc0 ? 2 : (byte & 0xf0) === 0xe0 ? 3 : (byte & 0xf8) === 0xf0 ? 4 : 1;
    return needed > back ? back : 0;
  }
  return 0;
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
        // The bytes a chunk ended in the middle of a character with. Held for
        // the next chunk rather than decoded into a replacement character.
        const carry: Record<OutputStream, Buffer> = { stdout: Buffer.alloc(0), stderr: Buffer.alloc(0) };
        launch(
          argv,
          options,
          (chunk, stream) => {
            // Decoded per RUN, not per chunk: a UTF-16 message from wsl.exe
            // itself can arrive between two UTF-8 pip lines ON THE SAME HANDLE
            // and in the SAME chunk.
            const whole = carry[stream].length === 0 ? chunk : Buffer.concat([carry[stream], chunk]);
            const hold = incompleteTailBytes(whole);
            carry[stream] = hold === 0 ? Buffer.alloc(0) : Buffer.from(whole.subarray(whole.length - hold));
            const usable = hold === 0 ? whole : whole.subarray(0, whole.length - hold);
            const { lines, rest } = splitLines(pending[stream] + decodeWslBytes(usable));
            pending[stream] = rest;
            for (const line of lines) if (line.trim().length > 0) options.onLine(line, stream);
          },
          (result) => {
            for (const stream of ['stdout', 'stderr'] as const) {
              // Whatever was held back at the end is decoded now: the process is
              // gone, so there is no next chunk to complete it and a truncated
              // last line is still what it said.
              const tail = carry[stream].length === 0 ? '' : decodeWslBytes(carry[stream]);
              const text = pending[stream] + tail;
              if (text.trim().length > 0) options.onLine(text, stream);
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
