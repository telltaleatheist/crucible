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
  /** Published working directory of an installed control command. */
  cwd?: string;
  /** Required; no call here may block forever. */
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
  /** Run `argv[0]` with `argv.slice(1)` and collect what it said; never throws. */
  run(argv: readonly string[], options: RunOptions): Promise<RunResult>;
  /** The same, but `onLine` sees every line as it arrives; never throws. */
  stream(argv: readonly string[], options: StreamOptions): Promise<RunResult>;
  fileExists(path: string): boolean;
  /** Throws on any failure; the caller names the refusal. */
  readFile(path: string): string;
  /** `fs.realpathSync.native`; throws when the path cannot be resolved. */
  realpathNative(path: string): string;
}

/** One run of bytes in a single encoding, inside a buffer that may hold both. */
export interface WslSegment {
  encoding: 'utf16le' | 'utf8';
  start: number;
  /** Exclusive end offset. */
  end: number;
}

/** Split a buffer into runs of UTF-16LE and runs of UTF-8. */
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

/** Decode wsl.exe bytes, choosing UTF-16LE or UTF-8 per run and stripping BOMs. */
export function decodeWslBytes(buffer: Buffer): string {
  let text = '';
  for (const segment of segmentWslBytes(buffer)) {
    text += buffer.subarray(segment.start, segment.end).toString(segment.encoding).replace(/^﻿/, '');
  }
  return text;
}

/** How many trailing bytes of a chunk must WAIT for the next one. */
export function incompleteTailBytes(buffer: Buffer): number {
  const segments = segmentWslBytes(buffer);
  const last = segments[segments.length - 1];
  if (last === undefined) return 0;
  const size = last.end - last.start;
  if (last.encoding === 'utf16le') return size % 2;
  const previous = segments[segments.length - 2];
  if (size === 1 && previous !== undefined && previous.encoding === 'utf16le') return 1;
  for (let back = 1; back <= Math.min(3, size); back += 1) {
    const byte = buffer[buffer.length - back] as number;
    if ((byte & 0xc0) === 0x80) continue;
    const needed = (byte & 0xe0) === 0xc0 ? 2 : (byte & 0xf0) === 0xe0 ? 3 : (byte & 0xf8) === 0xf0 ? 4 : 1;
    return needed > back ? back : 0;
  }
  return 0;
}

/**
 * Split text on `\n`, `\r\n` and bare `\r`, returning complete lines and the unterminated
 * remainder.
 */
export function splitLines(pending: string): { lines: string[]; rest: string } {
  const parts = pending.split(/\r?\n|\r/);
  const rest = parts.pop() ?? '';
  return { lines: parts, rest };
}

function killTree(child: ChildProcess, platform: NodeJS.Platform): void {
  if (child.pid === undefined || child.exitCode !== null) return;
  if (platform === 'win32') {
    try {
      spawnSync('taskkill', ['/pid', String(child.pid), '/t', '/f'], { windowsHide: true });
      return;
    } catch {
    }
  }
  try {
    child.kill('SIGTERM');
  } catch {
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
      cwd: options.cwd,
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

/** The real {@link Runner}. */
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
        const carry: Record<OutputStream, Buffer> = { stdout: Buffer.alloc(0), stderr: Buffer.alloc(0) };
        launch(
          argv,
          options,
          (chunk, stream) => {
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
