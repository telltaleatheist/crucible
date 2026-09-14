/**
 * The real runner, against real child processes (node itself): the two
 * decodings, the line splitting, the timeout, the spawn error.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decodeWslBytes, processRunner, splitLines } from '../src/index.js';

const NODE = process.execPath;

test('decodeWslBytes: UTF-16LE with a BOM (wsl.exe itself) decodes and loses the BOM', () => {
  const bytes = Buffer.concat([Buffer.from([0xff, 0xfe]), Buffer.from('* Ubuntu Running 2\r\n', 'utf16le')]);
  assert.equal(decodeWslBytes(bytes), '* Ubuntu Running 2\r\n');
});

test('decodeWslBytes: UTF-8 from inside the distro passes through, BOM or not', () => {
  assert.equal(decodeWslBytes(Buffer.from('conda=/home/owen/anaconda3\n', 'utf8')), 'conda=/home/owen/anaconda3\n');
  assert.equal(decodeWslBytes(Buffer.from('﻿hello', 'utf8')), 'hello');
  assert.equal(decodeWslBytes(Buffer.from('déjà vu — ok', 'utf8')), 'déjà vu — ok');
});

test('decodeWslBytes: an empty buffer is an empty string', () => {
  assert.equal(decodeWslBytes(Buffer.alloc(0)), '');
});

test('splitLines: \\n, \\r\\n and a bare \\r (pip repainting its bar) all end a line', () => {
  assert.deepEqual(splitLines('a\nb\r\nc\rd'), { lines: ['a', 'b', 'c'], rest: 'd' });
  assert.deepEqual(splitLines('whole\n'), { lines: ['whole'], rest: '' });
  assert.deepEqual(splitLines('partial'), { lines: [], rest: 'partial' });
});

test('processRunner.run collects stdout, stderr and the exit code', async () => {
  const runner = processRunner();
  const result = await runner.run([NODE, '-e', 'process.stdout.write("out"); process.stderr.write("err"); process.exit(3)'], { timeoutMs: 20_000 });
  assert.equal(result.code, 3);
  assert.equal(result.stdout, 'out');
  assert.equal(result.stderr, 'err');
  assert.equal(result.failure, null);
});

test('processRunner.run passes env through to the child', async () => {
  const runner = processRunner();
  const result = await runner.run([NODE, '-p', 'process.env.CRUCIBLE_HOME'], { timeoutMs: 20_000, env: { CRUCIBLE_HOME: '/srv/crucible' } });
  assert.equal(result.stdout.trim(), '/srv/crucible');
});

test('processRunner.stream hands over lines as they arrive, splitting on \\r too, and decoding UTF-16 chunks', async () => {
  const runner = processRunner();
  const seen: string[] = [];
  const script = [
    'process.stdout.write("Collecting torch\\r  10%\\r  20%\\n");',
    'process.stderr.write("WARNING: something\\n");',
    'process.stdout.write(Buffer.from("there is no distribution\\r\\n", "utf16le"));',
    'process.stdout.write("tail without newline");',
  ].join(' ');
  const result = await runner.stream([NODE, '-e', script], { timeoutMs: 20_000, onLine: (line, stream) => seen.push(`${stream}:${line}`) });
  assert.equal(result.code, 0);
  assert.deepEqual(seen.filter((s) => s.startsWith('stdout:')), [
    'stdout:Collecting torch',
    'stdout:  10%',
    'stdout:  20%',
    'stdout:there is no distribution',
    'stdout:tail without newline',
  ]);
  assert.deepEqual(seen.filter((s) => s.startsWith('stderr:')), ['stderr:WARNING: something']);
});

test('processRunner: a call that never returns is a reported failure, not a hang', async () => {
  const runner = processRunner();
  const started = Date.now();
  const result = await runner.run([NODE, '-e', 'setInterval(() => {}, 1000)'], { timeoutMs: 500 });
  assert.equal(result.code, null);
  assert.match(result.failure ?? '', /did not answer within 1s/);
  assert.ok(Date.now() - started < 10_000, 'the timeout fired');
});

test('processRunner: a program that does not exist is a reported failure', async () => {
  const runner = processRunner();
  const result = await runner.run(['definitely-not-a-program-7a1b', '--version'], { timeoutMs: 5_000 });
  assert.equal(result.code, null);
  assert.match(result.failure ?? '', /ENOENT/);
});

test('processRunner: an empty argv is refused as a failure', async () => {
  const runner = processRunner();
  const result = await runner.run([], { timeoutMs: 5_000 });
  assert.equal(result.failure, 'empty argv');
});
