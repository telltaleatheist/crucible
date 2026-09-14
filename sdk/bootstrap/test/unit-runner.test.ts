/**
 * The real runner, against real child processes (node itself): the two
 * decodings, the line splitting, the timeout, the spawn error.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { decodeWslBytes, incompleteTailBytes, processRunner, segmentWslBytes, splitLines } from '../src/index.js';

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

test('decodeWslBytes: BOTH encodings in ONE buffer, decoded per run — the defect the stream test caught', () => {
  // A pipe hands over whatever chunking the OS felt like, and wsl.exe's own
  // UTF-16 message can share a chunk with the guest's UTF-8 output. Deciding
  // the encoding per buffer decoded "tail" as UTF-16 and printed 慴汩.
  const mixed = Buffer.concat([
    Buffer.from('there is no distribution\r\n', 'utf16le'),
    Buffer.from('tail without newline', 'utf8'),
  ]);
  assert.equal(decodeWslBytes(mixed), 'there is no distributiontail without newline'.replace('distribution', 'distribution\r\n'));

  const sandwich = Buffer.concat([
    Buffer.from('Collecting torch\n', 'utf8'),
    Buffer.from([0xff, 0xfe]),
    Buffer.from('wsl: a message\r\n', 'utf16le'),
    Buffer.from('déjà vu\n', 'utf8'),
  ]);
  assert.equal(decodeWslBytes(sandwich), 'Collecting torch\nwsl: a message\r\ndéjà vu\n');
  assert.deepEqual(segmentWslBytes(sandwich).map((s) => s.encoding), ['utf8', 'utf16le', 'utf8']);
});

test('decodeWslBytes: a lone non-Latin code unit does not end a UTF-16 run', () => {
  // `—` is 14 20 in UTF-16LE: a non-NUL high byte inside an otherwise ASCII
  // message. One such pair must not split the run; two in a row (which is what
  // UTF-8 looks like) must.
  assert.equal(decodeWslBytes(Buffer.from('a — b\r\n', 'utf16le')), 'a — b\r\n');
});

test('incompleteTailBytes: half a UTF-16 code unit, or a split UTF-8 character, waits for the next chunk', () => {
  const utf16 = Buffer.from('hello', 'utf16le');
  assert.equal(incompleteTailBytes(utf16), 0);
  assert.equal(incompleteTailBytes(utf16.subarray(0, 9)), 1, 'an odd UTF-16 tail holds one byte');
  const utf8 = Buffer.from('déjà', 'utf8');
  assert.equal(incompleteTailBytes(utf8), 0);
  assert.equal(incompleteTailBytes(utf8.subarray(0, utf8.length - 1)), 1, 'half of à holds one byte');
  assert.equal(incompleteTailBytes(Buffer.from('plain ascii', 'utf8')), 0, 'a complete line is never delayed');
  assert.equal(incompleteTailBytes(Buffer.from('ends with newline\n', 'utf8')), 0);
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

test('processRunner.stream: a character split across two chunks arrives whole', async () => {
  const runner = processRunner();
  const seen: string[] = [];
  // Two writes, the second beginning in the middle of the UTF-16 code unit the
  // first ended in, and a UTF-8 multi-byte character split the same way.
  const script = [
    'const u16 = Buffer.from("wsl message\\r\\n", "utf16le");',
    'process.stdout.write(u16.subarray(0, 7));',
    'process.stdout.write(Buffer.concat([u16.subarray(7), Buffer.from("déjà vu\\n", "utf8").subarray(0, 3)]));',
    'process.stdout.write(Buffer.from("déjà vu\\n", "utf8").subarray(3));',
  ].join(' ');
  const result = await runner.stream([NODE, '-e', script], { timeoutMs: 20_000, onLine: (line) => seen.push(line) });
  assert.equal(result.code, 0);
  assert.deepEqual(seen, ['wsl message', 'déjà vu']);
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
