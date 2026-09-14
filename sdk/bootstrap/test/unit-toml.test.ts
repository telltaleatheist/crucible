/** The TOML reader, against what `tomli_w` writes and against what it does not. */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { parseToml, TomlError } from '../src/index.js';
import { GUEST_CONFIG } from './fake.js';

test('parses the config.toml crucible init wrote on the PC, arrays of tables included', () => {
  const table = parseToml(GUEST_CONFIG) as Record<string, Record<string, unknown>>;
  assert.equal(table['server']?.['name'], 'crucible@owens-pc-wsl');
  assert.equal(table['server']?.['port'], 7100);
  assert.equal(table['auth']?.['token'], 'test-token-not-a-secret');
  assert.equal(table['jobs']?.['enable_echo'], true);
  assert.equal(table['accelerator']?.['desktop_allowance_bytes'], 3221225472);
  const capability = table['capability'] as { classes: Record<string, unknown>[] };
  assert.equal(capability.classes.length, 1);
  assert.equal(capability.classes[0]?.['capability'], 'tts');
  assert.match(String(capability.classes[0]?.['reason']), /17\.7 GiB/);
});

test('strings: escapes, unicode, literal strings, comments and quoted keys', () => {
  const table = parseToml('a = "quote \\" back \\\\ nl \\n u \\u00e9" # trailing\n\'lit\' = \'C:\\raw\'\nb.c = \'x\'\n# whole line\n') as Record<string, unknown>;
  assert.equal(table['a'], 'quote " back \\ nl \n u é');
  assert.equal(table['lit'], 'C:\\raw');
  assert.deepEqual(table['b'], { c: 'x' });
});

test('numbers, booleans and single-line arrays', () => {
  const table = parseToml('i = 1_000\nn = -3\nf = 1.5e3\nt = true\nl = [1, "two", [3]]\ne = []\n') as Record<string, unknown>;
  assert.equal(table['i'], 1000);
  assert.equal(table['n'], -3);
  assert.equal(table['f'], 1500);
  assert.equal(table['t'], true);
  assert.deepEqual(table['l'], [1, 'two', [3]]);
  assert.deepEqual(table['e'], []);
});

test('a dotted header under an array of tables addresses its last element', () => {
  const table = parseToml('[[a]]\nx = 1\n[a.sub]\ny = 2\n[[a]]\nx = 3\n') as { a: Record<string, unknown>[] };
  assert.deepEqual(table.a, [{ x: 1, sub: { y: 2 } }, { x: 3 }]);
});

for (const [label, text, pattern] of [
  ['an inline table', 'a = { b = 1 }', /inline tables/],
  ['a multi-line string', 'a = """x"""', /multi-line strings/],
  ['a date', 'a = 2026-09-13T00:00:00Z', /dates/],
  ['a duplicate key', 'a = 1\na = 2', /defined twice/],
  ['a bare word', 'a = nope', /unsupported value/],
  ['a line that is not key = value', 'just text', /expected key = value/],
  ['an unterminated string', 'a = "open', /unterminated string/],
  ['an unterminated header', '[server', /unterminated table header/],
  ['a multi-line array', 'a = [1,\n2]', /unterminated array/],
  ['trailing text', 'a = 1 2', /trailing text/],
] as const) {
  test(`refuses ${label} by name, with the line number`, () => {
    assert.throws(() => parseToml(text), (err: unknown) => err instanceof TomlError && pattern.test(err.message) && err.line >= 1);
  });
}
