/**
 * The Windows/WSL plumbing facts, one test each (docs/FROM-FOUNDRY-WSL-VLLM.md section 2).
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { guestPathFor, guestUnpackArgv, networkPathBehind, parseWslList, shellQuote, toWslPath, wslArgv, wslListArgv } from '../src/index.js';
import { FakeRunner, refusal } from './fake.js';

test('wslArgv: always -d <distro> --exec, never the implicit shell, and never `--`', () => {
  assert.deepEqual(wslArgv('Ubuntu', ['bash', '-c', 'f=hi; echo $f']), ['wsl.exe', '-d', 'Ubuntu', '--exec', 'bash', '-c', 'f=hi; echo $f']);
  assert.throws(() => wslArgv('Ubuntu', []), /nothing to exec/);
});

test('wslArgv: backslashes are doubled once, because wsl.exe halves them once before bash exists', () => {
  const argv = wslArgv('Ubuntu', ['printf', '%s', 'C:\\Users\\owen', '\\\\TITAN\\iO']);
  assert.deepEqual(argv.slice(4), ['printf', '%s', 'C:\\\\Users\\\\owen', '\\\\\\\\TITAN\\\\iO']);
});

test('wslListArgv is `wsl.exe -l -v`', () => {
  assert.deepEqual(wslListArgv(), ['wsl.exe', '-l', '-v']);
});

test('parseWslList: the table wsl.exe prints, with the default starred', () => {
  const parsed = parseWslList('  NAME              STATE           VERSION\r\n* Ubuntu            Running         2\r\n  docker-desktop    Stopped         2\r\n  Legacy            Stopped         1\r\n');
  assert.deepEqual(parsed, {
    distros: [
      { name: 'Ubuntu', state: 'Running', version: 2, default: true },
      { name: 'docker-desktop', state: 'Stopped', version: 2, default: false },
      { name: 'Legacy', state: 'Stopped', version: 1, default: false },
    ],
    default: 'Ubuntu',
  });
});

test('parseWslList: no rows, no default', () => {
  assert.deepEqual(parseWslList('  NAME   STATE   VERSION\r\n'), { distros: [], default: null });
  assert.deepEqual(parseWslList(''), { distros: [], default: null });
});

test('toWslPath: C:\\a\\b → /mnt/c/a/b, drive letter lowercased, spaces kept', () => {
  assert.equal(toWslPath('C:\\Users\\Some One\\wheel.whl'), '/mnt/c/Users/Some One/wheel.whl');
  assert.equal(toWslPath('E:/books/x'), '/mnt/e/books/x');
});

test('toWslPath: a UNC path is refused by name, never mangled', async () => {
  const r = await refusal(Promise.resolve().then(() => toWslPath('\\\\TITAN\\iO\\bookforge\\x.whl')));
  assert.equal(r.code, 'network_path');
  assert.match(r.message, /no \/mnt mapping/);
});

test('toWslPath: a relative or guest path has no WSL spelling', async () => {
  const r = await refusal(Promise.resolve().then(() => toWslPath('wheel.whl')));
  assert.equal(r.code, 'not_a_windows_path');
  const g = await refusal(Promise.resolve().then(() => toWslPath('/home/owen/x.whl')));
  assert.equal(g.code, 'not_a_windows_path');
});

test('networkPathBehind: a mapped drive is caught by realpath.native, a local one is not, an unresolvable one is null', () => {
  const runner = new FakeRunner({ files: { 'C:\\local\\x.whl': '' }, realpaths: { 'Z:\\bookforge\\x.whl': '\\\\TITAN\\iO\\bookforge\\x.whl' } }, []);
  assert.equal(networkPathBehind(runner, 'Z:\\bookforge\\x.whl'), '\\\\TITAN\\iO\\bookforge\\x.whl');
  assert.equal(networkPathBehind(runner, 'C:\\local\\x.whl'), null);
  assert.equal(networkPathBehind(runner, 'C:\\nowhere\\x.whl'), null);
});

test('guestPathFor: refuses the mapped drive by name, maps the local one', async () => {
  const runner = new FakeRunner({ files: { 'C:\\local\\x.whl': '' }, realpaths: { 'Z:\\bookforge\\x.whl': '\\\\TITAN\\iO\\bookforge\\x.whl' } }, []);
  const r = await refusal(Promise.resolve().then(() => guestPathFor(runner, 'Z:\\bookforge\\x.whl')));
  assert.equal(r.code, 'network_path');
  assert.match(r.message, /TITAN\\iO/);
  assert.match(r.message, /fixed drives only/);
  assert.equal(guestPathFor(runner, 'C:\\local\\x.whl'), '/mnt/c/local/x.whl');
});

test('shellQuote: single quotes, with the one character they cannot hold spliced back', () => {
  assert.equal(shellQuote('/mnt/c/Users/Some One/x'), "'/mnt/c/Users/Some One/x'");
  assert.equal(shellQuote("it's"), "'it'\\''s'");
});

test('guestUnpackArgv: the distro\'s own tar reads the archive through /mnt, never \\\\wsl$', () => {
  const runner = new FakeRunner({ files: { 'C:\\dl\\env.tar.gz': '' } }, []);
  assert.deepEqual(guestUnpackArgv(runner, 'Ubuntu', 'C:\\dl\\env.tar.gz', '/home/owen/.crucible/envs/llm'), [
    'wsl.exe', '-d', 'Ubuntu', '--exec', 'bash', '-c',
    "mkdir -p '/home/owen/.crucible/envs/llm' && exec tar -xzf '/mnt/c/dl/env.tar.gz' -C '/home/owen/.crucible/envs/llm'",
  ]);
});
