/**
 * PHASE14 4c's table: every row detected, every row's answer, and the rule that
 * a facts probe does not reach the internet.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { detectWslState, elevatedArgv, probeArgv, wslStates } from '../src/index.js';
import { FakeRunner, WSL_LIST, WSL_LIST_WITH_CRUCIBLE } from './fake.js';

const INPUTS = { release: '0.6.0' };
const STATUS_OK = 'WSL version: 2.3.26.0\r\nKernel version: 5.15.167.4-1\r\nDefault Version: 2\r\n';
const MARKED = '# crucible-rootfs\n[boot]\nsystemd=true\n';
const P = (key: Parameters<typeof probeArgv>[0], inputs = INPUTS): string[] => probeArgv(key, inputs);

test('the table is total, ordered deepest-cause-first, and every row carries a sentence and an action', () => {
  const rows = wslStates({ release: '0.6.0', appDistro: 'Ubuntu', requiredBytes: 1, checkNetwork: true, guestUser: 'crucible' });
  assert.deepEqual(rows.map((row) => row.code), [
    'virtualization_disabled',
    'wsl_missing',
    'wsl1_only',
    'no_crucible_distro',
    'distro_not_systemd',
    'foreign_distro_not_systemd',
    'guest_no_network',
    'pack_disk',
    'linger_unreadable',
    'wsl_ready',
  ]);
  const empty = { code: 0, stdout: '', stderr: '', failure: null };
  const seen = { results: {}, distros: [] };
  for (const row of rows) {
    assert.ok(row.sentence(empty, seen).length > 20, `${row.code} has a sentence a person can read`);
    assert.ok(['run', 'run-elevated', 'instruct', 'link'].includes(row.action(empty, seen).kind), `${row.code} has an action`);
  }
  assert.equal(rows[rows.length - 1]?.means(empty, seen), true, 'the last row matches anything: the table must be total');
});

test('virtualization off in the firmware is detected BEFORE "WSL is not installed"', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: P('wsl-status'), code: 1, stderr: 'Error: 0x80370102 The virtual machine could not be started because a required feature is not installed.' },
  ]);
  const state = await detectWslState(INPUTS, runner);
  runner.assertDrained();
  assert.equal(state.code, 'virtualization_disabled');
  assert.match(state.sentence, /Windows cannot start a virtual machine/);
  assert.equal(state.action.kind, 'instruct');
  assert.match((state.action as { text: string }).text, /Intel VT-x \(Intel\) or SVM Mode \(AMD\)/);
});

test('no wsl.exe is one probe and an ELEVATED action the app runs, never this package', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [{ argv: P('wsl-status'), failure: 'spawn wsl.exe ENOENT' }]);
  const state = await detectWslState(INPUTS, runner);
  runner.assertDrained();
  assert.equal(state.code, 'wsl_missing');
  assert.deepEqual(state.action, { kind: 'run-elevated', argv: ['wsl.exe', '--install', '--no-distribution'] });
  assert.deepEqual(elevatedArgv(state.action), [
    'powershell.exe',
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-Command',
    "Start-Process -Verb RunAs -Wait -FilePath 'wsl.exe' -ArgumentList '--install','--no-distribution'",
  ]);
  // Nothing was elevated by the detection itself.
  assert.equal(runner.calls.some((call) => call.argv[0] === 'powershell.exe'), false);
});

test('elevatedArgv refuses to dress up an action that is not elevated', () => {
  assert.throws(() => elevatedArgv({ kind: 'run', argv: ['wsl.exe'] }), /not an elevated action/);
});

test('WSL1 as the default version is ours to fix', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [{ argv: P('wsl-status'), stdout: 'WSL version: 1.0\r\nDefault Version: 1\r\n' }]);
  const state = await detectWslState(INPUTS, runner);
  assert.equal(state.code, 'wsl1_only');
  assert.deepEqual(state.action, { kind: 'run', argv: ['wsl.exe', '--set-default-version', '2'] });
});

test('a healthy WSL with no crucible distro stops at that row and probes nothing further', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: P('wsl-status'), stdout: STATUS_OK },
    { argv: P('wsl-list'), stdout: WSL_LIST },
  ]);
  const state = await detectWslState(INPUTS, runner);
  runner.assertDrained();
  assert.equal(state.code, 'no_crucible_distro');
  assert.match(state.sentence, /touches nothing you already have/);
});

test('our distro without systemd is repaired without asking; somebody else\'s is not', async () => {
  const ours = new FakeRunner({ platform: 'win32' }, [
    { argv: P('wsl-status'), stdout: STATUS_OK },
    { argv: P('wsl-list'), stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: P('wsl-conf'), stdout: '[automount]\nenabled=true\n' },
  ]);
  const a = await detectWslState(INPUTS, ours);
  ours.assertDrained();
  assert.equal(a.code, 'distro_not_systemd');
  assert.match(a.sentence, /It is ours: this is repaired without asking/);

  const withApp = { release: '0.6.0', appDistro: 'Ubuntu' };
  const theirs = new FakeRunner({ platform: 'win32' }, [
    { argv: P('wsl-status', withApp), stdout: STATUS_OK },
    { argv: P('wsl-list', withApp), stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: P('wsl-conf', withApp), stdout: MARKED },
    { argv: P('app-distro-conf', withApp), stdout: '[automount]\nenabled=true\n' },
  ]);
  const b = await detectWslState(withApp, theirs);
  theirs.assertDrained();
  assert.equal(b.code, 'foreign_distro_not_systemd');
  assert.match(b.sentence, /will not write to a distribution you chose/);
  assert.equal(b.action.kind, 'instruct');
});

test('the network row is probed ONLY when the caller says it is about to download', async () => {
  const quiet = new FakeRunner({ platform: 'win32' }, [
    { argv: P('wsl-status'), stdout: STATUS_OK },
    { argv: P('wsl-list'), stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: P('wsl-conf'), stdout: MARKED },
    { argv: P('guest-root'), stdout: '0\n' },
  ]);
  assert.equal((await detectWslState(INPUTS, quiet)).code, 'wsl_ready');
  quiet.assertDrained();
  assert.equal(quiet.calls.some((call) => call.argv.includes('curl')), false, 'reading a machine\'s facts does not reach the internet');

  const inputs = { release: '0.6.0', checkNetwork: true };
  const offline = new FakeRunner({ platform: 'win32' }, [
    { argv: P('wsl-status', inputs), stdout: STATUS_OK },
    { argv: P('wsl-list', inputs), stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: P('wsl-conf', inputs), stdout: MARKED },
    { argv: P('guest-network', inputs), code: 6, stderr: 'curl: (6) Could not resolve host: github.com' },
  ]);
  const state = await detectWslState(inputs, offline);
  assert.equal(state.code, 'guest_no_network');
  assert.equal(state.action.kind, 'link');
  assert.match(state.sentence, /A VPN or a proxy on this machine usually explains it/);
});

test('the disk row is the numbers, and only when a number was given', async () => {
  const inputs = { release: '0.6.0', requiredBytes: 9 * 1024 ** 3 };
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: P('wsl-status', inputs), stdout: STATUS_OK },
    { argv: P('wsl-list', inputs), stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: P('wsl-conf', inputs), stdout: MARKED },
    { argv: P('guest-disk', inputs), stdout: '2000000\n' },
  ]);
  const state = await detectWslState(inputs, runner);
  runner.assertDrained();
  assert.equal(state.code, 'pack_disk');
  assert.match(state.sentence, /needs 9\.0 GiB free and the "crucible" distribution has 1\.9 GiB/);
  assert.match(state.sentence, /Nothing has been downloaded/);
});

test('a distro that will not give root is the one hand-over, with the exact loginctl line', async () => {
  const inputs = { release: '0.6.0', guestUser: 'crucible' };
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: P('wsl-status', inputs), stdout: STATUS_OK },
    { argv: P('wsl-list', inputs), stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: P('wsl-conf', inputs), stdout: MARKED },
    { argv: P('guest-root', inputs), code: 1, stderr: 'wsl: root is not available in this distribution' },
  ]);
  const state = await detectWslState(inputs, runner);
  assert.equal(state.code, 'linger_unreadable');
  assert.match((state.action as { text: string }).text, /sudo loginctl enable-linger crucible/);
});

test('a machine with nothing wrong answers wsl_ready rather than nothing', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: P('wsl-status'), stdout: STATUS_OK },
    { argv: P('wsl-list'), stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: P('wsl-conf'), stdout: MARKED },
    { argv: P('guest-root'), stdout: '0\n' },
  ]);
  const state = await detectWslState(INPUTS, runner);
  assert.equal(state.code, 'wsl_ready');
  assert.equal(state.action.kind, 'instruct');
});
