/** `ensureRunning()` — a no-op when up, `crucible service start` when not, never a spawn. */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ensureRunning, parseServiceStatus } from '../src/index.js';
import { guestProbeScript } from '../src/runtime.js';
import { CRUCIBLE_BIN, FakeRunner, GUEST_BARE, GUEST_INSTALLED, refusal, WSL_LIST } from './fake.js';

const W = (...argv: string[]): string[] => ['wsl.exe', '-d', 'Ubuntu', '--exec', ...argv];
/** win32: which distro, then what that distro has. There is no interpreter hunt any more. */
const LIST = { argv: ['wsl.exe', '-l', '-v'], stdout: WSL_LIST };
const PROBE = { argv: W('bash', '-c', guestProbeScript(undefined)), stdout: GUEST_INSTALLED };
const STATUS = W(CRUCIBLE_BIN, 'service', 'status', '--json');
const START = W(CRUCIBLE_BIN, 'service', 'start');

function status(overrides: Partial<Record<string, unknown>> = {}): string {
  return JSON.stringify({
    mechanism: 'systemd',
    definition: '/home/owen/.config/systemd/user/crucible.service',
    installed: true,
    running: true,
    pid: 123126,
    detail: 'active/running, unit file enabled',
    linger: true,
    ...overrides,
  });
}

test('parseServiceStatus: the shape crucible/service.py prints, and nothing looser', () => {
  assert.deepEqual(parseServiceStatus(status()), {
    mechanism: 'systemd', definition: '/home/owen/.config/systemd/user/crucible.service', installed: true, running: true, pid: 123126, detail: 'active/running, unit file enabled', linger: true,
  });
  assert.equal(parseServiceStatus(status({ pid: null, linger: null }))?.pid, null);
  assert.equal(parseServiceStatus('crucible: no config at /home/owen/.crucible/config.toml'), null);
  assert.equal(parseServiceStatus('{}'), null);
  assert.equal(parseServiceStatus(status({ mechanism: 'initd' })), null);
  assert.equal(parseServiceStatus(status({ pid: '123' })), null);
  assert.equal(parseServiceStatus('[]'), null);
  assert.equal(parseServiceStatus('null'), null);
});

test('already running: one status read, no start, started false', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [LIST, PROBE, { argv: STATUS, code: 0, stdout: status() }]);
  const result = await ensureRunning({ distro: 'Ubuntu' }, runner);
  runner.assertDrained();
  assert.deepEqual(result, {
    running: true,
    pid: 123126,
    mechanism: 'systemd',
    definition: '/home/owen/.config/systemd/user/crucible.service',
    linger: null,
    enableLinger: null,
    started: false,
  });
});

test('installed and stopped: `crucible service start`, then status again, started true', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    PROBE,
    { argv: STATUS, code: 1, stdout: status({ running: false, pid: null, detail: 'inactive/dead, unit file enabled' }) },
    { argv: START, stdout: 'started crucible.service\n' },
    { argv: STATUS, code: 0, stdout: status({ pid: 4242 }) },
  ]);
  const result = await ensureRunning({ distro: 'Ubuntu' }, runner);
  runner.assertDrained();
  assert.equal(result.started, true);
  assert.equal(result.pid, 4242);
});

test('launchd has no linger question: null, no command', async () => {
  const runner = new FakeRunner({ platform: 'darwin' }, [
    { argv: ['bash', '-c', guestProbeScript(undefined)], stdout: 'home=/Users/owen/.crucible\nuser=owen\nfree_kib=9000000\ncrucible=/Users/owen/.crucible/server/bin/crucible\nversion=crucible 0.6.0\n' },
    { argv: ['/Users/owen/.crucible/server/bin/crucible', 'service', 'status', '--json'], stdout: status({ mechanism: 'launchd', definition: '/Users/owen/Library/LaunchAgents/com.crucible.serve.plist', linger: null, pid: 31426 }) },
  ]);
  const result = await ensureRunning({}, runner);
  assert.equal(result.mechanism, 'launchd');
  assert.equal(result.linger, null);
  assert.equal(result.enableLinger, null);
  assert.equal(result.pid, 31426);
});

test('native linux: the fact is REPORTED with the sudo line, and nothing is run', async () => {
  const runner = new FakeRunner({ platform: 'linux', homedir: '/home/owen' }, [
    { argv: ['bash', '-c', guestProbeScript(undefined)], stdout: 'home=/home/owen/.crucible\nuser=owen\nfree_kib=9000000\ncrucible=/home/owen/.crucible/server/bin/crucible\nversion=crucible 0.6.0\n' },
    { argv: ['/home/owen/.crucible/server/bin/crucible', 'service', 'status', '--json'], stdout: status({ linger: false }) },
  ]);
  const result = await ensureRunning({}, runner);
  runner.assertDrained();
  assert.equal(result.linger, false);
  assert.equal(result.enableLinger, 'sudo loginctl enable-linger "$USER"');
});

test('not installed: service_not_installed naming the definition and `crucible service install`', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [LIST, PROBE, { argv: STATUS, code: 1, stdout: status({ installed: false, running: false, pid: null, detail: 'inactive/dead, unit file not-found' }) }]);
  const r = await refusal(ensureRunning({ distro: 'Ubuntu' }, runner));
  assert.equal(r.code, 'service_not_installed');
  assert.match(r.message, /\/home\/owen\/\.config\/systemd\/user\/crucible\.service does not exist/);
  assert.equal(r.command, `${CRUCIBLE_BIN} service install`);
  runner.assertDrained();
});

test('start refusing is service_failed with its words and where the logs are', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    PROBE,
    { argv: STATUS, code: 1, stdout: status({ running: false, pid: null }) },
    { argv: START, code: 1, stderr: 'crucible: systemd would not start crucible.service: Job for crucible.service failed' },
  ]);
  const r = await refusal(ensureRunning({ distro: 'Ubuntu' }, runner));
  assert.equal(r.code, 'service_failed');
  assert.match(r.message, /exited 1.*Job for crucible\.service failed/);
  assert.equal(r.command, 'journalctl --user -u crucible.service -n 100 --no-pager');
});

test('started and still not running is service_failed with the status detail', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    PROBE,
    { argv: STATUS, code: 1, stdout: status({ running: false, pid: null }) },
    { argv: START, stdout: 'started crucible.service\n' },
    { argv: STATUS, code: 1, stdout: status({ running: false, pid: null, detail: 'failed/failed, unit file enabled' }) },
  ]);
  const r = await refusal(ensureRunning({ distro: 'Ubuntu' }, runner));
  assert.equal(r.code, 'service_failed');
  assert.match(r.message, /started and is not running: failed\/failed/);
});

test('status with no config behind it is no_local_config; any other non-status answer is service_failed', async () => {
  const noConfig = new FakeRunner({ platform: 'win32' }, [LIST, PROBE, { argv: STATUS, code: 1, stderr: 'crucible: no config at /home/owen/.crucible/config.toml — run `crucible init`' }]);
  const n = await refusal(ensureRunning({ distro: 'Ubuntu' }, noConfig));
  assert.equal(n.code, 'no_local_config');

  const garbage = new FakeRunner({ platform: 'win32' }, [LIST, PROBE, { argv: STATUS, code: 1, stderr: 'crucible: this host detects backend cuda-linux, but config was initialised for mlx-darwin' }]);
  const g = await refusal(ensureRunning({ distro: 'Ubuntu' }, garbage));
  assert.equal(g.code, 'service_failed');
  assert.match(g.message, /initialised for mlx-darwin/);

  const dead = new FakeRunner({ platform: 'win32' }, [LIST, PROBE, { argv: STATUS, failure: 'wsl.exe did not answer within 60s' }]);
  const d = await refusal(ensureRunning({ distro: 'Ubuntu' }, dead));
  assert.equal(d.code, 'service_failed');
});

test('{home} travels as env CRUCIBLE_HOME= on every verb', async () => {
  const E = (...argv: string[]): string[] => W('env', 'CRUCIBLE_HOME=/srv/c', ...argv);
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    { argv: W('bash', '-c', guestProbeScript('/srv/c')), stdout: 'home=/srv/c\nuser=owen\nfree_kib=9000000\ncrucible=/home/owen/.crucible/server/bin/crucible\nversion=crucible 0.6.0\n' },
    { argv: E(CRUCIBLE_BIN, 'service', 'status', '--json'), code: 1, stdout: status({ running: false, pid: null }) },
    { argv: E(CRUCIBLE_BIN, 'service', 'start') },
    { argv: E(CRUCIBLE_BIN, 'service', 'status', '--json'), stdout: status() },
  ]);
  await ensureRunning({ distro: 'Ubuntu', home: '/srv/c' }, runner);
  runner.assertDrained();
});

test('no server runtime: the named refusal, nothing else asked', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [LIST, { argv: PROBE.argv, stdout: GUEST_BARE }]);
  const r = await refusal(ensureRunning({ distro: 'Ubuntu' }, runner));
  assert.equal(r.code, 'no_server_runtime');
  assert.match(r.message, /\/home\/owen\/\.crucible\/server\/bin\/crucible/);
  assert.match(r.message, /install\(\) downloads the pinned interpreter/);
  runner.assertDrained();
});

test('win32 with no crucible distro and no {distro} is no_wsl_distro', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [LIST]);
  const r = await refusal(ensureRunning({}, runner));
  assert.equal(r.code, 'no_wsl_distro');
  assert.match(r.message, /there is no "crucible" distro on this machine and no distro was named/);
});
