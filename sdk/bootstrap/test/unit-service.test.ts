/** `ensureRunning()` — a no-op when up, `crucible service start` when not, never a spawn. */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { ensureRunning, parseServiceStatus, DEFAULT_CONDA_ROOTS } from '../src/index.js';
import { interpreterScript } from '../src/host.js';
import { CRUCIBLE_BIN, FakeRunner, INTERPRETER_OK, refusal } from './fake.js';

const W = (...argv: string[]): string[] => ['wsl.exe', '-d', 'Ubuntu', '--exec', ...argv];
const INTERP = { argv: W('bash', '-c', interpreterScript(DEFAULT_CONDA_ROOTS)), stdout: INTERPRETER_OK };
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

test('already running: one status read, no start, started false, linger reported', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [INTERP, { argv: STATUS, code: 0, stdout: status() }]);
  const result = await ensureRunning({ distro: 'Ubuntu' }, runner);
  runner.assertDrained();
  assert.deepEqual(result, {
    running: true, pid: 123126, mechanism: 'systemd', definition: '/home/owen/.config/systemd/user/crucible.service', linger: true, enableLinger: null, started: false,
  });
});

test('installed and stopped: `crucible service start`, then status again, started true', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    INTERP,
    { argv: STATUS, code: 1, stdout: status({ running: false, pid: null, detail: 'inactive/dead, unit file enabled' }) },
    { argv: START, stdout: 'started crucible.service\n' },
    { argv: STATUS, code: 0, stdout: status({ pid: 4242 }) },
  ]);
  const result = await ensureRunning({ distro: 'Ubuntu' }, runner);
  runner.assertDrained();
  assert.equal(result.started, true);
  assert.equal(result.pid, 4242);
});

test('linger off is a fact beside success, with the command that is root\'s to run', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [INTERP, { argv: STATUS, stdout: status({ linger: false }) }]);
  const result = await ensureRunning({ distro: 'Ubuntu' }, runner);
  assert.equal(result.running, true);
  assert.equal(result.linger, false);
  assert.equal(result.enableLinger, 'sudo loginctl enable-linger "$USER"');
});

test('launchd has no linger question: null, no command', async () => {
  const runner = new FakeRunner({ platform: 'darwin' }, [
    { argv: ['bash', '-c', interpreterScript(DEFAULT_CONDA_ROOTS)], stdout: 'home=/Users/owen\nconda=/Users/owen/miniforge3\npython=/Users/owen/miniforge3/envs/crucible/bin/python\nversion=Python 3.11.13\n' },
    { argv: ['/Users/owen/miniforge3/envs/crucible/bin/crucible', 'service', 'status', '--json'], stdout: status({ mechanism: 'launchd', definition: '/Users/owen/Library/LaunchAgents/com.crucible.serve.plist', linger: null, pid: 31426 }) },
  ]);
  const result = await ensureRunning({}, runner);
  assert.equal(result.mechanism, 'launchd');
  assert.equal(result.linger, null);
  assert.equal(result.enableLinger, null);
  assert.equal(result.pid, 31426);
});

test('not installed: service_not_installed naming the definition and `crucible service install`', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [INTERP, { argv: STATUS, code: 1, stdout: status({ installed: false, running: false, pid: null, detail: 'inactive/dead, unit file not-found' }) }]);
  const r = await refusal(ensureRunning({ distro: 'Ubuntu' }, runner));
  assert.equal(r.code, 'service_not_installed');
  assert.match(r.message, /\/home\/owen\/\.config\/systemd\/user\/crucible\.service does not exist/);
  assert.equal(r.command, `${CRUCIBLE_BIN} service install`);
  runner.assertDrained();
});

test('start refusing is service_failed with its words and where the logs are', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    INTERP,
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
    INTERP,
    { argv: STATUS, code: 1, stdout: status({ running: false, pid: null }) },
    { argv: START, stdout: 'started crucible.service\n' },
    { argv: STATUS, code: 1, stdout: status({ running: false, pid: null, detail: 'failed/failed, unit file enabled' }) },
  ]);
  const r = await refusal(ensureRunning({ distro: 'Ubuntu' }, runner));
  assert.equal(r.code, 'service_failed');
  assert.match(r.message, /started and is not running: failed\/failed/);
});

test('status with no config behind it is no_local_config; any other non-status answer is service_failed', async () => {
  const noConfig = new FakeRunner({ platform: 'win32' }, [INTERP, { argv: STATUS, code: 1, stderr: 'crucible: no config at /home/owen/.crucible/config.toml — run `crucible init`' }]);
  const n = await refusal(ensureRunning({ distro: 'Ubuntu' }, noConfig));
  assert.equal(n.code, 'no_local_config');

  const garbage = new FakeRunner({ platform: 'win32' }, [INTERP, { argv: STATUS, code: 1, stderr: 'crucible: this host detects backend cuda-linux, but config was initialised for mlx-darwin' }]);
  const g = await refusal(ensureRunning({ distro: 'Ubuntu' }, garbage));
  assert.equal(g.code, 'service_failed');
  assert.match(g.message, /initialised for mlx-darwin/);

  const dead = new FakeRunner({ platform: 'win32' }, [INTERP, { argv: STATUS, failure: 'wsl.exe did not answer within 60s' }]);
  const d = await refusal(ensureRunning({ distro: 'Ubuntu' }, dead));
  assert.equal(d.code, 'service_failed');
});

test('{home} travels as env CRUCIBLE_HOME= on every verb', async () => {
  const E = (...argv: string[]): string[] => W('env', 'CRUCIBLE_HOME=/srv/c', ...argv);
  const runner = new FakeRunner({ platform: 'win32' }, [
    INTERP,
    { argv: E(CRUCIBLE_BIN, 'service', 'status', '--json'), code: 1, stdout: status({ running: false, pid: null }) },
    { argv: E(CRUCIBLE_BIN, 'service', 'start') },
    { argv: E(CRUCIBLE_BIN, 'service', 'status', '--json'), stdout: status() },
  ]);
  await ensureRunning({ distro: 'Ubuntu', home: '/srv/c' }, runner);
  runner.assertDrained();
});

test('no interpreter: the named refusal, nothing else asked', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [{ argv: INTERP.argv, stdout: 'home=/home/owen\n' }]);
  const r = await refusal(ensureRunning({ distro: 'Ubuntu' }, runner));
  assert.equal(r.code, 'no_conda');
  runner.assertDrained();
});

test('win32 without a distro is no_wsl_distro', async () => {
  const r = await refusal(ensureRunning({}, new FakeRunner({ platform: 'win32' }, [])));
  assert.equal(r.code, 'no_wsl_distro');
});
