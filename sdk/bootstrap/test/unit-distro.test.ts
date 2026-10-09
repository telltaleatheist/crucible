/** The Crucible-owned WSL distro: which one is `local`. Importing it is the Windows host's (crucible/host/installer.py). */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { CRUCIBLE_DISTRO, crucibleAppData, resolveDistro } from '../src/index.js';
import { FakeRunner, refusal, WSL_LIST, WSL_LIST_WITH_CRUCIBLE } from './fake.js';

const LIST = ['wsl.exe', '-l', '-v'];
const HAS_CONFIG = (distro: string): string[] => ['wsl.exe', '-d', distro, '--exec', 'bash', '-c', 'test -f "${CRUCIBLE_HOME:-$HOME/.crucible}/config.toml"'];
const ENV = { LOCALAPPDATA: 'C:\\Users\\owen\\AppData\\Local' };
const INSTALL_DIR = 'C:\\Users\\owen\\AppData\\Local\\Crucible\\wsl';

test('crucibleAppData reads LOCALAPPDATA and refuses to assemble one from a username', async () => {
  assert.equal(crucibleAppData(new FakeRunner({ platform: 'win32', env: ENV }, []), 'wsl'), INSTALL_DIR);
  const r = await refusal(Promise.resolve().then(() => crucibleAppData(new FakeRunner({ platform: 'win32', env: {} }, []), 'wsl')));
  assert.equal(r.code, 'host_unresponsive');
  assert.match(r.message, /LOCALAPPDATA is not set/);
});

test('resolveDistro: ours wins, the app\'s setting is the fallback, and neither is no_wsl_distro', async () => {
  const ours = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: HAS_CONFIG('Ubuntu'), code: 1 },
  ]);
  assert.equal(await resolveDistro(ours, { distro: 'Ubuntu' }), CRUCIBLE_DISTRO);

  const theirs = new FakeRunner({ platform: 'win32' }, [{ argv: LIST, stdout: WSL_LIST }]);
  assert.equal(await resolveDistro(theirs, { distro: 'Ubuntu' }), 'Ubuntu');

  const nothing = new FakeRunner({ platform: 'win32' }, [{ argv: LIST, stdout: WSL_LIST }]);
  const r = await refusal(resolveDistro(nothing, {}));
  assert.equal(r.code, 'no_wsl_distro');
});

test('resolveDistro: both holding a config is two_local_crucibles, with both ways out named', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: HAS_CONFIG('Ubuntu'), code: 0 },
  ]);
  const r = await refusal(resolveDistro(runner, { distro: 'Ubuntu' }));
  assert.equal(r.code, 'two_local_crucibles');
  assert.match(r.message, /\{distro: "Ubuntu", exact: true\}/);
  assert.match(r.message, /\{distro: "crucible", exact: true\}/);
});

test('resolveDistro: {exact} asks wsl.exe nothing at all', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, []);
  assert.equal(await resolveDistro(runner, { distro: 'Ubuntu', exact: true }), 'Ubuntu');
  assert.deepEqual(runner.calls, []);
  const r = await refusal(resolveDistro(runner, { exact: true }));
  assert.equal(r.code, 'no_wsl_distro');
});

test('resolveDistro: a distro that will not answer is wsl_read_failed, never a silent "no config"', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: HAS_CONFIG('Ubuntu'), failure: 'wsl.exe did not answer within 60s' },
  ]);
  const r = await refusal(resolveDistro(runner, { distro: 'Ubuntu' }));
  assert.equal(r.code, 'wsl_read_failed');
});
