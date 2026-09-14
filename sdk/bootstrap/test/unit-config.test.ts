/**
 * `readLocalConfig` — BookForge's `local.ts` rule, every named state, on both
 * sides of the WSL boundary.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { connectHost, localConfigPath, parseLocalConfig, readLocalConfig } from '../src/index.js';
import { FakeRunner, GUEST_CONFIG, refusal, WSL_LIST, WSL_LIST_WITH_CRUCIBLE } from './fake.js';

test('connectHost: 0.0.0.0, :: and empty bind to loopback; a specific address is used as written', () => {
  assert.equal(connectHost('0.0.0.0'), '127.0.0.1');
  assert.equal(connectHost('::'), '127.0.0.1');
  assert.equal(connectHost(''), '127.0.0.1');
  assert.equal(connectHost('127.0.0.1'), '127.0.0.1');
  assert.equal(connectHost('100.64.0.7'), '100.64.0.7');
  assert.equal(connectHost('::1'), '::1');
});

test('localConfigPath: $CRUCIBLE_HOME, else ~/.crucible, else the home the caller named', () => {
  // POSIX always: the file is only read natively on darwin and linux.
  assert.equal(localConfigPath({}, '/home/owen'), '/home/owen/.crucible/config.toml');
  assert.equal(localConfigPath({ CRUCIBLE_HOME: '/srv/c' }, '/home/owen'), '/srv/c/config.toml');
  assert.equal(localConfigPath({ CRUCIBLE_HOME: '' }, '/home/owen'), '/home/owen/.crucible/config.toml');
  assert.equal(localConfigPath({ CRUCIBLE_HOME: '/srv/c' }, '/home/owen', '/named'), '/named/config.toml');
});

test('parseLocalConfig: the guest config, bind→connect mapped, token verbatim', () => {
  const config = parseLocalConfig(GUEST_CONFIG.replace('host = "127.0.0.1"', 'host = "0.0.0.0"'), 'Ubuntu:/home/owen/.crucible/config.toml', 'wsl');
  assert.deepEqual(config, {
    name: 'crucible@owens-pc-wsl',
    url: 'http://127.0.0.1:7100',
    token: 'test-token-not-a-secret',
    configPath: 'Ubuntu:/home/owen/.crucible/config.toml',
    via: 'wsl',
  });
});

test('parseLocalConfig: an IPv6 bind is bracketed in the URL', () => {
  const config = parseLocalConfig('[server]\nname = "n"\nhost = "fd7a::1"\nport = 7100\n[auth]\ntoken = "t"\n', 'p', 'file');
  assert.equal(config.url, 'http://[fd7a::1]:7100');
});

for (const [label, text, code, pattern] of [
  ['not TOML', 'this is not = = toml', 'config_unreadable', /not TOML/],
  ['no [server]', '[auth]\ntoken = "t"\n', 'config_missing_key', /missing the \[server\] section/],
  ['no server.port', '[server]\nname = "n"\nhost = "h"\n[auth]\ntoken = "t"\n', 'config_missing_key', /missing server\.port/],
  ['a string port', '[server]\nname = "n"\nhost = "h"\nport = "7100"\n[auth]\ntoken = "t"\n', 'config_missing_key', /server\.port must be an integer, got string/],
  ['no [auth]', '[server]\nname = "n"\nhost = "h"\nport = 7100\n', 'config_missing_key', /missing the \[auth\] section/],
  ['an empty token', '[server]\nname = "n"\nhost = "h"\nport = 7100\n[auth]\ntoken = "  "\n', 'config_missing_key', /auth\.token is empty/],
] as const) {
  test(`parseLocalConfig refuses ${label} as ${code}, naming crucible init --force`, async () => {
    const r = await refusal(Promise.resolve().then(() => parseLocalConfig(text, 'cfg', 'file')));
    assert.equal(r.code, code);
    assert.match(r.message, pattern);
    assert.equal(r.command, 'crucible init --force');
  });
}

// --------------------------------------------------------------- native side

test('native: the file is read directly and via says so', async () => {
  const runner = new FakeRunner({ platform: 'darwin', homedir: '/Users/owen', files: { '/Users/owen/.crucible/config.toml': GUEST_CONFIG } }, []);
  const config = await readLocalConfig({}, runner);
  assert.equal(config.via, 'file');
  assert.equal(config.url, 'http://127.0.0.1:7100');
  assert.equal(config.configPath, '/Users/owen/.crucible/config.toml');
  assert.deepEqual(runner.calls, []);
});

test('native: no file is no_local_config, a named state', async () => {
  const runner = new FakeRunner({ platform: 'linux', homedir: '/home/owen' }, []);
  const r = await refusal(readLocalConfig({}, runner));
  assert.equal(r.code, 'no_local_config');
  assert.match(r.message, /\/home\/owen\/\.crucible\/config\.toml does not exist/);
});

test('native: a file that cannot be read is config_unreadable', async () => {
  const runner = new FakeRunner({ platform: 'linux', homedir: '/home/owen', unreadable: ['/home/owen/.crucible/config.toml'] }, []);
  const r = await refusal(readLocalConfig({}, runner));
  assert.equal(r.code, 'config_unreadable');
  assert.match(r.message, /EACCES/);
});

test('native: {home} names the CRUCIBLE_HOME to read', async () => {
  const runner = new FakeRunner({ platform: 'darwin', homedir: '/Users/owen', files: { '/srv/crucible/config.toml': GUEST_CONFIG } }, []);
  const config = await readLocalConfig({ home: '/srv/crucible' }, runner);
  assert.equal(config.configPath, '/srv/crucible/config.toml');
});

test('an unsupported platform is refused by name before anything runs', async () => {
  const runner = new FakeRunner({ platform: 'freebsd' }, []);
  const r = await refusal(readLocalConfig({}, runner));
  assert.equal(r.code, 'unsupported_platform');
});

// ------------------------------------------------------------------ win32

const READ_SCRIPT = 'p="${CRUCIBLE_HOME:-$HOME/.crucible}/config.toml"; if [ ! -f "$p" ]; then echo "$p" >&2; exit 3; fi; echo "$p"; cat "$p"';
/**
 * WHICH DISTRO comes first now (PHASE14 4b): the read asks `wsl -l -v`, and
 * the `crucible` distro wins over the app's setting when it exists.
 */
const LIST = { argv: ['wsl.exe', '-l', '-v'], stdout: WSL_LIST };

test('win32: read through wsl.exe -d <distro> --exec bash -c, resolving $CRUCIBLE_HOME in the guest', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    { argv: ['wsl.exe', '-d', 'Ubuntu', '--exec', 'bash', '-c', READ_SCRIPT], stdout: `/home/owen/.crucible/config.toml\n${GUEST_CONFIG}` },
  ]);
  const config = await readLocalConfig({ distro: 'Ubuntu' }, runner);
  assert.equal(config.via, 'wsl');
  assert.equal(config.configPath, 'Ubuntu:/home/owen/.crucible/config.toml');
  assert.equal(config.name, 'crucible@owens-pc-wsl');
  assert.equal(config.token, 'test-token-not-a-secret');
  runner.assertDrained();
});

test('win32: {home} is quoted into the script instead of the guest\'s $CRUCIBLE_HOME', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    {
      argv: ['wsl.exe', '-d', 'Ubuntu', '--exec', 'bash', '-c', "p='/srv/crucible'/config.toml; if [ ! -f \"$p\" ]; then echo \"$p\" >&2; exit 3; fi; echo \"$p\"; cat \"$p\""],
      stdout: `/srv/crucible/config.toml\n${GUEST_CONFIG}`,
    },
  ]);
  const config = await readLocalConfig({ distro: 'Ubuntu', home: '/srv/crucible' }, runner);
  assert.equal(config.configPath, 'Ubuntu:/srv/crucible/config.toml');
});

test('win32: no crucible distro and no distro named is no_wsl_distro, and no config is read', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [LIST]);
  const r = await refusal(readLocalConfig({}, runner));
  assert.equal(r.code, 'no_wsl_distro');
  assert.match(r.message, /there is no "crucible" distro/);
  runner.assertDrained();

  const blank = new FakeRunner({ platform: 'win32' }, [LIST]);
  assert.equal((await refusal(readLocalConfig({ distro: '  ' }, blank))).code, 'no_wsl_distro');
});

test('win32: the crucible distro wins over the app\'s setting, and both holding a config is refused', async () => {
  const ours = new FakeRunner({ platform: 'win32' }, [
    { argv: ['wsl.exe', '-l', '-v'], stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: ['wsl.exe', '-d', 'Ubuntu', '--exec', 'bash', '-c', 'test -f "${CRUCIBLE_HOME:-$HOME/.crucible}/config.toml"'], code: 1 },
    { argv: ['wsl.exe', '-d', 'crucible', '--exec', 'bash', '-c', READ_SCRIPT], stdout: `/home/crucible/.crucible/config.toml\n${GUEST_CONFIG}` },
  ]);
  const config = await readLocalConfig({ distro: 'Ubuntu' }, ours);
  ours.assertDrained();
  assert.equal(config.configPath, 'crucible:/home/crucible/.crucible/config.toml');

  const both = new FakeRunner({ platform: 'win32' }, [
    { argv: ['wsl.exe', '-l', '-v'], stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: ['wsl.exe', '-d', 'Ubuntu', '--exec', 'bash', '-c', 'test -f "${CRUCIBLE_HOME:-$HOME/.crucible}/config.toml"'], code: 0 },
  ]);
  const r = await refusal(readLocalConfig({ distro: 'Ubuntu' }, both));
  assert.equal(r.code, 'two_local_crucibles');
  assert.match(r.message, /which one is `local` is not guessed here/);
  both.assertDrained();

  // {exact} is the way out, and it does not even list.
  const exact = new FakeRunner({ platform: 'win32' }, [
    { argv: ['wsl.exe', '-d', 'Ubuntu', '--exec', 'bash', '-c', READ_SCRIPT], stdout: `/home/owen/.crucible/config.toml\n${GUEST_CONFIG}` },
  ]);
  assert.equal((await readLocalConfig({ distro: 'Ubuntu', exact: true }, exact)).configPath, 'Ubuntu:/home/owen/.crucible/config.toml');
  exact.assertDrained();
});

test('win32: exit 3 is no_local_config, naming the guest path', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [LIST, { argv: () => true, code: 3, stderr: '/home/owen/.crucible/config.toml\n' }]);
  const r = await refusal(readLocalConfig({ distro: 'Ubuntu' }, runner));
  assert.equal(r.code, 'no_local_config');
  assert.match(r.message, /\/home\/owen\/\.crucible\/config\.toml does not exist inside WSL distro "Ubuntu"/);
});

test('win32: wsl.exe not running at all is wsl_read_failed', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [{ argv: () => true, failure: 'wsl.exe did not answer within 30s' }]);
  const r = await refusal(readLocalConfig({ distro: 'Ubuntu' }, runner));
  assert.equal(r.code, 'wsl_read_failed');
  assert.match(r.message, /did not answer within 30s/);
});

test('win32: any other exit is wsl_read_failed with the guest\'s stderr', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [LIST, { argv: () => true, code: 1, stderr: 'There is no distribution with the supplied name.' }]);
  const r = await refusal(readLocalConfig({ distro: 'Nope' }, runner));
  assert.equal(r.code, 'wsl_read_failed');
  assert.match(r.message, /exit 1.*no distribution with the supplied name/);
});

test('win32: a guest that printed no path line is wsl_read_failed', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [LIST, { argv: () => true, code: 0, stdout: '' }]);
  const r = await refusal(readLocalConfig({ distro: 'Ubuntu' }, runner));
  assert.equal(r.code, 'wsl_read_failed');
  assert.match(r.message, /printed no config path/);
});
