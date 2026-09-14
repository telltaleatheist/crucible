/** The Crucible-owned WSL distro: which one is `local`, and the import that is idempotent. */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { CRUCIBLE_DISTRO, crucibleAppData, ensureDistro, importArgv, resolveDistro, terminateArgv, unregisterArgv, WSL_CONF_MARKER } from '../src/index.js';
import { FakeRunner, refusal, WSL_LIST, WSL_LIST_WITH_CRUCIBLE } from './fake.js';

const LIST = ['wsl.exe', '-l', '-v'];
const HAS_CONFIG = (distro: string): string[] => ['wsl.exe', '-d', distro, '--exec', 'bash', '-c', 'test -f "${CRUCIBLE_HOME:-$HOME/.crucible}/config.toml"'];
const CONF = (distro: string): string[] => ['wsl.exe', '-d', distro, '-u', 'root', '--exec', 'bash', '-c', 'test -f /etc/wsl.conf && cat /etc/wsl.conf || true'];
const MARKED = `${WSL_CONF_MARKER}\n[boot]\nsystemd=true\n[user]\ndefault=crucible\n`;
const ENV = { LOCALAPPDATA: 'C:\\Users\\owen\\AppData\\Local' };
const INSTALL_DIR = 'C:\\Users\\owen\\AppData\\Local\\Crucible\\wsl';
const DOWNLOAD_DIR = 'C:\\Users\\owen\\AppData\\Local\\Crucible\\downloads';
const ROOTFS = `${DOWNLOAD_DIR}\\crucible-rootfs-0.6.0.tar.zst`;
const BASE = 'https://github.com/telltaleatheist/crucible/releases/download/v0.6.0';

test('the argv this file builds: import at version 2, terminate ONE distro, unregister ours', () => {
  assert.deepEqual(importArgv(INSTALL_DIR, ROOTFS), ['wsl.exe', '--import', 'crucible', INSTALL_DIR, ROOTFS, '--version', '2']);
  assert.deepEqual(terminateArgv(), ['wsl.exe', '--terminate', 'crucible']);
  assert.deepEqual(unregisterArgv(), ['wsl.exe', '--unregister', 'crucible']);
  // NEVER --shutdown: a training run and a live server live in the other distro.
  assert.equal(JSON.stringify([importArgv(INSTALL_DIR, ROOTFS), terminateArgv(), unregisterArgv()]).includes('--shutdown'), false);
});

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

test('ensureDistro: a marked distro is used as it is, and nothing is downloaded or terminated', async () => {
  const runner = new FakeRunner({ platform: 'win32', env: ENV }, [
    { argv: LIST, stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: CONF('crucible'), stdout: MARKED },
  ]);
  const outcome = await ensureDistro({ release: '0.6.0' }, runner);
  runner.assertDrained();
  assert.deepEqual(outcome, { name: 'crucible', imported: false, installDir: null, repaired: false, detail: '"crucible" is already imported and marked' });
});

test('ensureDistro: absent — download, verify, import, mark, terminate once', async () => {
  const runner = new FakeRunner({ platform: 'win32', env: ENV }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: ['curl.exe', '-fL', '--retry', '3', '--create-dirs', '-o', ROOTFS, `${BASE}/crucible-rootfs-0.6.0.tar.zst`] },
    { argv: ['curl.exe', '-fsSL', '--retry', '3', `${BASE}/crucible-rootfs-0.6.0.tar.zst.sha256`], stdout: `${'f'.repeat(64)}  crucible-rootfs-0.6.0.tar.zst\n` },
    { argv: ['certutil', '-hashfile', ROOTFS, 'SHA256'], stdout: `SHA256 hash of ${ROOTFS}:\r\n${'f'.repeat(64)}\r\nCertUtil: -hashfile command completed successfully.\r\n` },
    { argv: importArgv(INSTALL_DIR, ROOTFS) },
    { argv: CONF('crucible'), stdout: MARKED },
    { argv: terminateArgv() },
  ]);
  const outcome = await ensureDistro({ release: '0.6.0' }, runner);
  runner.assertDrained();
  assert.equal(outcome.imported, true);
  assert.equal(outcome.installDir, INSTALL_DIR);
  assert.equal(outcome.repaired, false);
  assert.match(outcome.detail, /imported crucible-rootfs-0\.6\.0\.tar\.zst/);
});

test('ensureDistro: a rootfs whose digest is not the release\'s is pack_sha_mismatch, before the import', async () => {
  const runner = new FakeRunner({ platform: 'win32', env: ENV }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: () => true },
    { argv: () => true, stdout: `${'f'.repeat(64)}  x\n` },
    { argv: () => true, stdout: `SHA256 hash:\r\n${'0'.repeat(64)}\r\n` },
  ]);
  const r = await refusal(ensureDistro({ release: '0.6.0' }, runner));
  runner.assertDrained();
  assert.equal(r.code, 'pack_sha_mismatch');
  assert.match(r.message, /Delete .* and run this again/);
});

test('ensureDistro: an image the CALLER named is imported and marked by us, and says whose digest it is', async () => {
  const runner = new FakeRunner({ platform: 'win32', env: ENV }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: (argv) => argv[0] === 'curl.exe' && argv.includes('https://example.invalid/ubuntu.tar.gz') },
    { argv: importArgv(INSTALL_DIR, ROOTFS) },
    { argv: CONF('crucible'), stdout: '' },
    { argv: (argv) => argv[0] === 'wsl.exe' && argv[4] === 'root' && (argv[8] ?? '').includes('cat > /etc/wsl.conf') },
    { argv: terminateArgv() },
  ]);
  const lines: string[] = [];
  const outcome = await ensureDistro({ release: '0.6.0', rootfsUrl: 'https://example.invalid/ubuntu.tar.gz', onLine: (line) => lines.push(line) }, runner);
  runner.assertDrained();
  assert.equal(outcome.repaired, true, 'a stock image has no marker, so the import finishes by writing one');
  assert.ok(lines.some((line) => /digest is the caller's to vouch for/.test(line)));
});

test('ensureDistro: an unmarked distro with a config in it is NOT re-imported over', async () => {
  const runner = new FakeRunner({ platform: 'win32', env: ENV }, [
    { argv: LIST, stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: CONF('crucible'), stdout: '[boot]\nsystemd=true\n' },
    { argv: HAS_CONFIG('crucible'), code: 0 },
  ]);
  const r = await refusal(ensureDistro({ release: '0.6.0' }, runner));
  runner.assertDrained();
  assert.equal(r.code, 'distro_unmarked');
  assert.match(r.message, /would delete a server somebody is using/);
});

test('ensureDistro: an unmarked EMPTY distro is a partial import — unregistered and redone', async () => {
  const runner = new FakeRunner({ platform: 'win32', env: ENV }, [
    { argv: LIST, stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: CONF('crucible'), stdout: '' },
    { argv: HAS_CONFIG('crucible'), code: 1 },
    { argv: unregisterArgv() },
    { argv: (argv) => argv[0] === 'curl.exe' && argv.includes(ROOTFS) },
    { argv: (argv) => argv[0] === 'curl.exe' && argv[4]?.endsWith('.sha256') === true, stdout: `${'f'.repeat(64)}\n` },
    { argv: (argv) => argv[0] === 'certutil', stdout: `hash:\r\n${'f'.repeat(64)}\r\n` },
    { argv: importArgv(INSTALL_DIR, ROOTFS) },
    { argv: CONF('crucible'), stdout: MARKED },
    { argv: terminateArgv() },
  ]);
  const outcome = await ensureDistro({ release: '0.6.0' }, runner);
  runner.assertDrained();
  assert.equal(outcome.imported, true);
});

test('ensureDistro: a WSL1 crucible distro is refused with the one command that fixes it', async () => {
  const runner = new FakeRunner({ platform: 'win32', env: ENV }, [
    { argv: LIST, stdout: '  NAME      STATE     VERSION\r\n* crucible  Stopped   1\r\n' },
  ]);
  const r = await refusal(ensureDistro({ release: '0.6.0' }, runner));
  assert.equal(r.code, 'wsl1_only');
  assert.equal(r.command, 'wsl --set-version crucible 2');
});

test('ensureDistro: an import that fails is distro_import_failed with what wsl.exe said', async () => {
  const runner = new FakeRunner({ platform: 'win32', env: ENV }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: () => true },
    { argv: () => true, stdout: `${'f'.repeat(64)}\n` },
    { argv: () => true, stdout: `hash:\r\n${'f'.repeat(64)}\r\n` },
    { argv: importArgv(INSTALL_DIR, ROOTFS), code: 1, stderr: 'The system cannot find the path specified.' },
  ]);
  const r = await refusal(ensureDistro({ release: '0.6.0' }, runner));
  assert.equal(r.code, 'distro_import_failed');
  assert.match(r.message, /cannot find the path specified/);
});

test('ensureDistro is a Windows verb and says so anywhere else', async () => {
  const r = await refusal(ensureDistro({ release: '0.6.0' }, new FakeRunner({ platform: 'darwin' }, [])));
  assert.equal(r.code, 'unsupported_platform');
});
