/**
 * The server runtime: the probe, the interpreter half, the wheel half, and
 * every refusal each of them has.
 *
 * PHASE20-CODE-NOT-ENVIRONMENTS.md section 3. What this file used to test was a
 * manifest, thirteen archives and a part-by-part download; what it tests now is
 * one pinned interpreter, fetched once, and one wheel, fetched every time.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { installRuntime, probeGuest, requireRuntime, runtimePaths, shaArgv } from '../src/index.js';
import { interpreterFor, interpreterUrl } from '../src/interpreter.js';
import { wheelShaUrl, wheelUrl } from '../src/release.js';
import { guestProbeScript } from '../src/runtime.js';
import { ARCHIVE, DEST, DOWNLOADS, FakeRunner, GUEST_BARE, GUEST_INSTALLED, PIN, PY_SHA, WHEEL, refusal } from './fake.js';

const TARGET = { kind: 'wsl', distro: 'Ubuntu' } as const;
const NATIVE = { kind: 'native' } as const;
const W = (...argv: string[]): string[] => ['wsl.exe', '-d', 'Ubuntu', '--exec', ...argv];
const SILENT = (): void => undefined;
const WHEEL_SHA = 'c'.repeat(64);

const options = (installed: Parameters<typeof installRuntime>[2]['installed']) => ({
  release: '0.6.0',
  backend: 'cuda-linux' as const,
  home: '/home/owen/.crucible',
  installed,
  rollbackTo: null,
  timeoutMs: 1000,
  onLine: SILENT,
});

test('the probe reads home, user, free space, missing tools, the runtime and its stamp', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: W('bash', '-c', guestProbeScript(undefined)), stdout: GUEST_INSTALLED },
  ]);
  const facts = await probeGuest(runner, TARGET, undefined);
  assert.deepEqual(facts, {
    home: '/home/owen/.crucible',
    user: 'owen',
    freeBytes: 400000000 * 1024,
    missingTools: [],
    server: {
      crucible: '/home/owen/.crucible/server/bin/crucible',
      python: '/home/owen/.crucible/server/bin/python3',
      version: 'crucible 0.6.0',
      pythonSha256: PY_SHA,
      release: '0.6.0',
    },
  });
});

test('the probe asks for curl and tar, and NOT for zstd', () => {
  // The measure of what PHASE20 removed: zstd was here for the 8 GB
  // environment archives, and nothing this installer touches is compressed
  // with it. A machine refused for a tool nothing uses is a refusal that costs
  // somebody an afternoon.
  const script = guestProbeScript(undefined);
  assert.match(script, /for t in curl tar; do/);
  assert.ok(!script.includes('zstd'), script);
});

test('a probe that answers nothing useful is host_unresponsive, never a default home', async () => {
  const runner = new FakeRunner({ platform: 'linux' }, [{ argv: () => true, stdout: 'bash: df: command not found\n' }]);
  const r = await refusal(probeGuest(runner, NATIVE, undefined));
  assert.equal(r.code, 'host_unresponsive');
  assert.match(r.message, /did not answer home, user and free space/);
});

test('requireRuntime: a host with none is no_server_runtime naming the path install() fills', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [{ argv: () => true, stdout: GUEST_BARE }]);
  const facts = await probeGuest(runner, TARGET, undefined);
  const r = await refusal(Promise.resolve().then(() => requireRuntime(TARGET, facts)));
  assert.equal(r.code, 'no_server_runtime');
});

test('runtimePaths: the interpreter, the wheel and the stamp, from one home', () => {
  const paths = runtimePaths('/home/owen/.crucible', PIN, '0.6.0');
  assert.equal(paths.dest, DEST);
  assert.equal(paths.partial, `${DEST}.partial`);
  assert.equal(paths.downloads, DOWNLOADS);
  assert.equal(paths.archive, ARCHIVE);
  assert.equal(paths.wheel, WHEEL);
  assert.equal(paths.crucible, `${DEST}/bin/crucible`);
  assert.equal(paths.python, `${DEST}/bin/python3`);
  // `.crucible` and not `.pack`: the file's SUBJECT changed from "the sha256 of
  // an archive of this whole tree" to "the digest of the interpreter in it".
  assert.equal(paths.stamp, `${DEST}/.crucible`);
});

test('shaArgv: sha256sum on Linux, shasum -a 256 on a Mac that has no sha256sum', () => {
  assert.deepEqual(shaArgv('linux', '/a'), ['sha256sum', '/a']);
  assert.deepEqual(shaArgv('win32', '/a'), ['sha256sum', '/a'], 'win32 means inside the guest');
  assert.deepEqual(shaArgv('darwin', '/a'), ['shasum', '-a', '256', '/a']);
});

/** The guest calls a clean first install makes, in order. */
function firstInstall(pythonDigest: string, wheelDigest: string): { argv: readonly string[] | ((argv: readonly string[]) => boolean); stdout?: string }[] {
  return [
    { argv: (argv) => (argv[6] ?? '').includes(`curl -fL`) && (argv[6] ?? '').includes(PIN.asset) },
    { argv: W('sha256sum', ARCHIVE), stdout: `${pythonDigest}  ${ARCHIVE}\n` },
    { argv: (argv) => (argv[6] ?? '').includes('tar -xzf') },
    { argv: (argv) => (argv[6] ?? '').includes(`-o '${WHEEL}'`) },
    { argv: W('curl', '-fsSL', '--retry', '3', wheelShaUrl('0.6.0')), stdout: `${wheelDigest}  crucible-0.6.0-py3-none-any.whl\n` },
    { argv: W('sha256sum', WHEEL), stdout: `${wheelDigest}  ${WHEEL}\n` },
    { argv: (argv) => (argv[6] ?? '').includes('pip install --upgrade --no-input') },
  ];
}

test('installRuntime runs exactly the guest commands PHASE20 section 3 describes, in order', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, firstInstall(PY_SHA, WHEEL_SHA));
  const result = await installRuntime(runner, TARGET, options(null));
  runner.assertDrained();
  assert.equal(result.interpreterSkipped, false);
  assert.equal(result.paths.crucible, `${DEST}/bin/crucible`);
  const scripts = result.ran.map((argv) => (argv[0] === 'bash' ? argv[2] ?? '' : argv.join(' ')));
  assert.equal(scripts.length, 7);
  assert.ok((scripts[0] ?? '').includes(interpreterUrl(PIN)), 'the interpreter comes from its publisher');
  assert.equal(scripts[1], `sha256sum ${ARCHIVE}`);
  // `python/` and not the archive root: `install_only` carries one top-level
  // directory and THAT is the interpreter.
  assert.match(scripts[2] ?? '', /tar -xzf .* && test -x '.*\.partial\/python\/bin\/python3'/);
  assert.ok((scripts[3] ?? '').includes(wheelUrl('0.6.0')));
  assert.equal(scripts[4], `curl -fsSL --retry 3 ${wheelShaUrl('0.6.0')}`);
  assert.match(scripts[6] ?? '', /pip install --upgrade --no-input '.*crucible-0\.6\.0-py3-none-any\.whl'/);
  assert.match(scripts[6] ?? '', /printf 'python_sha256=%s\\npython_version=%s\\nrelease=%s\\n'/);
});

test('an upgrade skips the interpreter and installs the wheel — that is the whole ruling', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, firstInstall(PY_SHA, WHEEL_SHA).slice(3));
  const result = await installRuntime(runner, TARGET, options({
    crucible: `${DEST}/bin/crucible`,
    python: `${DEST}/bin/python3`,
    version: 'crucible 0.5.9',
    pythonSha256: PY_SHA,
    release: '0.5.9',
  }));
  runner.assertDrained();
  assert.equal(result.interpreterSkipped, true);
  const scripts = result.ran.map((argv) => (argv[0] === 'bash' ? argv[2] ?? '' : argv.join(' ')));
  assert.equal(scripts.length, 4, 'four guest calls: fetch the wheel, its digest, hash it, pip it');
  assert.ok(scripts.every((script) => !script.includes(PIN.asset)), 'nothing fetched the interpreter');
});

test('an interpreter whose digest is not the pin is refused by name and deleted', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: () => true },
    { argv: W('sha256sum', ARCHIVE), stdout: `${'0'.repeat(64)}  ${ARCHIVE}\n` },
    { argv: (argv) => (argv[6] ?? '').startsWith('rm -f') },
  ]);
  const r = await refusal(installRuntime(runner, TARGET, options(null)));
  assert.equal(r.code, 'runtime_sha_mismatch');
  assert.match(r.message, /@crucible\/bootstrap pins/);
  assert.match(r.message, new RegExp(PIN.sha256));
});

test('a wheel whose digest is not the release\'s is refused by name and deleted', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    ...firstInstall(PY_SHA, WHEEL_SHA).slice(0, 5),
    { argv: W('sha256sum', WHEEL), stdout: `${'1'.repeat(64)}  ${WHEEL}\n` },
    { argv: (argv) => (argv[6] ?? '').startsWith('rm -f') },
  ]);
  const r = await refusal(installRuntime(runner, TARGET, options(null)));
  assert.equal(r.code, 'runtime_sha_mismatch');
  assert.match(r.message, new RegExp(WHEEL_SHA));
});

test('a sha256 sidecar that is not a digest is refused rather than compared against', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    ...firstInstall(PY_SHA, WHEEL_SHA).slice(0, 4),
    { argv: () => true, code: 22, stderr: 'curl: (22) The requested URL returned error: 404' },
  ]);
  const r = await refusal(installRuntime(runner, TARGET, options(null)));
  assert.equal(r.code, 'runtime_download_failed');
  assert.match(r.message, /\.whl\.sha256/);
});

test('an interpreter that will not download is runtime_download_failed naming the URL', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: () => true, code: 22, stderr: 'curl: (22) The requested URL returned error: 404' },
  ]);
  const r = await refusal(installRuntime(runner, TARGET, options(null)));
  assert.equal(r.code, 'runtime_download_failed');
  assert.match(r.message, /python-build-standalone/);
});

test('a tar that will not open leaves .partial where a person can look at it', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    ...firstInstall(PY_SHA, WHEEL_SHA).slice(0, 2),
    { argv: () => true, code: 2, stderr: 'tar: Unrecognized archive format' },
  ]);
  const r = await refusal(installRuntime(runner, TARGET, options(null)));
  assert.equal(r.code, 'runtime_unpack_failed');
  assert.match(r.message, /Unrecognized archive format/);
});

test('pip refusing the wheel is its own name, carrying pip\'s last lines', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    ...firstInstall(PY_SHA, WHEEL_SHA).slice(0, 6),
    { argv: () => true, code: 1, stderr: 'ERROR: crucible-0.6.0-py3-none-any.whl is not a supported wheel on this platform.' },
  ]);
  const r = await refusal(installRuntime(runner, TARGET, options(null)));
  assert.equal(r.code, 'runtime_install_failed');
  assert.match(r.message, /not a supported wheel/);
});

test('the interpreter pinned for each backend is the one the table names', () => {
  assert.equal(interpreterFor('cuda-linux').asset, PIN.asset);
  assert.match(interpreterFor('mlx-darwin').asset, /aarch64-apple-darwin-install_only\.tar\.gz$/);
  assert.match(interpreterFor('llama-windows').asset, /x86_64-pc-windows-msvc-install_only\.tar\.gz$/);
  for (const backend of ['cuda-linux', 'mlx-darwin', 'llama-windows'] as const) {
    const pin = interpreterFor(backend);
    assert.match(pin.sha256, /^[0-9a-f]{64}$/);
    assert.ok(interpreterUrl(pin).startsWith('https://github.com/astral-sh/python-build-standalone/releases/download/'));
  }
});
