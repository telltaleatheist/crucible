/** The server pack: the probe, the disk maths, the fetch, and every refusal it has. */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { fetchManifest, installPack, packPaths, parseEnvpacks, probeGuest, requiredBytes, requirePack, shaArgv } from '../src/index.js';
import { envpacksUrl } from '../src/envpacks.js';
import { guestProbeScript } from '../src/pack.js';
import { ARCHIVE, DEST, DOWNLOADS, ENVPACKS_JSON, FakeRunner, GUEST_BARE, GUEST_INSTALLED, PACK_SHA, refusal } from './fake.js';

const TARGET = { kind: 'wsl', distro: 'Ubuntu' } as const;
const NATIVE = { kind: 'native' } as const;
const W = (...argv: string[]): string[] => ['wsl.exe', '-d', 'Ubuntu', '--exec', ...argv];
const URL = envpacksUrl('0.6.0');
const MANIFEST = parseEnvpacks(ENVPACKS_JSON, URL, '0.6.0');
const ENTRY = MANIFEST.packs[0] as NonNullable<(typeof MANIFEST.packs)[0]>;
const SILENT = (): void => undefined;

test('the probe reads home, user, free space, missing tools, the pack and its stamp', async () => {
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
      sha256: PACK_SHA,
      release: '0.6.0',
    },
  });
});

test('a probe that answers nothing useful is host_unresponsive, never a default home', async () => {
  const runner = new FakeRunner({ platform: 'linux' }, [{ argv: () => true, stdout: 'bash: df: command not found\n' }]);
  const r = await refusal(probeGuest(runner, NATIVE, undefined));
  assert.equal(r.code, 'host_unresponsive');
  assert.match(r.message, /did not answer home, user and free space/);
});

test('requirePack: a host with no pack is no_server_pack naming the path install() fills', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [{ argv: () => true, stdout: GUEST_BARE }]);
  const facts = await probeGuest(runner, TARGET, undefined);
  const r = await refusal(Promise.resolve().then(() => requirePack(TARGET, facts)));
  assert.equal(r.code, 'no_server_pack');
});

test('the disk pre-flight is unpacked + the archive + one part', () => {
  // 400 MB unpacked + 120 MB archive + one of two equal parts (60 MB).
  assert.equal(requiredBytes(ENTRY), 400_000_000 + 120_000_000 + 60_000_000);
});

test('packPaths: the server pack goes to <home>/server, a job env to <home>/envs/<name>', () => {
  const server = packPaths('/home/owen/.crucible', ENTRY, '0.6.0');
  assert.equal(server.dest, DEST);
  assert.equal(server.partial, `${DEST}.partial`);
  assert.equal(server.downloads, DOWNLOADS);
  assert.equal(server.archive, ARCHIVE);
  assert.equal(server.crucible, `${DEST}/bin/crucible`);
  assert.equal(server.stamp, `${DEST}/.pack`);
  const llm = packPaths('/home/owen/.crucible', MANIFEST.packs[2] as NonNullable<(typeof MANIFEST.packs)[0]>, '0.6.0');
  assert.equal(llm.dest, '/home/owen/.crucible/envs/llm');
});

test('shaArgv: sha256sum on Linux, shasum -a 256 on a Mac that has no sha256sum', () => {
  assert.deepEqual(shaArgv('linux', '/a'), ['sha256sum', '/a']);
  assert.deepEqual(shaArgv('win32', '/a'), ['sha256sum', '/a'], 'win32 means inside the guest');
  assert.deepEqual(shaArgv('darwin', '/a'), ['shasum', '-a', '256', '/a']);
});

test('fetchManifest: the guest\'s curl, and a network exit code is guest_no_network, not an unreadable manifest', async () => {
  const ok = new FakeRunner({ platform: 'win32' }, [{ argv: W('curl', '-fsSL', '--retry', '3', URL), stdout: ENVPACKS_JSON }]);
  assert.equal((await fetchManifest(ok, TARGET, '0.6.0', URL, 1000)).packs.length, 3);

  const offline = new FakeRunner({ platform: 'win32' }, [{ argv: () => true, code: 6, stderr: 'curl: (6) Could not resolve host: github.com' }]);
  const n = await refusal(fetchManifest(offline, TARGET, '0.6.0', URL, 1000));
  assert.equal(n.code, 'guest_no_network');
  assert.match(n.message, /Could not resolve host/);

  const missing = new FakeRunner({ platform: 'win32' }, [{ argv: () => true, code: 22, stderr: 'curl: (22) The requested URL returned error: 404' }]);
  const m = await refusal(fetchManifest(missing, TARGET, '0.6.0', URL, 1000));
  assert.equal(m.code, 'pack_manifest_unreadable');
});

function fetchExpectations(digest: string): { argv: readonly string[] | ((argv: readonly string[]) => boolean); stdout?: string }[] {
  return [
    { argv: W('bash', '-c', `rm -f '${ARCHIVE}' && mkdir -p '${DOWNLOADS}'`) },
    { argv: (argv) => (argv[6] ?? '').includes('.part00') },
    { argv: (argv) => (argv[6] ?? '').includes('.part01') },
    { argv: W('sha256sum', ARCHIVE), stdout: `${digest}  ${ARCHIVE}\n` },
    { argv: (argv) => (argv[6] ?? '').includes('tar --zstd -xf') },
    { argv: (argv) => (argv[6] ?? '').includes('--version') },
    { argv: (argv) => (argv[6] ?? '').includes('mv ') },
  ];
}

test('installPack records exactly the guest commands PHASE14 section 4 describes, in order', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, fetchExpectations(PACK_SHA));
  const result = await installPack(runner, TARGET, MANIFEST, 'server', {
    release: '0.6.0',
    backend: 'cuda-linux',
    home: '/home/owen/.crucible',
    freeBytes: 900_000_000,
    installed: null,
    timeoutMs: 1000,
    onLine: SILENT,
  });
  runner.assertDrained();
  assert.equal(result.skipped, false);
  assert.equal(result.paths.crucible, `${DEST}/bin/crucible`);
  const scripts = result.ran.map((argv) => (argv[0] === 'bash' ? argv[2] ?? '' : argv.join(' ')));
  assert.equal(scripts.length, 7);
  assert.match(scripts[1] ?? '', /^curl -fL --retry 3 --retry-delay 2 --continue-at - --create-dirs -o /);
  assert.equal(scripts[3], `sha256sum ${ARCHIVE}`);
  assert.match(scripts[4] ?? '', /rm -rf '.*\.partial' && mkdir -p '.*\.partial' && tar --zstd -xf/);
  assert.match(scripts[5] ?? '', /\/server\.partial\/bin\/crucible' --version$/);
  assert.match(scripts[6] ?? '', /printf 'sha256=%s\\nrelease=%s\\n' 'a{64}' '0\.6\.0'/);
});

test('installPack is a skip when the stamp is already the manifest\'s sha', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, []);
  const result = await installPack(runner, TARGET, MANIFEST, 'server', {
    release: '0.6.0',
    backend: 'cuda-linux',
    home: '/home/owen/.crucible',
    freeBytes: 900_000_000,
    installed: { crucible: `${DEST}/bin/crucible`, python: `${DEST}/bin/python3`, version: 'crucible 0.6.0', sha256: PACK_SHA, release: '0.6.0' },
    timeoutMs: 1000,
    onLine: SILENT,
  });
  assert.equal(result.skipped, true);
  assert.deepEqual(runner.calls, [], 'a matching stamp downloads nothing');
});

test('a part that will not download is pack_download_failed naming the URL', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: () => true },
    { argv: () => true, code: 22, stderr: 'curl: (22) The requested URL returned error: 404' },
  ]);
  const r = await refusal(installPack(runner, TARGET, MANIFEST, 'server', {
    release: '0.6.0', backend: 'cuda-linux', home: '/home/owen/.crucible', freeBytes: 900_000_000, installed: null, timeoutMs: 1000, onLine: SILENT,
  }));
  assert.equal(r.code, 'pack_download_failed');
  assert.match(r.message, /download https:\/\/github\.com\/telltaleatheist\/crucible\/releases\/download\/v0\.6\.0\/crucible-env-server-cuda-linux-0\.6\.0\.tar\.zst\.part00/);
});

test('a tar that will not open leaves .partial where a person can look at it', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    ...fetchExpectations(PACK_SHA).slice(0, 4),
    { argv: () => true, code: 2, stderr: 'tar: Unrecognized archive format' },
  ]);
  const r = await refusal(installPack(runner, TARGET, MANIFEST, 'server', {
    release: '0.6.0', backend: 'cuda-linux', home: '/home/owen/.crucible', freeBytes: 900_000_000, installed: null, timeoutMs: 1000, onLine: SILENT,
  }));
  assert.equal(r.code, 'pack_unpack_failed');
  assert.match(r.message, /Unrecognized archive format/);
});

test('a pack that unpacks but will not run its own --version never becomes the server', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    ...fetchExpectations(PACK_SHA).slice(0, 5),
    { argv: () => true, code: 127, stderr: 'bash: line 1: bin/crucible: cannot execute binary file' },
  ]);
  const r = await refusal(installPack(runner, TARGET, MANIFEST, 'server', {
    release: '0.6.0', backend: 'cuda-linux', home: '/home/owen/.crucible', freeBytes: 900_000_000, installed: null, timeoutMs: 1000, onLine: SILENT,
  }));
  assert.equal(r.code, 'pack_unpack_failed');
  assert.match(r.message, /cannot execute binary file/);
});
