/**
 * The release channel, and the two gates that stop a machine going backwards.
 *
 * INSTALL-UNINSTALL.md §6.5: "latest" has one owner and it is
 * `releases/latest`; an unreachable channel is a refusal by name and never a
 * reason to install something older out of a cache; and a pack install over a
 * NEWER pack is refused unless an operator named the exact older version.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  compareReleases, install, installPack, latestRelease, LATEST_RELEASE_URL, parseEnvpacks, parseLatestRelease,
} from '../src/index.js';
import { envpacksUrl } from '../src/envpacks.js';
import { ENVPACKS_JSON, FakeRunner, PACK_SHA, refusal } from './fake.js';

const TARGET = { kind: 'wsl', distro: 'Ubuntu' } as const;
const URL = envpacksUrl('0.6.0');
const MANIFEST = parseEnvpacks(ENVPACKS_JSON, URL, '0.6.0');
const SILENT = (): void => undefined;

/** A pack already on the disk, at whatever release the test is about. */
const stamped = (release: string, sha256: string) => ({
  crucible: '/home/owen/.crucible/server/bin/crucible',
  python: '/home/owen/.crucible/server/bin/python3',
  version: `crucible ${release}`,
  sha256,
  release,
});

const packOptions = (installed: ReturnType<typeof stamped> | null, rollbackTo: string | null) => ({
  release: '0.6.0',
  backend: 'cuda-linux' as const,
  home: '/home/owen/.crucible',
  freeBytes: 900_000_000_000,
  installed,
  rollbackTo,
  timeoutMs: 1000,
  onLine: SILENT,
});

// ------------------------------------------------------------------ the pointer

test('the channel pointer is releases/latest — the promoted release, never the newest tag', () => {
  assert.equal(LATEST_RELEASE_URL, 'https://api.github.com/repos/telltaleatheist/crucible/releases/latest');
  assert.ok(!LATEST_RELEASE_URL.includes('per_page'), 'per_page=1 answers "newest tag created", which is an unpromoted candidate');
});

test('the pointer is read as a version, with the tag\'s leading v removed', () => {
  assert.equal(parseLatestRelease(JSON.stringify({ tag_name: 'v1.0.2' }), LATEST_RELEASE_URL), '1.0.2');
  assert.equal(parseLatestRelease(JSON.stringify({ tag_name: '1.0.2' }), LATEST_RELEASE_URL), '1.0.2');
});

test('a channel that answers something else is release_channel_unreadable, never a fallback version', async () => {
  for (const body of ['<html>404</html>', '{}', JSON.stringify({ tag_name: 'nightly' }), JSON.stringify({ tag_name: '' })]) {
    const r = await refusal(Promise.resolve().then(() => parseLatestRelease(body, LATEST_RELEASE_URL)));
    assert.equal(r.code, 'release_channel_unreadable', `${body} should be refused by name`);
    assert.match(r.message, /releases\/latest/);
  }
});

test('a channel that will not answer at all is refused by name, carrying what failed', async () => {
  const offline = (async () => { throw new Error('getaddrinfo ENOTFOUND api.github.com'); }) as typeof fetch;
  const dead = await refusal(latestRelease(offline));
  assert.equal(dead.code, 'release_channel_unreadable');
  assert.match(dead.message, /ENOTFOUND/);

  const rateLimited = (async () => new Response('{"message":"API rate limit exceeded"}', { status: 403 })) as typeof fetch;
  const limited = await refusal(latestRelease(rateLimited));
  assert.equal(limited.code, 'release_channel_unreadable');
  assert.match(limited.message, /403/);

  let asked = '';
  const ok = (async (url: unknown) => { asked = String(url); return new Response(JSON.stringify({ tag_name: 'v1.0.2' })); }) as typeof fetch;
  assert.equal(await latestRelease(ok), '1.0.2');
  assert.equal(asked, LATEST_RELEASE_URL);
});

test('releases order by their three numbers, so "newer" is a comparison and not a string test', () => {
  assert.equal(compareReleases('1.0.2', '1.0.10') < 0, true, '10 is after 2, which a string compare gets wrong');
  assert.equal(compareReleases('1.0.2', '1.0.2'), 0);
  assert.equal(compareReleases('1.0.2', '1.0.1') > 0, true);
  assert.equal(compareReleases('v1.0.2', '1.0.2'), 0, 'a tag and a version are the same release');
  assert.equal(compareReleases('2.0.0', '1.99.99') > 0, true);
});

// -------------------------------------------- the bootstrapper's own never-older

test('installing over a NEWER pack is refused by name, and nothing is downloaded', async () => {
  // Nothing is scripted on the runner: a refusal that ran a command would show
  // up here as "unexpected call", which is the half of this that matters.
  const runner = new FakeRunner({ platform: 'win32' }, []);
  const r = await refusal(installPack(runner, TARGET, MANIFEST, 'server', packOptions(stamped('0.6.3', 'f'.repeat(64)), null)));
  assert.equal(r.code, 'install_would_downgrade');
  assert.match(r.message, /0\.6\.3/);
  assert.match(r.message, /0\.6\.0/);
  assert.match(r.message, /rollbackTo/);
  assert.equal(runner.calls.length, 0, 'a refused downgrade must not touch the guest');
});

test('an operator rollback naming the EXACT version is allowed, and only that version', async () => {
  const mismatched = new FakeRunner({ platform: 'win32' }, []);
  const r = await refusal(installPack(mismatched, TARGET, MANIFEST, 'server', packOptions(stamped('0.6.3', 'f'.repeat(64)), '0.5.9')));
  assert.equal(r.code, 'rollback_version_mismatch');
  assert.match(r.message, /0\.5\.9/);
  assert.equal(mismatched.calls.length, 0);

  // Named exactly, the downgrade proceeds: the first thing it does is clear the
  // download directory, which is how this test knows the gate opened.
  const rolling = new FakeRunner({ platform: 'win32' }, [{ argv: (argv) => argv.join(' ').includes('rm -f'), code: 0 }]);
  await assert.rejects(
    installPack(rolling, TARGET, MANIFEST, 'server', packOptions(stamped('0.6.3', 'f'.repeat(64)), '0.6.0')),
    /unexpected call/,
    'the rollback should have got past the gate and started downloading',
  );
  assert.ok(rolling.calls.length >= 1, 'an allowed rollback reaches the guest');
});

test('a rollback on win32 is refused by name, never dropped on the way to the host door', async () => {
  const runner = new FakeRunner({ platform: 'win32', env: { LOCALAPPDATA: 'C:\\Users\\owen\\AppData\\Local' } }, []);
  const r = await refusal(install({ jobTypes: ['echo'], release: '0.6.0', rollbackTo: '0.6.0', onLine: SILENT }, runner));
  // ITS OWN NAME, not `rollback_version_mismatch`: the two versions here AGREE,
  // and what refuses them is the host door having no field to carry either.
  assert.equal(r.code, 'host_rollback_unsupported');
  assert.match(r.command ?? '', /install\.ps1 -Release 0\.6\.0 -RollbackTo 0\.6\.0/);
  assert.equal(runner.calls.length, 0);
});

test('rollbackTo must name the release being installed, checked before any guest command', async () => {
  const runner = new FakeRunner({ platform: 'linux' }, []);
  const r = await refusal(install({ jobTypes: ['echo'], release: '0.6.0', rollbackTo: '0.5.9', onLine: SILENT }, runner));
  assert.equal(r.code, 'rollback_version_mismatch');
  assert.match(r.message, /0\.5\.9/);
  assert.match(r.message, /0\.6\.0/);
  assert.equal(runner.calls.length, 0);
});

test('the same pack already on the disk is still a skip, and an older one still installs', async () => {
  const same = new FakeRunner({ platform: 'win32' }, []);
  const skipped = await installPack(same, TARGET, MANIFEST, 'server', packOptions(stamped('0.6.0', PACK_SHA), null));
  assert.equal(skipped.skipped, true);
  assert.equal(same.calls.length, 0);

  const older = new FakeRunner({ platform: 'win32' }, [{ argv: (argv) => argv.join(' ').includes('rm -f'), code: 0 }]);
  await assert.rejects(
    installPack(older, TARGET, MANIFEST, 'server', packOptions(stamped('0.5.1', 'f'.repeat(64)), null)),
    /unexpected call/,
    'an upgrade is not gated at all',
  );
  assert.ok(older.calls.length >= 1);
});
