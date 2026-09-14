/** `health()` — the config's URL and token into the SDK, the SDK's errors into named refusals. */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { CrucibleAuthError, CrucibleNotACrucible, CrucibleUnreachable, CrucibleVersionError, type Activity } from '@crucible/client';

import { health, BOOTSTRAP_VERSION } from '../src/index.js';
import { FakeRunner, GUEST_CONFIG, refusal } from './fake.js';

const ACTIVITY = { server: { name: 'crucible@owens-pc-wsl', version: '0.5.0', apiVersion: 1, backend: 'cuda-linux', uptimeS: 12 } } as unknown as Activity;

function runner(): FakeRunner {
  return new FakeRunner({ platform: 'win32' }, [{ argv: () => true, stdout: `/home/owen/.crucible/config.toml\n${GUEST_CONFIG}` }]);
}

test('the client is built on the config\'s url and token, with the caller\'s name, and activity comes back as it is', async () => {
  let built: { url: string; token: string; clientName: string } | null = null;
  const result = await health({
    distro: 'Ubuntu',
    clientName: 'bookforge',
    clientFactory: (options) => {
      built = options;
      return { activity: async () => ACTIVITY };
    },
  }, runner());
  assert.deepEqual(built, { url: 'http://127.0.0.1:7100', token: 'test-token-not-a-secret', clientName: 'bookforge' });
  assert.equal(result, ACTIVITY);
});

test('with no clientName the bootstrap names itself, version included', async () => {
  let name = '';
  await health({ distro: 'Ubuntu', clientFactory: (o) => { name = o.clientName; return { activity: async () => ACTIVITY }; } }, runner());
  assert.equal(name, `crucible-bootstrap/${BOOTSTRAP_VERSION}`);
});

test('no local config is that named state, before any client is built', async () => {
  const r = await refusal(health({ distro: 'Ubuntu', clientFactory: () => assert.fail('no client should be built') }, new FakeRunner({ platform: 'win32' }, [{ argv: () => true, code: 3, stderr: '/home/owen/.crucible/config.toml' }])));
  assert.equal(r.code, 'no_local_config');
});

for (const [label, thrown, code, command] of [
  ['unreachable', new CrucibleUnreachable('http://127.0.0.1:7100', 'ECONNREFUSED'), 'unreachable', 'crucible service start'],
  ['a refused token', new CrucibleAuthError('bad_token', 'not this server\'s token'), 'wrong_token', 'crucible service install'],
  ['not a crucible', new CrucibleNotACrucible('http://127.0.0.1:7100', '<html>ollama</html>'), 'not_a_crucible', null],
  ['a version mismatch', new CrucibleVersionError('api_version', 'speak 2', 2, 1), 'version_mismatch', null],
] as const) {
  test(`${label} is ${code}`, async () => {
    const r = await refusal(health({ distro: 'Ubuntu', clientFactory: () => ({ activity: async () => { throw thrown; } }) }, runner()));
    assert.equal(r.code, code);
    assert.equal(r.command, command);
    assert.equal((r.error as Error & { cause?: unknown }).cause, thrown);
    assert.match(r.message, /crucible@owens-pc-wsl|127\.0\.0\.1:7100/);
  });
}

test('any other error is the SDK\'s and is rethrown as it is', async () => {
  const other = new TypeError('fetch is not defined');
  await assert.rejects(health({ distro: 'Ubuntu', clientFactory: () => ({ activity: async () => { throw other; } }) }, runner()), (err) => err === other);
});
