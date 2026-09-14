/** `envpacks.json` — the asset names, and a manifest that is read strictly or refused. */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { backendFor, envpacksUrl, findPack, packAssetName, parseEnvpacks, releaseAssetUrl, rootfsAssetName } from '../src/index.js';
import { ENVPACKS_JSON, PACK_SHA, refusal } from './fake.js';

const URL = envpacksUrl('0.6.0');

test('the asset names are PHASE14 section 1\'s, character for character', () => {
  assert.equal(packAssetName('llm', 'cuda-linux', '0.6.0'), 'crucible-env-llm-cuda-linux-0.6.0.tar.zst');
  assert.equal(packAssetName('tts-higgs-v3', 'mlx-darwin', '1.0.0'), 'crucible-env-tts-higgs-v3-mlx-darwin-1.0.0.tar.zst');
  assert.equal(rootfsAssetName('0.6.0'), 'crucible-rootfs-0.6.0.tar.zst');
  assert.equal(URL, 'https://github.com/telltaleatheist/crucible/releases/download/v0.6.0/envpacks.json');
  assert.equal(releaseAssetUrl('0.6.0', 'install.sh'), 'https://github.com/telltaleatheist/crucible/releases/download/v0.6.0/install.sh');
});

test('a release that is not a version has no assets to name', async () => {
  const r = await refusal(Promise.resolve().then(() => envpacksUrl('latest')));
  assert.equal(r.code, 'pack_manifest_unreadable');
  assert.match(r.message, /"latest" is not a Crucible version/);
});

test('backendFor: win32 means the WSL2 guest, which is cuda-linux; darwin is mlx; nothing else is a backend', async () => {
  assert.equal(backendFor('win32'), 'cuda-linux');
  assert.equal(backendFor('linux'), 'cuda-linux');
  assert.equal(backendFor('darwin'), 'mlx-darwin');
  const r = await refusal(Promise.resolve().then(() => backendFor('freebsd')));
  assert.equal(r.code, 'unsupported_platform');
});

test('a well-formed manifest parses, and the pack for a backend is found by name', () => {
  const manifest = parseEnvpacks(ENVPACKS_JSON, URL, '0.6.0');
  assert.equal(manifest.version, '0.6.0');
  assert.equal(manifest.packs.length, 3);
  const server = findPack(manifest, 'server', 'cuda-linux');
  assert.equal(server.sha256, PACK_SHA);
  assert.equal(server.parts.length, 2);
  assert.equal(server.recipeSha256, 'e'.repeat(64), 'the server pack hashes pyproject.toml, which owns its dependencies');
  // A manifest that omits it still parses: the field is the SERVER's to refuse
  // drift on (`pack_recipe_drift`), and nothing here compares recipes.
  assert.equal(parseEnvpacks(ENVPACKS_JSON.replace(/"recipe_sha256":"[a-f]+",/g, ''), URL, '0.6.0').packs[0]?.recipeSha256, null);
  assert.equal(findPack(manifest, 'llm', 'cuda-linux').recipeSha256, 'd'.repeat(64));
  assert.equal(findPack(manifest, 'server', 'mlx-darwin').parts.length, 1);
});

test('a (name, backend) the release does not publish is pack_not_published, listing what it does have', async () => {
  const manifest = parseEnvpacks(ENVPACKS_JSON, URL, '0.6.0');
  const r = await refusal(Promise.resolve().then(() => findPack(manifest, 'tts-higgs-v3', 'cuda-linux')));
  assert.equal(r.code, 'pack_not_published');
  assert.match(r.message, /publishes no "tts-higgs-v3" pack for cuda-linux/);
  assert.match(r.message, /lists server, llm for that backend/);
  assert.match(r.message, /never built here/);

  const none = await refusal(Promise.resolve().then(() => findPack(manifest, 'llm', 'mlx-darwin')));
  assert.match(none.message, /lists server for that backend/);
});

for (const [label, text, pattern] of [
  ['not JSON', '<html>404</html>', /it is not JSON/],
  ['an array', '[]', /the top level is not an object/],
  ['another schema', '{"schema": 2, "version": "0.6.0", "packs": []}', /schema must be 1/],
  ['another version', '{"schema": 1, "version": "0.5.0", "packs": []}', /it says version "0\.5\.0"/],
  ['no packs array', '{"schema": 1, "version": "0.6.0"}', /packs must be an array/],
  ['a pack with no parts', '{"schema":1,"version":"0.6.0","packs":[{"name":"server","backend":"cuda-linux","python":"3.11.13","bytes":1,"sha256":"'
    + 'a'.repeat(64) + '","parts":[],"unpacked_bytes":2}]}', /parts must list at least one asset/],
  ['a short digest', '{"schema":1,"version":"0.6.0","packs":[{"name":"server","backend":"cuda-linux","python":"3.11.13","bytes":1,"sha256":"abc","parts":["p"],"unpacked_bytes":2}]}', /not 64 lowercase hex/],
  ['a third backend', '{"schema":1,"version":"0.6.0","packs":[{"name":"server","backend":"win32","python":"3.11.13","bytes":1,"sha256":"'
    + 'a'.repeat(64) + '","parts":["p"],"unpacked_bytes":2}]}', /the two backends are cuda-linux and mlx-darwin/],
  ['a zero size', '{"schema":1,"version":"0.6.0","packs":[{"name":"server","backend":"cuda-linux","python":"3.11.13","bytes":0,"sha256":"'
    + 'a'.repeat(64) + '","parts":["p"],"unpacked_bytes":2}]}', /bytes must be a positive integer/],
  ['a part with a path in it', '{"schema":1,"version":"0.6.0","packs":[{"name":"server","backend":"cuda-linux","python":"3.11.13","bytes":1,"sha256":"'
    + 'a'.repeat(64) + '","parts":["../etc/passwd"],"unpacked_bytes":2}]}', /must be an asset name/],
] as const) {
  test(`a manifest that is ${label} is pack_manifest_unreadable, not a manifest with holes`, async () => {
    const r = await refusal(Promise.resolve().then(() => parseEnvpacks(text, URL, '0.6.0')));
    assert.equal(r.code, 'pack_manifest_unreadable');
    assert.match(r.message, pattern);
  });
}
