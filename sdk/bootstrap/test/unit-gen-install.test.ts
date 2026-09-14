/**
 * The generated installers: up to date with the step list, and made of the
 * same constants `install()` uses. PHASE14 4a's "cannot differ" is this test.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { GENERATED, generateInstallPs1, generateInstallSh } from '../scripts/gen-install-scripts.js';
import { CURL_ARGS, installSteps, TAR_ARGS } from '../src/index.js';
import { guestProbeScript } from '../src/pack.js';

const STANDALONE = { enableFlags: [], installs: [], bind: [], linger: true };

test('install.sh and install.ps1 on disk ARE what the generator writes (npm run gen:install)', () => {
  for (const file of GENERATED) {
    const onDisk = readFileSync(file.path, 'utf8').replace(/\r\n/g, '\n');
    assert.equal(onDisk, file.text, `${file.path} has drifted from src/steps.ts — run: npm run gen:install`);
  }
});

test('install.sh walks the same step list, in the same order, with the same names', () => {
  const sh = generateInstallSh();
  const names = installSteps(STANDALONE).map((step) => step.name);
  assert.deepEqual(names, ['host-facts', 'server-pack', 'init', 'service-install', 'linger', 'capability-write']);
  let at = -1;
  for (const name of names) {
    const found = sh.indexOf(`say "${name}"`);
    assert.ok(found > at, `${name} appears, and after the step before it`);
    at = found;
  }
});

test('install.sh carries NO job types and NO weights: a bare Crucible (4a)', () => {
  const sh = generateInstallSh();
  assert.equal(/crucible' install |--enable-llm|--narrator-engine|models pull/.test(sh), false);
  assert.match(sh, /"\$CRUCIBLE" token --url\n?$/, 'the last thing it prints is the pairing line');
});

test('install.sh uses the same probe script, curl flags and tar flags the TypeScript does', () => {
  const sh = generateInstallSh();
  assert.ok(sh.includes(guestProbeScript(undefined)), 'the host probe is ONE script, not two');
  assert.ok(sh.includes(`curl ${CURL_ARGS.join(' ')} -o "$downloads/$part"`));
  assert.ok(sh.includes(`tar ${TAR_ARGS.join(' ')} "$archive" -C "$partial"`));
  assert.ok(sh.includes('rm -rf "$dest" && mv "$partial" "$dest"'), 'the same rename-into-place');
  assert.ok(sh.includes('printf \'sha256=%s\\nrelease=%s\\n\''), 'the same stamp');
});

test('install.sh detects the two backends and refuses a third, and never assumes WSL2', () => {
  const sh = generateInstallSh();
  assert.match(sh, /Linux\/x86_64\)\s+BACKEND=cuda-linux; SHA_TOOL="sha256sum";/);
  assert.match(sh, /Darwin\/arm64\)\s+BACKEND=mlx-darwin; SHA_TOOL="shasum -a 256";/);
  assert.match(sh, /die "unsupported_platform/);
  assert.match(sh, /set -eu/);
});

test('install.sh names every refusal the TypeScript names for the same failure', () => {
  const sh = generateInstallSh();
  for (const code of [
    'guest_missing_tool',
    'pack_manifest_unreadable',
    'pack_not_published',
    'pack_disk',
    'pack_download_failed',
    'pack_sha_mismatch',
    'pack_unpack_failed',
  ]) {
    assert.ok(sh.includes(`${code}:`), `install.sh refuses ${code} by the same name`);
  }
});

test('install.sh mints its own token and keeps an existing config\'s', () => {
  const sh = generateInstallSh();
  assert.match(sh, /if \[ -f "\$CRUCIBLE_HOME\/config\.toml" \]; then/);
  assert.match(sh, /its token is kept/);
  assert.match(sh, /TOKEN="\$\(head -c 32 \/dev\/urandom \| base64 \| tr '\+\/' '-_' \| tr -d '='\)"/);
  assert.match(sh, /init' '--token' "\$TOKEN"/);
});

test('install.sh\'s linger step tries itself, then sudo -n, then prints the ONE hand-over line', () => {
  const sh = generateInstallSh();
  assert.match(sh, /loginctl show-user "\$GUEST_USER" -p Linger/);
  assert.match(sh, /sudo -n loginctl enable-linger "\$GUEST_USER"/);
  assert.match(sh, /run this once, by hand:  sudo loginctl enable-linger \$GUEST_USER/);
});

test('install.ps1 walks 4c in order and never runs --shutdown', () => {
  const ps1 = generateInstallPs1();
  const order = ['HCS_E_HYPERV_NOT_INSTALLED', 'Start-Process -Verb RunAs', 'Default Version:', '--import', '/etc/wsl.conf', 'v$Release/install.sh'];
  let at = -1;
  for (const marker of order) {
    const found = ps1.indexOf(marker);
    assert.ok(found > at, `${marker} appears after what comes before it`);
    at = found;
  }
  const code = ps1.split('\n').filter((line) => !line.trimStart().startsWith('#')).join('\n');
  assert.equal(code.includes('--shutdown'), false, 'never: the other distros are not ours to stop');
  assert.match(ps1, /wsl\.exe --terminate \$Distro/);
});

test('install.ps1 verifies the rootfs before importing it, and is Windows PowerShell 5.1 safe', () => {
  const ps1 = generateInstallPs1();
  assert.match(ps1, /Get-FileHash -Algorithm SHA256/);
  assert.match(ps1, /pack_sha_mismatch/);
  // 5.1 has no && / || chain operators, no ternary, no null-coalescing. The
  // bash this script hands to the guest is exempt: that is a string, and bash
  // has all three.
  const powershellOnly = ps1
    .split('\n')
    .filter((line) => !line.includes('bash -c') && !line.includes('| sh'))
    .join('\n')
    .replace(/https?:\/\//g, '');
  assert.equal(/&&|\|\||\?\?|\?\./.test(powershellOnly), false, powershellOnly.split('\n').filter((l) => /&&|\|\|/.test(l)).join('\n'));
  assert.match(ps1, /\$ErrorActionPreference = "Continue"/);
});

test('install.ps1 hands the guest the same install.sh, at the same release', () => {
  const ps1 = generateInstallPs1();
  assert.match(ps1, /releases\/download\/v\$Release\/install\.sh/);
  assert.match(ps1, /CRUCIBLE_RELEASE=\$Release curl -fsSL \$sh \| sh/);
});
