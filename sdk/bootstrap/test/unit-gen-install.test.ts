/**
 * The generated installers: up to date with the step list, and made of the
 * same constants `install()` uses. PHASE14 4a's "cannot differ" is this test.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { GENERATED, generateInstallPs1, generateInstallSh } from '../scripts/gen-install-scripts.js';
import { CURL_ARGS, ENVPACKS_ASSET, HOST_BACKEND, HOST_PACK, installSteps, TAR_ARGS } from '../src/index.js';
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

test('install.ps1 installs the HOST and stops — it no longer walks 4c itself (PHASE15 4.4)', () => {
  // This REPLACES the assertion that install.ps1 walks the state table.
  // PHASE15-HOST.md 4.4 moved that walk into `crucible host` (4.3), so that
  // one implementation of the sequence serves the page's engine switch, an
  // app's install() and a hand install alike. A .ps1 that still did it would
  // be the second walk this phase exists to delete.
  const ps1 = generateInstallPs1();
  const order = ['tar.exe --version', 'pack_not_published', 'pack_disk', 'Get-FileHash', 'Move-Item $Partial $HostDir', 'host --install-startup', 'Start-Process -WindowStyle Hidden'];
  let at = -1;
  for (const marker of order) {
    const found = ps1.indexOf(marker);
    assert.ok(found > at, `${marker} appears, and after what comes before it`);
    at = found;
  }
  const code = ps1.split('\n').filter((line) => !line.trimStart().startsWith('#')).join('\n');
  assert.equal(code.includes('--shutdown'), false, 'never: the other distros are not ours to stop');
  assert.equal(code.includes('wsl.exe --import'), false, 'the host imports the distro now, not this script');
  assert.equal(code.includes('/etc/wsl.conf'), false, 'ditto');
});

test('install.ps1 needs no admin, and starts the tray with pythonw rather than the .cmd', () => {
  const ps1 = generateInstallPs1();
  // 4.1: a per-user login item is what a tray program is. Nothing here may
  // raise UAC; the host asks for it later, by name, when a 4c row needs it.
  const code = ps1.split('\n').filter((line) => !line.trimStart().startsWith('#')).join('\n');
  assert.equal(code.includes('-Verb RunAs'), false, 'install.ps1 never elevates');
  assert.match(ps1, /\$Pythonw = Join-Path \$HostDir "pythonw\.exe"/);
  assert.match(ps1, /Start-Process -WindowStyle Hidden -FilePath \$Pythonw -ArgumentList "-m","crucible\.cli","host"/);
  // The Startup item has ONE owner and this script asks for it by verb.
  assert.match(ps1, /& \$Cmd host --install-startup/);
  assert.equal(/New-Object -ComObject WScript\.Shell/.test(ps1), false, 'it does not write a .lnk of its own');
});

test('install.ps1 downloads the host pack for THIS release, by the manifest, into %LOCALAPPDATA%', () => {
  const ps1 = generateInstallPs1();
  assert.ok(ps1.includes(`$entry.name -eq '${HOST_PACK}'`), 'it asks the manifest for the host pack');
  assert.ok(ps1.includes(`$entry.backend -eq '${HOST_BACKEND}'`), 'for this backend, by the shared constant');
  assert.match(ps1, /\$HostDir = Join-Path \$Root 'host'/);
  assert.match(ps1, /\[string\]\$Root = "\$env:LOCALAPPDATA\\Crucible"/);
  assert.ok(ps1.includes(ENVPACKS_ASSET), 'the same manifest asset name the TypeScript uses');
  assert.ok(ps1.includes(`& curl.exe ${CURL_ARGS.join(' ')} -o $partPath`), 'the same curl flags');
  assert.ok(ps1.includes(`& tar.exe ${TAR_ARGS.join(' ')} $archive -C $Partial`), 'the same tar flags');
});

test('install.ps1 checks that THIS machine tar carries zstd rather than assuming it', () => {
  // Measured 2026-09-14: C:\Windows\System32\tar.exe is bsdtar 3.8.1 with
  // libzstd 1.5.5, and no zstd.exe ships at all. A machine whose tar has no
  // zstd would half-unpack in silence.
  const ps1 = generateInstallPs1();
  assert.match(ps1, /tar\.exe --version/);
  assert.match(ps1, /\$tarVersion -notmatch "zstd"/);
  assert.match(ps1, /guest_missing_tool/);
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

test('install.ps1 unpacks BESIDE, proves the moved pack runs, and only then renames', () => {
  // PHASE14 7.2a's defect, in its Windows form: pip writes Scripts\*.exe
  // launchers with the BUILD tree's interpreter path inside the binary, which
  // no shebang rewrite can reach. The .cmd derives its interpreter from
  // %~dp0, and the only proof of that is running it somewhere else.
  const ps1 = generateInstallPs1();
  assert.match(ps1, /\$Partial = "\$HostDir\.partial"/);
  assert.match(ps1, /& \(Join-Path \$Partial "crucible\.cmd"\) --version/);
  assert.match(ps1, /Move-Item \$Partial \$HostDir/);
  assert.ok(
    ps1.indexOf('crucible.cmd") --version') < ps1.indexOf('Move-Item $Partial $HostDir'),
    'it runs BEFORE the rename, so a pack that cannot run never becomes the installed one',
  );
});

test('install.sh is unchanged by PHASE15: linux and darwin have no host', () => {
  const sh = generateInstallSh();
  assert.match(sh, /Linux\/x86_64\)  BACKEND=cuda-linux/);
  assert.match(sh, /Darwin\/arm64\)  BACKEND=mlx-darwin/);
  assert.equal(sh.includes('llama-windows'), false);
  assert.equal(sh.includes('crucible.cmd'), false);
});
