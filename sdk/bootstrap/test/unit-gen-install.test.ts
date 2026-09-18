/**
 * The generated installers: up to date with the step list, and made of the
 * same constants `install()` uses. PHASE14 4a's "cannot differ" is this test.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { asciiOnly, GENERATED, generateInstallPs1, generateInstallSh } from '../scripts/gen-install-scripts.js';
import { CURL_ARGS, ENVPACKS_ASSET, HOST_BACKEND, HOST_PACK, installSteps, TAR_ARGS } from '../src/index.js';
import { guestProbeScript } from '../src/pack.js';
import { installJobTypesSh, renderArgv, uninstallSh } from '../src/steps.js';

const STANDALONE = { enableFlags: [], installs: [], bind: [{ sh: '$BIND' }], linger: true };

test('install.sh and install.ps1 on disk ARE what the generator writes (npm run gen:install)', () => {
  for (const file of GENERATED) {
    const onDisk = readFileSync(file.path, 'utf8').replace(/\r\n/g, '\n');
    assert.equal(onDisk, file.text, `${file.path} has drifted from src/steps.ts — run: npm run gen:install`);
  }
});

/**
 * The install half of the script: everything from `host-facts` down.
 *
 * The `--uninstall` branch above it says `say "server-pack"` too — it removes
 * the pack this script unpacked — so an order assertion that used `indexOf`
 * over the whole file would find that one first. Slicing is the honest fix:
 * the two halves are two sequences, and only one of them is the step list.
 */
function installHalf(sh: string): string {
  const at = sh.indexOf('# --- host-facts');
  assert.ok(at > 0, 'the install half starts at host-facts');
  return sh.slice(at);
}

test('install.sh walks the same step list, in the same order, with the same names', () => {
  const sh = installHalf(generateInstallSh());
  const names = installSteps(STANDALONE).map((step) => step.name);
  assert.deepEqual(names, ['host-facts', 'server-pack', 'init', 'service-install', 'local-register', 'local-install-cli', 'local-install-desktop', 'linger', 'capability-write', 'local-start']);
  let at = -1;
  for (const name of names) {
    const found = sh.indexOf(`say "${name}"`);
    assert.ok(found > at, `${name} appears, and after the step before it`);
    at = found;
  }
});

test('install.sh installs NO job types and NO weights unless a person asks (4a)', () => {
  // 4a's "a bare Crucible that serves nothing" is about the DEFAULT, and it
  // still holds: `$JOB_TYPES` and `$BIND` are empty, so `curl … | sh` with no
  // arguments performs exactly the install it performed before the droplet
  // route existed. What is new is that a person at a terminal can say
  // otherwise, and weights are STILL never pulled by this script.
  const sh = generateInstallSh();
  assert.equal(/models pull|voices pull|rvc pull|--enable-llm/.test(sh), false, 'no weights, ever, and no enable flags of its own');
  assert.match(sh, /^JOB_TYPES=""$/m, 'no job types until --install names one');
  assert.match(sh, /^BIND=""$/m, 'loopback until --host says otherwise');
  assert.match(sh, /if \[ -n "\$JOB_TYPES" \]; then/, 'the loop is guarded by the flag');
  assert.match(sh, /"\$CRUCIBLE" token --url\n?$/, 'the last thing it prints is the pairing line');
});

test('install.sh takes --host/--port through to init, which is what a droplet needs', () => {
  const sh = generateInstallSh();
  assert.match(sh, /--host\) need \$# "--host"; shift; BIND="\$BIND --host \$1" ;;/);
  assert.match(sh, /--port\) need \$# "--port"; shift; BIND="\$BIND --port \$1" ;;/);
  // Unquoted on purpose: `$BIND` holds two words and must split into two.
  assert.match(sh, /init' '--token' "\$TOKEN" \$BIND \|\| die "step_failed: init"/);
});

test('install.sh takes --token, and mints one only when nobody brought one', () => {
  const sh = generateInstallSh();
  assert.match(sh, /--token\) need \$# "--token"; shift; TOKEN="\$1" ;;/);
  assert.match(sh, /if \[ -z "\$\{TOKEN:-\}" \]; then/);
  assert.match(sh, /TOKEN="\$\(head -c 32 \/dev\/urandom \| base64 \| tr '\+\/' '-_' \| tr -d '='\)"/);
});

test('install.sh --install <type> runs the SAME command install() runs', () => {
  const sh = generateInstallSh();
  assert.ok(sh.includes(installJobTypesSh().trimEnd()), 'the loop is steps.ts\'s, not a second spelling');
  assert.match(sh, /"\$CRUCIBLE" 'install' "\$type" \|\| die "step_failed: install-\$type"/);
  // tts names its engine, for the reason `crucible install tts` refuses without one.
  assert.match(sh, /"\$CRUCIBLE" 'install' "\$type" '--narrator-engine' "\$engine"/);
  assert.match(sh, /\*=\*\) type="\$\{entry%%=\*\}"; engine="\$\{entry#\*=\}" ;;/);
});

test('install.sh checks the droplet prerequisites BY NAME, before it downloads anything', () => {
  const sh = generateInstallSh();
  for (const code of ['no_nvidia_smi', 'no_nvidia_driver', 'no_cuda_arch', 'cuda_arch_too_old', 'no_ffmpeg', 'disk_too_small']) {
    assert.ok(sh.includes(`${code}:`), `${code} is refused by name`);
  }
  // Only on the backend that has a card to refuse over.
  assert.match(sh, /if \[ "\$BACKEND" = cuda-linux \]; then\n  command -v nvidia-smi/);
  // After the probe that measured the disk, before the pack that spends it.
  // On the install half, because the `--uninstall` branch says "server-pack"
  // too — that one is the pack being REMOVED.
  const half = installHalf(sh);
  const probe = half.indexOf('say "host-facts"');
  const prereq = half.indexOf('say "prerequisites"');
  const pack = half.indexOf('say "server-pack"\n');
  assert.ok(probe < prereq && prereq < pack, 'host-facts, then prerequisites, then the download');
  // ffmpeg is required only when something that decodes audio was asked for.
  assert.match(sh, /\*" tts"\*\|\*" asr"\*\|\*" rvc"\*\|\*" align"\*\|\*" denoise"\*\)/);
  assert.match(sh, /NO ffmpeg on PATH\. Nothing asked for today needs it/);
});

test('install.sh --from-source is a ROUTE and never a fallback for a failed download', () => {
  const sh = generateInstallSh();
  assert.match(sh, /if \[ -n "\$FROM_SOURCE" \]; then/);
  assert.match(sh, /git -C "\$src" checkout --detach "\$FROM_SOURCE"/);
  for (const code of ['from_source_clone_failed', 'from_source_ref_unknown', 'from_source_venv_failed', 'from_source_install_failed']) {
    assert.ok(sh.includes(`${code}:`), `${code} is refused by name`);
  }
  // The pack route still refuses its own failures by their own names — the
  // source build is reached by an argument, never by a download that failed.
  const fromSource = sh.indexOf('--from-source $FROM_SOURCE, building instead');
  const packFail = sh.indexOf('pack_download_failed');
  assert.ok(fromSource < packFail, 'the branch is chosen before anything is fetched');
  assert.equal(/pack_download_failed[\s\S]{0,200}FROM_SOURCE/.test(sh), false, 'no failure leads into the source build');
});

test('install.sh --uninstall calls the verb, then removes the pack the verb cannot', () => {
  const sh = generateInstallSh();
  const [firstLine = ''] = uninstallSh().trimEnd().split('\n');
  assert.ok(firstLine !== '' && sh.includes(firstLine), 'the block is steps.ts\'s');
  assert.match(sh, /"\$CRUCIBLE" 'uninstall' \$UNINSTALL_FLAGS \|\| die "step_failed: uninstall"/);
  // Order: the verb first, because it is running out of the directory the
  // next line deletes.
  const verb = sh.indexOf(`"$CRUCIBLE" 'uninstall'`);
  const pack = sh.indexOf('rm -rf "$CRUCIBLE_HOME/server"');
  assert.ok(verb > 0 && verb < pack, 'the interpreter runs before it is removed');
  // And the whole branch exits: an uninstall never falls through to an install.
  assert.match(sh, /say "uninstalled\."\n  exit 0\nfi/);
  assert.ok(sh.indexOf('if [ "$UNINSTALL" = 1 ]; then') < sh.indexOf('say "host-facts"'));
  assert.ok(sh.includes('not_installed:'), 'a machine with no Crucible is refused by name');
});

test('install.sh --uninstall keeps the weights, and the home, unless asked', () => {
  const sh = generateInstallSh();
  assert.match(sh, /if \[ "\$PURGE_WEIGHTS" = 1 \]; then UNINSTALL_FLAGS="\$UNINSTALL_FLAGS --purge-weights"; fi/);
  assert.match(sh, /if \[ "\$DRY_RUN" = 1 \]; then UNINSTALL_FLAGS="\$UNINSTALL_FLAGS --dry-run"; fi/);
  // The home goes only by rmdir, which cannot remove a directory with
  // anything in it. There is no `rm -rf "$CRUCIBLE_HOME"` anywhere.
  assert.match(sh, /if rmdir "\$CRUCIBLE_HOME" 2>\/dev\/null; then/);
  assert.equal(/rm -rf "\$CRUCIBLE_HOME"[^/]/.test(sh), false, 'the home is never recursively deleted');
  // --purge-weights on its own is a typo, not a request.
  assert.ok(sh.includes('flag_needs_uninstall:'));
});

test('install.sh --uninstall --dry-run describes the pack removal instead of doing it', () => {
  const sh = generateInstallSh();
  assert.match(sh, /say "server-pack: would remove \$CRUCIBLE_HOME\/server and \$CRUCIBLE_HOME\/downloads"/);
  assert.match(sh, /say "home: would remove \$CRUCIBLE_HOME if it were then empty"/);
});

test('install.sh refuses a flag it does not take, rather than ignoring it', () => {
  const sh = generateInstallSh();
  assert.ok(sh.includes('unknown_flag:'));
  assert.ok(sh.includes('flag_needs_value:'));
  assert.match(sh, /-h\|--help\) usage; exit 0 ;;/);
});

test('install.sh uses the same probe script, curl flags and tar flags the TypeScript does', () => {
  const sh = generateInstallSh();
  assert.ok(sh.includes(guestProbeScript(undefined)), 'the host probe is ONE script, not two');
  assert.ok(sh.includes(`curl ${CURL_ARGS.join(' ')} -o "$downloads/$part"`));
  assert.ok(sh.includes(`tar ${TAR_ARGS.join(' ')} "$archive" -C "$partial"`));
  assert.ok(sh.includes('activate_crucible_pack'), 'the shared verified activation transaction');
  assert.ok(sh.includes('local shutdown || return 1'), 'the runtime is stopped before replacement');
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
  // `& $Tar --version` and not `tar.exe --version`: `282871b` ("Name the
  // Windows tar where the installer is actually written") stopped resolving
  // `tar` through PATH, which from a Git Bash shell found GNU tar 1.32 and
  // refused a machine whose System32 bsdtar reads zstd perfectly well. The
  // probe and the unpack both go through the one named binary now, and these
  // three rows had been red since.
  const order = ['& $Tar --version', 'pack_not_published', 'pack_disk', 'Get-FileHash', 'Move-Item $Partial $HostDir', '& $Cmd local $action', 'Start-Process -WindowStyle Hidden'];
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
  assert.match(ps1, /& \$Cmd local \$action/);
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
  assert.ok(ps1.includes(`& $Tar ${TAR_ARGS.join(' ')} $archive -C $Partial`), 'the same tar flags');
});

test('install.ps1 checks that THIS machine tar carries zstd rather than assuming it', () => {
  // Measured 2026-09-14: C:\Windows\System32\tar.exe is bsdtar 3.8.1 with
  // libzstd 1.5.5, and no zstd.exe ships at all. A machine whose tar has no
  // zstd would half-unpack in silence.
  //
  // AND IT CHECKS THE TAR IT WILL USE. Measured 2026-09-17 deploying 0.6.8:
  // resolving `tar` through PATH from a Git Bash shell found GNU tar 1.32 in
  // Git's usr/bin and refused a machine whose System32 bsdtar reads zstd.
  // Checking one tool and unpacking with another is how a check passes and the
  // unpack still half-works, so the binary is NAMED once and both go through
  // it (`282871b`).
  const ps1 = generateInstallPs1();
  assert.match(ps1, /\$Tar = Join-Path \$env:SystemRoot "System32\\tar\.exe"/);
  assert.match(ps1, /& \$Tar --version/);
  assert.match(ps1, /\$tarVersion -notmatch "zstd"/);
  assert.match(ps1, /guest_missing_tool/);
  assert.equal(
    / tar\.exe /.test(ps1), false,
    'no bare tar.exe survives: PATH is the caller\'s, and the caller may be Git Bash',
  );
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

test('install.ps1 is ASCII, because Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI', () => {
  // MEASURED 2026-09-15: `[Parser]::ParseFile` on the generated script
  // reported "The string is missing the terminator" inside a Die message that
  // contained an em dash — 5.1 decoded the UTF-8 bytes as cp1252 and one of
  // the three characters that produced reads as a quote. `irm | iex` is not
  // the route that suffers (it decodes the HTTP charset); `.\install.ps1
  // -Uninstall` is, and that is the only way to pass a switch.
  const ps1 = generateInstallPs1();
  const offender = ps1.match(/[^\x00-\x7f]/);
  assert.equal(offender, null, `install.ps1 must be ASCII; found ${JSON.stringify(offender?.[0])}`);
});

test('asciiOnly refuses a character it has no spelling for, rather than dropping it', () => {
  assert.equal(asciiOnly('a — b … c', 'a test'), 'a  -  b ... c');
  assert.throws(() => asciiOnly('a ☃ b', 'a test'), /U\+2603/);
});

test('install.ps1 -Uninstall calls the verb, then removes the pack the verb cannot', () => {
  const ps1 = generateInstallPs1();
  assert.match(ps1, /\[switch\]\$Uninstall,/);
  assert.match(ps1, /\[switch\]\$PurgeWeights,/);
  assert.match(ps1, /\[switch\]\$DryRun,/);
  assert.match(ps1, /\[switch\]\$WslToo/);
  assert.match(ps1, /if \(\$Uninstall\) \{/);
  assert.match(ps1, /& \$Cmd @verb/, 'through the .cmd, which is the relocatable entry point (4.4)');
  // Order, and it is the only order that works: the verb runs out of the
  // directory the next lines delete.
  const verb = ps1.indexOf('& $Cmd @verb');
  const pack = ps1.indexOf('foreach ($gone in @($Partial, $DownloadDir, $HostDir))');
  assert.ok(verb > 0 && verb < pack);
  // And it exits before the install half.
  assert.ok(ps1.indexOf('if ($Uninstall) {') < ps1.indexOf('& $Tar --version'));
  assert.ok(ps1.includes('not_installed:'), 'a machine with no host pack is refused by name');
  assert.ok(ps1.includes('host_pack_locked:'), 'a file still held is refused by name, and re-running is the fix');
});

test('install.ps1 -Uninstall removes the home only when it is empty, never recursively-by-default', () => {
  const ps1 = generateInstallPs1();
  assert.match(ps1, /\$left = @\(Get-ChildItem -Force -Path \$Root -ErrorAction SilentlyContinue\)/);
  assert.match(ps1, /if \(\$left\.Count -eq 0\) \{/);
  assert.match(ps1, /home: KEPT \$Root/);
});

test('install.ps1 stays Windows PowerShell 5.1 safe with the uninstall branch in it', () => {
  // The same rule the install half is held to: no &&, ||, ??, ?. in 5.1.
  const ps1 = generateInstallPs1();
  const powershellOnly = ps1
    .split('\n')
    .filter((line) => !line.includes('bash -c') && !line.includes('| sh') && !line.includes('| iex'))
    .join('\n')
    .replace(/https?:\/\//g, '');
  assert.equal(/&&|\|\||\?\?|\?\./.test(powershellOnly), false);
});

test('a {sh} word is for the generated script only: renderArgv refuses one by name', () => {
  assert.throws(() => renderArgv(['crucible', { sh: '$BIND' }], {}), /shell fragment and there is no shell here/);
});

test('install.sh is unchanged by PHASE15: linux and darwin have no host', () => {
  const sh = generateInstallSh();
  assert.match(sh, /Linux\/x86_64\)  BACKEND=cuda-linux/);
  assert.match(sh, /Darwin\/arm64\)  BACKEND=mlx-darwin/);
  assert.equal(sh.includes('llama-windows'), false);
  assert.equal(sh.includes('crucible.cmd'), false);
});
