/**
 * The generated installers: up to date with the step list, and made of the
 * same constants `install()` uses. PHASE14 4a's "cannot differ" is this test.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { asciiOnly, GENERATED, generateInstallPs1, generateInstallSh } from '../scripts/gen-install-scripts.js';
import { CURL_ARGS, HOST_BACKEND, installSteps, interpreterFor, interpreterUrl, LATEST_RELEASE_URL, TAR_ARGS, wheelAssetName } from '../src/index.js';
import { UBUNTU_WSL_ROOTFS } from '../src/distro.js';
import { guestProbeScript } from '../src/runtime.js';
import { installJobTypesSh, renderArgv, uninstallSh } from '../src/steps.js';

/** The Windows host's pinned interpreter, from the one table both scripts read. */
const HOST_PIN = interpreterFor(HOST_BACKEND);

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
 * The `--uninstall` branch above it says `say "server"` too — it removes
 * the interpreter this script unpacked — so an order assertion that used `indexOf`
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
  assert.deepEqual(names, ['host-facts', 'server', 'init', 'env-patch-llm', 'service-install', 'local-register', 'local-install-cli', 'local-install-desktop', 'linger', 'capability-write', 'local-start']);
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
  // After the probe that measured the disk, before the download that spends it.
  // On the install half, because the `--uninstall` branch says "server" too —
  // that one is the runtime being REMOVED.
  const half = installHalf(sh);
  const probe = half.indexOf('say "host-facts"');
  const prereq = half.indexOf('say "prerequisites"');
  const server = half.indexOf('say "server"\n');
  assert.ok(probe < prereq && prereq < server, 'host-facts, then prerequisites, then the download');
  // ffmpeg is required only when something that decodes audio was asked for.
  assert.match(sh, /\*" tts"\*\|\*" asr"\*\|\*" rvc"\*\|\*" align"\*\|\*" denoise"\*\)/);
  assert.match(sh, /NO ffmpeg on PATH\. Nothing asked for today needs it/);
});

test('install.sh --from-source is a ROUTE and never a fallback for a failed download', () => {
  const sh = generateInstallSh();
  assert.match(sh, /if \[ -n "\$FROM_SOURCE" \]; then/);
  assert.match(sh, /git -C "\$src" checkout --detach "\$FROM_SOURCE"/);
  for (const code of ['from_source_clone_failed', 'from_source_ref_unknown', 'from_source_install_failed']) {
    assert.ok(sh.includes(`${code}:`), `${code} is refused by name`);
  }
  // IT REPLACES THE WHEEL HALF ONLY (PHASE20 section 3): the same pinned
  // interpreter is already at $dest either way, which is why this route needs
  // no python3 on the machine and builds no venv of its own.
  assert.equal(/--from-source needs a python3/.test(sh), false, 'the interpreter arrives before the branch');
  assert.equal(/python3 -m venv/.test(sh), false, 'no second venv: pip goes into the tree that is there');
  assert.match(sh, /"\$dest\/bin\/python3" -m pip install --upgrade --no-input "\$src"/);
  // The wheel route still refuses its own failures by their own names — the
  // source install is reached by an argument, never by a download that failed.
  const fromSource = sh.indexOf('--from-source $FROM_SOURCE, installing from a checkout');
  const wheelFail = sh.lastIndexOf('runtime_download_failed');
  assert.ok(fromSource > 0 && fromSource < wheelFail, 'the branch is chosen before the wheel is fetched');
  assert.equal(/runtime_download_failed[\s\S]{0,200}FROM_SOURCE/.test(sh), false, 'no failure leads into the source install');
});

test('install.sh --uninstall calls the verb, then removes the interpreter the verb cannot', () => {
  const sh = generateInstallSh();
  const [firstLine = ''] = uninstallSh().trimEnd().split('\n');
  assert.ok(firstLine !== '' && sh.includes(firstLine), 'the block is steps.ts\'s');
  assert.match(sh, /"\$CRUCIBLE" 'uninstall' \$UNINSTALL_FLAGS \|\| die "step_failed: uninstall"/);
  // Order: the verb first, because it is running out of the directory the
  // next line deletes.
  const verb = sh.indexOf(`"$CRUCIBLE" 'uninstall'`);
  const removal = sh.indexOf('rm -rf "$CRUCIBLE_HOME/server"');
  assert.ok(verb > 0 && verb < removal, 'the interpreter runs before it is removed');
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

test('install.sh --uninstall --dry-run describes the removal instead of doing it', () => {
  const sh = generateInstallSh();
  assert.match(sh, /say "server: would remove \$CRUCIBLE_HOME\/server and \$CRUCIBLE_HOME\/downloads"/);
  assert.match(sh, /say "home: would remove \$CRUCIBLE_HOME if it were then empty"/);
});

test('install.sh refuses a flag it does not take, rather than ignoring it', () => {
  const sh = generateInstallSh();
  assert.ok(sh.includes('unknown_flag:'));
  assert.ok(sh.includes('flag_needs_value:'));
  assert.match(sh, /-h\|--help\) usage; exit 0 ;;/);
});

test('install.sh uses the same probe script, pin, curl flags and tar flags the TypeScript does', () => {
  const sh = generateInstallSh();
  assert.ok(sh.includes(guestProbeScript(undefined)), 'the host probe is ONE script, not two');
  assert.ok(sh.includes(`curl ${CURL_ARGS.join(' ')} -o "$downloads/$py_asset"`));
  assert.ok(sh.includes(`tar ${TAR_ARGS.join(' ')} "$downloads/$py_asset" -C "$partial"`));
  // The PIN, not a copy of it: a script that spelled its own digest would be a
  // second owner of the one thing PHASE20 section 2 says has exactly one.
  const linux = interpreterFor('cuda-linux');
  assert.ok(sh.includes(`py_sha='${linux.sha256}'`), 'the cuda-linux digest is the table\'s');
  assert.ok(sh.includes(`py_url='${interpreterUrl(linux)}'`));
  assert.ok(sh.includes(interpreterFor('mlx-darwin').sha256), 'and the Mac\'s');
  assert.ok(sh.includes(`wheel="${wheelAssetName('$RELEASE')}"`), 'the wheel is named the one way');
  assert.ok(sh.includes('activate_crucible_runtime'), 'the shared verified activation transaction');
  assert.ok(sh.includes('local shutdown || return 1'), 'the runtime is stopped before replacement');
  assert.ok(sh.includes('printf \'python_sha256=%s\\npython_version=%s\\nrelease=%s\\n\''), 'the same stamp');
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
    'runtime_download_failed',
    'runtime_sha_mismatch',
    'runtime_unpack_failed',
    'runtime_install_failed',
    'release_channel_unreadable',
    'install_would_downgrade',
    'rollback_version_mismatch',
  ]) {
    assert.ok(sh.includes(`${code}:`), `install.sh refuses ${code} by the same name`);
  }
});

/**
 * INSTALL-UNINSTALL.md §6.5, in both hand installers.
 *
 * `releases?per_page=1` is the newest TAG, which between a cut and its
 * promotion is the unverified candidate `promote_release.py` exists to hold
 * back; `releases/latest` is the promoted one. And neither script may walk a
 * machine backwards without an operator naming the version.
 */
test('both installers read the channel\'s releases/latest, never the newest tag', () => {
  for (const [name, text] of [['install.sh', generateInstallSh()], ['install.ps1', generateInstallPs1()]] as const) {
    assert.ok(text.includes(LATEST_RELEASE_URL), `${name} does not read ${LATEST_RELEASE_URL}`);
    // What is FETCHED, not what is mentioned: both scripts name the old feed in
    // the comment that says why they stopped reading it.
    const fetched = text.split('\n').filter((line) => /curl/.test(line) && !line.trimStart().startsWith('#'));
    assert.equal(fetched.some((line) => line.includes('per_page')), false, `${name} still fetches the newest tag created`);
  }
});

test('both installers refuse to install over a newer release, and take an exact-version rollback', () => {
  const sh = generateInstallSh();
  // The FLAG is parsed and the value is what the gate reads — one assertion each,
  // because a script that takes `--rollback-to` and never reads it would pass a
  // regex that only looked for the word.
  assert.match(sh, /--rollback-to\) need \$# "--rollback-to"; shift; ROLLBACK_TO="\$1"/);
  assert.match(sh, /\[ "\$ROLLBACK_TO" = "\$RELEASE" \]/);
  assert.match(sh, /stamp_release=/, 'install.sh must read the release the stamp records');
  assert.match(sh, /crucible_older "\$RELEASE" "\$stamp_release"/);
  assert.match(sh, /install_would_downgrade: \$dest is the \$stamp_release release/);

  const ps1 = generateInstallPs1();
  assert.match(ps1, /\$RollbackTo/);
  assert.match(ps1, /\$haveRelease/, 'install.ps1 must read the release the stamp records');
  assert.match(ps1, /install_would_downgrade: \$HostDir is the \$haveRelease release/);
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
  const order = ['$Tar = Join-Path', '$PyUrl =', 'Get-FileHash', 'Move-Item $staged $HostDir', 'pip install --upgrade --no-input', '& $Cmd local $action', 'Start-Process -WindowStyle Hidden'];
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

test('install.ps1 installs the PINNED interpreter and THIS release\'s wheel into %LOCALAPPDATA%', () => {
  const ps1 = generateInstallPs1();
  // The same table install.sh reads, for the Windows backend. A .ps1 that
  // spelled its own digest would install a different interpreter under one
  // version number (PHASE20 section 2).
  assert.ok(ps1.includes(`$PySha = '${HOST_PIN.sha256}'`), 'the pin is the table\'s');
  assert.ok(ps1.includes(`$PyAsset = '${HOST_PIN.asset}'`));
  assert.ok(ps1.includes(interpreterUrl(HOST_PIN)), 'from python-build-standalone, not from our release');
  assert.match(ps1, /\$HostDir = Join-Path \$Root 'host'/);
  assert.match(ps1, /\[string\]\$Root = "\$env:LOCALAPPDATA\\Crucible"/);
  assert.ok(ps1.includes(`& curl.exe ${CURL_ARGS.join(' ')} -o $archive`), 'the same curl flags');
  assert.ok(ps1.includes(`& $Tar ${TAR_ARGS.join(' ')} $archive -C $Partial`), 'the same tar flags');
  assert.ok(ps1.includes(`$Wheel = "${wheelAssetName('$Release')}"`), 'the wheel is named the one way');
  assert.match(ps1, /pip install --upgrade --no-input \$WheelPath/);
  assert.match(ps1, /pip install pystray pillow/, 'the tray, which is not a wheel dependency');
});

test('install.ps1 names the tar it uses rather than resolving one through PATH', () => {
  // Measured 2026-09-17 deploying 0.6.8: resolving `tar` through PATH from a
  // Git Bash shell found GNU tar 1.32 in Git's usr/bin and refused a machine
  // whose System32 bsdtar was fine. The binary is NAMED once (`282871b`).
  //
  // THE ZSTD CHECK IS GONE WITH THE PACKS. python-build-standalone publishes
  // gzip, which every tar reads, so a probe for a format nothing downloads
  // would refuse a machine for a tool it does not need.
  const ps1 = generateInstallPs1();
  assert.match(ps1, /\$Tar = Join-Path \$env:SystemRoot "System32\\tar\.exe"/);
  assert.match(ps1, /guest_missing_tool/);
  assert.equal(ps1.includes('zstd'), false, 'nothing this installer fetches is zstd');
  assert.equal(
    / tar\.exe /.test(ps1), false,
    'no bare tar.exe survives: PATH is the caller\'s, and the caller may be Git Bash',
  );
});

test('install.ps1 verifies every download before it is used, and is Windows PowerShell 5.1 safe', () => {
  const ps1 = generateInstallPs1();
  assert.match(ps1, /Get-FileHash -Algorithm SHA256/);
  assert.match(ps1, /runtime_sha_mismatch/);
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

test('install.ps1 unpacks BESIDE, then moves, and pips ONLY into the final path', () => {
  // PHASE14 7.2a's defect, in its Windows form: pip writes Scripts\*.exe
  // launchers with the interpreter's absolute path inside the BINARY, which no
  // shebang rewrite can reach. PHASE20 removes the defect rather than
  // correcting it — pip runs from the tree at %LOCALAPPDATA%\Crucible\host,
  // which is where it stays, so the path the launcher bakes is the right one.
  const ps1 = generateInstallPs1();
  assert.match(ps1, /\$Partial = "\$HostDir\.partial"/);
  assert.match(ps1, /& \(Join-Path \$staged "python\.exe"\) --version/);
  assert.match(ps1, /Move-Item \$staged \$HostDir/);
  assert.ok(
    ps1.indexOf('Move-Item $staged $HostDir') < ps1.indexOf('pip install --upgrade --no-input'),
    'the tree is at its FINAL path before pip writes a launcher into it',
  );
  // And the .cmd, which every other part of Crucible spells (host/paths.py).
  assert.match(ps1, /\[System\.IO\.File\]::WriteAllText\(\$Cmd, \$shim/);
  assert.match(ps1, /%~dp0python\.exe/);
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

test('install.ps1 -Uninstall calls the verb, then removes the runtime the verb cannot', () => {
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
  const removal = ps1.indexOf('foreach ($gone in @($Partial, $DownloadDir, $HostDir))');
  assert.ok(verb > 0 && verb < removal);
  // And it exits before the install half.
  assert.ok(ps1.indexOf('if ($Uninstall) {') < ps1.indexOf('$Tar = Join-Path'));
  assert.ok(ps1.includes('not_installed:'), 'a machine with no host is refused by name');
  assert.ok(ps1.includes('host_runtime_locked:'), 'a file still held is refused by name, and re-running is the fix');
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

test('neither installer names an asset of ours that PHASE20 deleted', () => {
  // The grep `tests/test_no_packs.py` runs over the repo, applied to the two
  // GENERATED files — which that grep reads too, but only after this suite has
  // regenerated them.
  for (const [name, text] of [['install.sh', generateInstallSh()], ['install.ps1', generateInstallPs1()]] as const) {
    for (const gone of ['envpacks.json', 'crucible-env-', 'crucible-rootfs-']) {
      assert.equal(text.includes(gone), false, `${name} still names ${gone}`);
    }
  }
});

test('the WSL image comes from Canonical, and its digest from Canonical\'s own sums file', () => {
  // PHASE20 section 2. The host imports it (install.ps1 does not), so what this
  // asserts is the generated PYTHON table — the one owner crossing the seam.
  const py = GENERATED.find((file) => file.path.endsWith('wsl_states.py'));
  assert.ok(py !== undefined, 'the generator still writes the Python table');
  assert.ok(py.text.includes(UBUNTU_WSL_ROOTFS));
  assert.ok(py.text.includes('cloud-images.ubuntu.com/wsl/releases/24.04/current'));
  assert.ok(py.text.includes('SHA256SUMS'));
  assert.equal(py.text.includes('ROOTFS_ASSET_TEMPLATE'), false, 'no asset of ours any more');
  // And what the guest network probe reaches for is an asset that still exists.
  assert.ok(py.text.includes('crucible-{release}-py3-none-any.whl'), 'the probe curls the wheel');
  assert.equal(py.text.includes('envpacks.json'), false);
  // The four things Canonical's image does not have, as ONE script with ONE owner.
  assert.ok(py.text.includes('FINISH_IMPORT_SCRIPT'));
  assert.ok(py.text.includes('useradd --create-home --shell /bin/bash crucible'));
  assert.ok(py.text.includes('# crucible-rootfs'));
});
