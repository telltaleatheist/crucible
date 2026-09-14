/**
 * Generate `install.sh` and `install.ps1` from the step list and the WSL state
 * table — the two things PHASE14-ENVPACKS.md 4a and 4c say have ONE owner.
 *
 *   node build/scripts/gen-install-scripts.js            # write them
 *   node build/scripts/gen-install-scripts.js --check    # fail if they drift
 *
 * The `--check` mode is what `test/unit-gen-install.test.ts` asserts, so a
 * change to `src/steps.ts` that is not regenerated fails the suite rather than
 * shipping a hand installer that does something else than the app does.
 *
 * What the scripts deliberately do NOT do (4a): no job types, no weights. A
 * bare Crucible that serves nothing until an app or the operator page asks.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { CRUCIBLE_DISTRO, WSL_CONF_MARKER, WSL_CONF_TEXT } from '../src/distro.js';
import { ENVPACKS_ASSET, RELEASE_REPO, rootfsAssetName } from '../src/envpacks.js';
import { installSteps } from '../src/steps.js';
import { BOOTSTRAP_VERSION } from '../src/version.js';
import { wslStates, type WslStateDef } from '../src/wsl-states.js';

const HERE = dirname(fileURLToPath(import.meta.url));
/** build/scripts → the package root. Written next to this generator, in `scripts/`. */
const SCRIPTS = join(HERE, '..', '..', 'scripts');

const BANNER = (comment: string): string =>
  `${comment} GENERATED FILE — do not edit.\n`
  + `${comment} Written by sdk/bootstrap/scripts/gen-install-scripts.ts from src/steps.ts\n`
  + `${comment} and src/wsl-states.ts, so a hand install and an app-driven install cannot\n`
  + `${comment} differ (PHASE14-ENVPACKS.md 4a). Regenerate: npm run gen:install\n`;

/** The standalone installer's plan: no job types, no weights (4a). */
const STANDALONE = { enableFlags: [], installs: [], bind: [], linger: true } as const;

export function generateInstallSh(): string {
  const steps = installSteps(STANDALONE);
  const lines: string[] = [
    '#!/bin/sh',
    BANNER('#').trimEnd(),
    '#',
    '# Install a Crucible on this machine (Linux x86_64, macOS arm64, or inside a',
    '# WSL2 distro). Downloads the server pack from the release, initialises it,',
    '# installs the service, and prints the line that pairs an app with it.',
    '#',
    '#   curl -fsSL https://github.com/' + RELEASE_REPO + '/releases/latest/download/install.sh | sh',
    '#',
    '# CRUCIBLE_RELEASE=<version> picks a release other than the one this script',
    '# was cut with. Everything here is idempotent: run it again after a failure.',
    '',
    'set -eu',
    '',
    `RELEASE="\${CRUCIBLE_RELEASE:-${BOOTSTRAP_VERSION}}"`,
    '',
    "say() { printf 'crucible: %s\\n' \"$*\"; }",
    "die() { printf 'crucible: %s\\n' \"$*\" >&2; exit 1; }",
    '',
    '# --- backend -------------------------------------------------------------',
    '# Two backends and no third. Windows is never one: on Windows this script',
    '# runs INSIDE the WSL2 distro that install.ps1 imported.',
    'case "$(uname -s)/$(uname -m)" in',
    '  Linux/x86_64)  BACKEND=cuda-linux; SHA_TOOL="sha256sum";     MECHANISM=systemd ;;',
    '  Darwin/arm64)  BACKEND=mlx-darwin; SHA_TOOL="shasum -a 256"; MECHANISM=launchd ;;',
    '  *) die "unsupported_platform: $(uname -s)/$(uname -m) is not a Crucible backend '
      + '(cuda-linux on Linux x86_64, mlx-darwin on Apple Silicon)" ;;',
    'esac',
    'say "release $RELEASE, backend $BACKEND"',
    '',
  ];
  for (const step of steps) {
    lines.push(`# --- ${step.name} ${'-'.repeat(Math.max(0, 68 - step.name.length))}`);
    lines.push(`# ${step.what}`);
    lines.push(`say "${step.name}"`);
    lines.push(step.sh.trimEnd());
    lines.push('');
  }
  lines.push('# --- done ----------------------------------------------------------------');
  lines.push('say "installed. Pair an app with the line below."');
  lines.push('"$CRUCIBLE" token --url');
  lines.push('');
  return lines.join('\n');
}

/** The one row we need by name, refused loudly if the table is renamed under us. */
function row(states: WslStateDef[], code: string): WslStateDef {
  const found = states.find((state) => state.code === code);
  if (found === undefined) throw new Error(`gen-install-scripts: the WSL state table has no "${code}" row any more`);
  return found;
}

function psQuote(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

/**
 * `/etc/wsl.conf` as ONE bash command, from {@link WSL_CONF_TEXT}. `printf`
 * with a `%s\n` per line and the lines as arguments: wsl.exe halves
 * backslashes before bash exists, so the `\n` is doubled here to arrive whole.
 */
function wslConfPrintf(): string {
  const lines = WSL_CONF_TEXT.split('\n').filter((line) => line !== '');
  const args = lines.map((line) => `'${line.replace(/'/g, `'\\''`)}'`).join(' ');
  return `printf '%s\\\\n' ${args} > /etc/wsl.conf`;
}

export function generateInstallPs1(): string {
  const states = wslStates({ release: BOOTSTRAP_VERSION });
  const enable = row(states, 'wsl_missing');
  const version2 = row(states, 'wsl1_only');
  const firmware = row(states, 'virtualization_disabled');
  const noDistro = row(states, 'no_crucible_distro');
  const enableAction = enable.action({ code: 0, stdout: '', stderr: '', failure: null }, { results: {}, distros: [] });
  const version2Action = version2.action({ code: 0, stdout: '', stderr: '', failure: null }, { results: {}, distros: [] });
  if (enableAction.kind !== 'run-elevated' || version2Action.kind !== 'run') {
    throw new Error('gen-install-scripts: the WSL table changed shape; install.ps1 must be re-thought, not re-run');
  }
  const firmwareText = firmware.action({ code: 0, stdout: '', stderr: '', failure: null }, { results: {}, distros: [] });
  const asset = rootfsAssetName('$Release');
  const lines: string[] = [
    BANNER('#').trimEnd(),
    '#',
    '# Install a Crucible on Windows. Walks every WSL state in the order',
    '# PHASE14-ENVPACKS.md 4c lists them, imports a distribution Crucible owns',
    '# (your own WSL is never touched), then runs install.sh inside it.',
    '#',
    '#   irm https://github.com/' + RELEASE_REPO + '/releases/latest/download/install.ps1 | iex',
    '#',
    '# Idempotent and resumable: after the reboot WSL asks for, run it again.',
    '',
    '[CmdletBinding()]',
    'param(',
    `  [string]$Release = ${psQuote(BOOTSTRAP_VERSION)},`,
    '  [string]$InstallDir = "$env:LOCALAPPDATA\\Crucible\\wsl",',
    '  [string]$DownloadDir = "$env:LOCALAPPDATA\\Crucible\\downloads"',
    ')',
    '',
    '# Continue, not Stop: every call below is a native program whose exit code',
    '# is checked explicitly, and Windows PowerShell 5.1 turns a native command'
      + ' writing to stderr into a terminating error under Stop.',
    '$ErrorActionPreference = "Continue"',
    `$Distro = ${psQuote(CRUCIBLE_DISTRO)}`,
    '',
    'function Say($m) { Write-Host "crucible: $m" }',
    'function Die($m) { Write-Host "crucible: $m" -ForegroundColor Red; exit 1 }',
    '',
    '# wsl.exe writes UTF-16LE; PowerShell 5.1 reads it as UTF-8 and produces text',
    '# with a NUL between every character. Stripped here so -match works.',
    'function Get-WslText($output) { return (($output | Out-String) -replace "`0", "") }',
    '',
    '# --- 1. virtualization / the WSL feature ---------------------------------',
    'Say "checking WSL"',
    '$status = ""',
    '$wslOk = $false',
    'try { $status = Get-WslText (& wsl.exe --status); $wslOk = ($LASTEXITCODE -eq 0) } catch { $status = "$_"; $wslOk = $false }',
    'if (-not $wslOk -and ($status -match "HCS_E_HYPERV_NOT_INSTALLED" -or $status -match "0x80370102")) {',
    `  Die ${psQuote(firmwareText.kind === 'instruct' ? firmwareText.text : '')}`,
    '}',
    'if (-not $wslOk) {',
    `  Say ${psQuote(enable.sentence({ code: 1, stdout: '', stderr: '', failure: 'ENOENT' }, { results: {}, distros: [] }))}`,
    '  Say "A Windows permission prompt will appear."',
    `  Start-Process -Verb RunAs -Wait -FilePath ${psQuote(enableAction.argv[0] ?? 'wsl.exe')}`
      + ` -ArgumentList ${enableAction.argv.slice(1).map(psQuote).join(',')}`,
    '  Say "WSL is enabled. RESTART Windows, then run this script again."',
    '  exit 0',
    '}',
    '',
    '# --- 2. WSL2 as the default version --------------------------------------',
    'if ($status -match "Default Version:\\s*1") {',
    `  Say ${psQuote(version2.sentence({ code: 0, stdout: '', stderr: '', failure: null }, { results: {}, distros: [] }))}`,
    `  & ${version2Action.argv.map(psQuote).join(' ')}`,
    '}',
    '',
    '# --- 3. the distribution Crucible owns ------------------------------------',
    '$list = Get-WslText (& wsl.exe -l -q)',
    '$have = $false',
    'foreach ($line in ($list -split "`r?`n")) { if ($line.Trim() -eq $Distro) { $have = $true } }',
    'if (-not $have) {',
    `  Say ${psQuote(noDistro.sentence({ code: 0, stdout: '', stderr: '', failure: null }, { results: {}, distros: [] }))}`,
    `  $asset = "${asset}"`,
    '  $rootfs = Join-Path $DownloadDir $asset',
    `  $base = "https://github.com/${RELEASE_REPO}/releases/download/v$Release"`,
    '  New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null',
    '  Say "downloading $asset"',
    '  & curl.exe -fL --retry 3 --create-dirs -o $rootfs "$base/$asset"',
    '  if ($LASTEXITCODE -ne 0) { Die "pack_download_failed: $base/$asset" }',
    '  $wantRaw = & curl.exe -fsSL --retry 3 "$base/$asset.sha256"',
    '  if ($LASTEXITCODE -ne 0) { Die "pack_download_failed: $base/$asset.sha256" }',
    '  $want = ($wantRaw -split "\\s+")[0].ToLower()',
    '  $got = (Get-FileHash -Algorithm SHA256 -Path $rootfs).Hash.ToLower()',
    '  if ($got -ne $want) { Remove-Item $rootfs -Force; Die "pack_sha_mismatch: $asset hashes $got, the release says $want" }',
    '  New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null',
    '  Say "importing $Distro into $InstallDir"',
    '  & wsl.exe --import $Distro $InstallDir $rootfs --version 2',
    '  if ($LASTEXITCODE -ne 0) { Die "distro_import_failed: wsl --import exited $LASTEXITCODE" }',
    '}',
    '',
    '# --- 4. systemd in that distribution --------------------------------------',
    '$conf = Get-WslText (& wsl.exe -d $Distro -u root --exec bash -c "test -f /etc/wsl.conf && cat /etc/wsl.conf || true")',
    `if ($conf -notmatch ${psQuote(WSL_CONF_MARKER.replace('#', '').trim())}) {`,
    '  Say "writing /etc/wsl.conf in $Distro"',
    '  # One printf per line, because wsl.exe halves backslashes on the way in',
    '  # and a heredoc would need newlines inside one argument.',
    `  $confCmd = ${psQuote(wslConfPrintf())}`,
    '  & wsl.exe -d $Distro -u root --exec bash -c $confCmd',
    '  if ($LASTEXITCODE -ne 0) { Die "distro_not_systemd: could not write /etc/wsl.conf in $Distro" }',
    '  # --terminate, never --shutdown: the other distributions on this machine',
    '  # are not ours to stop.',
    '  & wsl.exe --terminate $Distro',
    '}',
    '',
    '# --- 5. the same install.sh, inside ---------------------------------------',
    'Say "running install.sh inside $Distro"',
    `$sh = "https://github.com/${RELEASE_REPO}/releases/download/v$Release/install.sh"`,
    '& wsl.exe -d $Distro --exec bash -c "CRUCIBLE_RELEASE=$Release curl -fsSL $sh | sh"',
    'if ($LASTEXITCODE -ne 0) { Die "install.sh exited $LASTEXITCODE inside $Distro" }',
    'Say "done."',
    '',
  ];
  return lines.join('\n');
}

export const GENERATED = [
  { path: join(SCRIPTS, 'install.sh'), text: generateInstallSh() },
  { path: join(SCRIPTS, 'install.ps1'), text: generateInstallPs1() },
];

function main(): void {
  const check = process.argv.includes('--check');
  let drifted = 0;
  for (const file of GENERATED) {
    if (check) {
      let onDisk = '';
      try {
        onDisk = readFileSync(file.path, 'utf8');
      } catch {
        onDisk = '';
      }
      if (onDisk.replace(/\r\n/g, '\n') !== file.text) {
        drifted += 1;
        console.error(`gen-install-scripts: ${file.path} is not what src/steps.ts says. Run: npm run gen:install`);
      }
    } else {
      writeFileSync(file.path, file.text, 'utf8');
      console.log(`gen-install-scripts: wrote ${file.path}`);
    }
  }
  if (drifted > 0) process.exitCode = 1;
}

// Only when run as a program; the test imports the two generators directly.
if (process.argv[1] !== undefined && process.argv[1].endsWith('gen-install-scripts.js')) main();
