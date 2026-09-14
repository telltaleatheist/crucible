/**
 * Generate `install.sh`, `install.ps1` and `crucible/host/wsl_states.py` from
 * the step list and the WSL state table — the things PHASE14-ENVPACKS.md 4a
 * and 4c, and PHASE15-HOST.md 4.3, say have ONE owner.
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
 *
 * THE THIRD OUTPUT IS PYTHON, AND THAT IS WHY IT IS HERE (PHASE15 4.3)
 * --------------------------------------------------------------------
 * `crucible host` drives the same 4c table, and it is Python. Two hand-written
 * copies of a ten-row table with ten sentences in it is exactly the shape
 * ARCHITECTURE.md R1 forbids, so the table's DATA is emitted into
 * `crucible/host/wsl_states.py` from this file, and `--check` refuses a drift
 * in it the same way it does for the two scripts. What is NOT emitted is the
 * `means` predicates, which are code rather than data — those live in
 * `crucible/host/wslstate.py`, one per code, and a pytest asserts the two sets
 * are equal. It is the seam `steps.ts` already uses for its three programs and
 * the one `envpack.SMOKE_IMPORT` uses for `cli.INSTALLABLE_JOB_TYPES`.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { CRUCIBLE_DISTRO, WSL_CONF_MARKER, WSL_CONF_TEXT } from '../src/distro.js';
import { ENVPACKS_ASSET, HOST_BACKEND, HOST_PACK, RELEASE_REPO } from '../src/envpacks.js';
import { CURL_ARGS, DOWNLOADS_SUBDIR, HOST_SUBDIR, PARTIAL_SUFFIX, STAMP_NAME, TAR_ARGS } from '../src/pack.js';
import type { RunResult } from '../src/runner.js';
import { installSteps } from '../src/steps.js';
import { BOOTSTRAP_VERSION } from '../src/version.js';
import { probeArgv, wslStates, type ProbeKey, type WslStateDef } from '../src/wsl-states.js';

const HERE = dirname(fileURLToPath(import.meta.url));
/** build/scripts → the package root. Written next to this generator, in `scripts/`. */
const SCRIPTS = join(HERE, '..', '..', 'scripts');
/** build/scripts → sdk/bootstrap → sdk → the repo root, where `crucible/` is. */
const REPO = join(HERE, '..', '..', '..', '..');

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
  const enableAction = enable.action({ code: 0, stdout: '', stderr: '', failure: null }, { results: {}, distros: [] });
  if (enableAction.kind !== 'run-elevated') {
    throw new Error('gen-install-scripts: the WSL table changed shape; install.ps1 must be re-thought, not re-run');
  }
  const base = `https://github.com/${RELEASE_REPO}/releases/download/v$Release`;
  const lines: string[] = [
    BANNER('#').trimEnd(),
    '#',
    '# Install Crucible on Windows: the HOST, and nothing else.',
    '#',
    '# PHASE15-HOST.md 4.4. This script used to walk the WSL states itself and',
    '# then run install.sh inside an imported distribution. It no longer does,',
    '# and that is the point of the phase: `crucible host` owns that sequence',
    '# (4.3), it can carry a reboot across because it starts at login, and the',
    '# page drives it as a task (4.7) — an app that asks for an install talks',
    '# to the SAME implementation through the host loopback door. Two walks of',
    '# one table was the thing being removed.',
    '#',
    '# So: download the host pack for this release, verify it, unpack it to',
    '# %LOCALAPPDATA%\\Crucible\\host\\, register the Startup item, start the',
    '# host, and STOP.',
    '#',
    '#   irm https://github.com/' + RELEASE_REPO + '/releases/latest/download/install.ps1 | iex',
    '#',
    '# No admin. Everything here is per-user and idempotent: run it again after',
    '# a failure and it resumes from whatever is already on disk.',
    '',
    '[CmdletBinding()]',
    'param(',
    `  [string]$Release = ${psQuote(BOOTSTRAP_VERSION)},`,
    '  [string]$Root = "$env:LOCALAPPDATA\\Crucible"',
    ')',
    '',
    '# Continue, not Stop: every call below is a native program whose exit code',
    '# is checked explicitly, and Windows PowerShell 5.1 turns a native command'
      + ' writing to stderr into a terminating error under Stop.',
    '$ErrorActionPreference = "Continue"',
    `$HostDir = Join-Path $Root ${psQuote(HOST_SUBDIR)}`,
    `$DownloadDir = Join-Path $Root ${psQuote(DOWNLOADS_SUBDIR)}`,
    `$Partial = "$HostDir${PARTIAL_SUFFIX}"`,
    `$Stamp = Join-Path $HostDir ${psQuote(STAMP_NAME)}`,
    '$Cmd = Join-Path $HostDir "crucible.cmd"',
    '$Pythonw = Join-Path $HostDir "pythonw.exe"',
    '',
    'function Say($m) { Write-Host "crucible: $m" }',
    'function Die($m) { Write-Host "crucible: $m" -ForegroundColor Red; exit 1 }',
    '',
    '# --- 0. this machine can hold the host -----------------------------------',
    '# 64-bit x86 only: the pinned interpreter is',
    '# x86_64-pc-windows-msvc-install_only and there is no second pin (4.4).',
    'if ([System.Environment]::Is64BitOperatingSystem -ne $true) {',
    '  Die "unsupported_platform: the Crucible Windows host is 64-bit x86 only."',
    '}',
    'if (-not $env:LOCALAPPDATA) {',
    '  Die "host_no_localappdata: LOCALAPPDATA is not set, so there is no per-user place to install into."',
    '}',
    '',
    '# A pack is a zstd tarball. Windows 10 1803+ and Windows 11 ship bsdtar',
    '# linked with libzstd, so no zstd.exe is needed — MEASURED on 2026-09-14:',
    '# bsdtar 3.8.1 / libarchive 3.8.1 / libzstd 1.5.5. A machine whose tar has',
    '# no zstd would half-unpack in silence, so it is CHECKED, not assumed.',
    '$tarVersion = ""',
    'try { $tarVersion = (& tar.exe --version | Out-String) } catch { $tarVersion = "" }',
    'if ($tarVersion -notmatch "zstd") {',
    '  Die "guest_missing_tool: this machine tar cannot read zstd (tar --version said: $($tarVersion.Trim())). Windows 10 1803+ and Windows 11 ship one that can."',
    '}',
    '',
    '# --- 1. which pack -------------------------------------------------------',
    'Say "release $Release"',
    `$manifestUrl = "${base}/${ENVPACKS_ASSET}"`,
    '$manifestRaw = & curl.exe -fsSL --retry 3 "$manifestUrl"',
    'if ($LASTEXITCODE -ne 0) { Die "pack_manifest_unreadable: could not fetch $manifestUrl" }',
    'try { $manifest = $manifestRaw | Out-String | ConvertFrom-Json } catch { Die "pack_manifest_unreadable: $manifestUrl is not JSON" }',
    'if ($manifest.schema -ne 1) { Die "pack_manifest_unreadable: $manifestUrl declares schema $($manifest.schema), this installer reads 1" }',
    '$pack = $null',
    `foreach ($entry in $manifest.packs) { if ($entry.name -eq ${psQuote(HOST_PACK)} -and $entry.backend -eq ${psQuote(HOST_BACKEND)}) { $pack = $entry } }`,
    `if ($null -eq $pack) { Die "pack_not_published: the $Release release publishes no ${HOST_PACK} pack for ${HOST_BACKEND}" }`,
    '',
    '# --- 2. already installed? ------------------------------------------------',
    '# The stamp is the same two lines the guest-side install writes, read the',
    '# same way: a matching sha means these bytes are already unpacked.',
    '$have = ""',
    'if (Test-Path $Stamp) {',
    '  foreach ($line in (Get-Content $Stamp)) { if ($line -match "^sha256=(.+)$") { $have = $Matches[1].Trim() } }',
    '}',
    'if ($have -eq $pack.sha256 -and (Test-Path $Cmd)) {',
    '  Say "host-pack: already installed ($($pack.sha256))"',
    '} else {',
    '  # --- 3. disk ------------------------------------------------------------',
    '  # The same sum pack.ts requiredBytes() uses: unpacked + the whole archive',
    '  # + one part, a part being the archive over the part count.',
    '  $need = $pack.unpacked_bytes + $pack.bytes + [math]::Floor($pack.bytes / $pack.parts.Count)',
    '  $drive = (Get-Item $env:LOCALAPPDATA).PSDrive',
    '  if ($drive.Free -lt $need) {',
    '    Die "pack_disk: the host pack needs $([math]::Round($need/1GB,1)) GiB free on $($drive.Name): and there is $([math]::Round($drive.Free/1GB,1)) GiB. Nothing has been downloaded."',
    '  }',
    '',
    '  # --- 4. download, join, verify -------------------------------------------',
    '  New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null',
    '  $archiveName = $pack.parts[0] -replace "\\.part[0-9]+$", ""',
    '  $archive = Join-Path $DownloadDir $archiveName',
    '  if (Test-Path $archive) { Remove-Item $archive -Force }',
    '  foreach ($part in $pack.parts) {',
    '    Say "host-pack: $part"',
    '    $partPath = Join-Path $DownloadDir $part',
    `    & curl.exe ${CURL_ARGS.join(' ')} -o $partPath "${base}/$part"`,
    `    if ($LASTEXITCODE -ne 0) { Die "pack_download_failed: ${base}/$part" }`,
    '    # Byte-for-byte append, then delete: peak extra disk is ONE part and',
    '    # not the whole set. Add-Content would re-encode the bytes as text.',
    '    $in = [System.IO.File]::OpenRead($partPath)',
    '    $out = [System.IO.File]::Open($archive, "Append", "Write")',
    '    try { $in.CopyTo($out) } finally { $out.Close(); $in.Close() }',
    '    Remove-Item $partPath -Force',
    '  }',
    '  $got = (Get-FileHash -Algorithm SHA256 -Path $archive).Hash.ToLower()',
    '  if ($got -ne $pack.sha256) {',
    '    Remove-Item $archive -Force',
    '    Die "pack_sha_mismatch: $archiveName hashes $got, the manifest says $($pack.sha256)"',
    '  }',
    '',
    '  # --- 5. unpack beside, prove it runs, THEN move --------------------------',
    '  if (Test-Path $Partial) { Remove-Item $Partial -Recurse -Force }',
    '  New-Item -ItemType Directory -Force -Path $Partial | Out-Null',
    `  & tar.exe ${TAR_ARGS.join(' ')} $archive -C $Partial`,
    '  if ($LASTEXITCODE -ne 0) { Die "pack_unpack_failed: tar would not open $archive" }',
    '  # The .cmd and not the .exe: pip Scripts\\*.exe launchers bake the build',
    '  # tree interpreter path into the binary and do not survive this move',
    '  # (PHASE15-HOST.md 4.4, and PHASE14 7.2a for the POSIX half of it).',
    '  & (Join-Path $Partial "crucible.cmd") --version | Out-Null',
    '  if ($LASTEXITCODE -ne 0) { Die "pack_unpack_failed: crucible.cmd in $Partial would not run" }',
    '  if (Test-Path $HostDir) { Remove-Item $HostDir -Recurse -Force }',
    '  Move-Item $Partial $HostDir',
    '  Set-Content -Path $Stamp -Encoding ascii -Value @("sha256=$($pack.sha256)", "release=$Release")',
    '  Remove-Item $archive -Force',
    '  Say "host-pack: unpacked $($pack.parts.Count) part(s) into $HostDir (Python $($pack.python))"',
    '}',
    '',
    '# --- 6. start at login ----------------------------------------------------',
    '# The host OWNS that shortcut (4.1). This script asks for it by verb rather',
    '# than writing a .lnk of its own, so there is one spelling of what it points',
    '# at and one place that changes when it moves.',
    '& $Cmd host --install-startup',
    'if ($LASTEXITCODE -ne 0) { Die "the Startup item could not be written (crucible host --install-startup exited $LASTEXITCODE)" }',
    '',
    '# --- 7. start the host, and stop ------------------------------------------',
    '# pythonw, not the .cmd: a tray program has no console window (4.1).',
    'Say "starting the tray"',
    'Start-Process -WindowStyle Hidden -FilePath $Pythonw -ArgumentList "-m","crucible.cli","host"',
    'Say "Crucible is in your notification area. Open its menu to install the WSL2 engine."',
    '',
  ];
  return lines.join('\n');
}

// ------------------------------------------------- the table, as Python data

/**
 * Sentinels. Every row's `sentence` is a FUNCTION of the evidence, so the way
 * to get a template out of it is to call it with values that cannot occur and
 * then swap those for `{placeholders}` — and to ASSERT each swap fired, so a
 * reworded row fails this generator instead of shipping a sentence with a
 * sentinel in it. The alternative, a second table of sentences in Python, is
 * the thing this file exists to prevent.
 */
const SAID = 'XXSAIDXX';
const APP_DISTRO = 'XXAPPDISTROXX';
const GUEST_USER = 'XXGUESTUSERXX';
const RELEASE_MARK = '424.242.424';
/** gib(REQUIRED_SENTINEL) is "424242.0 GiB", which occurs in no real sentence. */
const REQUIRED_SENTINEL = 424242 * 1024 ** 3;
/** The `df -Pk` reply the disk row is given: gib() renders it "953.7 GiB". */
const FREE_KIB_SENTINEL = '999999999';

const SUBSTITUTIONS: { from: string; to: string; required: boolean }[] = [
  { from: SAID, to: '{said}', required: false },
  { from: APP_DISTRO, to: '{app_distro}', required: false },
  { from: GUEST_USER, to: '{guest_user}', required: false },
  { from: RELEASE_MARK, to: '{release}', required: false },
  { from: '424242.0 GiB', to: '{required}', required: false },
  { from: '953.7 GiB', to: '{free}', required: false },
];

function sentinelResult(): RunResult {
  // `said()` prefers stderr, so putting the sentinel there is what makes every
  // row that quotes the evidence come back with it.
  return { code: 1, stdout: FREE_KIB_SENTINEL, stderr: SAID, failure: 'ENOENT' };
}

function pyString(value: string): string {
  return JSON.stringify(value);
}

function pyList(values: readonly string[]): string {
  return `(${values.map((value) => `${pyString(value)}, `).join('')})`;
}

/** Every sentinel swapped for its `{placeholder}`, and none left behind. */
function templated(text: string, code: string, fired: Set<string>): string {
  let out = text;
  for (const swap of SUBSTITUTIONS) {
    if (out.includes(swap.from)) {
      fired.add(swap.to);
      out = out.split(swap.from).join(swap.to);
    }
  }
  for (const sentinel of [SAID, APP_DISTRO, GUEST_USER, RELEASE_MARK, FREE_KIB_SENTINEL, '424242']) {
    if (out.includes(sentinel)) {
      throw new Error(
        `gen-install-scripts: the sentinel ${sentinel} survived into ${code}: ${JSON.stringify(out)}. `
        + 'A sentence that reaches a person with a sentinel in it is worse than a generator that refuses.',
      );
    }
  }
  return out;
}

export function generateWslStatesPy(): string {
  const inputs = {
    release: RELEASE_MARK,
    appDistro: APP_DISTRO,
    guestUser: GUEST_USER,
    requiredBytes: REQUIRED_SENTINEL,
    checkNetwork: true,
  };
  const states = wslStates(inputs);
  // The SAME table with nothing the caller measured, so that `optional` is
  // read off the rows themselves rather than written down here: a row that is
  // disabled when nobody asked for a disk figure or a network probe is a row
  // whose probe costs something, and `detect()` must not run it unasked.
  const bare = wslStates({ release: RELEASE_MARK });
  const optionalCodes = new Set(bare.filter((state) => state.enabled === false).map((state) => state.code));
  const seen = { results: {}, distros: [] as { name: string; version: number }[] };
  const result = sentinelResult();
  const fired = new Set<string>();

  const rows = states.map((state: WslStateDef) => {
    const sentence = templated(state.sentence(result, seen), state.code, fired);
    const action = state.action(result, seen);
    const argv = probeArgv(state.probe as ProbeKey, inputs).map((word) => templated(word, `${state.code}.probe_argv`, fired));
    const actionArgv = (action.kind === 'run' || action.kind === 'run-elevated' ? action.argv : [])
      .map((word) => templated(word, `${state.code}.action_argv`, fired));
    const parts = [
      `        code=${pyString(state.code)},`,
      `        probe=${pyString(state.probe)},`,
      `        probe_argv=${pyList(argv)},`,
      `        sentence=${pyString(sentence)},`,
      `        action_kind=${pyString(action.kind)},`,
      `        action_argv=${pyList(actionArgv)},`,
      `        action_text=${pyString(action.kind === 'instruct' ? templated(action.text, `${state.code}.action_text`, fired) : '')},`,
      `        action_url=${pyString(action.kind === 'link' ? templated(action.url, `${state.code}.action_url`, fired) : '')},`,
      `        optional=${optionalCodes.has(state.code) ? 'True' : 'False'},`,
    ];
    return `    WslStateDef(\n${parts.join('\n')}\n    ),`;
  });

  // Every placeholder the table CAN produce must actually have been produced,
  // or the Python side would show a sentence with a hole nobody fills.
  for (const wanted of ['{said}', '{app_distro}', '{guest_user}', '{release}', '{required}', '{free}']) {
    if (!fired.has(wanted)) {
      throw new Error(
        `gen-install-scripts: nothing in the WSL table produces ${wanted} any more. `
        + 'Either a row was reworded and crucible/host/wslstate.py must be re-thought, or this generator is substituting for a fact that no longer exists.',
      );
    }
  }

  return [
    BANNER('#').trimEnd(),
    '#',
    '# The WSL state table of PHASE14-ENVPACKS.md 4c, as DATA, for `crucible host`',
    '# (PHASE15-HOST.md 4.3). The ORDER is the order they are tried: deepest cause',
    '# first, so "virtualization is off in the firmware" is never reported as "WSL',
    '# is not installed".',
    '#',
    '# `sentence`, `action_text`, `action_url` and `probe_argv` carry `{said}` /',
    '# `{app_distro}` / `{guest_user}` / `{release}` / `{required}` / `{free}`',
    '# where the TypeScript interpolated something the caller measures. The',
    '# PREDICATES are not here: they are code, and they live in',
    '# `crucible/host/wslstate.py`, one per code, tied to this file by a test.',
    '',
    'from __future__ import annotations',
    '',
    'from dataclasses import dataclass',
    '',
    '#: The distro Crucible owns. One name, and its owner is sdk/bootstrap/src/distro.ts.',
    `CRUCIBLE_DISTRO = ${pyString(CRUCIBLE_DISTRO)}`,
    '',
    '#: The line /etc/wsl.conf carries in the Crucible rootfs and nowhere else.',
    `WSL_CONF_MARKER = ${pyString(WSL_CONF_MARKER)}`,
    '',
    '#: /etc/wsl.conf, exactly as the rootfs ships it and as the repair writes it.',
    `WSL_CONF_TEXT = ${pyString(WSL_CONF_TEXT)}`,
    '',
    '',
    '@dataclass(frozen=True)',
    'class WslStateDef:',
    '    """One row of 4c. `optional` rows are only probed when the caller asks."""',
    '',
    '    code: str',
    '    probe: str',
    '    probe_argv: tuple[str, ...]',
    '    sentence: str',
    '    action_kind: str',
    '    action_argv: tuple[str, ...]',
    '    action_text: str',
    '    action_url: str',
    '    optional: bool',
    '',
    '',
    'WSL_STATES: tuple[WslStateDef, ...] = (',
    ...rows,
    ')',
    '',
    '#: Every code, in table order. `crucible/host/wslstate.py` must have a',
    '#: predicate for each, and no others.',
    'WSL_STATE_CODES: tuple[str, ...] = tuple(state.code for state in WSL_STATES)',
    '',
  ].join('\n');
}

export const GENERATED = [
  { path: join(SCRIPTS, 'install.sh'), text: generateInstallSh() },
  { path: join(SCRIPTS, 'install.ps1'), text: generateInstallPs1() },
  { path: join(REPO, 'crucible', 'host', 'wsl_states.py'), text: generateWslStatesPy() },
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
