import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { LATEST_RELEASE_URL } from '../src/channel.js';
import { CRUCIBLE_DISTRO, finishImportScript, UBUNTU_WSL_ROOTFS, UBUNTU_WSL_SERIES, rootfsSumsUrl, rootfsUrl, WSL_CONF_MARKER, WSL_CONF_TEXT, WSL_OUTCOME_NAME } from '../src/distro.js';
import { DESKTOP_PACKAGES, interpreterFor, interpreterUrl } from '../src/interpreter.js';
import { HOST_BACKEND, RELEASE_REPO, wheelAssetName, wheelShaAssetName } from '../src/release.js';
import { activateRuntimeSh, CURL_ARGS, DOWNLOADS_SUBDIR, HOST_SUBDIR, PARTIAL_SUFFIX, SERVER_SUBDIR, STAMP_NAME, TAR_ARGS } from '../src/runtime.js';
import type { RunResult } from '../src/runner.js';
import { installJobTypesSh, installSteps, interpreterSh, serverPreludeSh, uninstallSh, wheelFetchSh, wheelInstallSh, type StepPlan } from '../src/steps.js';
import { BOOTSTRAP_VERSION } from '../src/version.js';
import { probeArgv, wslStates, type ProbeKey, type WslStateDef } from '../src/wsl-states.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const SCRIPTS = join(HERE, '..', '..', 'scripts');
const REPO = join(HERE, '..', '..', '..', '..');

const ASCII_FOR: Readonly<Record<string, string>> = {
  '—': ' - ',
  '–': '-',
  '‘': "'",
  '’': "'",
  '“': '"',
  '”': '"',
  '…': '...',
  ' ': ' ',
  '×': 'x',
  '→': '->',
  '·': '-',
};

export function asciiOnly(text: string, what: string): string {
  const out = text.replace(/[^\x00-\x7f]/g, (character) => {
    const replacement = ASCII_FOR[character];
    if (replacement === undefined) {
      throw new Error(
        `gen-install-scripts: ${what} contains U+${character.codePointAt(0)!.toString(16).toUpperCase().padStart(4, '0')} `
        + `(${JSON.stringify(character)}), which has no ASCII spelling in ASCII_FOR. Windows PowerShell 5.1 reads a `
        + 'BOM-less .ps1 as the ANSI code page, so a character outside ASCII does not arrive as itself - add it to the '
        + 'table with the spelling you mean, or write the sentence in ASCII.',
      );
    }
    return replacement;
  });
  const left = out.match(/[^\x00-\x7f]/);
  if (left !== null) throw new Error(`gen-install-scripts: ${what} is still not ASCII after transliteration: ${JSON.stringify(left[0])}`);
  return out;
}

function indent(text: string): string {
  return text
    .split('\n')
    .map((line) => (line === '' ? '' : `  ${line}`))
    .join('\n');
}

const STANDALONE: StepPlan = {
  enableFlags: [],
  installs: [],
  bind: [{ sh: '$BIND' }],
  linger: true,
};

function argumentsSh(): string {
  return [
    'usage() {',
    "  cat <<'USAGE'",
    'crucible install.sh — install or remove a Crucible on this machine.',
    '',
    'Install:',
    '  --token <t>          use this bearer token instead of minting one',
    '  --host <addr>        bind address for the server (default 127.0.0.1;',
    '                       a rented box is reached over the network, so it',
    '                       wants 0.0.0.0 — the bearer token is the lock)',
    '  --port <n>           bind port (default 7100)',
    '  --install <type>     also install this job type, from its recipe. Repeatable.',
    '                       tts names its engine: --install tts=higgs-v3',
    '  --from-source <ref>  install the server from a git ref instead of the',
    "                       release's wheel (a branch, a tag or a sha)",
    '  --release <version>  install this exact release rather than the channel\'s',
    '                       latest. The one override; it still downloads from that',
    '                       release, so it is a pin and not an offline install.',
    '  --rollback-to <ver>  an operator rollback: install this EXACT older release',
    '                       over a newer one already on this disk. Must name the',
    '                       same version as --release; there is no other way down.',
    '  --min-free-gib <n>   refuse unless this much disk is free for the weights',
    '',
    'Remove:',
    '  --uninstall          undo the install, in the inverse order',
    '  --purge-weights      with --uninstall: delete the weights too',
    '  --dry-run            with --uninstall: print every step and touch nothing',
    '',
    'USAGE',
    '}',
    '',
    'UNINSTALL=0',
    'PURGE_WEIGHTS=0',
    'DRY_RUN=0',
    'TOKEN=""',
    'BIND=""',
    'JOB_TYPES=""',
    'FROM_SOURCE=""',
    'MIN_FREE_GIB=""',
    'ROLLBACK_TO=""',
    'need() { [ "$1" -ge 2 ] || die "flag_needs_value: $2 takes a value"; }',
    'while [ $# -gt 0 ]; do',
    '  case "$1" in',
    '    --uninstall) UNINSTALL=1 ;;',
    '    --purge-weights) PURGE_WEIGHTS=1 ;;',
    '    --dry-run) DRY_RUN=1 ;;',
    '    --token) need $# "--token"; shift; TOKEN="$1" ;;',
    '    --host) need $# "--host"; shift; BIND="$BIND --host $1" ;;',
    '    --port) need $# "--port"; shift; BIND="$BIND --port $1" ;;',
    '    --install) need $# "--install"; shift; JOB_TYPES="$JOB_TYPES $1" ;;',
    '    --from-source) need $# "--from-source"; shift; FROM_SOURCE="$1" ;;',
    '    --release) need $# "--release"; shift; RELEASE="$1" ;;',
    '    --rollback-to) need $# "--rollback-to"; shift; ROLLBACK_TO="$1" ;;',
    '    --min-free-gib) need $# "--min-free-gib"; shift; MIN_FREE_GIB="$1" ;;',
    '    -h|--help) usage; exit 0 ;;',
    '    *) die "unknown_flag: $1 is not a flag this installer takes; run with --help" ;;',
    '  esac',
    '  shift',
    'done',
    'if [ "$UNINSTALL" = 0 ] && [ "$PURGE_WEIGHTS" = 1 ]; then',
    '  die "flag_needs_uninstall: --purge-weights deletes weights and only means something with --uninstall"',
    'fi',
    'if [ "$UNINSTALL" = 1 ] && [ -n "$ROLLBACK_TO" ]; then',
    '  die "flag_needs_install: --rollback-to names a release to INSTALL and means nothing with --uninstall"',
    'fi',
  ].join('\n');
}

function prerequisitesSh(): string {
  return [
    'if [ "$BACKEND" = cuda-linux ]; then',
    '  command -v nvidia-smi >/dev/null 2>&1 || die "no_nvidia_smi: there is no nvidia-smi on PATH. cuda-linux runs vLLM, SGLang and torch on an NVIDIA card; a box without the driver is not this backend"',
    '  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d \'[:space:]\')"',
    '  [ -n "$driver" ] || die "no_nvidia_driver: nvidia-smi is on PATH and named no driver. On a rented GPU box that usually means the image has the CUDA userland and not the kernel module"',
    '  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d \'[:space:].\')"',
    '  [ -n "$cap" ] || die "no_cuda_arch: this driver would not report a compute capability, so nothing here can say whether the engines will build for this card"',
    '  [ "$cap" -ge 70 ] || die "cuda_arch_too_old: this card reports compute capability $cap (7.0 is the floor: vLLM and SGLang ship no kernels below it, and torch\'s wheels drop it too). Rent a card at 7.0 or newer"',
    '  card="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"',
    '  say "prerequisites: $card, driver $driver, compute capability $cap"',
    'fi',
    'if [ "$BACKEND" = cuda-linux ]; then',
    '  say "prerequisites: ffmpeg is Crucible\'s own; the first job type that decodes audio places it in $CRUCIBLE_HOME/tools/bin"',
    'else',
    '  case " $JOB_TYPES " in',
    '    *" tts"*|*" asr"*|*" rvc"*|*" align"*|*" denoise"*)',
    '      command -v ffmpeg >/dev/null 2>&1 || die "no_ffmpeg: --install named a job type that decodes audio and this Mac has no ffmpeg on PATH. Install Homebrew\'s first (brew install ffmpeg); crucible would otherwise install cleanly and refuse its first job" ;;',
    '    *)',
    '      if command -v ffmpeg >/dev/null 2>&1; then',
    '        say "prerequisites: ffmpeg present"',
    '      else',
    '        say "prerequisites: NO ffmpeg on PATH. Nothing asked for today needs it; on this Mac tts, asr, align, rvc and denoise will refuse until Homebrew\'s is installed (brew install ffmpeg)"',
    '      fi ;;',
    '  esac',
    'fi',
    'if [ -n "$MIN_FREE_GIB" ]; then',
    '  have_gib=$(( free_kib / 1048576 ))',
    '  [ "$have_gib" -ge "$MIN_FREE_GIB" ] || die "disk_too_small: $CRUCIBLE_HOME has ${have_gib} GiB free and --min-free-gib asked for $MIN_FREE_GIB"',
    'fi',
    'if [ -n "${WSL_DISTRO_NAME:-}" ]; then',
    '  say "prerequisites: WSL\'s virtual disk can grow to $(( free_kib / 1048576 )) GiB; the real limit is the free space on the Windows drive it lives on"',
    'else',
    '  say "prerequisites: $(( free_kib / 1048576 )) GiB free at $CRUCIBLE_HOME. Weights are pulled later and priced then — a 9B model is ~18 GiB, a Higgs voice ~8.5 GiB"',
    'fi',
  ].join('\n');
}

function fromSourceSh(): string {
  return [
    `say "server: --from-source $FROM_SOURCE, installing from a checkout instead of the wheel"`,
    'command -v git >/dev/null 2>&1 || die "guest_missing_tool: --from-source needs git"',
    'src="$CRUCIBLE_HOME/src"',
    'rm -rf "$src"',
    `git clone --filter=blob:none "https://github.com/${RELEASE_REPO}" "$src" || die "from_source_clone_failed: https://github.com/${RELEASE_REPO}"`,
    'git -C "$src" checkout --detach "$FROM_SOURCE" || die "from_source_ref_unknown: the checkout has no ref called $FROM_SOURCE"',
    'crucible_quiesce || say "server: the running server did not stop; installing the checkout and stopping it with that"',
    '"$dest/bin/python3" -m pip install --upgrade --no-input "$src" || die "from_source_install_failed: pip would not install $src into $dest"',
    `if [ "$(uname -s)" = Darwin ]; then "$dest/bin/python3" -m pip install ${DESKTOP_PACKAGES.join(' ')} || die "from_source_install_failed: desktop packages could not be installed"; fi`,
    `printf 'python_sha256=%s\\npython_version=%s\\nrelease=%s\\n' "$py_sha" "$py_version" "$(git -C "$src" rev-parse HEAD)" > "$dest/${STAMP_NAME}"`,
    'CRUCIBLE="$dest/bin/crucible"',
    'crucible_quiesce_after',
    'say "server: installed $("$CRUCIBLE" --version) from $(git -C "$src" rev-parse --short HEAD)"',
  ].join('\n');
}

export function generateInstallSh(): string {
  const steps = installSteps(STANDALONE);
  const lines: string[] = [
    '#!/bin/sh',
    '',
    'set -eu',
    '',
    "RELEASE=\"${CRUCIBLE_RELEASE:-}\"",
    '',
    "say() { printf 'crucible: %s\\n' \"$*\"; }",
    "die() { printf 'crucible: %s\\n' \"$*\" >&2; exit 1; }",
    '',
    argumentsSh(),
    '',
    'case "$(uname -s)/$(uname -m)" in',
    '  Linux/x86_64)  BACKEND=cuda-linux; SHA_TOOL="sha256sum";     MECHANISM=systemd ;;',
    '  Darwin/arm64)  BACKEND=mlx-darwin; SHA_TOOL="shasum -a 256"; MECHANISM=launchd ;;',
    '  *) die "unsupported_platform: $(uname -s)/$(uname -m) is not a Crucible backend '
      + '(cuda-linux on Linux x86_64, mlx-darwin on Apple Silicon)" ;;',
    'esac',
    'newest_release() {',
    `  curl -fsSL --retry 3 -H "Accept: application/vnd.github+json" "${LATEST_RELEASE_URL}" |`,
    `    grep -o '"tag_name": *"[^"]*"' | head -n 1 | cut -d'"' -f4`,
    '}',
    'if [ "$UNINSTALL" = 1 ]; then',
    '  say "backend $BACKEND, uninstalling"',
    'else',
    '  if [ -z "$RELEASE" ]; then',
    '    tag="$(newest_release)" || tag=""',
    '    RELEASE="${tag#v}"',
    '  fi',
    `  [ -n "$RELEASE" ] || die "release_channel_unreadable: could not read the release channel at ${LATEST_RELEASE_URL} -- name a release with --release <version> or CRUCIBLE_RELEASE=<version>"`,
    '  [ -z "$ROLLBACK_TO" ] || [ "$ROLLBACK_TO" = "$RELEASE" ] || die "rollback_version_mismatch: --rollback-to names $ROLLBACK_TO and the release being installed is $RELEASE; a rollback names the exact Crucible you want back"',
    '  say "release $RELEASE, backend $BACKEND"',
    'fi',
    '',
    'if [ "$UNINSTALL" = 1 ]; then',
    '  say "uninstall"',
    indent(uninstallSh().trimEnd()),
    '  say "uninstalled."',
    '  exit 0',
    'fi',
    '',
  ];
  for (const step of steps) {
    lines.push(`say "${step.name}"`);
    if (step.name === 'server') {
      lines.push(serverPreludeSh().trimEnd());
      lines.push('if [ -z "$FROM_SOURCE" ]; then');
      lines.push(indent(wheelFetchSh().trimEnd()));
      lines.push('fi');
      lines.push(interpreterSh().trimEnd());
      lines.push('if [ -n "$FROM_SOURCE" ]; then');
      lines.push(indent(fromSourceSh()));
      lines.push('else');
      lines.push(indent(wheelInstallSh().trimEnd()));
      lines.push('fi');
    } else {
      lines.push(step.sh.trimEnd());
    }
    lines.push('');
    if (step.name === 'host-facts') {
      lines.push('say "prerequisites"');
      lines.push(prerequisitesSh());
      lines.push('');
    }
    if (step.name === 'host-facts') {
      lines.push('FRESH=0');
      lines.push('[ -f "$CRUCIBLE_HOME/config.toml" ] || FRESH=1');
      lines.push('');
    }
    if (step.name === 'init') {
      lines.push(installJobTypesSh().trimEnd());
      lines.push('');
    }
  }
  lines.push(openAppSh());
  lines.push('say "installed. Pair an app with the line below."');
  lines.push('"$CRUCIBLE" token --url');
  lines.push('');
  return lines.join('\n');
}

export function openAppSh(): string {
  return [
    `if [ "$(uname -s)" = Darwin ] && [ -d "${MAC_APP}" ]; then`,
    `  say "Crucible is in your Applications folder: ${MAC_APP}"`,
    '  if [ "$FRESH" = 1 ] && [ -t 1 ] && [ -z "${SSH_CONNECTION:-}" ]; then',
    `    open "${MAC_APP}" || say "open Crucible from your Applications folder"`,
    '  fi',
    'fi',
  ].join('\n');
}

export function openAppPs1(): string[] {
  return [
    'Say "Crucible is in your Start Menu: search for Crucible."',
    '$Interactive = [Environment]::UserInteractive -and -not $FromApp -and -not $env:SSH_CONNECTION',
    'if ($Fresh -and $Interactive) {',
    '  Say "opening Crucible"',
    '  Start-Process -FilePath $Pythonw -ArgumentList "-m","crucible.cli","app"',
    '}',
  ];
}

function row(states: WslStateDef[], code: string): WslStateDef {
  const found = states.find((state) => state.code === code);
  if (found === undefined) throw new Error(`gen-install-scripts: the WSL state table has no "${code}" row any more`);
  return found;
}

function psQuote(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

const PIP_QUIET = '--quiet --disable-pip-version-check --no-warn-script-location';

const APP_SCRIPT_NAME = 'crucible-install.ps1';

const MAC_APP = '$HOME/Applications/Crucible.app';

function wslConfPrintf(): string {
  const lines = WSL_CONF_TEXT.split('\n').filter((line) => line !== '');
  const args = lines.map((line) => `'${line.replace(/'/g, `'\\''`)}'`).join(' ');
  return `printf '%s\\\\n' ${args} > /etc/wsl.conf`;
}

const ORDINARY_POWERSHELL = 'Open an ordinary PowerShell window (Start, type PowerShell, press Enter) and run it there.';

export function packagedRefusalPs1(what: string): string {
  return `packaged_shell: ${what} is running inside the Windows app package $PackageName (an app installed from the Store `
    + 'or as an MSIX, such as the Claude desktop app, and anything started from a terminal inside it). Windows quietly '
    + "redirects what such a process writes under AppData into that app's own private folder, so Crucible would land where "
    + 'only that app can see it, and would not start when you sign in. Nothing has been written. '
    + ORDINARY_POWERSHELL;
}

export function packagedCheckPs1(): string[] {
  return [
    'try {',
    "  Add-Type -ErrorAction Stop -TypeDefinition @'",
    'using System.Runtime.InteropServices;',
    'using System.Text;',
    'public static class CruciblePackage {',
    '  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]',
    '  static extern int GetCurrentPackageFullName(ref uint length, StringBuilder name);',
    '  public static int Ask(out string name) {',
    '    uint length = 0;',
    '    name = "";',
    '    int code = GetCurrentPackageFullName(ref length, null);',
    '    if (code != 122) { return code; }',
    '    StringBuilder buffer = new StringBuilder((int)length);',
    '    code = GetCurrentPackageFullName(ref length, buffer);',
    '    name = buffer.ToString();',
    '    return code;',
    '  }',
    '}',
    "'@",
    '} catch {',
    `  Die "package_check_failed: Windows could not be asked whether this PowerShell runs inside an app package ($($_.Exception.Message)), and a Crucible written from inside one is a Crucible nothing can start. ${ORDINARY_POWERSHELL}"`,
    '}',
    "$PackageName = ''",
    '$PackageCode = [CruciblePackage]::Ask([ref]$PackageName)',
    `if ($PackageCode -eq 0) { Die "${packagedRefusalPs1('this installer')}" }`,
    `if ($PackageCode -ne 15700) { Die "package_check_failed: GetCurrentPackageFullName returned $PackageCode, so it is not known whether this PowerShell runs inside an app package. ${ORDINARY_POWERSHELL}" }`,
  ];
}

export function generateInstallPs1(): string {
  const states = wslStates({ release: BOOTSTRAP_VERSION });
  const enable = row(states, 'wsl_missing');
  const enableAction = enable.action({ code: 0, stdout: '', stderr: '', failure: null }, { results: {}, distros: [] });
  if (enableAction.kind !== 'run-elevated') {
    throw new Error('gen-install-scripts: the WSL table changed shape; install.ps1 must be re-thought, not re-run');
  }
  const base = `https://github.com/${RELEASE_REPO}/releases/download/v$Release`;
  const pin = interpreterFor(HOST_BACKEND);
  const lines: string[] = [
    '[CmdletBinding()]',
    'param(',
    "  [string]$Release = '',",
    "  [string]$RollbackTo = '',",
    '  [string]$Root = "$env:LOCALAPPDATA\\Crucible",',
    '  [switch]$Uninstall,',
    '  [switch]$PurgeWeights,',
    '  [switch]$DryRun,',
    '  [switch]$WslToo,',
    "  [string]$PythonArchive = '',",
    "  [string]$WheelFile = '',",
    "  [string]$WheelSha = ''",
    ')',
    '',
    '$ErrorActionPreference = "Continue"',
    '$Root = [System.IO.Path]::GetFullPath($Root)',
    '$env:CRUCIBLE_HOME = $Root',
    `$HostDir = Join-Path $Root ${psQuote(HOST_SUBDIR)}`,
    `$DownloadDir = Join-Path $Root ${psQuote(DOWNLOADS_SUBDIR)}`,
    `$Partial = "$HostDir${PARTIAL_SUFFIX}"`,
    `$Stamp = Join-Path $HostDir ${psQuote(STAMP_NAME)}`,
    '$Cmd = Join-Path $HostDir "crucible.cmd"',
    '$Pythonw = Join-Path $HostDir "pythonw.exe"',
    '$Fresh = -not (Test-Path -LiteralPath $Cmd)',
    '',
    'function Say($m) { Write-Host "crucible: $m" }',
    'function Die($m) { Write-Host "crucible: $m" -ForegroundColor Red; if ($PSCommandPath) { exit 1 } else { throw "crucible: $m" } }',
    'function Native([scriptblock]$Command) {',
    '  & $Command 2>&1 | ForEach-Object {',
    '    if ($_ -is [System.Management.Automation.ErrorRecord]) { "$($_.Exception.Message)" } else { "$_" }',
    "  } | Where-Object { $_.Trim() -ne '' }",
    '}',
    'function Show { process { Write-Host "  $_" } }',
    '$Previous = "$HostDir.previous"',
    'foreach ($target in @($HostDir, $Partial, $Previous, $DownloadDir)) {',
    '  $absolute = [System.IO.Path]::GetFullPath($target)',
    '  if (-not $absolute.StartsWith($Root.TrimEnd("\\") + "\\", [System.StringComparison]::OrdinalIgnoreCase)) { Die "unsafe_install_path: $absolute is outside $Root" }',
    '}',
    '',
    'if ([System.Environment]::Is64BitOperatingSystem -ne $true) {',
    '  Die "unsupported_platform: the Crucible Windows host is 64-bit x86 only."',
    '}',
    'if (-not $env:LOCALAPPDATA) {',
    '  Die "host_no_localappdata: LOCALAPPDATA is not set, so there is no per-user place to install into."',
    '}',
    ...packagedCheckPs1(),
    '',
    'if ($Uninstall) {',
    '  if (-not (Test-Path $Cmd)) {',
    '    Die "not_installed: there is no $Cmd on this machine, so there is no Crucible host here to remove."',
    '  }',
    '  $verb = @("uninstall")',
    '  if ($DryRun) { $verb += "--dry-run" }',
    '  if ($PurgeWeights) { $verb += "--purge-weights" }',
    '  if ($WslToo) { $verb += "--wsl-too" }',
    '  Say "uninstall: $Cmd $($verb -join \' \')"',
    '  & $Cmd @verb',
    '  if ($LASTEXITCODE -ne 0) { Die "step_failed: uninstall (crucible uninstall exited $LASTEXITCODE; nothing of the runtime has been removed)" }',
    '  if ($DryRun) {',
    '    Say "host: would remove $HostDir and $DownloadDir"',
    '    Say "home: would remove $Root if it were then empty"',
    '    exit 0',
    '  }',
    '  Say "host"',
    '  foreach ($gone in @($Partial, $DownloadDir, $HostDir)) {',
    '    if (Test-Path $gone) {',
    '      try {',
    '        Remove-Item $gone -Recurse -Force -ErrorAction Stop',
    '      } catch {',
    '        Die "host_runtime_locked: $gone could not be removed ($($_.Exception.Message)). Something still holds a file in it — the tray was just ended, so log out and back in, then run this again. It is idempotent."',
    '      }',
    '    }',
    '  }',
    '  Say "host: removed $HostDir"',
    '  $left = @(Get-ChildItem -Force -Path $Root -ErrorAction SilentlyContinue)',
    '  if ($left.Count -eq 0) {',
    '    Remove-Item $Root -Force -Recurse',
    '    Say "home: removed $Root"',
    '  } else {',
    '    Say "home: KEPT $Root — it still holds $($left.Name -join \', \')"',
    '    Say "home: weights are kept unless -PurgeWeights; nothing else there was Crucible\'s to delete"',
    '  }',
    '  Say "uninstalled."',
    '  exit 0',
    '}',
    '',
    '$Tar = Join-Path $env:SystemRoot "System32\\tar.exe"',
    'if (-not (Test-Path $Tar)) {',
    '  Die "guest_missing_tool: there is no $Tar on this machine. Windows 10 1803+ and Windows 11 ship a bsdtar there, and the interpreter archive cannot be unpacked without one."',
    '}',
    '',
    'if (-not $Release) {',
    `  $feed = "${LATEST_RELEASE_URL}"`,
    '  $feedRaw = & curl.exe -fsSL --retry 3 -H "Accept: application/vnd.github+json" "$feed"',
    '  if ($LASTEXITCODE -ne 0) { Die "release_channel_unreadable: could not read the release channel at $feed -- name a release with -Release <version>" }',
    '  try { $feedJson = $feedRaw | Out-String | ConvertFrom-Json } catch { Die "release_channel_unreadable: $feed is not JSON" }',
    "  $Release = ($feedJson.tag_name) -replace '^v',''",
    '  if (-not $Release) { Die "release_channel_unreadable: $feed named no release" }',
    '}',
    'if ($RollbackTo -and $RollbackTo -ne $Release) { Die "rollback_version_mismatch: -RollbackTo names $RollbackTo and the release being installed is $Release; a rollback names the exact Crucible you want back" }',
    'Say "release $Release"',
    '',
    `$PyAsset = ${psQuote(pin.asset)}`,
    `$PySha = ${psQuote(pin.sha256)}`,
    `$PyVersion = ${psQuote(pin.version)}`,
    `$PyUrl = ${psQuote(interpreterUrl(pin))}`,
    '$PythonExe = Join-Path $HostDir "python.exe"',
    '$have = ""',
    '$haveRelease = ""',
    'if (Test-Path $Stamp) {',
    '  foreach ($line in (Get-Content $Stamp)) {',
    '    if ($line -match "^python_sha256=(.+)$") { $have = $Matches[1].Trim() }',
    '    if ($line -match "^release=(.+)$") { $haveRelease = $Matches[1].Trim() }',
    '  }',
    '}',
    'if ($haveRelease) {',
    '  $onDisk = $null; $wanted = $null',
    '  if ([version]::TryParse($haveRelease, [ref]$onDisk) -and [version]::TryParse($Release, [ref]$wanted) -and $wanted -lt $onDisk) {',
    '    if ($RollbackTo -ne $Release) {',
    '      Die "install_would_downgrade: $HostDir is the $haveRelease release and this would install $Release over it. Nothing has been downloaded. An operator who means to go back names the version: -RollbackTo $Release"',
    '    }',
    '  }',
    '}',
    'if ($have -eq $PySha -and (Test-Path $PythonExe)) {',
    '  Say "host: python $PyVersion is already at $HostDir"',
    '} else {',
    '  New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null',
    '  $archive = Join-Path $DownloadDir $PyAsset',
    '  if (Test-Path $archive) { Remove-Item $archive -Force }',
    '  if ($PythonArchive) {',
    '    Say "host: python $PyVersion from $PythonArchive"',
    '    Copy-Item -LiteralPath $PythonArchive -Destination $archive -Force',
    '  } else {',
    '    Say "host: python $PyVersion from python-build-standalone"',
    '    Native { & curl.exe ' + CURL_ARGS.join(' ') + ' -sS -o $archive "$PyUrl" } | Show',
    '    if ($LASTEXITCODE -ne 0) { Die "runtime_download_failed: $PyUrl" }',
    '  }',
    '  $got = (Get-FileHash -Algorithm SHA256 -Path $archive).Hash.ToLower()',
    '  if ($got -ne $PySha) {',
    '    Remove-Item $archive -Force',
    '    Die "runtime_sha_mismatch: $PyAsset hashes $got, this installer pins $PySha. The download was deleted"',
    '  }',
    '',
    '  if (Test-Path $Partial) { Remove-Item $Partial -Recurse -Force }',
    '  New-Item -ItemType Directory -Force -Path $Partial | Out-Null',
    `  & $Tar ${TAR_ARGS.join(' ')} $archive -C $Partial`,
    '  if ($LASTEXITCODE -ne 0) { Die "runtime_unpack_failed: tar would not open $archive" }',
    '  $staged = Join-Path $Partial "python"',
    '  & (Join-Path $staged "python.exe") --version | Out-Null',
    '  if ($LASTEXITCODE -ne 0) { Die "runtime_unpack_failed: python.exe in $staged would not run" }',
    '  if (Test-Path -LiteralPath $Previous) { Die "upgrade_recovery_required: $Previous exists from an interrupted upgrade; restore or inspect it before retrying" }',
    '  if (Test-Path -LiteralPath $HostDir) {',
    '    if (Test-Path -LiteralPath $Cmd) {',
    '      $said = @(Native { & $Cmd local shutdown })',
    '      if ($LASTEXITCODE -ne 0) { $said | Show; Die "upgrade_stop_failed: the old runtime was kept because Crucible did not stop cleanly" }',
    '    }',
    '    Move-Item -LiteralPath $HostDir -Destination $Previous -ErrorAction Stop',
    '  }',
    '  try {',
    '    Move-Item $staged $HostDir -ErrorAction Stop',
    '    & $PythonExe --version | Out-Null',
    '    if ($LASTEXITCODE -ne 0) { throw "the installed interpreter failed its startup check" }',
    '  } catch {',
    '    if ((Test-Path -LiteralPath $Previous) -and (Test-Path -LiteralPath $HostDir)) { Remove-Item -LiteralPath $HostDir -Recurse -Force }',
    '    if ((Test-Path -LiteralPath $Previous) -and -not (Test-Path -LiteralPath $HostDir)) { Move-Item -LiteralPath $Previous -Destination $HostDir }',
    '    Die "upgrade_swap_failed: $_. The previous runtime is retained at $Previous when present."',
    '  }',
    '  if (Test-Path -LiteralPath $Previous) { Remove-Item -LiteralPath $Previous -Recurse -Force }',
    '  Remove-Item $Partial -Recurse -Force',
    '  Remove-Item $archive -Force',
    '  Say "host: python $PyVersion at $HostDir"',
    '}',
    '',
    'New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null',
    `$Wheel = "${wheelAssetName('$Release')}"`,
    '$WheelPath = Join-Path $DownloadDir $Wheel',
    'if (Test-Path $WheelPath) { Remove-Item $WheelPath -Force }',
    'Say "host: $Wheel"',
    'if ($WheelFile) {',
    '  Copy-Item -LiteralPath $WheelFile -Destination $WheelPath -Force',
    '} else {',
    `  Native { & curl.exe ${CURL_ARGS.join(' ')} -sS -o $WheelPath "${base}/$Wheel" } | Show`,
    `  if ($LASTEXITCODE -ne 0) { Die "runtime_download_failed: ${base}/$Wheel" }`,
    '}',
    'if ($WheelSha) {',
    '  $want = $WheelSha.Trim().ToLower()',
    `  if ($want -notmatch '^[0-9a-f]{64}$') { Die "runtime_download_failed: -WheelSha $WheelSha is not a sha256" }`,
    '} else {',
    `  $wantRaw = & curl.exe -fsSL --retry 3 "${base}/${wheelShaAssetName('$Release')}"`,
    `  if ($LASTEXITCODE -ne 0) { Die "runtime_download_failed: ${base}/${wheelShaAssetName('$Release')}" }`,
    "  $want = ($wantRaw | Out-String).Trim().Split()[0].ToLower()",
    `  if ($want -notmatch '^[0-9a-f]{64}$') { Die "runtime_download_failed: ${base}/${wheelShaAssetName('$Release')} is not a sha256" }`,
    '}',
    '$gotWheel = (Get-FileHash -Algorithm SHA256 -Path $WheelPath).Hash.ToLower()',
    'if ($gotWheel -ne $want) {',
    '  Remove-Item $WheelPath -Force',
    '  Die "runtime_sha_mismatch: $Wheel hashes $gotWheel, the release says $want. The download was deleted"',
    '}',
    'if (Test-Path -LiteralPath $Cmd) {',
    '  $Stage = Join-Path $DownloadDir "stage"',
    '  if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }',
    '  $stopped = $false',
    '  $null = @(Native { & $PythonExe -m pip install --quiet --no-deps --no-input --target $Stage $WheelPath })',
    '  if ($LASTEXITCODE -eq 0) {',
    '    $keptPath = $env:PYTHONPATH',
    '    $env:PYTHONPATH = $Stage',
    '    $said = @(Native { & $PythonExe -m crucible.cli local shutdown })',
    '    $stopped = ($LASTEXITCODE -eq 0)',
    '    $env:PYTHONPATH = $keptPath',
    '  }',
    '  if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }',
    '  if (-not $stopped) {',
    '    $said = @(Native { & $Cmd local shutdown })',
    '    $stopped = ($LASTEXITCODE -eq 0)',
    '  }',
    '  if (-not $stopped) { Say "host: the running Crucible did not stop cleanly; installing over it:"; $said | Show }',
    '}',
    'Say "host: installing Crucible into $HostDir (about a minute)"',
    `Native { & $PythonExe -m pip install ${PIP_QUIET} --upgrade --no-input $WheelPath } | Show`,
    'if ($LASTEXITCODE -ne 0) { Die "runtime_install_failed: pip would not install $Wheel into $HostDir" }',
    `Native { & $PythonExe -m pip install ${PIP_QUIET} ${DESKTOP_PACKAGES.join(' ')} } | Show`,
    'if ($LASTEXITCODE -ne 0) { Die "runtime_install_failed: the tray packages would not install" }',
    'Remove-Item $WheelPath -Force',
    '',
    '$shim = "@echo off`r`n""%~dp0python.exe"" -m crucible.cli %*`r`n"',
    '[System.IO.File]::WriteAllText($Cmd, $shim, [System.Text.Encoding]::ASCII)',
    '& $Cmd --version | Out-Null',
    'if ($LASTEXITCODE -ne 0) { Die "runtime_install_failed: $Cmd would not run" }',
    'Set-Content -Path $Stamp -Encoding ascii -Value @("python_sha256=$PySha", "python_version=$PyVersion", "release=$Release")',
    'Say "host: $Release installed at $HostDir (Python $PyVersion)"',
    '',
    'foreach ($action in @("register", "install-cli", "install-desktop")) {',
    '  $said = @(Native { & $Cmd local $action })',
    '  if ($LASTEXITCODE -ne 0) { $said | Show; Die "local setup failed: $action (exit $LASTEXITCODE). Run this installer again; it carries on from where it stopped." }',
    '}',
    '',
    'Say "starting the tray"',
    "$Began = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')",
    'Start-Process -WindowStyle Hidden -FilePath $Pythonw -ArgumentList "-m","crucible.cli","orchestrator"',
    '$said = @(Native { & $Cmd local start })',
    'if ($LASTEXITCODE -ne 0) { $said | Show; Die "Crucible is installed, but its engine did not start. Run this installer again; it carries on from where it stopped." }',
    '',
    'Say "Crucible is ready in your notification area."',
    '$FromApp = [bool]($PSCommandPath -and ([System.IO.Path]::GetFileName($PSCommandPath) -eq '
      + `${psQuote(APP_SCRIPT_NAME)}) -and [Console]::IsOutputRedirected)`,
    "$Watch = @('-m', 'crucible.host.installwatch', '--home', $Root, '--since', $Began)",
    "if ($FromApp) { $Watch += '--brief' }",
    'Native { & $PythonExe @Watch } | ForEach-Object { Write-Host $_ }',
    'if ($LASTEXITCODE -ne 0) {',
    '  Say "The Linux engine sets itself up in the background, which takes several minutes; there is nothing to click. If it stops or needs a Windows restart, the menu of the Crucible icon by the clock says so."',
    '}',
    ...openAppPs1(),
    '',
  ];
  return asciiOnly(lines.join('\n'), 'install.ps1');
}

const SAID = 'XXSAIDXX';
const APP_DISTRO = 'XXAPPDISTROXX';
const GUEST_USER = 'XXGUESTUSERXX';
const RELEASE_MARK = '424.242.424';
const REQUIRED_SENTINEL = 424242 * 1024 ** 3;
const FREE_KIB_SENTINEL = '999999999';

const SUBSTITUTIONS: { from: string; to: string; required: boolean }[] = [
  { from: SAID, to: '{said}', required: false },
  { from: APP_DISTRO, to: '{app_distro}', required: false },
  { from: RELEASE_MARK, to: '{release}', required: false },
  { from: '424242.0 GiB', to: '{required}', required: false },
  { from: '953.7 GiB', to: '{free}', required: false },
];

function sentinelResult(): RunResult {
  return { code: 1, stdout: FREE_KIB_SENTINEL, stderr: SAID, failure: 'ENOENT' };
}

function pyString(value: string): string {
  return JSON.stringify(value);
}

function pyList(values: readonly string[]): string {
  return `(${values.map((value) => `${pyString(value)}, `).join('')})`;
}

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
    requiredBytes: REQUIRED_SENTINEL,
    checkNetwork: true,
  };
  const states = wslStates(inputs);
  const bare = wslStates({ release: RELEASE_MARK });
  const optionalCodes = new Set(bare.filter((state) => state.enabled === false).map((state) => state.code));
  const seen = { results: {}, distros: [] as { name: string; version: number }[] };
  const result = sentinelResult();
  const fired = new Set<string>();

  const rows = states.map((state: WslStateDef) => {
    const sentence = templated(state.sentence(result, seen), state.code, fired);
    const action = state.action(result, seen);
    if (state.code !== 'wsl_ready') {
      const carried = action.kind === 'run' || action.kind === 'run-elevated';
      if (state.automatic !== carried) {
        throw new Error(
          `gen-install-scripts: ${state.code} says automatic=${state.automatic} and its action is `
          + `"${action.kind}". A row the tray can carry is one whose action is something we RUN; `
          + 'wsl_ready is the only exception and this is not it.',
        );
      }
    }
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
      `        automatic=${state.automatic ? 'True' : 'False'},`,
      `        optional=${optionalCodes.has(state.code) ? 'True' : 'False'},`,
    ];
    return `    WslStateDef(\n${parts.join('\n')}\n    ),`;
  });

  for (const wanted of ['{said}', '{app_distro}', '{release}', '{required}', '{free}']) {
    if (!fired.has(wanted)) {
      throw new Error(
        `gen-install-scripts: nothing in the WSL table produces ${wanted} any more. `
        + 'Either a row was reworded and crucible/host/wslstate.py must be re-thought, or this generator is substituting for a fact that no longer exists.',
      );
    }
  }

  return [
    'from __future__ import annotations',
    '',
    'from dataclasses import dataclass',
    '',
    `CRUCIBLE_DISTRO = ${pyString(CRUCIBLE_DISTRO)}`,
    `RELEASE_REPOSITORY = ${pyString(RELEASE_REPO)}`,
    '',
    `UBUNTU_WSL_SERIES = ${pyString(UBUNTU_WSL_SERIES)}`,
    `UBUNTU_WSL_ROOTFS = ${pyString(UBUNTU_WSL_ROOTFS)}`,
    `UBUNTU_WSL_ROOTFS_URL = ${pyString(rootfsUrl())}`,
    `UBUNTU_WSL_SUMS_URL = ${pyString(rootfsSumsUrl())}`,
    '',
    `WSL_OUTCOME_NAME = ${pyString(WSL_OUTCOME_NAME)}`,
    '',
    `WSL_CONF_MARKER = ${pyString(WSL_CONF_MARKER)}`,
    '',
    `WSL_CONF_TEXT = ${pyString(WSL_CONF_TEXT)}`,
    '',
    `FINISH_IMPORT_SCRIPT = ${pyString(finishImportScript())}`,
    '',
    '',
    '@dataclass(frozen=True)',
    'class WslStateDef:',
    '    code: str',
    '    probe: str',
    '    probe_argv: tuple[str, ...]',
    '    sentence: str',
    '    action_kind: str',
    '    action_argv: tuple[str, ...]',
    '    action_text: str',
    '    action_url: str',
    '    automatic: bool',
    '    optional: bool',
    '',
    '',
    'WSL_STATES: tuple[WslStateDef, ...] = (',
    ...rows,
    ')',
    '',
    'WSL_STATE_CODES: tuple[str, ...] = tuple(state.code for state in WSL_STATES)',
    '',
  ].join('\n');
}

export const GENERATED = [
  { path: join(SCRIPTS, 'install.sh'), text: generateInstallSh() },
  { path: join(SCRIPTS, 'install.ps1'), text: generateInstallPs1() },
  { path: join(REPO, 'crucible', 'platform', 'wsl_table.py'), text: generateWslStatesPy() },
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

if (process.argv[1] !== undefined && process.argv[1].endsWith('gen-install-scripts.js')) main();
