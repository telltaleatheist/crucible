# GENERATED FILE — do not edit.
# Written by sdk/bootstrap/scripts/gen-install-scripts.ts from src/steps.ts
# and src/wsl-states.ts, so a hand install and an app-driven install cannot
# differ (PHASE14-ENVPACKS.md 4a). Regenerate: npm run gen:install
#
# Install a Crucible on Windows. Walks every WSL state in the order
# PHASE14-ENVPACKS.md 4c lists them, imports a distribution Crucible owns
# (your own WSL is never touched), then runs install.sh inside it.
#
#   irm https://github.com/telltaleatheist/crucible/releases/latest/download/install.ps1 | iex
#
# Idempotent and resumable: after the reboot WSL asks for, run it again.

[CmdletBinding()]
param(
  [string]$Release = '0.6.0',
  [string]$InstallDir = "$env:LOCALAPPDATA\Crucible\wsl",
  [string]$DownloadDir = "$env:LOCALAPPDATA\Crucible\downloads"
)

# Continue, not Stop: every call below is a native program whose exit code
# is checked explicitly, and Windows PowerShell 5.1 turns a native command writing to stderr into a terminating error under Stop.
$ErrorActionPreference = "Continue"
$Distro = 'crucible'

function Say($m) { Write-Host "crucible: $m" }
function Die($m) { Write-Host "crucible: $m" -ForegroundColor Red; exit 1 }

# wsl.exe writes UTF-16LE; PowerShell 5.1 reads it as UTF-8 and produces text
# with a NUL between every character. Stripped here so -match works.
function Get-WslText($output) { return (($output | Out-String) -replace "`0", "") }

# --- 1. virtualization / the WSL feature ---------------------------------
Say "checking WSL"
$status = ""
$wslOk = $false
try { $status = Get-WslText (& wsl.exe --status); $wslOk = ($LASTEXITCODE -eq 0) } catch { $status = "$_"; $wslOk = $false }
if (-not $wslOk -and ($status -match "HCS_E_HYPERV_NOT_INSTALLED" -or $status -match "0x80370102")) {
  Die 'Virtualization is turned off in this machine''s firmware. Restart, open the BIOS/UEFI setup (usually Del or F2 during boot), and enable Intel VT-x (Intel) or SVM Mode (AMD). Then run Enable WSL again.'
}
if (-not $wslOk) {
  Say 'This machine has no wsl.exe: the Windows Subsystem for Linux has never been enabled.'
  Say "A Windows permission prompt will appear."
  Start-Process -Verb RunAs -Wait -FilePath 'wsl.exe' -ArgumentList '--install','--no-distribution'
  Say "WSL is enabled. RESTART Windows, then run this script again."
  exit 0
}

# --- 2. WSL2 as the default version --------------------------------------
if ($status -match "Default Version:\s*1") {
  Say 'WSL is set to version 1, which has no GPU. Crucible needs WSL2.'
  & 'wsl.exe' '--set-default-version' '2'
}

# --- 3. the distribution Crucible owns ------------------------------------
$list = Get-WslText (& wsl.exe -l -q)
$have = $false
foreach ($line in ($list -split "`r?`n")) { if ($line.Trim() -eq $Distro) { $have = $true } }
if (-not $have) {
  Say 'Crucible has no Linux of its own on this machine yet (the "crucible" distribution). Installing one takes a download and touches nothing you already have.'
  $asset = "crucible-rootfs-$Release.tar.zst"
  $rootfs = Join-Path $DownloadDir $asset
  $base = "https://github.com/telltaleatheist/crucible/releases/download/v$Release"
  New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
  Say "downloading $asset"
  & curl.exe -fL --retry 3 --create-dirs -o $rootfs "$base/$asset"
  if ($LASTEXITCODE -ne 0) { Die "pack_download_failed: $base/$asset" }
  $wantRaw = & curl.exe -fsSL --retry 3 "$base/$asset.sha256"
  if ($LASTEXITCODE -ne 0) { Die "pack_download_failed: $base/$asset.sha256" }
  $want = ($wantRaw -split "\s+")[0].ToLower()
  $got = (Get-FileHash -Algorithm SHA256 -Path $rootfs).Hash.ToLower()
  if ($got -ne $want) { Remove-Item $rootfs -Force; Die "pack_sha_mismatch: $asset hashes $got, the release says $want" }
  New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
  Say "importing $Distro into $InstallDir"
  & wsl.exe --import $Distro $InstallDir $rootfs --version 2
  if ($LASTEXITCODE -ne 0) { Die "distro_import_failed: wsl --import exited $LASTEXITCODE" }
}

# --- 4. systemd in that distribution --------------------------------------
$conf = Get-WslText (& wsl.exe -d $Distro -u root --exec bash -c "test -f /etc/wsl.conf && cat /etc/wsl.conf || true")
if ($conf -notmatch 'crucible-rootfs') {
  Say "writing /etc/wsl.conf in $Distro"
  # One printf per line, because wsl.exe halves backslashes on the way in
  # and a heredoc would need newlines inside one argument.
  $confCmd = 'printf ''%s\\n'' ''# crucible-rootfs'' ''[boot]'' ''systemd=true'' ''[user]'' ''default=crucible'' > /etc/wsl.conf'
  & wsl.exe -d $Distro -u root --exec bash -c $confCmd
  if ($LASTEXITCODE -ne 0) { Die "distro_not_systemd: could not write /etc/wsl.conf in $Distro" }
  # --terminate, never --shutdown: the other distributions on this machine
  # are not ours to stop.
  & wsl.exe --terminate $Distro
}

# --- 5. the same install.sh, inside ---------------------------------------
Say "running install.sh inside $Distro"
$sh = "https://github.com/telltaleatheist/crucible/releases/download/v$Release/install.sh"
& wsl.exe -d $Distro --exec bash -c "CRUCIBLE_RELEASE=$Release curl -fsSL $sh | sh"
if ($LASTEXITCODE -ne 0) { Die "install.sh exited $LASTEXITCODE inside $Distro" }
Say "done."
