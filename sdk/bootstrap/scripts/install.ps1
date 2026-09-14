# GENERATED FILE — do not edit.
# Written by sdk/bootstrap/scripts/gen-install-scripts.ts from src/steps.ts
# and src/wsl-states.ts, so a hand install and an app-driven install cannot
# differ (PHASE14-ENVPACKS.md 4a). Regenerate: npm run gen:install
#
# Install Crucible on Windows: the HOST, and nothing else.
#
# PHASE15-HOST.md 4.4. This script used to walk the WSL states itself and
# then run install.sh inside an imported distribution. It no longer does,
# and that is the point of the phase: `crucible host` owns that sequence
# (4.3), it can carry a reboot across because it starts at login, and the
# page drives it as a task (4.7) — an app that asks for an install talks
# to the SAME implementation through the host loopback door. Two walks of
# one table was the thing being removed.
#
# So: download the host pack for this release, verify it, unpack it to
# %LOCALAPPDATA%\Crucible\host\, register the Startup item, start the
# host, and STOP.
#
#   irm https://github.com/telltaleatheist/crucible/releases/latest/download/install.ps1 | iex
#
# No admin. Everything here is per-user and idempotent: run it again after
# a failure and it resumes from whatever is already on disk.

[CmdletBinding()]
param(
  [string]$Release = '0.6.0',
  [string]$Root = "$env:LOCALAPPDATA\Crucible"
)

# Continue, not Stop: every call below is a native program whose exit code
# is checked explicitly, and Windows PowerShell 5.1 turns a native command writing to stderr into a terminating error under Stop.
$ErrorActionPreference = "Continue"
$HostDir = Join-Path $Root 'host'
$DownloadDir = Join-Path $Root 'downloads'
$Partial = "$HostDir.partial"
$Stamp = Join-Path $HostDir '.pack'
$Cmd = Join-Path $HostDir "crucible.cmd"
$Pythonw = Join-Path $HostDir "pythonw.exe"

function Say($m) { Write-Host "crucible: $m" }
function Die($m) { Write-Host "crucible: $m" -ForegroundColor Red; exit 1 }

# --- 0. this machine can hold the host -----------------------------------
# 64-bit x86 only: the pinned interpreter is
# x86_64-pc-windows-msvc-install_only and there is no second pin (4.4).
if ([System.Environment]::Is64BitOperatingSystem -ne $true) {
  Die "unsupported_platform: the Crucible Windows host is 64-bit x86 only."
}
if (-not $env:LOCALAPPDATA) {
  Die "host_no_localappdata: LOCALAPPDATA is not set, so there is no per-user place to install into."
}

# A pack is a zstd tarball. Windows 10 1803+ and Windows 11 ship bsdtar
# linked with libzstd, so no zstd.exe is needed — MEASURED on 2026-09-14:
# bsdtar 3.8.1 / libarchive 3.8.1 / libzstd 1.5.5. A machine whose tar has
# no zstd would half-unpack in silence, so it is CHECKED, not assumed.
$tarVersion = ""
try { $tarVersion = (& tar.exe --version | Out-String) } catch { $tarVersion = "" }
if ($tarVersion -notmatch "zstd") {
  Die "guest_missing_tool: this machine tar cannot read zstd (tar --version said: $($tarVersion.Trim())). Windows 10 1803+ and Windows 11 ship one that can."
}

# --- 1. which pack -------------------------------------------------------
Say "release $Release"
$manifestUrl = "https://github.com/telltaleatheist/crucible/releases/download/v$Release/envpacks.json"
$manifestRaw = & curl.exe -fsSL --retry 3 "$manifestUrl"
if ($LASTEXITCODE -ne 0) { Die "pack_manifest_unreadable: could not fetch $manifestUrl" }
try { $manifest = $manifestRaw | Out-String | ConvertFrom-Json } catch { Die "pack_manifest_unreadable: $manifestUrl is not JSON" }
if ($manifest.schema -ne 1) { Die "pack_manifest_unreadable: $manifestUrl declares schema $($manifest.schema), this installer reads 1" }
$pack = $null
foreach ($entry in $manifest.packs) { if ($entry.name -eq 'host' -and $entry.backend -eq 'llama-windows') { $pack = $entry } }
if ($null -eq $pack) { Die "pack_not_published: the $Release release publishes no host pack for llama-windows" }

# --- 2. already installed? ------------------------------------------------
# The stamp is the same two lines the guest-side install writes, read the
# same way: a matching sha means these bytes are already unpacked.
$have = ""
if (Test-Path $Stamp) {
  foreach ($line in (Get-Content $Stamp)) { if ($line -match "^sha256=(.+)$") { $have = $Matches[1].Trim() } }
}
if ($have -eq $pack.sha256 -and (Test-Path $Cmd)) {
  Say "host-pack: already installed ($($pack.sha256))"
} else {
  # --- 3. disk ------------------------------------------------------------
  # The same sum pack.ts requiredBytes() uses: unpacked + the whole archive
  # + one part, a part being the archive over the part count.
  $need = $pack.unpacked_bytes + $pack.bytes + [math]::Floor($pack.bytes / $pack.parts.Count)
  $drive = (Get-Item $env:LOCALAPPDATA).PSDrive
  if ($drive.Free -lt $need) {
    Die "pack_disk: the host pack needs $([math]::Round($need/1GB,1)) GiB free on $($drive.Name): and there is $([math]::Round($drive.Free/1GB,1)) GiB. Nothing has been downloaded."
  }

  # --- 4. download, join, verify -------------------------------------------
  New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
  $archiveName = $pack.parts[0] -replace "\.part[0-9]+$", ""
  $archive = Join-Path $DownloadDir $archiveName
  if (Test-Path $archive) { Remove-Item $archive -Force }
  foreach ($part in $pack.parts) {
    Say "host-pack: $part"
    $partPath = Join-Path $DownloadDir $part
    & curl.exe -fL --retry 3 --retry-delay 2 --continue-at - --create-dirs -o $partPath "https://github.com/telltaleatheist/crucible/releases/download/v$Release/$part"
    if ($LASTEXITCODE -ne 0) { Die "pack_download_failed: https://github.com/telltaleatheist/crucible/releases/download/v$Release/$part" }
    # Byte-for-byte append, then delete: peak extra disk is ONE part and
    # not the whole set. Add-Content would re-encode the bytes as text.
    $in = [System.IO.File]::OpenRead($partPath)
    $out = [System.IO.File]::Open($archive, "Append", "Write")
    try { $in.CopyTo($out) } finally { $out.Close(); $in.Close() }
    Remove-Item $partPath -Force
  }
  $got = (Get-FileHash -Algorithm SHA256 -Path $archive).Hash.ToLower()
  if ($got -ne $pack.sha256) {
    Remove-Item $archive -Force
    Die "pack_sha_mismatch: $archiveName hashes $got, the manifest says $($pack.sha256)"
  }

  # --- 5. unpack beside, prove it runs, THEN move --------------------------
  if (Test-Path $Partial) { Remove-Item $Partial -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $Partial | Out-Null
  & tar.exe --zstd -xf $archive -C $Partial
  if ($LASTEXITCODE -ne 0) { Die "pack_unpack_failed: tar would not open $archive" }
  # The .cmd and not the .exe: pip Scripts\*.exe launchers bake the build
  # tree interpreter path into the binary and do not survive this move
  # (PHASE15-HOST.md 4.4, and PHASE14 7.2a for the POSIX half of it).
  & (Join-Path $Partial "crucible.cmd") --version | Out-Null
  if ($LASTEXITCODE -ne 0) { Die "pack_unpack_failed: crucible.cmd in $Partial would not run" }
  if (Test-Path $HostDir) { Remove-Item $HostDir -Recurse -Force }
  Move-Item $Partial $HostDir
  Set-Content -Path $Stamp -Encoding ascii -Value @("sha256=$($pack.sha256)", "release=$Release")
  Remove-Item $archive -Force
  Say "host-pack: unpacked $($pack.parts.Count) part(s) into $HostDir (Python $($pack.python))"
}

# --- 6. start at login ----------------------------------------------------
# The host OWNS that shortcut (4.1). This script asks for it by verb rather
# than writing a .lnk of its own, so there is one spelling of what it points
# at and one place that changes when it moves.
& $Cmd host --install-startup
if ($LASTEXITCODE -ne 0) { Die "the Startup item could not be written (crucible host --install-startup exited $LASTEXITCODE)" }

# --- 7. start the host, and stop ------------------------------------------
# pythonw, not the .cmd: a tray program has no console window (4.1).
Say "starting the tray"
Start-Process -WindowStyle Hidden -FilePath $Pythonw -ArgumentList "-m","crucible.cli","host"
Say "Crucible is in your notification area. Open its menu to install the WSL2 engine."
