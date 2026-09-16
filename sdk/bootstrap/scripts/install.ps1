# GENERATED FILE  -  do not edit.
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
# page drives it as a task (4.7)  -  an app that asks for an install talks
# to the SAME implementation through the host loopback door. Two walks of
# one table was the thing being removed.
#
# So: download the host pack for this release, verify it, unpack it to
# %LOCALAPPDATA%\Crucible\host\, register the Startup item, start the
# host, and STOP.
#
#   irm https://github.com/telltaleatheist/crucible/releases/latest/download/install.ps1 | iex
#
# And to take it off again, keeping the weights (the -Uninstall branch
# below): download it to a file first, because `irm | iex` has no way to
# pass a switch.
#
#   irm https://github.com/telltaleatheist/crucible/releases/latest/download/install.ps1 -OutFile install.ps1
#   .\install.ps1 -Uninstall            # weights kept
#   .\install.ps1 -Uninstall -WslToo    # and the guest engine with it
#
# No admin. Everything here is per-user and idempotent: run it again after
# a failure and it resumes from whatever is already on disk.

[CmdletBinding()]
param(
  [string]$Release = '0.6.3',
  [string]$Root = "$env:LOCALAPPDATA\Crucible",
  # The inverse. `crucible uninstall` does the work inside the home; this
  # script removes the host pack, because this script is what unpacked it.
  [switch]$Uninstall,
  [switch]$PurgeWeights,
  [switch]$DryRun,
  [switch]$WslToo
)

# Continue, not Stop: every call below is a native program whose exit code
# is checked explicitly, and Windows PowerShell 5.1 turns a native command writing to stderr into a terminating error under Stop.
$ErrorActionPreference = "Continue"
$Root = [System.IO.Path]::GetFullPath($Root)
$env:CRUCIBLE_HOME = $Root
$HostDir = Join-Path $Root 'host'
$DownloadDir = Join-Path $Root 'downloads'
$Partial = "$HostDir.partial"
$Stamp = Join-Path $HostDir '.pack'
$Cmd = Join-Path $HostDir "crucible.cmd"
$Pythonw = Join-Path $HostDir "pythonw.exe"

function Say($m) { Write-Host "crucible: $m" }
function Die($m) { Write-Host "crucible: $m" -ForegroundColor Red; exit 1 }
$Previous = "$HostDir.previous"
foreach ($target in @($HostDir, $Partial, $Previous, $DownloadDir)) {
  $absolute = [System.IO.Path]::GetFullPath($target)
  if (-not $absolute.StartsWith($Root.TrimEnd("\") + "\", [System.StringComparison]::OrdinalIgnoreCase)) { Die "unsafe_install_path: $absolute is outside $Root" }
}

# --- 0. this machine can hold the host -----------------------------------
# 64-bit x86 only: the pinned interpreter is
# x86_64-pc-windows-msvc-install_only and there is no second pin (4.4).
if ([System.Environment]::Is64BitOperatingSystem -ne $true) {
  Die "unsupported_platform: the Crucible Windows host is 64-bit x86 only."
}
if (-not $env:LOCALAPPDATA) {
  Die "host_no_localappdata: LOCALAPPDATA is not set, so there is no per-user place to install into."
}

# --- the inverse, which exits ---------------------------------------------
# `crucible uninstall` stops the tray, removes the Startup item and takes
# %LOCALAPPDATA%\Crucible apart step by named step  -  everything except the
# interpreter it is itself running from. THIS script unpacked that, so this
# script removes it, after the verb has returned. Weights are kept unless
# -PurgeWeights; -WslToo runs the guest's own uninstall first.
if ($Uninstall) {
  if (-not (Test-Path $Cmd)) {
    Die "not_installed: there is no $Cmd on this machine, so there is no Crucible host here to remove."
  }
  $verb = @("uninstall")
  if ($DryRun) { $verb += "--dry-run" }
  if ($PurgeWeights) { $verb += "--purge-weights" }
  if ($WslToo) { $verb += "--wsl-too" }
  Say "uninstall: $Cmd $($verb -join ' ')"
  & $Cmd @verb
  if ($LASTEXITCODE -ne 0) { Die "step_failed: uninstall (crucible uninstall exited $LASTEXITCODE; nothing of the pack has been removed)" }
  if ($DryRun) {
    Say "host-pack: would remove $HostDir and $DownloadDir"
    Say "home: would remove $Root if it were then empty"
    exit 0
  }
  Say "host-pack"
  foreach ($gone in @($Partial, $DownloadDir, $HostDir)) {
    if (Test-Path $gone) {
      try {
        Remove-Item $gone -Recurse -Force -ErrorAction Stop
      } catch {
        Die "host_pack_locked: $gone could not be removed ($($_.Exception.Message)). Something still holds a file in it  -  the tray was just ended, so log out and back in, then run this again. It is idempotent."
      }
    }
  }
  Say "host-pack: removed $HostDir"
  $left = @(Get-ChildItem -Force -Path $Root -ErrorAction SilentlyContinue)
  if ($left.Count -eq 0) {
    Remove-Item $Root -Force -Recurse
    Say "home: removed $Root"
  } else {
    Say "home: KEPT $Root  -  it still holds $($left.Name -join ', ')"
    Say "home: weights are kept unless -PurgeWeights; nothing else there was Crucible's to delete"
  }
  Say "uninstalled."
  exit 0
}

# A pack is a zstd tarball. Windows 10 1803+ and Windows 11 ship bsdtar
# linked with libzstd, so no zstd.exe is needed  -  MEASURED on 2026-09-14:
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
  # Stop with the new staged control code before touching the installed runtime.
  # A shutdown failure leaves both the old runtime and verified staging intact.
  if (Test-Path -LiteralPath $Previous) { Die "upgrade_recovery_required: $Previous exists from an interrupted upgrade; restore or inspect it before retrying" }
  if (Test-Path -LiteralPath $HostDir) {
    & (Join-Path $Partial "crucible.cmd") local shutdown
    if ($LASTEXITCODE -ne 0) { Die "upgrade_stop_failed: the old runtime was kept because Crucible did not stop cleanly" }
    Move-Item -LiteralPath $HostDir -Destination $Previous -ErrorAction Stop
  }
  try {
    Move-Item $Partial $HostDir -ErrorAction Stop
    & $Cmd --version | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "the installed runtime failed its startup check" }
  } catch {
    if ((Test-Path -LiteralPath $Previous) -and (Test-Path -LiteralPath $HostDir) -and -not (Test-Path -LiteralPath $Partial)) { Move-Item -LiteralPath $HostDir -Destination $Partial -ErrorAction Stop }
    if ((Test-Path -LiteralPath $Previous) -and -not (Test-Path -LiteralPath $HostDir)) { Move-Item -LiteralPath $Previous -Destination $HostDir }
    Die "upgrade_swap_failed: $_. The previous runtime is retained at $Previous when present."
  }
  if (Test-Path -LiteralPath $Previous) { Remove-Item -LiteralPath $Previous -Recurse -Force }
  Set-Content -Path $Stamp -Encoding ascii -Value @("sha256=$($pack.sha256)", "release=$Release")
  Remove-Item $archive -Force
  Say "host-pack: unpacked $($pack.parts.Count) part(s) into $HostDir (Python $($pack.python))"
}

# --- 6. start at login ----------------------------------------------------
# The host OWNS that shortcut (4.1). This script asks for it by verb rather
# than writing a .lnk of its own, so there is one spelling of what it points
# at and one place that changes when it moves.
foreach ($action in @("register", "install-cli", "install-desktop")) {
  & $Cmd local $action
  if ($LASTEXITCODE -ne 0) { Die "local setup failed: $action (exit $LASTEXITCODE)" }
}

# --- 7. start the host, and stop ------------------------------------------
# pythonw, not the .cmd: a tray program has no console window (4.1).
Say "starting the tray"
Start-Process -WindowStyle Hidden -FilePath $Pythonw -ArgumentList "-m","crucible.cli","host"
& $Cmd local start
if ($LASTEXITCODE -ne 0) { Die "Crucible installed but did not become ready. Run crucible local status for the named failure." }
Say "Crucible is ready in your notification area. The Windows engine works now; the optional Linux engine is available from its console."
