[CmdletBinding()]
param(
  [string]$Release = '',
  [string]$RollbackTo = '',
  [string]$Root = "$env:LOCALAPPDATA\Crucible",
  [switch]$Uninstall,
  [switch]$PurgeWeights,
  [switch]$DryRun,
  [switch]$WslToo,
  [string]$PythonArchive = '',
  [string]$WheelFile = '',
  [string]$WheelSha = ''
)

$ErrorActionPreference = "Continue"
$Root = [System.IO.Path]::GetFullPath($Root)
$env:CRUCIBLE_HOME = $Root
$HostDir = Join-Path $Root 'host'
$DownloadDir = Join-Path $Root 'downloads'
$Partial = "$HostDir.partial"
$Stamp = Join-Path $HostDir '.crucible'
$Cmd = Join-Path $HostDir "crucible.cmd"
$Pythonw = Join-Path $HostDir "pythonw.exe"
$Fresh = -not (Test-Path -LiteralPath $Cmd)

function Say($m) { Write-Host "crucible: $m" }
function Die($m) { Write-Host "crucible: $m" -ForegroundColor Red; if ($PSCommandPath) { exit 1 } else { throw "crucible: $m" } }
function Native([scriptblock]$Command) {
  & $Command 2>&1 | ForEach-Object {
    if ($_ -is [System.Management.Automation.ErrorRecord]) { "$($_.Exception.Message)" } else { "$_" }
  } | Where-Object { $_.Trim() -ne '' }
}
function Show { process { Write-Host "  $_" } }
$Previous = "$HostDir.previous"
foreach ($target in @($HostDir, $Partial, $Previous, $DownloadDir)) {
  $absolute = [System.IO.Path]::GetFullPath($target)
  if (-not $absolute.StartsWith($Root.TrimEnd("\") + "\", [System.StringComparison]::OrdinalIgnoreCase)) { Die "unsafe_install_path: $absolute is outside $Root" }
}

if ([System.Environment]::Is64BitOperatingSystem -ne $true) {
  Die "unsupported_platform: the Crucible Windows host is 64-bit x86 only."
}
if (-not $env:LOCALAPPDATA) {
  Die "host_no_localappdata: LOCALAPPDATA is not set, so there is no per-user place to install into."
}
try {
  Add-Type -ErrorAction Stop -TypeDefinition @'
using System.Runtime.InteropServices;
using System.Text;
public static class CruciblePackage {
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
  static extern int GetCurrentPackageFullName(ref uint length, StringBuilder name);
  public static int Ask(out string name) {
    uint length = 0;
    name = "";
    int code = GetCurrentPackageFullName(ref length, null);
    if (code != 122) { return code; }
    StringBuilder buffer = new StringBuilder((int)length);
    code = GetCurrentPackageFullName(ref length, buffer);
    name = buffer.ToString();
    return code;
  }
}
'@
} catch {
  Die "package_check_failed: Windows could not be asked whether this PowerShell runs inside an app package ($($_.Exception.Message)), and a Crucible written from inside one is a Crucible nothing can start. Open an ordinary PowerShell window (Start, type PowerShell, press Enter) and run it there."
}
$PackageName = ''
$PackageCode = [CruciblePackage]::Ask([ref]$PackageName)
if ($PackageCode -eq 0) { Die "packaged_shell: this installer is running inside the Windows app package $PackageName (an app installed from the Store or as an MSIX, such as the Claude desktop app, and anything started from a terminal inside it). Windows quietly redirects what such a process writes under AppData into that app's own private folder, so Crucible would land where only that app can see it, and would not start when you sign in. Nothing has been written. Open an ordinary PowerShell window (Start, type PowerShell, press Enter) and run it there." }
if ($PackageCode -ne 15700) { Die "package_check_failed: GetCurrentPackageFullName returned $PackageCode, so it is not known whether this PowerShell runs inside an app package. Open an ordinary PowerShell window (Start, type PowerShell, press Enter) and run it there." }

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
  if ($LASTEXITCODE -ne 0) { Die "step_failed: uninstall (crucible uninstall exited $LASTEXITCODE; nothing of the runtime has been removed)" }
  if ($DryRun) {
    Say "host: would remove $HostDir and $DownloadDir"
    Say "home: would remove $Root if it were then empty"
    exit 0
  }
  Say "host"
  foreach ($gone in @($Partial, $DownloadDir, $HostDir)) {
    if (Test-Path $gone) {
      try {
        Remove-Item $gone -Recurse -Force -ErrorAction Stop
      } catch {
        Die "host_runtime_locked: $gone could not be removed ($($_.Exception.Message)). Something still holds a file in it  -  the tray was just ended, so log out and back in, then run this again. It is idempotent."
      }
    }
  }
  Say "host: removed $HostDir"
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

$Tar = Join-Path $env:SystemRoot "System32\tar.exe"
if (-not (Test-Path $Tar)) {
  Die "guest_missing_tool: there is no $Tar on this machine. Windows 10 1803+ and Windows 11 ship a bsdtar there, and the interpreter archive cannot be unpacked without one."
}

if (-not $Release) {
  $feed = "https://api.github.com/repos/telltaleatheist/crucible/releases/latest"
  $feedRaw = & curl.exe -fsSL --retry 3 -H "Accept: application/vnd.github+json" "$feed"
  if ($LASTEXITCODE -ne 0) { Die "release_channel_unreadable: could not read the release channel at $feed -- name a release with -Release <version>" }
  try { $feedJson = $feedRaw | Out-String | ConvertFrom-Json } catch { Die "release_channel_unreadable: $feed is not JSON" }
  $Release = ($feedJson.tag_name) -replace '^v',''
  if (-not $Release) { Die "release_channel_unreadable: $feed named no release" }
}
if ($RollbackTo -and $RollbackTo -ne $Release) { Die "rollback_version_mismatch: -RollbackTo names $RollbackTo and the release being installed is $Release; a rollback names the exact Crucible you want back" }
Say "release $Release"

$PyAsset = 'cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only.tar.gz'
$PySha = '6be524fa6752af802146a4adc7d098565425b0b1c166e19a5a7a4c8cccb86bf6'
$PyVersion = '3.11.16'
$PyUrl = 'https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only.tar.gz'
$PythonExe = Join-Path $HostDir "python.exe"
$have = ""
$haveRelease = ""
if (Test-Path $Stamp) {
  foreach ($line in (Get-Content $Stamp)) {
    if ($line -match "^python_sha256=(.+)$") { $have = $Matches[1].Trim() }
    if ($line -match "^release=(.+)$") { $haveRelease = $Matches[1].Trim() }
  }
}
if ($haveRelease) {
  $onDisk = $null; $wanted = $null
  if ([version]::TryParse($haveRelease, [ref]$onDisk) -and [version]::TryParse($Release, [ref]$wanted) -and $wanted -lt $onDisk) {
    if ($RollbackTo -ne $Release) {
      Die "install_would_downgrade: $HostDir is the $haveRelease release and this would install $Release over it. Nothing has been downloaded. An operator who means to go back names the version: -RollbackTo $Release"
    }
  }
}
if ($have -eq $PySha -and (Test-Path $PythonExe)) {
  Say "host: python $PyVersion is already at $HostDir"
} else {
  New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
  $archive = Join-Path $DownloadDir $PyAsset
  if (Test-Path $archive) { Remove-Item $archive -Force }
  if ($PythonArchive) {
    Say "host: python $PyVersion from $PythonArchive"
    Copy-Item -LiteralPath $PythonArchive -Destination $archive -Force
  } else {
    Say "host: python $PyVersion from python-build-standalone"
    Native { & curl.exe -fL --retry 3 --retry-delay 2 --create-dirs -sS -o $archive "$PyUrl" } | Show
    if ($LASTEXITCODE -ne 0) { Die "runtime_download_failed: $PyUrl" }
  }
  $got = (Get-FileHash -Algorithm SHA256 -Path $archive).Hash.ToLower()
  if ($got -ne $PySha) {
    Remove-Item $archive -Force
    Die "runtime_sha_mismatch: $PyAsset hashes $got, this installer pins $PySha. The download was deleted"
  }

  if (Test-Path $Partial) { Remove-Item $Partial -Recurse -Force }
  New-Item -ItemType Directory -Force -Path $Partial | Out-Null
  & $Tar -xzf $archive -C $Partial
  if ($LASTEXITCODE -ne 0) { Die "runtime_unpack_failed: tar would not open $archive" }
  $staged = Join-Path $Partial "python"
  & (Join-Path $staged "python.exe") --version | Out-Null
  if ($LASTEXITCODE -ne 0) { Die "runtime_unpack_failed: python.exe in $staged would not run" }
  if (Test-Path -LiteralPath $Previous) { Die "upgrade_recovery_required: $Previous exists from an interrupted upgrade; restore or inspect it before retrying" }
  if (Test-Path -LiteralPath $HostDir) {
    if (Test-Path -LiteralPath $Cmd) {
      $said = @(Native { & $Cmd local shutdown })
      if ($LASTEXITCODE -ne 0) { $said | Show; Die "upgrade_stop_failed: the old runtime was kept because Crucible did not stop cleanly" }
    }
    Move-Item -LiteralPath $HostDir -Destination $Previous -ErrorAction Stop
  }
  try {
    Move-Item $staged $HostDir -ErrorAction Stop
    & $PythonExe --version | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "the installed interpreter failed its startup check" }
  } catch {
    if ((Test-Path -LiteralPath $Previous) -and (Test-Path -LiteralPath $HostDir)) { Remove-Item -LiteralPath $HostDir -Recurse -Force }
    if ((Test-Path -LiteralPath $Previous) -and -not (Test-Path -LiteralPath $HostDir)) { Move-Item -LiteralPath $Previous -Destination $HostDir }
    Die "upgrade_swap_failed: $_. The previous runtime is retained at $Previous when present."
  }
  if (Test-Path -LiteralPath $Previous) { Remove-Item -LiteralPath $Previous -Recurse -Force }
  Remove-Item $Partial -Recurse -Force
  Remove-Item $archive -Force
  Say "host: python $PyVersion at $HostDir"
}

New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
$Wheel = "crucible-$Release-py3-none-any.whl"
$WheelPath = Join-Path $DownloadDir $Wheel
if (Test-Path $WheelPath) { Remove-Item $WheelPath -Force }
Say "host: $Wheel"
if ($WheelFile) {
  Copy-Item -LiteralPath $WheelFile -Destination $WheelPath -Force
} else {
  Native { & curl.exe -fL --retry 3 --retry-delay 2 --create-dirs -sS -o $WheelPath "https://github.com/telltaleatheist/crucible/releases/download/v$Release/$Wheel" } | Show
  if ($LASTEXITCODE -ne 0) { Die "runtime_download_failed: https://github.com/telltaleatheist/crucible/releases/download/v$Release/$Wheel" }
}
if ($WheelSha) {
  $want = $WheelSha.Trim().ToLower()
  if ($want -notmatch '^[0-9a-f]{64}$') { Die "runtime_download_failed: -WheelSha $WheelSha is not a sha256" }
} else {
  $wantRaw = & curl.exe -fsSL --retry 3 "https://github.com/telltaleatheist/crucible/releases/download/v$Release/crucible-$Release-py3-none-any.whl.sha256"
  if ($LASTEXITCODE -ne 0) { Die "runtime_download_failed: https://github.com/telltaleatheist/crucible/releases/download/v$Release/crucible-$Release-py3-none-any.whl.sha256" }
  $want = ($wantRaw | Out-String).Trim().Split()[0].ToLower()
  if ($want -notmatch '^[0-9a-f]{64}$') { Die "runtime_download_failed: https://github.com/telltaleatheist/crucible/releases/download/v$Release/crucible-$Release-py3-none-any.whl.sha256 is not a sha256" }
}
$gotWheel = (Get-FileHash -Algorithm SHA256 -Path $WheelPath).Hash.ToLower()
if ($gotWheel -ne $want) {
  Remove-Item $WheelPath -Force
  Die "runtime_sha_mismatch: $Wheel hashes $gotWheel, the release says $want. The download was deleted"
}
if (Test-Path -LiteralPath $Cmd) {
  $Stage = Join-Path $DownloadDir "stage"
  if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
  $stopped = $false
  $null = @(Native { & $PythonExe -m pip install --quiet --no-deps --no-input --target $Stage $WheelPath })
  if ($LASTEXITCODE -eq 0) {
    $keptPath = $env:PYTHONPATH
    $env:PYTHONPATH = $Stage
    $said = @(Native { & $PythonExe -m crucible.cli local shutdown })
    $stopped = ($LASTEXITCODE -eq 0)
    $env:PYTHONPATH = $keptPath
  }
  if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
  if (-not $stopped) {
    $said = @(Native { & $Cmd local shutdown })
    $stopped = ($LASTEXITCODE -eq 0)
  }
  if (-not $stopped) { Say "host: the running Crucible did not stop cleanly; installing over it:"; $said | Show }
}
Say "host: installing Crucible into $HostDir (about a minute)"
Native { & $PythonExe -m pip install --quiet --disable-pip-version-check --no-warn-script-location --upgrade --no-input $WheelPath } | Show
if ($LASTEXITCODE -ne 0) { Die "runtime_install_failed: pip would not install $Wheel into $HostDir" }
Native { & $PythonExe -m pip install --quiet --disable-pip-version-check --no-warn-script-location pystray pillow } | Show
if ($LASTEXITCODE -ne 0) { Die "runtime_install_failed: the tray packages would not install" }
Remove-Item $WheelPath -Force

$shim = "@echo off`r`n""%~dp0python.exe"" -m crucible.cli %*`r`n"
[System.IO.File]::WriteAllText($Cmd, $shim, [System.Text.Encoding]::ASCII)
& $Cmd --version | Out-Null
if ($LASTEXITCODE -ne 0) { Die "runtime_install_failed: $Cmd would not run" }
Set-Content -Path $Stamp -Encoding ascii -Value @("python_sha256=$PySha", "python_version=$PyVersion", "release=$Release")
Say "host: $Release installed at $HostDir (Python $PyVersion)"

foreach ($action in @("register", "install-cli", "install-desktop")) {
  $said = @(Native { & $Cmd local $action })
  if ($LASTEXITCODE -ne 0) { $said | Show; Die "local setup failed: $action (exit $LASTEXITCODE). Run this installer again; it carries on from where it stopped." }
}

Say "starting the tray"
$Began = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
Start-Process -WindowStyle Hidden -FilePath $Pythonw -ArgumentList "-m","crucible.cli","orchestrator"
$said = @(Native { & $Cmd local start })
if ($LASTEXITCODE -ne 0) { $said | Show; Die "Crucible is installed, but its engine did not start. Run this installer again; it carries on from where it stopped." }

Say "Crucible is ready in your notification area."
$FromApp = [bool]($PSCommandPath -and ([System.IO.Path]::GetFileName($PSCommandPath) -eq 'crucible-install.ps1') -and [Console]::IsOutputRedirected)
$Watch = @('-m', 'crucible.host.installwatch', '--home', $Root, '--since', $Began)
if ($FromApp) { $Watch += '--brief' }
Native { & $PythonExe @Watch } | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) {
  Say "The Linux engine sets itself up in the background, which takes several minutes; there is nothing to click. If it stops or needs a Windows restart, the menu of the Crucible icon by the clock says so."
}
Say "Crucible is in your Start Menu: search for Crucible."
$Interactive = [Environment]::UserInteractive -and -not $FromApp -and -not $env:SSH_CONNECTION
if ($Fresh -and $Interactive) {
  Say "opening Crucible"
  Start-Process -FilePath $Pythonw -ArgumentList "-m","crucible.cli","app"
}
