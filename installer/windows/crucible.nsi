Unicode true
ManifestDPIAware true
RequestExecutionLevel user
SetCompressor /SOLID lzma

!addplugindir /x86-unicode "${PLUGINS}"

!include "MUI2.nsh"
!include "LogicLib.nsh"

!define UNINSTALL_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\Crucible"
!define UNINSTALLER "Uninstall Crucible.exe"
!define SCRIPT "crucible-install.ps1"
!define POWERSHELL "$SYSDIR\WindowsPowerShell\v1.0\powershell.exe"
!define APP_ICON "host\Lib\site-packages\crucible\desktop_app\assets\crucible.ico"

Name "Crucible"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Crucible"
BrandingText "Crucible ${VERSION}"
ShowInstDetails show
ShowUninstDetails show

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "Crucible"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "FileDescription" "Crucible setup"
VIAddVersionKey "CompanyName" "Owen Morgan"
VIAddVersionKey "LegalCopyright" "Owen Morgan"

!define MUI_ICON "${ICON}"
!define MUI_UNICON "${ICON}"
!define MUI_WELCOMEFINISHPAGE_BITMAP "${WELCOME}"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "${WELCOME}"
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "Install Crucible"
!define MUI_WELCOMEPAGE_TEXT "Crucible runs AI models on this computer: language models, narration voices, transcription and more.$\r$\n$\r$\nSetup downloads Python ${PY_VERSION} and Crucible ${VERSION} (about 60 MB), checks each download against the fingerprint built into this setup, and installs them for you only. No administrator rights are needed.$\r$\n$\r$\nClick Next to continue."
!define MUI_DIRECTORYPAGE_TEXT_TOP "Crucible keeps its programs, settings and downloaded models in this folder. Models are large: choose a disk with plenty of free space."
!define MUI_FINISHPAGE_TITLE "Crucible is installed"
!define MUI_FINISHPAGE_TEXT "Crucible is in your Start Menu, and its icon sits by the clock. It is now setting up its Linux engine by itself, in the background, which takes several minutes; there is nothing to click. If that stops or needs a Windows restart, the menu of the Crucible icon by the clock says so."
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Open Crucible"
!define MUI_FINISHPAGE_RUN_FUNCTION OpenCrucible

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Var Url
Var Target
Var Pinned

Function Fetch
  DetailPrint "Downloading $Url"
  NScurl::http GET "$Url" "$Target" /INSIST /CANCEL /TIMEOUT 60s /END
  Pop $0
  ${If} $0 != "OK"
    Delete "$Target"
    MessageBox MB_ICONSTOP "download_failed: $Url could not be downloaded ($0).$\r$\n$\r$\nCheck that this computer is online, then run this setup again."
    Abort
  ${EndIf}
  NScurl::sha256 -file "$Target"
  Pop $0
  ${If} $0 != $Pinned
    Delete "$Target"
    MessageBox MB_ICONSTOP "download_mismatch: $Url arrived as $0, but this setup was built for $Pinned. The download was deleted.$\r$\n$\r$\nRun this setup again; if it says this twice, download a fresh setup from https://github.com/telltaleatheist/crucible/releases/latest"
    Abort
  ${EndIf}
  DetailPrint "Checked $Target"
FunctionEnd

Section "Crucible"
  InitPluginsDir
  SetOutPath "$PLUGINSDIR"
  File "/oname=${SCRIPT}" "${INSTALL_PS1}"
  CreateDirectory "$PLUGINSDIR\downloads"

  StrCpy $Url "${PY_URL}"
  StrCpy $Target "$PLUGINSDIR\downloads\${PY_ASSET}"
  StrCpy $Pinned "${PY_SHA}"
  Call Fetch

  StrCpy $Url "${WHEEL_URL}"
  StrCpy $Target "$PLUGINSDIR\downloads\${WHEEL_NAME}"
  StrCpy $Pinned "${WHEEL_SHA}"
  Call Fetch

  DetailPrint "Installing Crucible into $INSTDIR (a few minutes)"
  nsExec::ExecToLog '"${POWERSHELL}" -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\${SCRIPT}" -Release ${VERSION} -Root "$INSTDIR" -PythonArchive "$PLUGINSDIR\downloads\${PY_ASSET}" -WheelFile "$PLUGINSDIR\downloads\${WHEEL_NAME}" -WheelSha ${WHEEL_SHA}'
  Pop $0
  ${If} $0 != 0
    MessageBox MB_ICONSTOP "Crucible did not finish installing (the install step ended with $0). What it said is in the list above.$\r$\n$\r$\nRun this setup again: it carries on from where it stopped."
    Abort
  ${EndIf}

  SetOutPath "$INSTDIR"
  WriteUninstaller "$INSTDIR\${UNINSTALLER}"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayName" "Crucible"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "Publisher" "Owen Morgan"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "URLInfoAbout" "https://github.com/telltaleatheist/crucible"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayIcon" "$INSTDIR\${APP_ICON}"
  WriteRegStr HKCU "${UNINSTALL_KEY}" "UninstallString" '"$INSTDIR\${UNINSTALLER}"'
  WriteRegStr HKCU "${UNINSTALL_KEY}" "QuietUninstallString" '"$INSTDIR\${UNINSTALLER}" /S'
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoRepair" 1
SectionEnd

Function OpenCrucible
  System::Call 'Kernel32::SetEnvironmentVariable(t "CRUCIBLE_HOME", t "$INSTDIR")'
  Exec '"$INSTDIR\host\pythonw.exe" -m crucible.cli app'
FunctionEnd

Section "Uninstall"
  InitPluginsDir
  SetOutPath "$PLUGINSDIR"
  File "/oname=${SCRIPT}" "${INSTALL_PS1}"
  ${If} ${FileExists} "$INSTDIR\host\crucible.cmd"
    DetailPrint "Removing Crucible with its own uninstall"
    nsExec::ExecToLog '"${POWERSHELL}" -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\${SCRIPT}" -Uninstall -Root "$INSTDIR"'
    Pop $0
    ${If} $0 != 0
      MessageBox MB_ICONSTOP "Crucible's uninstall stopped (it ended with $0), and nothing more was removed. What it said is in the list above.$\r$\n$\r$\nClose the Crucible window and its icon by the clock, then run this uninstaller again from Settings, Apps."
      Abort
    ${EndIf}
  ${EndIf}
  SetOutPath "$TEMP"
  Delete "$INSTDIR\${UNINSTALLER}"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINSTALL_KEY}"
SectionEnd
