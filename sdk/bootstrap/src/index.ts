/**
 * `@crucible/bootstrap` — the app-side installer and ensurer for a local Crucible.
 *
 * A local Crucible is a SERVICE on the machine and no app owns it (Owen,
 * 2026-09-13; PHASE5-APPS.md section 6.0). An app's job is to make sure this
 * machine has one and make sure it is running, and that is the whole surface:
 *
 * ```ts
 * import { detectHost, install, ensureRunning, readLocalConfig, health } from '@crucible/bootstrap';
 * ```
 *
 * Every verb is idempotent. Every missing prerequisite is a named refusal
 * carrying the command the HOST must run. Nothing is spawned as a child that
 * should be a service. The token is never logged.
 */

export { consoleScriptBeside, detectHost, parseNvidiaSmi, pickProbeDistro, WSL_NVIDIA_SMI } from './host.js';
export type { DetectOptions, GpuFacts, HostFacts, WslFacts } from './host.js';

export { install, mintToken, planJobTypes, DEFAULT_INSTALL_TIMEOUTS, INSTALLABLE_JOB_TYPES, JOB_TYPES } from './install.js';
export type { InstallOptions, InstallResult, InstallStep, InstallTimeouts, JobType, JobTypeRequest } from './install.js';

export { compareReleases, latestRelease, LATEST_RELEASE_URL, parseLatestRelease } from './channel.js';

export {
  backendFor,
  envpacksUrl,
  ENVPACKS_ASSET,
  findPack,
  HOST_BACKEND,
  HOST_PACK,
  packAssetName,
  parseEnvpacks,
  releaseAssetUrl,
  RELEASE_REPO,
  rootfsAssetName,
  SERVER_PACK,
} from './envpacks.js';
export type { EnvPacks, PackBackend, PackEntry } from './envpacks.js';

export {
  CURL_ARGS,
  DOWNLOADS_SUBDIR,
  fetchManifest,
  guestProbeScript,
  installPack,
  HOST_SUBDIR,
  packPaths,
  probeGuest,
  requiredBytes,
  requirePack,
  SERVER_SUBDIR,
  shaArgv,
  STAMP_NAME,
  TAR_ARGS,
} from './pack.js';
export type { GuestFacts, InstalledPack, PackInstallOptions, PackInstallResult, PackPaths } from './pack.js';

export {
  CRUCIBLE_DISTRO,
  crucibleAppData,
  ensureDistro,
  importArgv,
  listDistros,
  readWslConf,
  resolveDistro,
  terminateArgv,
  unregisterArgv,
  writeWslConf,
  WSL_CONF_MARKER,
  WSL_CONF_TEXT,
} from './distro.js';
export type { DistroChoiceOptions, DistroOutcome, EnsureDistroOptions } from './distro.js';

export {
  HOST_DOOR_PORT,
  HOST_DOOR_URL,
  HOST_ENTRY_POINT,
  HOST_EVENT_KINDS,
  HOST_INSTALL_PATH,
  HOST_INSTALL_TARGET,
  hostConfigPath,
  hostInstallCommand,
  hostInstalled,
  hostPackDir,
  hostToken,
  requestHostInstall,
} from './hostdoor.js';
export type {
  HostDoneData,
  HostDoneStep,
  HostEvent,
  HostEventKind,
  HostFailedData,
  HostFetch,
  HostInstallOptions,
  HostInstallRequestBody,
  HostLineData,
  HostProgressData,
  HostStateData,
  HostStepData,
} from './hostdoor.js';

export { detectWslState, elevatedArgv, probeArgv, wslStates } from './wsl-states.js';
export type { Evidence, ProbeKey, WslAction, WslState, WslStateDef, WslStateInputs } from './wsl-states.js';

export { installSteps, renderArgv, renderSh, SHELL_VARIABLE } from './steps.js';
export type { RefName, StepDef, StepPlan, Word } from './steps.js';

export { ensureLinger, lingerCommand, parseLinger } from './linger.js';
export type { LingerOutcome } from './linger.js';

export { ensureRunning, parseServiceStatus } from './service.js';
export type { EnsureRunningOptions, RunningService, ServiceStatus } from './service.js';

export {
  connectHost,
  CRUCIBLE_HOME_ENV,
  LOCAL_SERVER_NAME,
  localConfigPath,
  parseLocalConfig,
  readLocalConfig,
} from './config.js';
export type { LocalConfig, LocalConfigOptions } from './config.js';

export { health } from './health.js';
export type { ActivityClient, HealthOptions } from './health.js';

export { BootstrapRefusal, BootstrapStepFailed } from './errors.js';
export type { BootstrapRefusalCode, BootstrapRefusalOptions } from './errors.js';

export { decodeWslBytes, incompleteTailBytes, processRunner, segmentWslBytes, splitLines } from './runner.js';
export type { OutputStream, RunOptions, RunResult, Runner, StreamOptions, WslSegment } from './runner.js';

export { guestPathFor, networkPathBehind, parseWslList, shellQuote, toWslPath, wslArgv, wslListArgv, wslRootArgv } from './wsl.js';
export type { WslDistro } from './wsl.js';

export { parseToml, TomlError } from './toml.js';
export type { TomlTable, TomlValue } from './toml.js';

export { BOOTSTRAP_VERSION } from './version.js';
export { readLocalInstallation, localStatus, startLocal, stopLocal, localUninstallCommand, LocalInstallationError } from './local.js';
export type { LocalInstallation, LocalStatus, LocalOptions, LocalCommand, LocalUninstallFlags } from './local.js';
