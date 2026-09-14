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

export { detectHost, DEFAULT_CONDA_ROOTS, SERVER_ENV_NAME, SERVER_PYTHON, WSL_NVIDIA_SMI } from './host.js';
export type { DetectOptions, GpuFacts, HostFacts, PythonFacts, WslFacts } from './host.js';

export { install, mintToken, planJobTypes, DEFAULT_INSTALL_TIMEOUTS, INSTALLABLE_JOB_TYPES, JOB_TYPES } from './install.js';
export type { InstallOptions, InstallResult, InstallStep, InstallTimeouts, JobType, JobTypeRequest } from './install.js';

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

export { decodeWslBytes, processRunner, splitLines } from './runner.js';
export type { OutputStream, RunOptions, RunResult, Runner, StreamOptions } from './runner.js';

export { guestPathFor, guestUnpackArgv, networkPathBehind, parseWslList, shellQuote, toWslPath, wslArgv, wslListArgv, wslRootArgv } from './wsl.js';
export type { WslDistro } from './wsl.js';

export { parseToml, TomlError } from './toml.js';
export type { TomlTable, TomlValue } from './toml.js';

export { BOOTSTRAP_VERSION } from './version.js';
