/**
 * `@crucible/client` — the TypeScript client for a Crucible inference server.
 *
 * ```ts
 * import { CrucibleClient } from '@crucible/client';
 *
 * const crucible = new CrucibleClient({
 *   url: 'http://127.0.0.1:7100',
 *   token: process.env.CRUCIBLE_TOKEN!,
 *   clientName: 'bookforge',
 * });
 * ```
 *
 * Zero runtime dependencies. Node 20+, bun, and the Electron main process.
 */

export { CrucibleClient, engineOf, readRenderResult } from './client.js';
export type {
  CrucibleClientOptions,
  EventsOptions,
  WriteArtifactsOptions,
} from './client.js';

export {
  ACCELERATOR_UNREADABLE,
  CAPABILITY_ROUTE_MISSING,
  CAPABILITY_ROUTE_UNKNOWN,
  CAPABILITY_UNDECIDED,
  CrucibleAcceleratorUnreadable,
  CrucibleAuthError,
  CrucibleBusy,
  CrucibleCapabilityUndecided,
  CrucibleCardHeld,
  CrucibleConfigError,
  CrucibleError,
  CrucibleLeased,
  CrucibleNotACrucible,
  CruciblePairingError,
  CruciblePairingFileError,
  CrucibleProtocolError,
  CrucibleRefused,
  CrucibleServerError,
  CrucibleUnreachable,
  CrucibleVersionError,
  INVALID_PAIRING,
  LEASE_NOT_NEEDED,
  LEASED,
  PAIRING_FILE_MALFORMED,
  ROUTE_BAD_MODEL,
  ROUTE_NOT_ROUTABLE,
  ROUTE_UPSTREAM_UNCONFIGURED,
  SERVER_BUSY,
  SUBJECT_IN_USE,
  SUBJECT_NOT_INSTALLED,
  SUBJECT_REMOVE_FAILED,
  SUBJECT_UNKNOWN,
  UNKNOWN_UPSTREAM,
  UPSTREAM_BAD_FIELD,
  UPSTREAM_IN_USE,
  UPSTREAM_RATE_LIMITED,
  UPSTREAM_REJECTED,
  UPSTREAM_TEST_REFUSALS,
  UPSTREAM_UNCONFIGURED,
  UPSTREAM_UNREACHABLE,
  VOICES_NEEDS_REFERENCE_MISSING,
  VOICES_NEEDS_REFERENCE_UNKNOWN,
  isServerSpecificRefusal,
} from './errors.js';

export { PAIRING_SCHEME, parsePairing } from './pairing.js';
export type { Pairing } from './pairing.js';

// NODE ONLY, and re-exported from its own module so a browser bundle that
// wants `parsePairing` does not pull `node:fs` in behind it
// (PHASE15-HOST.md section 3.8).
export {
  CRUCIBLE_HOME_ENV,
  PAIRING_FILE,
  WINDOWS_HOME_DIRNAME,
  cruciblePairingPath,
  readPairingFile,
} from './pairing-file.js';

export {
  isStreamAudio,
  isStreamDone,
  isStreamRestart,
  isStreamRowError,
  openTtsStream,
} from './stream.js';
export type {
  CancelOutcome,
  StreamAudio,
  StreamEvent,
  StreamOptions,
  StreamRestart,
  StreamRowDone,
  StreamRowError,
  StreamTransport,
  TtsStreamSession,
} from './stream.js';

export {
  API_VERSION,
  TASK_TERMINAL_STATES,
  TERMINAL_EVENTS,
  isLlmCapability,
  isTaskBytesProgress,
  isTaskLineProgress,
  isTtsCapability,
} from './types.js';
export type {
  AcceleratorGpu,
  Activity,
  ActivityChat,
  ActivityJob,
  ActivityLease,
  ActivitySlot,
  ActivityStreaming,
  AcceleratorHolder,
  AcceleratorResident,
  AcceleratorState,
  ArtifactData,
  ArtifactWrite,
  AsrOptions,
  CancelResult,
  CancelledData,
  Capability,
  CapabilityRecord,
  CapabilityRow,
  CatalogRow,
  CrucibleModule,
  ChatMessage,
  ChatOptions,
  ChatResponse,
  ChatUsage,
  ChunkData,
  DoneData,
  EstimateBasis,
  FailedData,
  GpuInfo,
  Health,
  EngineTaskRequest,
  InstallTaskRequest,
  JobCapability,
  JobEvent,
  JobFailure,
  JobInput,
  JobRequest,
  JobState,
  JobStatus,
  Lease,
  LlmCapability,
  LoadVoiceOptions,
  CrucibleRole,
  EngineOwner,
  EngineRef,
  EngineRestartTaskRequest,
  ManagedBy,
  ModelDescriptor,
  ModelInfo,
  ModuleTaskRequest,
  Ping,
  PullTaskRequest,
  ProgressData,
  Provenance,
  QueuedData,
  RenderChunk,
  RenderFailure,
  RenderOptions,
  RenderResult,
  ResponseFormat,
  RouteSetting,
  ServerInfo,
  ServerSetup,
  SettingsDocument,
  SettingsPatch,
  SubjectKind,
  TaskBytesProgress,
  TaskCancelResult,
  TaskEvent,
  TaskLineProgress,
  TaskProgressData,
  TaskRequest,
  TaskSkippedData,
  TaskState,
  TaskStatus,
  UnmetNeed,
  TaskStepData,
  TerminalEventName,
  TtsCapability,
  UploadResult,
  UpstreamName,
  UpstreamSetting,
  UpstreamTestResult,
  VoiceInfo,
  VoiceKind,
  VoicePace,
  VoiceReference,
  WarmingData,
  WrittenArtifact,
} from './types.js';

export { SDK_VERSION } from './version.js';
export { startPairing, pollPairing, crucibleAddress, DEFAULT_CRUCIBLE_PORT, CrucibleConnectionError } from './connect.js';
export type { PairingRequest, PairingResult, PairingOptions, PendingPairing } from './connect.js';
