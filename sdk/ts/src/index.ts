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

export { CrucibleClient, readRenderResult } from './client.js';
export type {
  CrucibleClientOptions,
  EventsOptions,
  WriteArtifactsOptions,
} from './client.js';

export {
  ACCELERATOR_UNREADABLE,
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
  CrucibleProtocolError,
  CrucibleRefused,
  CrucibleServerError,
  CrucibleUnreachable,
  CrucibleVersionError,
  INVALID_PAIRING,
  LEASED,
  SERVER_BUSY,
  isServerSpecificRefusal,
} from './errors.js';

export { PAIRING_SCHEME, parsePairing } from './pairing.js';
export type { Pairing } from './pairing.js';

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
  ServerInfo,
  ServerSetup,
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
  TaskStepData,
  TerminalEventName,
  TtsCapability,
  UploadResult,
  VoiceInfo,
  VoiceKind,
  VoicePace,
  WarmingData,
  WrittenArtifact,
} from './types.js';

export { SDK_VERSION } from './version.js';
