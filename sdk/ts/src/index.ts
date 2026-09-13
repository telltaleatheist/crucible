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

export { CrucibleClient } from './client.js';
export type { CrucibleClientOptions, EventsOptions } from './client.js';

export {
  CrucibleAuthError,
  CrucibleConfigError,
  CrucibleError,
  CrucibleNotACrucible,
  CrucibleProtocolError,
  CrucibleRefused,
  CrucibleServerError,
  CrucibleUnreachable,
  CrucibleVersionError,
} from './errors.js';

export { API_VERSION, TERMINAL_EVENTS, isLlmCapability } from './types.js';
export type {
  ArtifactData,
  CancelResult,
  CancelledData,
  Capability,
  ChatMessage,
  ChatOptions,
  ChatResponse,
  ChatUsage,
  DoneData,
  FailedData,
  GpuInfo,
  Health,
  JobCapability,
  JobEvent,
  JobFailure,
  JobInput,
  JobRequest,
  JobState,
  JobStatus,
  LlmCapability,
  ModelDescriptor,
  ModelInfo,
  Ping,
  ProgressData,
  Provenance,
  QueuedData,
  ServerInfo,
  TerminalEventName,
  UploadResult,
  WarmingData,
} from './types.js';

export { SDK_VERSION } from './version.js';
