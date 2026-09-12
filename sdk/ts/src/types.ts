/**
 * The shapes API v1 returns, hand-written from `docs/DESIGN.md` section 4 and
 * `crucible/api.py`. The e2e test is what keeps them honest.
 *
 * Convention: everything the client *models* is camelCase, because it is a
 * TypeScript API. The one exception is {@link Provenance}, which is the sidecar
 * document verbatim — clients must persist it beside the artifact (DESIGN.md
 * section 7), and rewriting its keys would corrupt the thing being persisted.
 */

/** The API contract version this client speaks. Sent as `X-Crucible-Api`. */
export const API_VERSION = 1;

/** `GET /v1/ping` — the only unauthenticated route. */
export interface Ping {
  /** Always `true`; anything else is a {@link CrucibleNotACrucible}. */
  readonly crucible: true;
  readonly name: string;
  readonly apiVersion: number;
}

/** One model a job type can serve, as advertised by `GET /v1/info`. */
export interface ModelDescriptor {
  readonly id: string;
  readonly revision: string;
  readonly source: string;
  readonly resident: boolean;
  readonly vramBytes: number;
}

/** One job type this server offers, with the models it can serve. */
export interface Capability {
  readonly jobType: string;
  readonly models: readonly ModelDescriptor[];
}

/** The accelerator the server owns. */
export interface GpuInfo {
  readonly vendor: string;
  readonly name: string;
  readonly vramBytes: number;
}

/** `GET /v1/info`. */
export interface ServerInfo {
  readonly server: {
    readonly name: string;
    readonly version: string;
    readonly apiVersion: number;
  };
  readonly host: {
    readonly platform: string;
    readonly arch: string;
    /** `cuda-linux` or `mlx-darwin`. Windows is never a backend. */
    readonly backend: string;
    readonly gpu: GpuInfo;
  };
  readonly capabilities: readonly Capability[];
}

/** `GET /v1/health`. */
export interface Health {
  readonly status: 'ok' | 'warming' | 'busy';
  readonly queueDepth: number;
  readonly residentModels: readonly string[];
}

/** `POST /v1/uploads`. */
export interface UploadResult {
  readonly blobId: string;
  readonly bytes: number;
  readonly sha256: string;
}

/** One named input on a job: either an uploaded blob or bytes carried inline. */
export type JobInput = { readonly blobId: string } | { readonly inline: Uint8Array };

/** The body of `POST /v1/jobs`. */
export interface JobRequest {
  readonly type: string;
  /** Omit for a job type that serves no models; the server refuses a mismatch. */
  readonly model?: string;
  readonly params: Readonly<Record<string, unknown>>;
  readonly inputs: Readonly<Record<string, JobInput>>;
}

/** A job's lifecycle state. `done`, `failed` and `cancelled` are terminal. */
export type JobState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

/** The server's named refusal or failure, as carried on a job and in events. */
export interface JobFailure {
  readonly code: string;
  readonly message: string;
}

/** `GET /v1/jobs/{id}`. */
export interface JobStatus {
  readonly jobId: string;
  readonly type: string;
  readonly model: string | null;
  readonly status: JobState;
  readonly progress: number;
  /** 0 while running, 1-based place in line while queued, `null` once terminal. */
  readonly position: number | null;
  readonly error: JobFailure | null;
  readonly artifacts: readonly string[];
  readonly created: string;
  readonly started: string | null;
  readonly finished: string | null;
}

/** `DELETE /v1/jobs/{id}`. A running job ends `cancelled` at its next checkpoint. */
export interface CancelResult {
  readonly jobId: string;
  readonly status: 'cancelled' | 'cancelling';
}

/**
 * One SSE event from `GET /v1/jobs/{id}/events`.
 *
 * `id` is the server's monotonic event counter, starting at 1; hand the last one
 * you saw back as `lastEventId` to resume a stream without losing an event.
 */
export type JobEvent =
  | { readonly id: number; readonly event: 'queued'; readonly data: QueuedData }
  | { readonly id: number; readonly event: 'warming'; readonly data: WarmingData }
  | { readonly id: number; readonly event: 'progress'; readonly data: ProgressData }
  | { readonly id: number; readonly event: 'artifact'; readonly data: ArtifactData }
  | { readonly id: number; readonly event: 'done'; readonly data: DoneData }
  | { readonly id: number; readonly event: 'failed'; readonly data: FailedData }
  | { readonly id: number; readonly event: 'cancelled'; readonly data: CancelledData };

export interface QueuedData {
  readonly position: number | null;
}

/**
 * A model is being loaded for this job. DESIGN.md names the event but not its
 * payload, and the phase-1 server never emits it (nothing loads a model yet), so
 * the payload is left open rather than invented.
 */
export interface WarmingData {
  readonly [key: string]: unknown;
}

export interface ProgressData {
  readonly fraction: number;
  readonly message: string;
}

export interface ArtifactData {
  readonly name: string;
}

export interface DoneData {
  readonly artifacts: readonly string[];
}

export interface FailedData {
  readonly error: JobFailure;
}

export interface CancelledData {
  readonly status: 'cancelled';
}

/** The event names that end a stream. */
export const TERMINAL_EVENTS = ['done', 'failed', 'cancelled'] as const;

export type TerminalEventName = (typeof TERMINAL_EVENTS)[number];

/**
 * `<artifact>.provenance.json`, verbatim as the server wrote it (DESIGN.md
 * section 7). Persist this file beside the artifact: a finished audiobook says
 * which server rendered it. Keys are the server's, not camelCased, so that
 * `JSON.stringify(provenance)` round-trips the document.
 */
export interface Provenance {
  readonly server: { readonly name: string; readonly version: string };
  readonly backend: string;
  readonly job_type: string;
  readonly model: { readonly id: string; readonly revision: string | null } | null;
  readonly params: Readonly<Record<string, unknown>>;
  readonly started: string | null;
  readonly finished: string;
}
