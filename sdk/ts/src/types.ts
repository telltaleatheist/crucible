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

/**
 * One job type this server offers, with the models it can serve.
 *
 * Most capabilities describe their models with DESIGN.md section 4's row. The
 * `llm` capability is the exception the contract makes on purpose: its rows are
 * `GET /v1/models`' rows, the same shape from the same producer, so a model has
 * one description wherever a client finds it (PHASE2-LLM.md section 5). Narrow
 * with {@link isLlmCapability} before reading a row.
 */
export type Capability = LlmCapability | JobCapability;

/** Any capability other than `llm`. */
export interface JobCapability {
  readonly jobType: string;
  readonly models: readonly ModelDescriptor[];
}

/** The `llm` capability: `GET /v1/models`' rows, carried inside `info()`. */
export interface LlmCapability {
  readonly jobType: 'llm';
  readonly models: readonly ModelInfo[];
}

/** Narrow a capability to the `llm` one, whose rows are {@link ModelInfo}. */
export function isLlmCapability(capability: Capability): capability is LlmCapability {
  return capability.jobType === 'llm';
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
 * A model is being loaded for this job: one line of the engine's own readiness,
 * streamed as it happens. PHASE2-LLM.md section 5 pins the payload to
 * `{message}`; a `warming` frame without one is a protocol error, not an empty
 * message.
 */
export interface WarmingData {
  readonly message: string;
}

export interface ProgressData {
  readonly fraction: number;
  readonly message: string;
}

export interface ArtifactData {
  readonly name: string;
}

/**
 * The terminal `done` event's payload.
 *
 * Different job types finish with different news: a producing job (`echo`,
 * later `tts`, `vlm-pages`) reports the artifacts it wrote, and `load-model`
 * reports the model that is now resident (PHASE2-LLM.md section 5). There is no
 * discriminant inside the frame — the caller already knows which job it
 * submitted — so this is one record with both fields optional rather than a
 * union the caller would have to narrow before reading `artifacts`.
 *
 * It is still checked, not loose: `artifacts` must be an array of strings and
 * `resident` a string or `null` wherever either appears, and a `done` frame
 * carrying *neither* is a {@link CrucibleProtocolError}.
 */
export interface DoneData {
  /** What a producing job wrote. Fetch each with `artifact(jobId, name)`. */
  readonly artifacts?: readonly string[];
  /**
   * What is resident **now**, on a `load-model` or `unload-model` job: the
   * model that was loaded, or `null` after an unload, when nothing is
   * (PHASE2-LLM.md section 5). Phase 2 keeps one model resident at a time, so
   * this is the whole story, not one entry of it.
   */
  readonly resident?: string | null;
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

// --------------------------------------------------------------------- llm

/**
 * One model this server knows about, as `GET /v1/models` describes it
 * (PHASE2-LLM.md section 5).
 *
 * The four booleans are four different facts and none of them implies another:
 * `backendSupported` is "the manifest has a block for this host's backend",
 * `installed` is "the weights are on disk", `resident` is "an engine is serving
 * it right now", and `loadable` is "asking for it now would succeed" — which
 * also depends on the accelerator guard (section 4), so a model can be
 * installed and supported and still not loadable because someone else's process
 * holds the card.
 */
export interface ModelInfo {
  /** Crucible's id, stable across backends, e.g. `qwen3.5-9b`. */
  readonly id: string;
  readonly family: string;
  readonly paramsB: number;
  /**
   * The commit the manifest pins for **this host's** backend — the same sha the
   * weights were pulled at, so a client can record what it talked to. `null`
   * when `backendSupported` is false: a model this host cannot serve has no
   * revision here to name.
   */
  readonly revision: string | null;
  readonly backendSupported: boolean;
  readonly installed: boolean;
  readonly resident: boolean;
  readonly loadable: boolean;
  /**
   * Why it is not loadable, in the server's words. Always present when
   * `loadable` is false — a refusal with no reason is a protocol error — and
   * absent when it is true.
   */
  readonly reason?: string;
  /**
   * Weights plus KV at `contextDefault`, measured on the host, not guessed.
   * `null` when `backendSupported` is false, for the same reason
   * {@link ModelInfo.revision} is: the figure lives in this backend's block,
   * and there is no block. Never `0` — a zero would read as "needs nothing".
   */
  readonly memoryBytesEstimate: number | null;
  readonly contextDefault: number;
}

/** One turn of a chat. `content` is text; this client sends no other part types. */
export interface ChatMessage {
  readonly role: 'system' | 'user' | 'assistant';
  readonly content: string;
}

/** What {@link CrucibleClient.chat} and {@link CrucibleClient.chatStream} take. */
export interface ChatOptions {
  /**
   * The model to talk to. Must be the resident one: the server never loads
   * implicitly, and answers 409 `model_not_resident` naming what is resident
   * instead (PHASE2-LLM.md section 5).
   */
  readonly model: string;
  readonly messages: readonly ChatMessage[];
  readonly temperature?: number;
  readonly topP?: number;
  readonly maxTokens?: number;
  readonly stop?: readonly string[];
  /**
   * Whether a reasoning model thinks before it answers.
   *
   * Qwen3.5 and its kind emit `reasoning` first and `content` after, so a short
   * `maxTokens` spends the whole budget thinking and returns a message with no
   * `content` at all. `false` sends `chat_template_kwargs: {enable_thinking:
   * false}` — read per request by mlx-lm's server and honoured by vLLM under
   * the same name — and `true` sends the same field set to `true`. Omit it and
   * nothing is sent: the model's own default stands.
   *
   * A model whose chat template does not know `enable_thinking` ignores it.
   */
  readonly thinking?: boolean;
  /** Aborts the request. See the README: the abort surfaces as a DOM `AbortError`. */
  readonly signal?: AbortSignal;
}

/** The tokens a completion cost, as the engine counted them. */
export interface ChatUsage {
  readonly promptTokens: number;
  readonly completionTokens: number;
  readonly totalTokens: number;
}

/**
 * One non-streamed completion: OpenAI's `chat.completion` read down to the part
 * a caller actually uses. The first choice is the only one — this client never
 * asks for `n > 1`.
 */
export interface ChatResponse {
  readonly id: string;
  readonly model: string;
  readonly content: string;
  /** `stop`, `length`, ... — the engine's own word for why it stopped. */
  readonly finishReason: string;
  readonly usage: ChatUsage;
}
