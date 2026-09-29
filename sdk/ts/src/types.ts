/** The API contract version this client speaks. */
export const API_VERSION = 1;

/** `GET /v1/ping` — the only unauthenticated route. */
export interface Ping {
  /** Always `true`; anything else is a {@link CrucibleNotACrucible}. */
  readonly crucible: true;
  readonly name: string;
  readonly apiVersion: number;
}

/**
 * One model a job type can serve, as `GET /v1/info` lists it for capabilities other than `llm` and
 * `tts`.
 */
export interface ModelDescriptor {
  readonly id: string;
  readonly revision: string;
  readonly source: string;
  readonly installed: boolean;
  readonly resident: boolean;
  readonly vramBytes: number;
}

/** One job type this server offers, with the models it can serve. */
export type Capability = LlmCapability | TtsCapability | JobCapability | RawCapability;

/** Any capability other than `llm` and `tts`, whose rows are descriptors. */
export interface JobCapability {
  readonly jobType: string;
  readonly models: readonly ModelDescriptor[];
}

export interface RawCapability {
  readonly jobType: string;
  readonly models: readonly Readonly<Record<string, unknown>>[];
  readonly unreadable: string;
}

/**
 * One `llm` or `tts` row inside `info()` that this build could not read, carried aside rather than
 * failing the whole call.
 */
export interface UnreadableRow {
  /** Where the row sat in the capability's `models` array. */
  readonly index: number;
  /** The row's `id`, when it had a string one, for a log to name it. */
  readonly id: string | null;
  /** The row exactly as it arrived. */
  readonly raw: unknown;
  /** Why it could not be read, for a log rather than a branch. */
  readonly unreadable: string;
}

/** The `llm` capability: `GET /v1/models`' rows, carried inside `info()`. */
export interface LlmCapability {
  readonly jobType: 'llm';
  /** Every row that could be read. */
  readonly models: readonly ModelInfo[];
  /** Every row that could not, with its raw data and the reason. */
  readonly unreadableRows: readonly UnreadableRow[];
}

/** The `tts` capability: `GET /v1/voices`' rows, carried inside `info()`. */
export interface TtsCapability {
  readonly jobType: 'tts';
  /** Every row that could be read. */
  readonly models: readonly VoiceInfo[];
  /** Every row that could not, with its raw data and the reason. */
  readonly unreadableRows: readonly UnreadableRow[];
}

/** Narrow a capability to the `llm` one, whose rows are {@link ModelInfo}. */
export function isLlmCapability(capability: Capability): capability is LlmCapability {
  return capability.jobType === 'llm';
}

/** Narrow a capability to the `tts` one, whose rows are {@link VoiceInfo}. */
export function isTtsCapability(capability: Capability): capability is TtsCapability {
  return capability.jobType === 'tts';
}

/** The accelerator the server owns. */
export interface GpuInfo {
  readonly vendor: string;
  readonly name: string;
  readonly vramBytes: number;
}

/** Which half of the orchestrator/engine relation a process is. */
export type CrucibleRole = 'engine' | 'orchestrator';

/** How an orchestrator HOLDS its engine, and therefore what it may do to it. */
export type EngineOwner = 'wsl-unit' | 'child' | 'found';

/** Who manages an engine, from {@link ServerInfo.managedBy}. */
export interface ManagedBy {
  readonly name: string;
  readonly url: string;
}

/** The engine an orchestrator manages, from {@link ServerInfo.engine}. */
export interface EngineRef {
  readonly name: string | null;
  readonly url: string;
  readonly backend: string | null;
  readonly owner: EngineOwner;
}

/** What a page request IS, from {@link PagesEngine.request}. */
export interface PageRequest {
  /** The Crucible model id to send as `model`. */
  readonly model: string;
  /** What the CLIENT rasterises at. */
  readonly dpi: number;
  /** The processor's own pixel limit — the frame the model's boxes are in. */
  readonly maxPixels: number;
  /** The most tokens a page may take; re-read a truncated page at this ceiling. */
  readonly maxTokens: number;
  /** The sampling temperature to send. */
  readonly temperature: number;
  /** The model card's prompt, byte for byte. */
  readonly prompt: string;
  /** What the answer is shaped like, e.g. `dots-json`. */
  readonly dialect: string;
  /** How many page requests a client may have open at once. */
  readonly concurrency: number;
  /** The `finish_reason` that means the model was still writing. */
  readonly truncatedFinishReason: string;
}

/**
 * `GET /v1/info`'s `pages_engine` — which engine reads a page HERE, and what a page request is
 * anywhere.
 */
export interface PagesEngine {
  /** `vllm`, `llama-server`, `mlx-vlm` — or `null`, meaning this host serves none. */
  readonly engine: string | null;
  readonly installed: boolean;
  /** Why, in words, for an operator. */
  readonly detail: string;
  /** Published whether or not this host can answer one. */
  readonly request: PageRequest;
}

/** `GET /v1/info`. */
export interface ServerInfo {
  readonly server: {
    readonly name: string;
    /** The release, for a person to read. */
    readonly version: string;
    readonly apiVersion: number;
  };
  /** The machine, described for a person. */
  readonly host: {
    readonly platform: string;
    readonly arch: string;
    /** `cuda-linux` or `mlx-darwin`. */
    readonly backend: string;
    readonly gpu: GpuInfo;
  };
  /** What this server will accept as a `type` in `POST /v1/jobs`. */
  readonly jobTypes: readonly string[];
  /** One entry per capability, not one per postable job type. */
  readonly capabilities: readonly Capability[];
  /** Which half of the relation answered. */
  readonly role: CrucibleRole;
  /** On an ENGINE: which orchestrator claimed it, or `null`. */
  readonly managedBy: ManagedBy | null;
  /** On an orchestrator, the one engine it manages, or `null`; read it with {@link engineOf}. */
  readonly engine: EngineRef | null;
  /** What a page request is on this server, or `null` from an orchestrator. */
  readonly pagesEngine: PagesEngine | null;
}

/**
 * What this server asked to stop and has not been told is gone; read it before believing an idle
 * server is free.
 */
export interface Stopping {
  /** `llm`, `tts`, … — what sort of thing was on the card. */
  readonly kind: string;
  /** The model or voice id that was resident. */
  readonly id: string;
  /** When the stop was asked for, in {@link ActivityJob.started}'s format. */
  readonly since: string;
  /** The pids still holding the card, ascending. */
  readonly pids: readonly number[];
}

/** `GET /v1/health`. */
export interface Health {
  readonly status: string;
  readonly queueDepth: number;
  /** The ids of whatever is on the card. */
  readonly residentModels: readonly string[];
  /** What sort of thing holds the card (`llm`, `tts`, …), or `null` when nothing does. */
  readonly residentKind: string | null;
  /** What was told to go and has not, or `null`. */
  readonly stopping: Stopping | null;
}

/** One job as a bench reads it. */
export interface ActivityJob {
  readonly jobId: string;
  readonly type: string;
  readonly model: string | null;
  readonly status: string;
  readonly position: number | null;
  /** Fraction done, 0..1. */
  readonly progress: number;
  readonly message: string | null;
  readonly created: string;
  readonly started: string | null;
  /** The submitting User-Agent. */
  readonly client: string | null;
}

/** An open TTS streaming session, as a bench reads it. */
export interface ActivityStreaming {
  readonly sessionId: string;
  readonly voice: string;
  readonly language: string;
  readonly narratorEngine: string;
  /** When the session opened, in {@link ActivityJob.started}'s format. */
  readonly since: string;
  /** Who opened it. */
  readonly client: string | null;
  /** Always null: a session has no denominator. */
  readonly progress: null;
  /** Rows this session has been asked to say, ever. */
  readonly said: number;
  readonly finished: number;
  readonly inFlight: number;
  /** Seconds of audio delivered, measured from the bytes. */
  readonly seconds: number;
  readonly chars: number;
}

/** One chat completion, while it is happening. */
export interface ActivityChat {
  readonly id: number;
  /**
   * What this completion is — a capability class from `X-Crucible-Act` — or null when the client
   * did not say.
   */
  readonly act: string | null;
  readonly model: string;
  /** The calling User-Agent. */
  readonly client: string | null;
  readonly since: string;
}

/** A client's declared intention to keep using what is resident; a refusal, not a reservation. */
export interface Lease {
  readonly leaseId: string;
  /** Which resident kind it holds: `llm`, `tts` or `align`. */
  readonly kind: string;
  /** The id it is held on — a model, a voice or an aligner — which is always the resident one. */
  readonly subject: string;
  /** The holder's User-Agent, as `/v1/activity` reports it. */
  readonly client: string | null;
  /** What the run IS: a capability class name. */
  readonly act: string;
  /** When it was taken, in {@link ActivityJob.started}'s format. */
  readonly since: string;
  /** When it stops being open unless something heartbeats it. */
  readonly expiresAt: string;
}

/** The open lease, as `/v1/activity` reports it. */
export type ActivityLease = Omit<Lease, 'subject'>;

/** `GET /v1/activity` — what is on this server and how far along. */
export interface Activity {
  readonly server: {
    readonly name: string;
    readonly version: string;
    readonly apiVersion: number;
    readonly backend: string;
    readonly uptimeS: number;
  };
  readonly resident: {
    readonly kind: string;
    readonly id: string;
    readonly since: string;
    readonly memoryBytesEstimate: number;
    /** What holds this resident thing, or `null` — which is a stranded card, not an idle one. */
    readonly heldBy: {
      readonly fact: string;
      readonly who: string;
      readonly details: Record<string, unknown>;
    } | null;
    /** Since when nothing has held it, or `null` because something does. */
    readonly unclaimedSince: string | null;
    /** The code the resident model's engine EXITED with, or `null` while it runs. */
    readonly engineExitCode: number | null;
  } | null;
  /** What was told to go and has not, or `null`. */
  readonly stopping: Stopping | null;
  /** The id of a model being loaded right now, or null. */
  readonly warming: string | null;
  /** Who holds narrator's wire, or null. */
  readonly claim: { readonly heldBy: string } | null;
  /** The open streaming session, or null. */
  readonly streaming: ActivityStreaming | null;
  /** Chat completions open right now, and what this engine will admit at once. */
  readonly chat: {
    readonly inFlight: number;
    /** What this engine's chat door admits at once, or `null` (never "unlimited"). */
    readonly maxInFlight: number | null;
    /** Where `maxInFlight` came from, in a sentence. */
    readonly maxInFlightBasis: string | null;
    readonly rows: readonly ActivityChat[];
  };
  /** The open lease on whatever is resident, or null. */
  readonly lease: ActivityLease | null;
  readonly slots: { readonly accelerated: ActivitySlot };
  readonly running: readonly ActivityJob[];
  readonly queued: readonly ActivityJob[];
}

export interface ActivitySlot {
  /** Jobs on the lane; a stream does not take it. */
  readonly busy: number;
  readonly of: number;
  readonly queueDepth: number;
  /** The lane is free and nobody holds the card; still not a reservation. */
  readonly acceptsWork: boolean;
}

/** `POST /v1/uploads`. */
export interface UploadResult {
  readonly blobId: string;
  readonly bytes: number;
  readonly sha256: string;
}

/**
 * One named input on a job: an uploaded blob, bytes carried inline, or an artifact of a previous
 * job on the SAME server ({@link CrucibleClient.artifactRef}).
 */
export type JobInput =
  | { readonly blobId: string }
  | { readonly inline: Uint8Array }
  | { readonly artifact: { readonly jobId: string; readonly name: string } };

/** `POST /v1/jobs/{id}/hold`'s answer: a done job's artifacts kept for a chain. */
export interface ArtifactHold {
  readonly jobId: string;
  /** The job's status now: a hold may be taken at any status. */
  readonly status: string;
  readonly held: boolean;
  readonly heldBy: string | null;
  readonly heldSince: string;
  /** When the collector takes this job whatever holds it (ISO-8601), or null until the job ends. */
  readonly gcAt: string | null;
  /** Every artifact a later job may name with {@link CrucibleClient.artifactRef}. */
  readonly artifacts: readonly string[];
}

/** The body of `POST /v1/jobs`. */
export interface JobRequest {
  readonly type: string;
  /** Omit for a job type that serves no models; the server refuses a mismatch. */
  readonly model?: string;
  readonly params: Readonly<Record<string, unknown>>;
  readonly inputs: Readonly<Record<string, JobInput>>;
  /** Your own name for this work. */
  readonly clientRef?: string;
  /**
   * Hold this job's artifacts from birth, for a later job to take by {@link
   * CrucibleClient.artifactRef}.
   */
  readonly hold?: boolean;
}

/**
 * A job's lifecycle state; `interrupted` means the server stopped mid-job, not that the work
 * failed.
 */
export type JobState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted';

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
  /** The lease this job opened, `null` if it opened none. */
  readonly leaseId: string | null;
  /** The client's own name for this work, echoed back. */
  readonly clientRef: string | null;
  /** When the server was found to have stopped while this job was running. */
  readonly interruptedAt: string | null;
  /** The client holding this job's artifacts for a chain, or null. */
  readonly heldBy: string | null;
  /** When the hold was taken (ISO-8601), or null when nothing holds it. */
  readonly heldSince: string | null;
  /** The chunk index of every artifact this job published, ascending. */
  readonly chunksDone: readonly number[];
  /**
   * How many indexed chunks the job was asked for, or `null` for a job whose artifacts are not
   * chunks.
   */
  readonly chunksTotal: number | null;
  /** When the last chunk artifact landed (ISO-8601 UTC); `null` before any did. */
  readonly chunkAt: string | null;
  /**
   * The resume journal this job writes, to send as `params.resume` if it does not finish; `null`
   * for a job type that keeps none.
   */
  readonly resumeId: string | null;
  /** True when this job was itself a resume of an earlier job's journal. */
  readonly resumed: boolean;
}

/** One journal, as `GET /v1/resumable` lists it; reading it resumes nothing. */
export interface Resumable {
  /** Send this as `params.resume` to continue the work. */
  readonly resumeId: string;
  readonly jobType: string;
  /** The model and the exact revision the journal was written with. */
  readonly model: { readonly id: string | null; readonly revision: string | null };
  /** The job type's unit format; a resume under another is refused. */
  readonly formatVersion: number;
  /** Each input by name, sha256 and size: a resume must send the same bytes. */
  readonly inputs: readonly { readonly name: string; readonly sha256: string; readonly bytes: number }[];
  /** The output-affecting params it was written under, resolved. */
  readonly params: Readonly<Record<string, unknown>>;
  readonly unitsDone: number;
  readonly unitsTotal: number | null;
  /** A sentence a person reads: "2,400 of 3,015 pieces done (...)". */
  readonly progress: string;
  readonly created: string;
  readonly lastSaved: string;
  /** When the server's `retention_days` collector takes it (ISO-8601). */
  readonly expiresAt: string;
  /** The job that started the journal. */
  readonly jobId: string;
  /** The job that last wrote it, and how that ended. */
  readonly lastJobId: string;
  readonly state: 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted' | string;
}

/** `DELETE /v1/resumable/{id}`'s receipt. */
export interface ResumableDiscarded {
  readonly resumeId: string;
  readonly discarded: boolean;
  readonly unitsDone: number;
  readonly unitsTotal: number | null;
}

/** `DELETE /v1/jobs/{id}`. */
export interface CancelResult {
  readonly jobId: string;
  readonly status: 'cancelled' | 'cancelling';
}

/** One SSE event from `GET /v1/jobs/{id}/events`. */
export type JobEvent =
  | { readonly id: number; readonly event: 'queued'; readonly data: QueuedData }
  | { readonly id: number; readonly event: 'warming'; readonly data: WarmingData }
  | { readonly id: number; readonly event: 'progress'; readonly data: ProgressData }
  | { readonly id: number; readonly event: 'chunk'; readonly data: ChunkData }
  | { readonly id: number; readonly event: 'artifact'; readonly data: ArtifactData }
  | { readonly id: number; readonly event: 'done'; readonly data: DoneData }
  | { readonly id: number; readonly event: 'failed'; readonly data: FailedData }
  | { readonly id: number; readonly event: 'cancelled'; readonly data: CancelledData }
  | UnknownEvent;

export interface UnknownEvent {
  readonly id: number;
  readonly event: 'unknown';
  readonly kind: string;
  readonly data: Readonly<Record<string, unknown>>;
}

export interface QueuedData {
  readonly position: number;
}

/**
 * A model is being loaded for this job: one line of the engine's own readiness, streamed as it
 * happens.
 */
export interface WarmingData {
  readonly message: string;
}

export interface ProgressData {
  readonly fraction: number;
  readonly message: string;
  /** Every other key the job type put on this frame, verbatim. */
  readonly extra: Readonly<Record<string, unknown>>;
}

/** One rendered chunk of a `tts` job: the server's measurements and the engine's verdict. */
export interface ChunkData {
  /** The client's own chunk index, and the name of its artifact (`<index>.flac`). */
  readonly index: number;
  /** The duration of the audio that arrived, measured from its bytes. */
  readonly seconds: number;
  /** How many characters were sent, counted by the server. */
  readonly chars: number;
  /** `chars / seconds`. */
  readonly charsPerSec: number;
  /** How many tokens the engine spent, or null when narrator did not say. */
  readonly tokens: number | null;
  /**
   * Whether generation hit the frame cap rather than finishing; null means narrator did not say,
   * never false.
   */
  readonly capped: boolean | null;
  /** Which rung of the voice's take ladder this render asked for. */
  readonly take: number;
  /**
   * The verdict the engine's own retake ladder reached about this chunk, or null when narrator sent
   * none.
   */
  readonly guard: Readonly<Record<string, unknown>> | null;
}

export interface ArtifactData {
  readonly name: string;
}

/** The terminal `done` event's payload. */
export interface DoneData {
  /** What a producing job wrote. */
  readonly artifacts?: readonly string[];
  /** What is resident now after a `load-model` or `unload-model` job, or `null` after an unload. */
  readonly resident?: string | null;
  /** Every other key the job type put on its `done` frame, verbatim. */
  readonly extra: Readonly<Record<string, unknown>>;
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
 * `<artifact>.provenance.json`, verbatim as the server wrote it; persist it beside the artifact.
 */
export interface Provenance {
  readonly server: { readonly name: string; readonly version: string };
  readonly backend: string;
  readonly job_type: string;
  readonly model: {
    readonly id: string;
    readonly revision: string | null;
    /** `<id>@<revision>`: which weights, not merely which model. */
    readonly fingerprint: string | null;
  } | null;
  readonly params: Readonly<Record<string, unknown>>;
  readonly started: string | null;
  readonly finished: string;
}

/** One model this server knows about, as `GET /v1/models` describes it. */
export interface ModelInfo {
  /** Crucible's id, stable across backends, e.g. `qwen3.5-9b`. */
  readonly id: string;
  readonly family: string;
  readonly paramsB: number;
  /**
   * The commit the manifest pins for this host's backend, or null when `backendSupported` is false.
   */
  readonly revision: string | null;
  /** `<id>@<revision>` — what to write down when you record what you talked to. */
  readonly fingerprint: string | null;
  /** What a chat's content parts may carry (`text`, `image`); never null. */
  readonly modalities: readonly string[];
  readonly backendSupported: boolean;
  readonly installed: boolean;
  /** The model whose download this one's weights are, or null for a model that owns its own. */
  readonly weightsOf: string | null;
  readonly resident: boolean;
  readonly loadable: boolean;
  /**
   * Why it is not loadable, in the server's words, or null on a loadable model, whose row carries
   * no `reason`.
   */
  readonly reason: string | null;
  /** Weights plus KV at `contextDefault`, measured on the host, not guessed. */
  readonly memoryBytesEstimate: number | null;
  /** The manifest's intent: the context this host would serve this model at. */
  readonly contextDefault: number;
  /** The context in force right now; size requests against this. */
  readonly maxModelLen: number | null;
}

/** OpenAI's structured-output request, passed through to the engine verbatim. */
export type ResponseFormat =
  | { readonly type: 'text' }
  | { readonly type: 'json_object' }
  | {
      readonly type: 'json_schema';
      readonly json_schema: {
        readonly name: string;
        readonly schema: Readonly<Record<string, unknown>>;
        /** Whether the engine must follow the schema exactly. */
        readonly strict?: boolean;
        readonly description?: string;
      };
    };

/** One turn of a chat. */
export interface ChatMessage {
  readonly role: 'system' | 'user' | 'assistant';
  readonly content: string;
}

/** What {@link CrucibleClient.chat} and {@link CrucibleClient.chatStream} take. */
export interface ChatOptions {
  /** The model to talk to. */
  readonly model: string;
  readonly messages: readonly ChatMessage[];
  readonly temperature?: number;
  readonly topP?: number;
  readonly maxTokens?: number;
  readonly stop?: readonly string[];
  /** The engine's sampling seed. */
  readonly seed?: number;
  /** OpenAI's `response_format`, forwarded to the engine exactly as given. */
  readonly responseFormat?: ResponseFormat;
  /** Whether a reasoning model thinks before it answers. */
  readonly thinking?: boolean;
  /** The context window an `ollama/<tag>` chat runs at, sent as `context_tokens`. */
  readonly contextTokens?: number;
  /**
   * What this chat is, as a capability class sent in the `X-Crucible-Act` header; omitted, no
   * header is sent.
   */
  readonly act?: string;
  /** Aborts the request. */
  readonly signal?: AbortSignal;
}

/** The tokens a completion cost, as the engine counted them. */
export interface ChatUsage {
  readonly promptTokens: number | null;
  readonly completionTokens: number | null;
  readonly totalTokens: number | null;
}

/**
 * One non-streamed completion: OpenAI's `chat.completion` read down to the part a caller actually
 * uses.
 */
export interface ChatResponse {
  readonly id: string | null;
  /** Crucible's model id — the one you asked for. */
  readonly model: string | null;
  readonly content: string;
  /** The engine's own word for why it stopped (`stop`, `length`, …); `length` means truncated. */
  readonly finishReason: string;
  readonly usage: ChatUsage | null;
}

/** Pick one of 2–26 named options. */
export interface DecideChoiceQuestion {
  readonly type: 'choice';
  readonly instructions: string;
  /** Option name → the description the model reads. */
  readonly options: Readonly<Record<string, string>>;
}

/**
 * Place the state on 2–10 unique, ORDERED levels; the answer's `score` is the expected 1-based
 * level.
 */
export interface DecideScoreQuestion {
  readonly type: 'score';
  readonly instructions: string;
  readonly levels: readonly string[];
}

/** A statement the state does or does not make true; the answer is P(Yes). */
export interface DecideYesNoQuestion {
  readonly type: 'yesno';
  readonly instructions: string;
}

export type DecideQuestion = DecideChoiceQuestion | DecideScoreQuestion | DecideYesNoQuestion;

export interface DecideRequest {
  /** The model to read the decision from. */
  readonly model: string;
  /** What the questions are about: a string, or any JSON value. */
  readonly state: unknown;
  /** Image FILES, base64-encoded (at most 8; more is 400 `too_many_images`). */
  readonly images?: readonly string[];
  /** Question name → question. */
  readonly questions: Readonly<Record<string, DecideQuestion>>;
  /** What to do when a label is not among the top tokens the engine returned. */
  readonly missing?: DecideMissing;
}

/** `DecideRequest.missing`: refuse a decision with a label outside the top-K, or report it. */
export type DecideMissing = 'refuse' | 'report';

export interface DecideOptions {
  /** What this decision is, sent as `X-Crucible-Act` exactly as {@link ChatOptions.act}. */
  readonly act?: string;
  /** Aborts the request; the abort surfaces as a DOM `AbortError`. */
  readonly signal?: AbortSignal;
}

/** What every answer carries past its distribution. */
export interface DecideAnswerCommon {
  readonly labelMass: number;
  readonly missingLabels?: readonly string[];
}

/** The answer to a `choice`. */
export interface DecideChoiceAnswer extends DecideAnswerCommon {
  readonly type: 'choice';
  readonly choice: string;
  readonly probabilities: Readonly<Record<string, number | null>>;
  readonly logprobs: Readonly<Record<string, number | null>>;
  readonly confidence: number;
}

/**
 * The answer to a `score`: `score` = Σ(1-based level index × p) over the levels returned, `level`
 * the likeliest.
 */
export interface DecideScoreAnswer extends DecideAnswerCommon {
  readonly type: 'score';
  readonly score: number;
  readonly level: string;
  readonly probabilities: Readonly<Record<string, number | null>>;
  readonly logprobs: Readonly<Record<string, number | null>>;
  readonly confidence: number;
}

/**
 * The answer to a `yesno`: `p` is the renormalised P(Yes), `logprob` ln of it (not calibrated;
 * `null` when `p` is exactly 0).
 */
export interface DecideYesNoAnswer extends DecideAnswerCommon {
  readonly type: 'yesno';
  readonly p: number;
  readonly logprob: number | null;
}

export type DecideAnswer = DecideChoiceAnswer | DecideScoreAnswer | DecideYesNoAnswer;

/** One completion the door sent the engine, timed by Crucible's wall clock. */
export interface DecideCallTiming {
  readonly wallMs: number;
  /** `usage.prompt_tokens`, as the engine counted it. */
  readonly promptTokens: number;
  /** `usage.prompt_tokens_details.cached_tokens`, or null (never 0) when the engine did not say. */
  readonly cachedTokens: number | null;
}

export interface DecideTiming {
  readonly total: number;
  readonly perQuestion: Readonly<Record<string, DecideCallTiming>>;
  /**
   * The shared-prefix prime, sent only when there is more than one question; `null` when none was.
   */
  readonly prime: DecideCallTiming | null;
}

export interface DecideResponse {
  /** Which weights made this decision. */
  readonly model: {
    readonly id: string;
    /** The revision the resident engine was started on. */
    readonly revision: string;
    /** `<id>@<revision>`. */
    readonly fingerprint: string;
  };
  /** The engine kind that answered (`vllm`, `llama-server`, …). */
  readonly engine: string;
  /** One answer per question asked, keyed by the question's name. */
  readonly answers: Readonly<Record<string, DecideAnswer>>;
  /** How long it took. */
  readonly timingMs: DecideTiming;
  /** What it cost in tokens. */
  readonly tokens: {
    readonly perQuestion: Readonly<Record<string, number>>;
    readonly images: number;
  };
}

/** One item of {@link DecideItemsRequest}: its text, and optionally its own options. */
export interface DecideItem {
  readonly text: string;
  readonly options?: Readonly<Record<string, string>>;
}

/** The items form of `POST /v1/decide`: choice questions about one state, answered in item order. */
export interface DecideItemsRequest {
  readonly model: string;
  readonly state: unknown;
  readonly images?: readonly string[];
  readonly instructions?: string;
  readonly options?: Readonly<Record<string, string>>;
  readonly items: readonly DecideItem[];
  readonly missing?: DecideMissing;
}

/** The items form's reply: one choice answer per item, in item order. */
export interface DecideItemsResponse {
  readonly model: DecideResponse['model'];
  readonly engine: string;
  readonly answers: readonly DecideChoiceAnswer[];
  readonly timingMs: { readonly total: number; readonly engineRequests: number };
  readonly tokens: {
    readonly shared: number | null;
    readonly perItem: readonly number[];
    readonly images: number;
  };
}

/** The band a client packs its chunks to, as the voice's manifest declares it. */
export interface VoicePace {
  /**
   * The measured pace this voice reads at; with the two edges it is all three or none, and null
   * means unmeasured.
   */
  readonly paceCharsPerSec: number | null;
  readonly maxCharsPerSec: number | null;
  readonly minCharsPerSec: number | null;
  readonly targetChars: number | null;
  readonly safeMinChars: number | null;
  readonly safeMaxChars: number | null;
}

/** How a voice is conditioned. */
export type VoiceKind = 'checkpoint' | 'zeroshot' | 'token';

/** Where a voice's `memoryBytesEstimate` came from: `measured` or `declared`. */
export type EstimateBasis = 'measured' | 'declared';

/** One voice this server knows about, as `GET /v1/voices` describes it. */
export interface VoiceInfo {
  /** Crucible's voice id, stable across backends, e.g. `deathstalker`. */
  readonly id: string;
  /** The name to put in front of a person. */
  readonly display: string;
  /**
   * The voice's kind (`checkpoint`, `zeroshot`, `token`) as the server's own word, or null on a
   * pinned voice this host cannot read.
   */
  readonly kind: string | null;
  /** The manifest's language tag, e.g. `en`. */
  readonly language: string | null;
  /** Which of narrator's engines serves this voice, e.g. `higgs-v3`. */
  readonly narratorEngine: string | null;
  readonly backendSupported: boolean;
  readonly installed: boolean;
  readonly resident: boolean;
  /** A local (`path`) voice that nothing holds: not resident, not leased, not named by a job. */
  readonly orphan: boolean;
  readonly loadable: boolean;
  /** Why it is not loadable, in the server's words; `null` when it is loadable. */
  readonly reason: string | null;
  /** The commit this host's backend block pins, or null when `backendSupported` is false. */
  readonly revision: string | null;
  /** `<id>@<revision>`, joined by the server. */
  readonly fingerprint: string | null;
  /** Estimated memory to load it, or null when `backendSupported` is false. */
  readonly memoryBytesEstimate: number | null;
  /** {@link EstimateBasis} today, as the server's own word. */
  readonly estimateBasis: string | null;
  /**
   * The most characters this voice may be handed in one chunk on this backend, or null when
   * `backendSupported` is false.
   */
  readonly maxChars: number | null;
  /**
   * The sample rate of the audio this voice produces, or null on a pinned voice this host cannot
   * read.
   */
  readonly sampleRate: number | null;
  /** How many rungs this voice's take ladder has; `0` on a pinned voice this host cannot read. */
  readonly takes: number;
  /**
   * What the server under narrator is sized by, or `null` for a voice that declares no serving
   * table.
   */
  readonly serving: VoiceServing | null;
  /** Whether loading this voice requires a reference clip ({@link LoadVoiceOptions.reference}). */
  readonly needsReference: boolean;
  /** The band a client packs its chunks to, or null on a pinned voice this host cannot read. */
  readonly pace: VoicePace | null;
  /** The tag this voice follows on its repo (e.g. `crucible`), or null for an exact-sha pin. */
  readonly ref: string | null;
  /** The commit the tag named at the last explicit check, or null when never checked. */
  readonly latestRevision: string | null;
  /** Whether `crucible voices pull <id>` (or a pull task) would move this voice to `latestRevision`. */
  readonly updateAvailable: boolean;
  /** When the tag was last looked up. */
  readonly updateCheckedAt: string | null;
  /** Why the last look-up failed; the voice stays on the revision it has. */
  readonly updateError: string | null;
  /** The sampling this backend's arm renders take 0 with, or null when `backendSupported` is false. */
  readonly sampling: VoiceSampling | null;
  /** The fades at each chunk edge on this arm, or null when the manifest states none. */
  readonly edgeFadeMs: VoiceEdgeFade | null;
  /** The silence a client adds after each chunk, or null when the manifest states none. */
  readonly chunkGap: VoiceChunkGap | null;
  /** The most reference-clip audio this arm takes, in seconds, or null when unstated. */
  readonly referenceSecondsCap: number | null;
  /** The inline control tokens this arm allows (`[]` allows none), or null when unstated. */
  readonly allowedControls: readonly string[] | null;
}

/** The sampling a voice's arm renders take 0 with. */
export interface VoiceSampling {
  readonly temperature: number;
  readonly topP: number;
  readonly topK: number;
}

/** Raised-cosine fades applied at each chunk edge, in milliseconds. */
export interface VoiceEdgeFade {
  readonly in: number;
  readonly out: number;
}

/** The sentence gap after each chunk: `injectS` is added net of the model's own tail. */
export interface VoiceChunkGap {
  readonly injectS: number;
  readonly targetJoinS: number;
  readonly modelSelfTailS: number;
  readonly readerSentenceGapS: number | null;
  readonly modelInternalGapS: number | null;
  readonly rule: string;
  readonly method: string;
  readonly source: string;
  readonly measuredOn: string;
}

/** The recording a zero-shot voice is cloned from, sent with {@link CrucibleClient.loadVoice}. */
export interface VoiceReference {
  /**
   * The wav's bytes, base64, with no `data:` prefix and no whitespace; at most 30 seconds of audio.
   */
  readonly data: string;
  /** The book-exact text spoken in the clip, never an ASR guess. */
  readonly transcript: string;
  /** A short label `/v1/activity` shows for which clip is resident. */
  readonly name?: string;
}

/** What {@link CrucibleClient.loadVoice} takes beyond the voice id. */
export interface LoadVoiceOptions {
  /**
   * Required when the voice's row says {@link VoiceInfo.needsReference}; refused on any other kind.
   */
  readonly reference?: VoiceReference;
  /** Hold the voice from the instant it is resident. */
  readonly lease?: LeaseOnLoad;
}

/** Hold what a load makes resident, from the instant it exists. */
export interface LeaseOnLoad {
  /** What the run is for: one of the capability classes. */
  readonly act: string;
  /** How long the lease outlives silence, in seconds (30-3600); each heartbeat extends it. */
  readonly ttlSeconds: number;
}

/** Options for {@link CrucibleClient.loadModel}. */
export interface LoadModelOptions {
  /** Hold the model from the instant it is resident. */
  readonly lease?: LeaseOnLoad;
  /** Tokens: the context to start the engine with (`params.context`). */
  readonly context?: number;
}

/** `[voice.serving]` — what the server under narrator is sized by. */
export interface VoiceServing {
  readonly maxNumSeqs: number;
  readonly maxNumSeqsNote: string;
  readonly memFraction: number | null;
  readonly memFractionNote: string | null;
  readonly contextLength: number | null;
  readonly contextLengthNote: string | null;
}

export interface RenderChunk {
  /** The client's number for this chunk, and the name of its artifact (`<index>.flac`). */
  readonly index: number;
  readonly text: string;
}

/** What {@link CrucibleClient.render} takes; nothing has a default. */
export interface RenderOptions {
  /**
   * Hold this job's artifacts from birth, for a later job to take by {@link
   * CrucibleClient.artifactRef}.
   */
  readonly hold?: boolean;
  /** The voice id; unlike a chat's model it need not already be resident. */
  readonly voice: string;
  /** The manifest's language tag for this text, e.g. `en`. */
  readonly language: string;
  /** Which rung of the voice's take ladder, and which seed lane, to render at. */
  readonly take: number;
  /** At least one. */
  readonly chunks: readonly RenderChunk[];
  /** `true` renders on narrator's guarded arm (needs `band`); absent or `false` is the bare arm. */
  readonly retake?: boolean;
  /** The pace band the guarded arm measures against, in the voice manifest's spelling. */
  readonly band?: {
    readonly pace_chars_per_sec: number;
    readonly max_chars_per_sec: number;
    readonly min_chars_per_sec: number;
  };
  /**
   * How many of this job's chunks may be in flight at once; absent keeps the width the engine was
   * started at.
   */
  readonly width?: number;
  /** Aborts the submit itself, not a job that was already queued. */
  readonly signal?: AbortSignal;
}

/** One chunk of a render that produced no audio, as `done` names it. */
export interface RenderFailure {
  readonly index: number;
  /** narrator's own words: `No audio generated`, `cancelled`, an exception. */
  readonly message: string;
}

/** A finished `tts` job's terminal news, read by {@link readRenderResult}. */
export interface RenderResult {
  /** How many chunks produced audio. */
  readonly rendered: number;
  /** Every chunk that did not, and why. */
  readonly failed: readonly RenderFailure[];
  /** The rung this render actually ran at. */
  readonly take: number;
  /**
   * The full sampling the engine applied: the voice's take-0 numbers with this take's rung laid
   * over them.
   */
  readonly sampling: Readonly<Record<string, number>>;
  /** Which weights ran, in the `/v1/voices` row's own words. */
  readonly voice: {
    readonly id: string;
    readonly identity: string;
    readonly identityBasis: string;
  };
  /** The width the request stated, or `null` when it stated none. */
  readonly width: number | null;
  /** The rate the voice was loaded at, and the rate every FLAC was written at. */
  readonly sampleRate: number;
  /** The artifacts the job published — one `<index>.flac` per rendered chunk. */
  readonly artifacts: readonly string[];
}

/**
 * One artifact {@link CrucibleClient.writeArtifactsTo} has finished writing, with its provenance
 * sidecar already on disk beside it.
 */
export interface WrittenArtifact {
  /** The artifact's name on the server, e.g. `41.flac`. */
  readonly name: string;
  /** Where it was written, e.g. `Z:\books\the-mutineer\41.flac`. */
  readonly path: string;
  readonly bytes: number;
  /** Where its sidecar was written: `<path>.provenance.json`. */
  readonly provenancePath: string;
  /** The sidecar, parsed. */
  readonly provenance: Provenance;
}

/**
 * What {@link CrucibleClient.writeArtifactsTo} yields: the job's own events, unchanged, interleaved
 * with the files it has written.
 */
export type ArtifactWrite =
  | { readonly kind: 'event'; readonly event: JobEvent }
  | { readonly kind: 'written'; readonly written: WrittenArtifact };

/** One process the driver says is holding accelerator memory. */
export interface AcceleratorHolder {
  readonly pid: number;
  readonly name: string;
  /** What this process holds, or null (never zero) where the driver will not say. */
  readonly bytes: number | null;
  /** Whether this pid is one of Crucible's own engine processes. */
  readonly ownedByCrucible: boolean;
}

/** What Crucible itself has on the card, from {@link AcceleratorState}. */
export interface AcceleratorResident {
  /** The family of the resident thing (`llm`, `tts`, …), not the job type that loaded it. */
  readonly kind: string;
  readonly id: string;
  /** When it was loaded, ISO-8601. */
  readonly since: string;
  readonly memoryBytesEstimate: number;
}

/** The accelerator, at the moment it was probed. */
export interface AcceleratorGpu {
  readonly vendor: string;
  readonly name: string;
  /** The live total from the probe, not the figure detection recorded at start-up. */
  readonly totalBytes: number;
}

/** `GET /v1/accelerator` — what is on the card right now, and which of it is Crucible's. */
export interface AcceleratorState {
  readonly backend: string;
  readonly gpu: AcceleratorGpu;
  readonly freeBytes: number;
  readonly usedBytes: number;
  /** What this server holds back for the desktop, from its own config. */
  readonly desktopAllowanceBytes: number;
  /**
   * VRAM in use that no listed holder accounts for, past the desktop allowance; null on mlx-darwin.
   */
  readonly unattributedBytes: number | null;
  /** What Crucible has loaded, or `null` when it has nothing loaded. */
  readonly resident: AcceleratorResident | null;
  /** Every compute process the driver listed. */
  readonly holders: readonly AcceleratorHolder[];
  /** The probe's own one-line summary, for a log. */
  readonly detail: string;
}

/** One capability class's verdict on this host; `enabled: false` is an answer, not an error. */
export interface CapabilityRow {
  /** The class name a client asks by: `clean`, `translate`, `tts`, `asr`, … */
  readonly capability: string;
  readonly enabled: boolean;
  /** The candidate that won, or `''` when none did. */
  readonly selected: string;
  /** Why, in the server's words, whichever way it went. */
  readonly reason: string;
  /** How much more memory the SMALLEST candidate would have needed, or 0. */
  readonly shortfallBytes: number;
  /** Where this class's work runs: `local` or `upstream`. */
  readonly route: 'local' | 'upstream';
  /**
   * The working context this row's fit was computed for, or `null` for a class that is not
   * token-shaped (a voice, an aligner).
   */
  readonly work: CapabilityWork | null;
  /**
   * Every candidate's context ceiling on this host for a client-sized class, or `null` on every
   * other.
   */
  readonly contextCeilings: readonly ContextCeiling[] | null;
}

/** One row's working context, and where it came from. */
export interface CapabilityWork {
  readonly tokens: number;
  readonly concurrency: number;
  /** Prose: whose number this is. */
  readonly source: string;
  readonly from: string;
}

/** The longest request one candidate can serve on this host. */
export interface ContextCeiling {
  readonly model: string;
  readonly tokens: number;
  readonly boundBy: string;
  readonly servedContext: number;
  /** `null` where the model's block is not taken apart into memory terms. */
  readonly memoryContext: number | null;
  readonly concurrency: number;
}

/** `capability()`'s sizing, for a CLIENT-SIZED class only (`generate`). */
export interface CapabilitySizing {
  /** The class to size, e.g. `'generate'`. */
  readonly class: string;
  /** Tokens per request. */
  readonly contextTokens?: number;
  /** Requests in flight at once. */
  readonly concurrency?: number;
}

/** `GET /v1/capability` — what this server can hold, per class, and why not. */
export interface CapabilityRecord {
  /** `cuda-linux` or `mlx-darwin`: the backend the decision was made for. */
  readonly backendKind: string;
  /** The pool the decision was made on — the card, or unified memory on a Mac. */
  readonly totalBytes: number;
  /** The host's own reserve, subtracted before any candidate was measured. */
  readonly desktopAllowanceBytes: number;
  /** One verdict per class, in the server's report order. */
  readonly classes: readonly CapabilityRow[];
}

/** What {@link CrucibleClient.asr} takes. */
export interface AsrOptions {
  /** Which transcriber; there is no default. */
  readonly model: string;
  /** The audio, as an uploaded blob or bytes carried inline. */
  readonly audio: JobInput;
  /** What to call that file on the server. */
  readonly filename: string;
  /** A language code (`en`, `de`, …), or `"auto"` to detect it. */
  readonly language: string;
  /** Whether whisper's voice-activity filter runs. */
  readonly vadFilter: boolean;
  /** Whether whisper emits per-word timestamps. */
  readonly wordTimestamps: boolean;
  /** Text a whisper model is primed with, such as the spelling of names; `null` for none. */
  readonly initialPrompt?: string | null;
  /**
   * Qwen3-ASR's system-turn context, an instruction and vocabulary read before every piece of
   * audio; `null` for none.
   */
  readonly context?: string | null;
  /**
   * Remove long stretches without speech before the model hears them; times stay on the original
   * timeline.
   */
  readonly speechOnly?: boolean;
  /**
   * With {@link speechOnly}: the detector score, 0.1 to 0.7, at which a 32 ms frame counts as
   * speech.
   */
  readonly speechThreshold?: number | null;
  /**
   * With {@link speechOnly}: seconds of audio kept either side of every stretch of speech, 0.1 to
   * 2.
   */
  readonly speechPadS?: number | null;
  /**
   * With {@link speechOnly}: the shortest stretch without speech that is ever taken out, 1 to 60
   * seconds; anything shorter stays in as context.
   */
  readonly speechMinGapS?: number | null;
  /** The resume id of a journal to continue instead of starting fresh (Qwen3-ASR only). */
  readonly resume?: string | null;
}

/** One window to align: its audio and exactly the text spoken in it. */
export interface AlignWindow {
  /** The window's key. */
  readonly index: number;
  /**
   * The words spoken in this window, as the caller holds them (an ebook's text, not a
   * transcript's).
   */
  readonly text: string;
  /** The window's audio, uploaded or inline. */
  readonly audio: JobInput;
  /** The audio's container extension, without the dot: `flac`, `wav`, `m4a`. */
  readonly extension: string;
}

export interface AlignOptions {
  /** Which aligner, e.g. `qwen3-aligner`; there is no default. */
  readonly model: string;
  /** One of the aligner's languages as an ISO code: en de fr es it pt ru ja ko zh yue. */
  readonly language: string;
  /** Every window of the run, in one job. */
  readonly windows: readonly AlignWindow[];
}

/** One placed item, in seconds FROM THE START OF ITS WINDOW'S AUDIO. */
export interface AlignItem {
  /** A token of the aligner's own tokenization of the window's text. */
  readonly text: string;
  readonly start: number;
  readonly end: number;
}

/** One window's outcome: exactly one of `items` and `error` is non-null. */
export interface AlignWindowResult {
  readonly index: number;
  readonly items: readonly AlignItem[] | null;
  readonly error: string | null;
}

/** `alignment.json`, read by {@link readAlignment}. */
export interface Alignment {
  /** The aligner id that placed these items. */
  readonly model: string;
  /** In the order the job listed its windows. */
  readonly windows: readonly AlignWindowResult[];
}

/** What an `image` job makes; a field left out takes the server's default (1024x1024, 40 steps, a random seed). */
export interface ImageOptions {
  readonly model: string;
  readonly prompt: string;
  readonly negativePrompt?: string | null;
  readonly width?: number;
  readonly height?: number;
  readonly seed?: number;
  readonly steps?: number;
  readonly guidance?: number;
  readonly imageStrength?: number | null;
  readonly image?: JobInput | null;
  readonly imageName?: string;
  /** Hold the model from the moment it is loaded, for a batch; `act` must be `image`. */
  readonly lease?: LeaseOnLoad;
}

/** Options for {@link CrucibleClient.loadImage}. */
export interface LoadImageOptions {
  /** Hold the model from the moment it is loaded; `act` must be `image`. */
  readonly lease?: LeaseOnLoad;
}

/** An `image` job's effective parameters and measurements, read by {@link readImageResult}. */
export interface ImageResult {
  readonly model: string;
  readonly hfRepo: string;
  readonly revision: string;
  readonly backend: string;
  readonly engine: string;
  readonly dtype: string;
  readonly prompt: string;
  readonly negativePrompt: string | null;
  readonly width: number;
  readonly height: number;
  readonly seed: number;
  readonly steps: number;
  readonly guidance: number;
  readonly imageStrength: number | null;
  readonly input: string | null;
  readonly seconds: number | null;
  readonly stageSeconds: Readonly<Record<string, number>>;
  readonly peakBytes: number | null;
  readonly stagePeakBytes: Readonly<Record<string, number>>;
  readonly memoryBytesEstimate: number;
  readonly memoryBasis: string;
  readonly artifacts: readonly string[];
  /** Whether the prompt's embeddings came from the loaded model's cache; null from a server that does not say. */
  readonly promptCache: 'hit' | 'miss' | null;
  /** The lease this job opened or renewed from `lease`, else null. */
  readonly leaseId: string | null;
}

/** `GET /v1/setup` — everything an app needs to be pointed at this server, token included. */
export interface ServerSetup {
  /** The server's name, e.g. `crucible@mac-studio`. */
  readonly name: string;
  readonly version: string;
  /** `cuda-linux` or `mlx-darwin`. */
  readonly backend: string;
  /** What this process bound, e.g. `http://0.0.0.0:7100`. */
  readonly bind: string;
  /**
   * The addresses this server is reachable on: one per non-loopback IPv4 interface for a wildcard
   * bind.
   */
  readonly urls: readonly string[];
  readonly token: string;
  /** One `crucible://` line per {@link ServerSetup.urls} entry, in order. */
  readonly pairing: readonly string[];
  /** `/v1/info`'s list, repeated so a page draws from one read. */
  readonly jobTypes: readonly string[];
  readonly configPath: string;
}

/** The kinds of thing a subject can be. */
export type SubjectKind =
  | 'model'
  | 'voice'
  | 'rvc'
  | 'rvc-base'
  | 'denoise'
  | 'engine';

/** One row of `GET /v1/catalog`: a pullable thing and where it stands here. */
export interface CatalogRow {
  readonly kind: SubjectKind;
  readonly id: string;
  /** The manifest's display name, or null where a manifest carries none. */
  readonly name: string | null;
  /** Which job type this subject belongs to: `llm`, `asr`, `align`, `tts`, … */
  readonly jobType: string;
  readonly installed: boolean;
  /** Bytes on disk, or null when it is not installed. */
  readonly installedBytes: number | null;
  /** Bytes a pull will fetch where the manifest declares them, else null. */
  readonly expectedBytes: number | null;
  /** The model whose download this row's weights are, or null. */
  readonly sharesWeightsOf: string | null;
  /** An alias's own files that are not on disk, or null on a row that is not an alias. */
  readonly missingFiles: readonly string[] | null;
  /**
   * The capability classes this model is the FLOOR for — the smallest model the class may run on at
   * all.
   */
  readonly floors: readonly string[];
  /** Always null: no manifest carries a licence. */
  readonly license: string | null;
  /** `hf:<repo>` — where the bytes come from. */
  readonly source: string;
  /** Is this the thing on the card right now? */
  readonly resident: boolean;
}

/** One `pull`: fetch a subject's weights. */
export interface PullTaskRequest {
  readonly type: 'pull';
  readonly kind: SubjectKind;
  readonly id: string;
}

/** One `install`: build a job type's env, then make it live. */
export interface InstallTaskRequest {
  readonly type: 'install';
  readonly jobType: string;
  /** The narrator engine to install for `tts`; required there and refused for anything else. */
  readonly narratorEngine?: string;
}

/** An app's statement of what it needs from a server, as the JSON file it vendors. */
export interface CrucibleModule {
  readonly name: string;
  /** Derived by the generator from the content, never typed by hand. */
  readonly version: string;
  readonly job_types: readonly {
    readonly type: string;
    readonly narrator_engine?: string;
  }[];
  /** Capability classes, which the server resolves to its own models. */
  readonly needs: readonly { readonly class: string }[];
  /** Ids an app chose explicitly: a voice, a whisper size, the rvc base. */
  readonly subjects: readonly {
    readonly kind: SubjectKind;
    readonly id: string;
  }[];
}

/** One class a `module` named that this engine does not serve. */
export interface UnmetNeed {
  readonly class: string;
  /** The capability row's OWN sentence, verbatim. */
  readonly reason: string;
}

/** One `module`: an ordered list of installs and pulls, validated whole. */
export interface ModuleTaskRequest {
  readonly type: 'module';
  readonly module: CrucibleModule;
}

/** One `engine`: move this Windows machine to the WSL2 engine, run by the host. */
export interface EngineTaskRequest {
  readonly type: 'engine';
  readonly target: 'wsl';
}

/** Restart this machine's engine, through its orchestrator. */
export interface EngineRestartTaskRequest {
  readonly type: 'engine-restart';
}

export type TaskRequest =
  | PullTaskRequest
  | InstallTaskRequest
  | ModuleTaskRequest
  | EngineTaskRequest
  | EngineRestartTaskRequest;

/** A task's state; there is no `queued`, because a second task is refused rather than parked. */
export type TaskState = 'running' | 'done' | 'failed' | 'cancelled';

export const TASK_TERMINAL_STATES = ['done', 'failed', 'cancelled'] as const;

/** `GET /v1/tasks/{id}`. */
export interface TaskStatus {
  readonly taskId: string;
  /** `pull`, `install`, `module` or `engine`. */
  readonly type: string;
  /** The request body, echoed, in the server's own spelling. */
  readonly request: Readonly<Record<string, unknown>>;
  readonly state: TaskState;
  readonly error: JobFailure | null;
  readonly created: string;
  readonly started: string;
  readonly finished: string | null;
  /** Classes a `module` named that this engine does not serve (5.3a). */
  readonly unmet: readonly UnmetNeed[];
  /**
   * What a task started by `POST /v1/jobs` is doing, in words; `null` for every other task and once
   * it ends.
   */
  readonly message: string | null;
}

/**
 * `details` of a `409 installing` refusal from `POST /v1/jobs`: the server started installing what
 * the job needs.
 */
export interface InstallingDetails {
  readonly job_type: string;
  readonly task_id: string;
  /** `installing` (this job's install task) or `task_busy` (another task has the lane). */
  readonly reason: 'installing' | 'task_busy';
  /** What the install is doing, in words. */
  readonly message: string;
  /** The install modal's sentences for this card (`GET /v1/capability/plan`), or null. */
  readonly plan: string | null;
  readonly steps: readonly string[];
  readonly step: { readonly name: string | null; readonly index: number | null; readonly total: number | null } | null;
  /** 0..1 while the step reports bytes, else null. */
  readonly progress: number | null;
  readonly line: string | null;
}

/** One step of a task. */
export interface TaskStepData {
  readonly name: string;
  /** The step's 1-based position. */
  readonly index: number;
  readonly total: number;
  /** What this server now offers, on the `reload` step only. */
  readonly jobTypes?: readonly string[];
}

/** A pull's progress: bytes of one file. */
export interface TaskBytesProgress {
  readonly bytesDone: number;
  readonly bytesTotal: number | null;
  readonly file: string;
}

/** An install's progress: one line of the installer's own output. */
export interface TaskLineProgress {
  readonly line: string;
}

export type TaskProgressData = TaskBytesProgress | TaskLineProgress;

/** Narrow a `progress` frame to the installer's lines. */
export function isTaskLineProgress(data: TaskProgressData): data is TaskLineProgress {
  return 'line' in data;
}

/** Narrow a `progress` frame to a pull's byte counts. */
export function isTaskBytesProgress(data: TaskProgressData): data is TaskBytesProgress {
  return 'bytesDone' in data;
}

/** A module entry that was already true, so nothing was done for it. */
export interface TaskSkippedData {
  /** Why, in the server's words. */
  readonly reason: string;
}

/** One frame of `GET /v1/tasks/{id}/events`. */
export type TaskEvent =
  | { readonly id: number; readonly event: 'started'; readonly data: { readonly type: string } }
  | { readonly id: number; readonly event: 'step'; readonly data: TaskStepData }
  | { readonly id: number; readonly event: 'progress'; readonly data: TaskProgressData }
  | { readonly id: number; readonly event: 'skipped'; readonly data: TaskSkippedData }
  | { readonly id: number; readonly event: 'done'; readonly data: Readonly<Record<string, unknown>> }
  | { readonly id: number; readonly event: 'failed'; readonly data: JobFailure }
  | { readonly id: number; readonly event: 'cancelled'; readonly data: Readonly<Record<string, unknown>> }
  | UnknownEvent;

/** What `DELETE /v1/tasks/{id}` answers. */
export interface TaskCancelResult {
  readonly taskId: string;
  /** `cancelling`: the runner stops when it sees the flag; watch for the `cancelled` event. */
  readonly status: 'cancelling';
}

/** The three upstreams a Crucible speaks to. */
export type UpstreamName = 'anthropic' | 'openai' | 'ollama';

/** Where one capability class's work runs on a server. */
export interface RouteSetting {
  /** `local` (this server's card) or `upstream` (the operator's account). */
  readonly route: 'local' | 'upstream';
  /**
   * The selected local model (`null` when none fits or nothing is decided), or the
   * `<upstream>/<model>` id.
   */
  readonly model: string | null;
}

/** One upstream card. */
export interface UpstreamSetting {
  readonly configured: boolean;
  /** `…` plus the key's last four characters, for `anthropic` and `openai`; absent for `ollama`. */
  readonly keyHint?: string | null;
  /** The address, for `ollama` only. */
  readonly url?: string | null;
}

/** An eligible local model, computed by the engine for a capability class. */
export interface LocalModelChoice {
  readonly id: string;
  readonly memoryBytesEstimate: number;
  readonly fits: boolean;
  readonly installed: boolean;
}

/** `GET /v1/settings` — the whole of what an app's settings window draws. */
export interface SettingsDocument {
  /** Which local model serves each capability class. */
  readonly localModels: Readonly<Record<string, string | null>>;
  /** Empty when the engine has not measured its card yet; never absent. */
  readonly localModelChoices: Readonly<Record<string, readonly LocalModelChoice[]>>;
  /** One entry per routable capability class. */
  readonly routes: Readonly<Record<string, RouteSetting>>;
  /** One card per upstream. */
  readonly upstreams: Readonly<Record<UpstreamName, UpstreamSetting>>;
  readonly desktopAllowanceBytes: number;
  readonly backendKind: string;
}

/** A partial `PUT /v1/settings` patch; a refusal applies nothing. */
export interface SettingsPatch {
  /** Explicit local selection; null restores the engine's automatic decision. */
  readonly localModels?: Readonly<Record<string, string | null>>;
  /** Class → `'local'` or an upstream model id. */
  readonly routes?: Readonly<Record<string, string>>;
  /** Name → its one field, or `null` to remove the upstream. */
  readonly upstreams?: Readonly<
    Partial<Record<UpstreamName, { key?: string; url?: string } | null>>
  >;
  readonly desktopAllowanceBytes?: number;
}

/** What `testUpstream` answers; the three test refusals come back as results, not exceptions. */
export type UpstreamTestResult =
  | { readonly ok: true; readonly models: string[] }
  | {
      readonly ok: false;
      readonly code:
        | 'upstream_unreachable'
        | 'upstream_rejected'
        | 'upstream_unconfigured';
      readonly message: string;
    };
