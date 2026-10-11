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
  /**
   * What this server's API offers, by name (`queue.sessions`, `events`, `tts.stream`, …): check a
   * name here, or with {@link CrucibleClient.has}, instead of comparing versions.
   */
  readonly features: readonly string[];
  /** The embed and rerank verbs, by name; null from a server before them. */
  readonly verbs: Readonly<Record<string, VerbInfo>> | null;
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

/**
 * A deploy's hold on this server (`POST /v1/server/updating`): while it stands, every door that
 * creates work answers `503 server_updating` (the SDK's `CrucibleUpdating`) and admits nothing. It ends
 * with the restart, a `DELETE`, or by itself at `until`.
 */
export interface UpdateHold {
  /** The release the deploy is installing, or `null` when it did not say. */
  readonly release: string | null;
  /** Who took the hold, or `null` when it did not say. */
  readonly by: string | null;
  /** When it was taken, in {@link ActivityJob.started}'s format. */
  readonly since: string;
  /** When it lapses by itself if nothing ends it first. */
  readonly until: string;
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
  /**
   * `true` while a running job's client has cancelled it and Crucible is still stopping it (the
   * `message` then says so); `null` on a row that does not say, which is a waiting call or a
   * server from before the field.
   */
  readonly cancelling: boolean | null;
  /** Seconds this job has waited in the server's queue; `null` unless it is waiting there. */
  readonly waitedS: number | null;
  /** How long it may wait before it is removed `expired`; `null` unless it is waiting. */
  readonly maxWaitS: number | null;
  /** Why it waits although the lane is free (a card held by someone else); `null` otherwise. */
  readonly waitingFor: QueueWaitingFor | null;
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

/** Where a queue session stands: waiting in the line, holding the server, or ended. */
export type QueueSessionStatus = 'queued' | 'open' | 'closed';

/**
 * Why a queue session ended. `client`: its client closed it. `idle`: `idle_s` passed with nothing
 * in flight, no item and no touch. `operator`: a person ended it. `max_hold`: the server's
 * configured maximum hold. `server_restart`: the server stopped. A session that never opened ends
 * `expired` (it waited `max_wait_s`, or nobody followed it), `load_failed` (its model would not
 * load), or with one of the reasons above.
 */
export type QueueSessionReason =
  | 'client'
  | 'idle'
  | 'operator'
  | 'max_hold'
  | 'server_restart'
  | 'expired'
  | 'load_failed';

/**
 * `GET /v1/queue/sessions/{id}`: one app's turn holding the server for a run of requests. Not a
 * TTS stream session.
 */
export interface QueueSessionState {
  readonly sessionId: string;
  readonly status: QueueSessionStatus;
  /** The capability class the run is for. */
  readonly act: string;
  /** Who holds it (`X-Crucible-Client`, else `User-Agent`). */
  readonly client: string | null;
  /** The model it opened with resident, or null when it named none. */
  readonly model: string | null;
  /** 1 is next, while it waits; null once open. */
  readonly position: number | null;
  readonly idleS: number;
  readonly maxWaitS: number;
  readonly created: string;
  readonly openedAt: string | null;
  /** When it closes `idle` unless something arrives; null while anything is in flight. */
  readonly idleDeadline: string | null;
  /** When it closes `max_hold`; null unless the server sets a maximum. */
  readonly maxHoldDeadline: string | null;
  readonly itemsRun: number;
  /** What it has running or waiting: jobs, chats, queued calls, stream rows being said. */
  readonly inFlight: readonly Readonly<Record<string, unknown>>[];
  /** The TTS stream open in it, or null. */
  readonly streamSession: Readonly<Record<string, unknown>> | null;
  /** The `load-model` job that made its model resident, or null. */
  readonly loadJob: string | null;
  readonly closedAt: string | null;
  /** Why it ended, once it has: one of {@link QueueSessionReason}; a newer server may name another. */
  readonly reason: string | null;
  readonly message: string | null;
  /** The load's error, for a session that ended `load_failed`. */
  readonly error: Readonly<Record<string, unknown>> | null;
}

/** Where a queued session stands in the line: 1 is next, of `of` waiting. */
export interface QueuePosition {
  readonly position: number;
  readonly of: number;
}

/** How a queue session ended, as its `closed` event said. */
export interface QueueSessionEnd {
  /** One of {@link QueueSessionReason}; a newer server may name another. */
  readonly reason: string;
  /** A sentence a person reads. */
  readonly message: string;
  /** How many items ran in it; null when the server could not say (it stopped). */
  readonly itemsRun: number | null;
  /** How long it held the server, in seconds; null when the server could not say. */
  readonly heldS: number | null;
}

/** Options for {@link CrucibleClient.session}. */
export interface SessionOptions {
  /** The capability class the run is for, as `X-Crucible-Act` names it. Required. */
  readonly act: string;
  /** A model to have resident when the session opens; the server loads it for the session. */
  readonly model?: string;
  /**
   * Close the session after this many seconds with nothing in flight, no item and no touch:
   * 10..86400, the server's default 300. A running job or an answer in flight always counts as
   * activity; work on the app's own side (a cloud call, a file copy) does not, so call
   * `touch()` across a long gap.
   */
  readonly idleS?: number;
  /** How long it may wait in the line to open: 10..86400 seconds, the server's default an hour. */
  readonly maxWaitS?: number;
  /** Called with the session's place in the line whenever it joins or moves. */
  readonly onQueue?: (position: QueuePosition) => void;
  /**
   * Called while the session waits at the front of the line because its model's load found the
   * accelerator held by a process the server does not own: when the wait begins, when the holder
   * changes, and every minute while it does not (1.0.83+). Never called once the session is open.
   */
  readonly onWaiting?: (waiting: CardWaitData) => void;
  /** Aborts the wait: a session still in the line leaves it, and the abort is thrown. */
  readonly signal?: AbortSignal;
}

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
  /** The open queue session, which holds the server until it closes, or null. */
  readonly session: QueueSessionState | null;
  readonly slots: { readonly accelerated: ActivitySlot };
  /**
   * The deploy hold, while one stands: new work is refused until the restart. `null` when none
   * stands, and from a server from before the field.
   */
  readonly updating: UpdateHold | null;
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
  /**
   * How this job waits while the server is busy. Left out: the client's own `queue`, else it waits
   * in the server's line (an hour; a day inside a session). `{maxWaitS}` changes the wait; `false`
   * refuses at once with `server_busy` instead.
   */
  readonly queue?: QueueChoice;
}

/**
 * How a request waits in the server's line while the server is busy. Waiting is the default, so
 * there is nothing to say to wait: leave `queue` out. `{maxWaitS}` (10..86400 seconds) changes the
 * wait from the server's default of an hour; `false` refuses at once instead of waiting.
 */
export type QueueChoice = false | { readonly maxWaitS: number };

/**
 * A job's lifecycle state; `interrupted` means the server stopped mid-job, not that the work
 * failed. `removed` means the job left the server's queue without running (an operator removed
 * it, you cancelled it, it waited too long, or the server restarted) — not a failure: show it
 * and offer to submit again.
 */
export type JobState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted' | 'removed';

/** Why a job left the server's queue without running. */
export type RemovalReason = 'operator' | 'client' | 'expired' | 'server_restart' | 'session_closed';

/** The `removed` event's payload, and {@link JobStatus.removal}. */
export interface RemovedData {
  /** One of {@link RemovalReason}; a newer server may name another. */
  readonly reason: string;
  /** A sentence a person reads. */
  readonly message: string;
  /** How long it waited, or `null` where the server could not say (after a crash). */
  readonly waitedS: number | null;
  readonly at: string;
}

/** The server's named refusal or failure, as carried on a job and in events. */
export interface JobFailure {
  readonly code: string;
  readonly message: string;
  /**
   * Facts to act on beside the sentence, for the failures that have some: a song refused
   * for its length (`song_length_out_of_range`, `instrumental_length_not_reached`) carries
   * `score_seconds`, the range, `ratio_needed` and its `attempts` (docs/AUDIO.md "Song
   * length"). Absent (undefined) otherwise.
   */
  readonly details?: Readonly<Record<string, unknown>>;
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
  /** Why the job left the queue without running, when `status` is `removed`; otherwise null. */
  readonly removal: RemovedData | null;
  /**
   * An audio job's request while it runs and after it ends anything but `done`: `params` with
   * the seed it uses written in (submit `type`, `model` and `params` again to reproduce it),
   * `seed`, `seed_chosen_by`, `settled`, `low_vram` and a `reproduce` sentence. Null once it is
   * `done` (its `audio` is the record) and for every other job type (docs/AUDIO.md).
   */
  readonly request: Readonly<Record<string, unknown>> | null;
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
  /** `removed` when the job was still waiting in the server's queue. */
  readonly status: 'cancelled' | 'cancelling' | 'removed';
}

/** One SSE event from `GET /v1/jobs/{id}/events`. */
export type JobEvent =
  | { readonly id: number; readonly event: 'queued'; readonly data: QueuedData }
  | { readonly id: number; readonly event: 'started'; readonly data: StartedData }
  | { readonly id: number; readonly event: 'removed'; readonly data: RemovedData }
  | { readonly id: number; readonly event: 'warming'; readonly data: WarmingData }
  | { readonly id: number; readonly event: 'progress'; readonly data: ProgressData }
  | { readonly id: number; readonly event: 'chunk'; readonly data: ChunkData }
  | { readonly id: number; readonly event: 'waiting'; readonly data: CardWaitData }
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

/**
 * Where the job stands: 1 is next. A job waiting in the server's queue gets one of these when it
 * joins and again whenever its place changes.
 */
export interface QueuedData {
  readonly position: number;
  /** How many jobs are waiting in the queue, or `null` on a job that never waited. */
  readonly of: number | null;
}

/** A job submitted with `queue` was admitted to the lane. */
export interface StartedData {
  /** Seconds it waited in the queue; 0 when the server was free. */
  readonly waitedS: number;
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
  /**
   * The interior pauses narrator cut down in this chunk (its interior-pause cap,
   * `NARRATOR_MAX_INTERIOR_PAUSE_S`), in order; `[]` when it cut none, null when narrator did not
   * say (an older narrator, or a server before 1.0.80). The audio and `seconds` are already after
   * the cuts.
   */
  readonly pauseCuts: readonly PauseCut[] | null;
}

/** One interior pause narrator shortened: where it was, how long it was, how long it is now. */
export interface PauseCut {
  /** Where the pause starts in the chunk's audio as delivered, in seconds. */
  readonly atS: number;
  /** How long the pause was before the cut, in seconds. */
  readonly fromS: number;
  /** How long it is now, in seconds. */
  readonly toS: number;
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
export const TERMINAL_EVENTS = ['done', 'failed', 'cancelled', 'removed'] as const;

/**
 * Why an item at the front waits although the lane is free: memory on the accelerator is held by a
 * process the server does not own (`accelerator_busy`). It is checked again at `nextCheckAt` and runs
 * the moment the memory is let go, or is removed `expired` when its wait runs out (docs/QUEUE.md).
 */
export interface CardWaitData {
  /** `accelerator_busy`. */
  readonly code: string;
  /** The server's sentence: how often it checks again, until when, and who holds the card. */
  readonly message: string;
  readonly details: Readonly<Record<string, unknown>> | null;
  /** When this wait began; the same on every repeat of it. */
  readonly since: string;
  /** When the card is checked again; null from a 1.0.82 server, which did not say. */
  readonly nextCheckAt: string | null;
}

/**
 * Why an item at the front waits although the lane is free (see {@link QueueWaitingFor}), as its
 * `waiting` event says it: when the wait begins, whenever who holds the card changes, and again
 * every minute while it does not (from 1.0.83), so a client watching for silence sees it is alive.
 */
export interface QueueWaitingFor {
  readonly code: string;
  /** The guard's sentence naming the holder: pid, name and memory, or the unattributed bytes. */
  readonly message: string;
  readonly details: Readonly<Record<string, unknown>> | null;
  readonly since: string;
  readonly nextCheckAt: string;
}

/** One job waiting in the server's queue, as `GET /v1/queue` lists it. */
export interface QueueItem {
  /** 1 is next. */
  readonly position: number;
  readonly jobId: string;
  readonly type: string;
  readonly model: string | null;
  /** Who queued it (its User-Agent or `X-Crucible-Client`), or null when it did not say. */
  readonly client: string | null;
  readonly clientRef: string | null;
  readonly submitted: string;
  readonly waitedS: number;
  readonly maxWaitS: number;
  /** When it is removed `expired` if it has not started. */
  readonly expiresAt: string;
  /** The queue session this item belongs to (its items go ahead of the line), or null. */
  readonly session: string | null;
  /**
   * `call` for a queued chat or decision (a `call-…` `jobId`, no job record); `session` for a queue
   * session waiting to open (its `jobId` is the session's `ses-…` id).
   */
  readonly kind: 'job' | 'call' | 'session';
  /** Why it waits although the lane is free (a card held by someone else); `null` otherwise. */
  readonly waitingFor: QueueWaitingFor | null;
}

/** `GET /v1/queue`. */
export interface QueueList {
  readonly items: readonly QueueItem[];
  readonly depth: number;
  readonly limits: {
    readonly perClient: number;
    readonly total: number;
    readonly maxWaitS: { readonly default: number; readonly min: number; readonly max: number };
    readonly abandonAfterS: number;
  };
}

/** `DELETE /v1/queue/{id}`. */
export interface QueueRemoved {
  readonly jobId: string;
  /** `closed` when the id was the open queue session's, which this ended. */
  readonly status: 'removed' | 'closed';
  readonly reason: 'operator';
}

/**
 * One event from `GET /v1/queue/events`: a `snapshot` first, then every change. `depth` is how
 * many jobs wait after the change.
 */
export type QueueEvent =
  | { readonly id: number; readonly event: 'snapshot'; readonly items: readonly QueueItem[]; readonly depth: number }
  | {
      readonly id: number;
      readonly event: 'added' | 'moved' | 'started' | 'removed';
      readonly jobId: string;
      readonly depth: number;
      /** Every other key the server put on the frame (`position`, `reason`, `waited_s`, …). */
      readonly data: Readonly<Record<string, unknown>>;
    }
  | UnknownEvent;

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
  /**
   * For a model that comes in more than one form (precision) of the same weights: the form this
   * server's card takes (the best that fits), which is what a load that names no form loads and
   * what `installed`, `revision` and `memoryBytesEstimate` describe. Null for a model with one.
   */
  readonly form: string | null;
  /** Why this card takes `form`, with the numbers; null for a model with one form. */
  readonly formReason: string | null;
  /** Every form, best first; null for a model with one form. */
  readonly forms: readonly ModelFormInfo[] | null;
  /** The verbs (capability classes) it serves; null from a server before embed and rerank. */
  readonly verbs: readonly string[] | null;
  /** The optional package it is in (`retrieval`), or null. */
  readonly package: string | null;
  /** Whether that package is installed here; null for a model in none. */
  readonly packageInstalled: boolean | null;
  /** An embedding model's facts and limits; null for any other. */
  readonly embed: ModelEmbedInfo | null;
  /** How it reranks (a reranker, or a decide model on the general template); null otherwise. */
  readonly rerank: ModelRerankInfo | null;
}

/** One form of a model (`ModelInfo.forms`). */
export interface ModelFormInfo {
  /** What `form` on a chat, a decision or {@link LoadModelOptions.form} says, e.g. `bf16`. */
  readonly name: string;
  readonly bits: number;
  readonly file: string;
  readonly memoryBytesEstimate: number;
  /** Whether it fits this server's card; null where the server knows no card. */
  readonly fits: boolean | null;
  readonly installed: boolean;
  /** The form this card takes. */
  readonly picked: boolean;
  /** The form on the card right now. */
  readonly resident: boolean;
  /** The operator command that pulls this form here. */
  readonly pullCommand: string;
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

/** {@link ChatOptions.jsonWhitespace}: compact JSON, or the default flexible whitespace. */
export type JsonWhitespace = 'compact' | 'flexible';

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
  /**
   * The whitespace of a JSON answer, sent as `json_whitespace`. `'compact'`: no whitespace between
   * JSON tokens, whitespace only inside strings (for a model trained on compact JSON).
   * `'flexible'` (the server's default): whitespace wherever JSON allows it. Needs a JSON
   * `responseFormat`; refused `json_whitespace_without_json` otherwise. Compact is kept on vLLM,
   * mlx-lm and llama-server on a Linux server (with thinking stated off there) and refused
   * `json_whitespace_not_served` on llama-server on native Windows, mlx-vlm and upstream
   * models; a schema whose `x-guidance` states whitespace itself is `json_whitespace_conflict`.
   */
  readonly jsonWhitespace?: JsonWhitespace;
  /** Whether a reasoning model thinks before it answers. */
  readonly thinking?: boolean;
  /**
   * The start of the answer: the model writes on from this text, and the reply's `content` is what
   * it wrote AFTER it (join the two yourself). Sent as `prefill`. Served on vLLM and llama-server;
   * needs thinking off (`thinking: false`, or the model's manifest) and no `responseFormat`, and
   * must not begin or end with whitespace. Refused by name otherwise (`prefill_not_served`,
   * `prefill_with_thinking`, `prefill_with_grammar`, `prefill_conflict`, `invalid_request`).
   */
  readonly prefill?: string;
  /** The context window an `ollama/<tag>` chat runs at, sent as `context_tokens`. */
  readonly contextTokens?: number;
  /**
   * Which form of the model serves this, for a model that comes in more than one form (precision)
   * of the same weights (`ModelInfo.forms`). Omitted: whichever form is resident, and the form the
   * server's card takes when one is loaded — what almost every caller wants. Named: that form;
   * another form on the card is a reload. A name the model does not have is 400 `unknown_form`.
   */
  readonly form?: string;
  /**
   * What this chat is, as a capability class sent in the `X-Crucible-Act` header; omitted, no
   * header is sent.
   */
  readonly act?: string;
  /**
   * While the model is not resident or every slot on its engine is taken, the chat waits in the
   * server's line and the server loads the model when its turn comes. Omitted: the client's own
   * `queue`, else it waits (an hour). `{maxWaitS}` changes the wait; `false` refuses at once
   * (`model_not_resident`, `chat_queue_full`, `session_open`) instead.
   */
  readonly queue?: QueueChoice;
  /** Aborts the request. */
  readonly signal?: AbortSignal;
}

/** The tokens a completion cost, as the engine counted them. */
export interface ChatUsage {
  readonly promptTokens: number | null;
  readonly completionTokens: number | null;
  readonly totalTokens: number | null;
  /**
   * Prompt tokens the engine read from its prefix cache instead of computing
   * (`usage.prompt_tokens_details.cached_tokens`); `null` when the engine did not say.
   */
  readonly cachedTokens: number | null;
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

/**
 * Score 2–256 free-text candidate replies by how likely the model is to say each, instead of
 * generating one: nothing is decoded, so nothing fails to parse or runs away. Each candidate is
 * scored as the start of the model's reply to `instructions` (thinking off), at most 256 tokens.
 * Served on vLLM, llama-server, mlx-lm and mlx-vlm (images on mlx-vlm only).
 */
export interface DecideLikelihoodQuestion {
  readonly type: 'likelihood';
  readonly instructions: string;
  /** Candidate name → the reply text scored: unique, no leading or trailing whitespace. */
  readonly candidates: Readonly<Record<string, string>>;
  /** Which measure picks `winner`. Omitted: `total`. */
  readonly rankBy?: DecideRankBy;
  /** What each candidate's `probability` is. Omitted: `softmax`. */
  readonly normalize?: DecideNormalize;
}

/**
 * `total`: the summed log-probability, for variants of the same content (spellings, OCR readings
 * of one line). `mean`: per token, for candidates whose lengths differ by content (titles).
 */
export type DecideRankBy = 'total' | 'mean';

/**
 * `softmax`: the totals renormalised over the candidates offered (they sum to 1), for picking the
 * one answer among mutually exclusive replies. `none`: exp(`logprob`), each reply's own
 * probability, independent of the others. Neither says "which of these apply": ask that as the
 * items form, one yes/no item per option.
 */
export type DecideNormalize = 'softmax' | 'none';

export type DecideQuestion =
  | DecideChoiceQuestion
  | DecideScoreQuestion
  | DecideYesNoQuestion
  | DecideLikelihoodQuestion;

export interface DecideRequest {
  /**
   * The model to read the decision from. Omitted: the model the server registered for `decide`,
   * or with `images`, the one it registered for a decision with images (the response's `model`
   * names which served it).
   */
  readonly model?: string;
  /** What the questions are about: a string, or any JSON value. */
  readonly state: unknown;
  /** Image FILES, base64-encoded (at most 8; more is 400 `too_many_images`). */
  readonly images?: readonly string[];
  /** Question name → question. */
  readonly questions: Readonly<Record<string, DecideQuestion>>;
  /** What to do when a label is not among the top tokens the engine returned. */
  readonly missing?: DecideMissing;
  /**
   * Which form of the model serves this, for a model that comes in more than one form (precision)
   * of the same weights (`ModelInfo.forms`). Omitted: whichever form is resident, and the form the
   * server's card takes when one is loaded — what almost every caller wants. Named: that form;
   * another form on the card is a reload. A name the model does not have is 400 `unknown_form`.
   */
  readonly form?: string;
}

/** `DecideRequest.missing`: refuse a decision with a label outside the top-K, or report it. */
export type DecideMissing = 'refuse' | 'report';

export interface DecideOptions {
  /** What this decision is, sent as `X-Crucible-Act` exactly as {@link ChatOptions.act}. */
  readonly act?: string;
  /** How the decision waits for its model, exactly as {@link ChatOptions.queue}. */
  readonly queue?: QueueChoice;
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

/** One candidate's reading in a {@link DecideLikelihoodAnswer}. */
export interface DecideCandidateScore {
  /** ln P(this reply | the context), summed over its tokens. Not calibrated. */
  readonly logprob: number;
  /** The tokens scored. */
  readonly tokens: number;
  /** `logprob / tokens`. */
  readonly meanLogprob: number;
  /**
   * By the question's `normalize`: a softmax over the candidates' totals (`softmax`), or
   * exp(`logprob`) on its own (`none`); from the total whatever `rankBy` says.
   */
  readonly probability: number;
}

/** The answer to a `likelihood`: every candidate's log-likelihood, and the winner by `rankBy`. */
export interface DecideLikelihoodAnswer {
  readonly type: 'likelihood';
  /** The best candidate by `rankBy`; on a tie the first asked. */
  readonly winner: string;
  readonly rankBy: DecideRankBy;
  /** What each `probability` is, echoed from the question. */
  readonly normalize: DecideNormalize;
  /** Candidate name → its reading, in the order asked. */
  readonly candidates: Readonly<Record<string, DecideCandidateScore>>;
  /** The context's tokens: the state, the request and the opened reply. */
  readonly contextTokens: number;
  /** Context tokens re-read because a candidate's first characters merged into the last (0 or 1). */
  readonly boundaryTokens: number;
}

export type DecideAnswer = DecideChoiceAnswer | DecideScoreAnswer | DecideYesNoAnswer | DecideLikelihoodAnswer;

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

/** `EmbedRequest.inputType`: a search query (the model's instruction prefix) or a document. */
export type EmbedInputType = 'query' | 'document';

/** How the vectors come back: JSON numbers, float32 little-endian base64, or IEEE half base64. */
export type EmbedEncoding = 'float' | 'base64' | 'base64_float16';

/**
 * `POST /v1/embed`: texts to unit-length vectors. Vectors are comparable only with vectors of the
 * same `fingerprint`: store {@link EmbedModelInfo.fingerprint} beside them and send it back here on
 * every later call, and a server that would write anything else refuses `409 fingerprint_mismatch`.
 */
export interface EmbedRequest {
  /** The texts, in order; 1 to 256, each non-empty. */
  readonly inputs: readonly string[];
  /** `query` is written with the model's instruction prefix; `document` as it is. */
  readonly inputType: EmbedInputType;
  /** What the query is for, in one sentence; refused on a document. Omitted: the model's default. */
  readonly instruction?: string;
  /** The Crucible model id. Omitted: the model this server registered for embed. */
  readonly model?: string;
  /** Which form of `model`, for a model with more than one. */
  readonly form?: string;
  /** An earlier answer's `model.fingerprint`: serve exactly that identity or refuse. */
  readonly fingerprint?: string;
  /** A ceiling in billions of parameters, when no model is named. */
  readonly maxParamsB?: number;
  /** A Matryoshka prefix length (renormalised); omitted, the model's own length. */
  readonly dimensions?: number;
  /** Omitted: `float`. */
  readonly encodingFormat?: EmbedEncoding;
}

/** What wrote the vectors: everything that changes a float. */
export interface EmbedModelInfo {
  readonly id: string;
  readonly revision: string;
  /** The weights file read (a GGUF), or null for a whole repo. */
  readonly file: string | null;
  readonly form: string | null;
  readonly bits: number | null;
  readonly engine: string;
  readonly engineBuild: string;
  /** Crucible's own reading of a vector. */
  readonly scheme: number;
  /** All of the above in one string: store it beside the vectors. */
  readonly fingerprint: string;
  /** The model's own vector length. */
  readonly dimensions: number;
}

/** Ms the call waited in the server's line, apart from how long it ran. */
export interface VerbTiming {
  readonly total: number;
  /** Null in a timing built outside the door. */
  readonly queued: number | null;
}

export interface EmbedResponse {
  readonly model: EmbedModelInfo;
  /** Each vector's length. */
  readonly dimensions: number;
  readonly inputType: EmbedInputType;
  /** The instruction the queries were written with; null for documents. */
  readonly instruction: string | null;
  readonly encodingFormat: EmbedEncoding;
  /** Unit-length vectors: numbers for `float`, base64 strings otherwise ({@link decodeEmbedding}). */
  readonly embeddings: readonly (readonly number[])[] | readonly string[];
  readonly tokens: { readonly perInput: readonly number[]; readonly total: number };
  readonly timingMs: VerbTiming;
}

/** `POST /v1/rerank`: a relevance probability per document for one query. */
export interface RerankRequest {
  readonly query: string;
  /** 1 to 256 documents, each non-empty. */
  readonly documents: readonly string[];
  /** What relevant means here, in one sentence. Omitted: the model's default. */
  readonly instruction?: string;
  /** A dedicated reranker or any decide model. Omitted: the model registered for rerank. */
  readonly model?: string;
  readonly form?: string;
  /** A ceiling in billions of parameters, when no model is named. */
  readonly maxParamsB?: number;
}

export interface RerankModelInfo {
  readonly id: string;
  readonly revision: string;
  readonly file: string | null;
  readonly form: string | null;
  readonly engine: string;
  readonly engineBuild: string;
  /** `model` (the reranker's own prompt) or `crucible-general-1` (a decide model). */
  readonly template: string;
  /** Scores compare across calls with the same fingerprint. */
  readonly fingerprint: string;
}

export interface RerankResult {
  readonly index: number;
  /** P(yes) / (P(yes) + P(no)), 0 to 1. */
  readonly relevanceScore: number;
}

export interface RerankResponse {
  readonly model: RerankModelInfo;
  readonly instruction: string;
  /** Each document's relevance, in the request's order. */
  readonly scores: readonly number[];
  /** Every document, most relevant first. */
  readonly results: readonly RerankResult[];
  readonly tokens: {
    readonly perDocument: readonly number[];
    /** Every document's prompt with its query, as llama-server is sent it (once per candidate). */
    readonly total: number;
    /** Of `total`, what no pass read again (the shared query, a held cache): `total - cached`
     * is what the engine read. Null when the engine did not say. */
    readonly cached: number | null;
  };
  readonly timingMs: VerbTiming;
}

/** Per-call options for {@link CrucibleClient.embed} and {@link CrucibleClient.rerank}. */
export interface VerbOptions {
  /** Sent as `X-Crucible-Act` exactly as {@link ChatOptions.act}. */
  readonly act?: string;
  /** How the call waits for its model, exactly as {@link ChatOptions.queue}. */
  readonly queue?: QueueChoice;
  /** Aborts the request; the abort surfaces as a DOM `AbortError`. */
  readonly signal?: AbortSignal;
}

export type EmbedOptions = VerbOptions;
export type RerankOptions = VerbOptions;

/** A `GET /v1/models` row's `embed`: what an embedding model writes and what a call may carry. */
export interface ModelEmbedInfo {
  readonly dimensions: number;
  /** The `dimensions` a request may ask for, `[low, high]`. */
  readonly dimensionsRange: readonly [number, number];
  readonly matryoshka: boolean;
  readonly pooling: string;
  readonly normalized: boolean;
  readonly inputTypes: readonly string[];
  readonly queryTakesInstruction: boolean;
  readonly defaultInstruction: string | null;
  readonly queryTemplate: string;
  readonly documentTemplate: string;
  readonly source: string;
  readonly maxInputs: number;
  /** Tokens one input may be (the served context); null where the backend has no block. */
  readonly maxInputTokens: number | null;
}

/** A `GET /v1/models` row's `rerank`: the template it is judged with and the limits. */
export interface ModelRerankInfo {
  /** `model` (its own prompt) or `crucible-general-1`. */
  readonly template: string;
  readonly defaultInstruction: string;
  readonly maxDocuments: number;
  readonly maxTokens: number | null;
  /** The reranker's own prompt parts; null for a decide model on the general template. */
  readonly prefixTemplate: string | null;
  readonly documentTemplate: string | null;
  readonly yes: string | null;
  readonly no: string | null;
  readonly source: string | null;
}

/** `GET /v1/info`'s `verbs` entry: whether this server serves the verb now, and with what. */
export interface VerbInfo {
  readonly route: string;
  readonly available: boolean;
  readonly registered: string | null;
  readonly reason: string | null;
  readonly package: string | null;
  readonly packageInstalled: boolean;
  /** The models a request may name here, in the order the automatic pick ranks them. */
  readonly models: readonly string[];
  readonly limits: { readonly maxInputs: number | null; readonly maxDocuments: number | null };
}

/** One item of {@link DecideItemsRequest}: its text, and optionally its own options. */
export interface DecideItem {
  readonly text: string;
  readonly options?: Readonly<Record<string, string>>;
}

/** The items form of `POST /v1/decide`: choice questions about one state, answered in item order. */
export interface DecideItemsRequest {
  /** As {@link DecideRequest.model}: omitted, the server's registered decide model. */
  readonly model?: string;
  readonly state: unknown;
  readonly images?: readonly string[];
  readonly instructions?: string;
  readonly options?: Readonly<Record<string, string>>;
  readonly items: readonly DecideItem[];
  readonly missing?: DecideMissing;
  /**
   * Which form of the model serves this, for a model that comes in more than one form (precision)
   * of the same weights (`ModelInfo.forms`). Omitted: whichever form is resident, and the form the
   * server's card takes when one is loaded — what almost every caller wants. Named: that form;
   * another form on the card is a reload. A name the model does not have is 400 `unknown_form`.
   */
  readonly form?: string;
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
  /** A local (`path`) voice that nothing holds: not resident, not named by a job. */
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
}

/** Options for {@link CrucibleClient.loadModel}. */
export interface LoadModelOptions {
  /** Tokens: the context to start the engine with (`params.context`). */
  readonly context?: number;
  /**
   * The form to load (`params.form`), for a model with more than one (`ModelInfo.forms`).
   * Omitted: the form the server's card takes.
   */
  readonly form?: string;
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
  /** Image-to-image: how much of the input survives, 0 to 1 exclusive (higher keeps more). With `mask`, optional: left out, the masked region is regenerated from scratch. */
  readonly imageStrength?: number | null;
  readonly image?: JobInput | null;
  /** The input name the picture is sent under; default `input.png`. */
  readonly imageName?: string;
  /** Inpainting and outpainting: a picture the size of `image`, white where to regenerate, black where to keep. Needs `image`. */
  readonly mask?: JobInput | null;
  /** The input name the mask is sent under (it becomes `params.mask`); default `mask.png`. */
  readonly maskName?: string;
  /** How many pixels inside the mask's edge the new picture fades into the kept one; 0 to 256, the server's default 8. Only with `mask`. */
  readonly maskBlur?: number;
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
  /** The mask input's name for an inpainting job, else null (and null from a server without masks). */
  readonly mask: string | null;
  /** The feather the mask was pasted back with, in pixels; null without a mask. */
  readonly maskBlur: number | null;
  /** The share of the picture the mask selected, 0 to 1; null without a mask. */
  readonly maskCoverage: number | null;
  /** How many denoising steps put the input back outside the mask; null without a mask or from an older server. */
  readonly maskBlendSteps: number | null;
  /** Mean absolute difference, 0 to 255, between the model's picture and the input outside the mask before the paste-back: a few units when the blend held; null without a mask or from an older server. */
  readonly maskOutsideDrift: number | null;
  readonly seconds: number | null;
  readonly stageSeconds: Readonly<Record<string, number>>;
  readonly peakBytes: number | null;
  readonly stagePeakBytes: Readonly<Record<string, number>>;
  readonly memoryBytesEstimate: number;
  readonly memoryBasis: string;
  readonly artifacts: readonly string[];
  /** Whether the prompt's embeddings came from the loaded model's cache; null from a server that does not say. */
  readonly promptCache: 'hit' | 'miss' | null;
}

/** What an `audio` job makes: `prompt` for sound effects and music, `tags` and `lyrics` for songs. */
export interface AudioOptions {
  readonly model: string;
  readonly prompt?: string;
  readonly tags?: string;
  readonly lyrics?: string;
  readonly negativePrompt?: string;
  readonly durationS?: number;
  readonly seed?: number;
  readonly steps?: number;
  readonly cfg?: number;
  /** `mp3` is 192 kbps CBR. */
  readonly format?: 'flac' | 'wav' | 'mp3';
  /**
   * A song model only (YuE2): write the melody, then play it on an instrument instead of
   * singing it. Lyrics become optional and only shape the sections. A server whose song model
   * cannot do it refuses by name.
   */
  readonly instrumental?: boolean;
  /**
   * With `instrumental` only, and never beside `lyrics`: words the score is planned from and
   * never sung, so the melody has a sung song's bounded phrases (sections tagged `[Verse]`,
   * `[Chorus]` and so on; at most 36 lines and 2000 characters). Left out, the server plans
   * from a set of its own pool picked by the seed, so the same params and seed plan the same
   * song. {@link AudioResult.planningLyrics} says which.
   */
  readonly planningLyrics?: string;
  /**
   * With `instrumental` only, never beside `planningLyrics` or `lyrics`: the id of the
   * server's pool set to plan from (the `planning_set` options in `GET /v1/playground`), so an
   * album gives each track its own structure. Left out, the seed picks the set.
   */
  readonly planningSet?: string;
  /**
   * A song model only (YuE2): the shortest and longest the song may be, in seconds (each 30 to
   * 360; either alone). The score is checked before anything is composed: an instrumental from
   * the server's pool is re-planned to land in the range; a song from the client's words that
   * misses fails `song_length_out_of_range` with the ratio its words need in
   * {@link JobFailure.details}. {@link AudioResult.length} says what the score measured.
   */
  readonly minDurationS?: number;
  readonly maxDurationS?: number;
}

/** One score planned for a song: its structure (when resized), its nominal seconds and whether it held. */
export interface AudioLengthAttempt {
  readonly attempt: number;
  /** Body sections of the pool set planned from; null for words not resized. */
  readonly bodySections: number | null;
  /** The section labels in order; null for words not resized. */
  readonly structure: readonly string[] | null;
  readonly lines: number | null;
  /** The score's bars at its tempo; null (with `unread`) only for a score that could not be timed. */
  readonly scoreSeconds: number | null;
  readonly unread: string | null;
  readonly scoreTokens: number;
  readonly scoreEnded: 'eos' | 'cap';
  /** Null when no range was asked. */
  readonly inRange: boolean | null;
}

/** A song's score length against the range asked (docs/AUDIO.md "Song length"). */
export interface AudioLength {
  readonly minDurationS: number | null;
  readonly maxDurationS: number | null;
  /** The nominal seconds of the score the song was composed from: its bars at its tempo. */
  readonly scoreSeconds: number | null;
  readonly inRange: boolean | null;
  readonly attempts: readonly AudioLengthAttempt[];
}

/** What an instrumental's score was planned from (never sung): the client's words or a pool set. */
export interface AudioPlanningLyrics {
  /** `request`: the client's `planningLyrics`. `pool`: the set the seed picked from the server's pool. */
  readonly source: 'pool' | 'request';
  /** The pool set's id; null for the client's own. */
  readonly id: string | null;
  /** True when the client named the set (`planningSet`), false when the seed picked it; null for the client's own words. */
  readonly requested: boolean | null;
  /** True when the set was grown or cut by whole sections to land in a length range; `lyrics` is then the text planned from. */
  readonly resized: boolean;
  /** The text, which sent back as `planningLyrics` plans the same song whatever the pool says later. */
  readonly lyrics: string;
}

/** An `audio` job's effective parameters and measurements, read by {@link readAudioResult}. */
export interface AudioResult {
  readonly model: string;
  readonly kind: 'sfx' | 'music' | 'song';
  readonly hfRepo: string;
  readonly revision: string;
  readonly backend: string;
  readonly engine: string;
  readonly dtype: string;
  readonly prompt: string | null;
  readonly tags: string | null;
  readonly lyrics: string | null;
  readonly durationS: number | null;
  readonly seed: number;
  readonly steps: number | null;
  readonly cfg: number | null;
  readonly format: 'flac' | 'wav' | 'mp3';
  /** Whether the song was made instrumental; null from a server older than `instrumental`. */
  readonly instrumental: boolean | null;
  /**
   * What an instrumental's score was planned from; null for a sung song, a sound without a
   * score, and an instrumental shaped by section tags in its `lyrics`.
   */
  readonly planningLyrics: AudioPlanningLyrics | null;
  /** A song's score length against the range asked, with every score planned; null for Stable Audio. */
  readonly length: AudioLength | null;
  /** The audio artifact's name: `audio.flac`, `audio.wav` or `audio.mp3`. */
  readonly artifact: string;
  /** `score.abc`, the song's ABC score, when the model wrote one; else null. */
  readonly score: string | null;
  readonly audioSeconds: number | null;
  readonly sampleRate: number | null;
  readonly channels: number | null;
  readonly seconds: number | null;
  readonly stageSeconds: Readonly<Record<string, number>>;
  readonly peakBytes: number | null;
  readonly stagePeakBytes: Readonly<Record<string, number>>;
  readonly memoryBytesEstimate: number;
  readonly memoryBasis: string;
  /** Whether `[audio] low_vram` held half of YuE2 on the card at a time for this render. */
  readonly lowVram: boolean;
  /**
   * Each token-decoding stage's own account, in order (`scoring`, then `composing` for YuE2);
   * null for an engine that decodes no tokens (Stable Audio).
   */
  readonly decodeStages: Readonly<Record<string, AudioDecodeStage>> | null;
  /**
   * The stages that ran to their token cap without the model ending them: `[]` when every
   * stage ended itself, null when nothing decodes tokens. The job still succeeds - the audio
   * is only longer than the model meant - and the server never re-runs it.
   */
  readonly stagesAtCap: readonly string[] | null;
  readonly artifacts: readonly string[];
}

/** One autoregressive stage of an `audio` job, as yue2-infer accounted for it. */
export interface AudioDecodeStage {
  /** Every token the model wrote, its end token included. */
  readonly tokens: number;
  /** The most the stage may write: 4096 for the score, 9000 for the song. */
  readonly cap: number;
  /** `eos`: the model ended the stage. `cap`: it ran to `cap` without ending - a runaway. */
  readonly ended: 'eos' | 'cap';
  /** `cuda_graph` or `eager`. */
  readonly execution: string;
  readonly attention: string;
  readonly lowVram: boolean;
  readonly prefixTokens: number;
  readonly cfgBranches: number;
  readonly seconds: number;
  readonly prefillSeconds: number;
  readonly tokensPerSecond: number;
}

/** One field of a playground form, as `GET /v1/playground` states it. */
export interface PlaygroundField {
  readonly name: string;
  readonly label: string;
  /** `text`, `tags`, `integer`, `number`, `boolean` or `choice`. */
  readonly kind: string;
  readonly required: boolean;
  readonly default: unknown;
  readonly placeholder: string | null;
  readonly hint: string | null;
  readonly min: number | null;
  readonly max: number | null;
  readonly step: number | null;
  readonly options: readonly unknown[] | null;
  /** A `tags` field's suggested tags, by group. */
  readonly suggestions: readonly PlaygroundTagGroup[] | null;
  /** A `tags` field's contradictions: lower-cased tag -> the tags it clashes with, each with why. */
  readonly conflicts: Readonly<Record<string, readonly PlaygroundTagConflict[]>> | null;
}

export interface PlaygroundTagGroup {
  readonly group: string;
  readonly tags: readonly string[];
}

export interface PlaygroundTagConflict {
  readonly tag: string;
  readonly why: string;
}

/** One model's playground page: its form, and whether it can run here. */
export interface PlaygroundPage {
  readonly jobType: string;
  readonly id: string;
  readonly name: string;
  readonly media: string;
  readonly kind: string;
  readonly makes: string;
  /** `ready`; `download` (its first job installs what it lacks); `unavailable` (see `reason`). */
  readonly standing: 'ready' | 'download' | 'unavailable';
  readonly available: boolean;
  readonly reason: string | null;
  readonly downloadBytes: number | null;
  readonly fields: readonly PlaygroundField[];
}

/** A named preset of a playground form, kept on the server: the form's params, never a seed. */
export interface PlaygroundPreset {
  readonly name: string;
  readonly params: Readonly<Record<string, string | number | boolean>>;
  readonly savedAt: string;
}

/** One click for a `select` model, in the input's own pixels from its top-left corner. */
export interface SegmentPoint {
  readonly x: number;
  readonly y: number;
  /** 1 keeps what is under the point; 0 leaves it out. */
  readonly label: 0 | 1;
}

/**
 * What a `segment` job cuts out of one picture. `birefnet` (class `cutout`) finds the main
 * subject by itself and takes no `points` or `box`; `sam2.1-hiera-large` (class `select`) needs
 * `points`, `box` or both.
 */
export interface SegmentOptions {
  readonly model: string;
  /** The picture: a PNG, JPEG or WebP, inline, an uploaded blob, or another job's artifact. */
  readonly image: JobInput;
  /** The input's name on the job (and in `SegmentResult.input`); default `input.png`. */
  readonly imageName?: string;
  readonly points?: readonly SegmentPoint[];
  /** `[x0, y0, x1, y1]` in input pixels, top-left corner first. */
  readonly box?: readonly [number, number, number, number];
}

/** A `segment` job's effective parameters and measurements, read by {@link readSegmentResult}. */
export interface SegmentResult {
  readonly model: string;
  readonly kind: 'cutout' | 'select';
  readonly hfRepo: string;
  readonly revision: string;
  readonly backend: string;
  readonly engine: string;
  readonly dtype: string;
  /** The input's name on the job. */
  readonly input: string;
  /** The input's size, which is also the mask's and the cutout's. */
  readonly width: number;
  readonly height: number;
  readonly points: readonly SegmentPoint[] | null;
  readonly box: readonly number[] | null;
  /** `mask.png`: 8-bit greyscale, 255 selected, 0 not. */
  readonly mask: string;
  /** `cutout.png`: the input as RGBA with the mask as its alpha. */
  readonly cutout: string;
  /** SAM's predicted IoU of the mask it returned, 0 to 1; null for a `cutout` model. */
  readonly score: number | null;
  /** True when one point and no box made SAM choose the best of three; null for a `cutout` model. */
  readonly multimask: boolean | null;
  /** The mask's mean, 0 to 1: the share of the picture selected. Near 0 means nothing was found. */
  readonly coverage: number | null;
  readonly seconds: number | null;
  readonly stageSeconds: Readonly<Record<string, number>>;
  readonly peakBytes: number | null;
  readonly stagePeakBytes: Readonly<Record<string, number>>;
  readonly memoryBytesEstimate: number;
  readonly memoryBasis: string;
  readonly artifacts: readonly string[];
}

/** What a `video` job makes: a clip with sound from `prompt`, or from `prompt` and a start `image`. */
export interface VideoOptions {
  readonly model: string;
  readonly prompt: string;
  /** Send both or neither; multiples of 32. The model's default is 1280x704. */
  readonly width?: number;
  readonly height?: number;
  /** Seconds, rounded to the model's 8k+1 frame grid. Send this or `numFrames`, not both. */
  readonly durationS?: number;
  /** An exact frame count on the 8k+1 grid (49, 97, 121, 145, …). */
  readonly numFrames?: number;
  readonly fps?: number;
  readonly seed?: number;
  /** The distilled model runs exactly 8; any other value is refused. */
  readonly steps?: number;
  /** `false` makes a silent clip. Default true. */
  readonly audio?: boolean;
  /** The first frame, for image-to-video: a PNG, JPEG or WebP. */
  readonly image?: JobInput | null;
  readonly imageName?: string;
}

/** A `video` job's effective parameters and measurements, read by {@link readVideoResult}. */
export interface VideoResult {
  readonly model: string;
  readonly hfRepo: string;
  readonly revision: string;
  /**
   * The quantized transformer that ran, when it came from its own repo (the PC's GGUF): its
   * repo, revision, file and sha256. Null on the Mac, whose transformer is in `hfRepo` itself.
   */
  readonly transformer: Readonly<Record<string, string>> | null;
  readonly backend: string;
  readonly engine: string;
  readonly dtype: string;
  readonly mode: 'text-to-video' | 'image-to-video';
  readonly prompt: string;
  /** The start picture's input name, for image-to-video; else null. */
  readonly input: string | null;
  readonly width: number;
  readonly height: number;
  readonly numFrames: number;
  readonly fps: number;
  readonly durationS: number;
  readonly videoTokens: number;
  readonly seed: number;
  readonly steps: number;
  /** The full-size refining pass's steps after the half-size pass (the Mac's arm); else null. */
  readonly refineSteps: number | null;
  readonly audio: boolean;
  readonly audioSeconds: number | null;
  readonly audioSampleRate: number | null;
  readonly audioChannels: number | null;
  /** The artifact's name, `video.mp4`. */
  readonly artifact: string;
  readonly bytes: number | null;
  /** The H.264 encoder ffmpeg used, e.g. `libopenh264`. */
  readonly encoder: string | null;
  readonly seconds: number | null;
  readonly stageSeconds: Readonly<Record<string, number>>;
  readonly peakBytes: number | null;
  readonly stagePeakBytes: Readonly<Record<string, number>>;
  readonly memoryBytesEstimate: number;
  readonly memoryBasis: string;
  readonly stageMemoryBytes: Readonly<Record<string, number>>;
  readonly promptCache: 'hit' | 'miss' | null;
  readonly artifacts: readonly string[];
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
  /**
   * Whether other devices can reach this server, as the server itself reports it; `null` from a
   * server that predates the report.
   */
  readonly network: ServerNetwork | null;
}

/** `GET /v1/setup`'s `network`: who can reach this server, and what opens it when nobody else can. */
export interface ServerNetwork {
  /** Whether a device other than this machine can reach it. */
  readonly reachable: boolean;
  /** The addresses other devices reach it on; empty when only this machine can. */
  readonly urls: readonly string[];
  /** The state, said for a person. */
  readonly sentence: string;
  /** When it is not reachable: what opens it, said for a person. */
  readonly how: string | null;
  /** The one command that opens it, when there is one (`crucible lan enable` on Windows). */
  readonly command: string | null;
  /** What opening it changes on that machine, including any administrator prompt. */
  readonly changes: string | null;
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
   * Always empty: the class floors were retired on 2026-10-09 (no manifest carries
   * `minimum_for`). The server still sends `[]` so SDKs that read it keep working.
   */
  readonly floors: readonly string[];
  /** Always null: no manifest carries a licence. */
  readonly license: string | null;
  /** `hf:<repo>` — where the bytes come from. */
  readonly source: string;
  /** Is this the thing on the card right now? */
  readonly resident: boolean;
  /**
   * For a model that comes in more than one form: the form this card takes, which is what
   * `installed`, the size and a pull are about. Null otherwise.
   */
  readonly form: string | null;
  /** Why this card takes `form`; null for a row without forms. */
  readonly formReason: string | null;
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


/** A topic of `GET /v1/events`; `server` is always sent. */
export type ServerEventTopic =
  | 'job'
  | 'queue'
  | 'session'
  | 'card'
  | 'chat'
  | 'task'
  | 'settings'
  | 'server';

/** Options for the server-wide {@link CrucibleClient.events} (no job id). */
export interface ServerEventsOptions {
  /** Only these topics; leave it out for all of them. */
  readonly topics?: readonly ServerEventTopic[];
  /** Resume after this event id: the server replays what came after it, or opens with a `gap` snapshot. */
  readonly lastEventId?: number;
  /** Ends the iteration: the stream is closed and no reconnect is attempted. */
  readonly signal?: AbortSignal;
}

/** The first event of every connection that is not a resume: what to draw before the first change. */
export interface ServerSnapshot {
  readonly id: number;
  readonly event: 'snapshot';
  /** True when a resume asked for an id the server no longer has: replace what you drew with this. */
  readonly gap: boolean;
  readonly topics: readonly string[];
  /** `GET /v1/activity`, exactly. */
  readonly activity: Activity;
  /** The waiting line, as `GET /v1/queue` lists it. */
  readonly queue: { readonly items: readonly QueueItem[]; readonly depth: number };
  /** The recent tasks, newest first, as `GET /v1/tasks` lists them. */
  readonly tasks: readonly TaskStatus[];
}

/**
 * The stream fell too far behind and the server dropped it. {@link CrucibleClient.events}
 * reconnects at once with `Last-Event-ID: lastEventId`; you lose nothing while that is still in
 * the server's history, and get a `gap` snapshot when it is not.
 */
export interface ServerOverflow {
  readonly id: number;
  readonly event: 'overflow';
  readonly lastEventId: number;
  readonly limit: number;
  readonly message: string;
}

/**
 * The server is stopping, and the stream ends after this. {@link CrucibleClient.events} waits and
 * reconnects with backoff until the server is back (or the signal aborts).
 */
export interface ServerStopping {
  /** Null on a stream other than `/v1/events`, whose stopping frame carries no id. */
  readonly id: number | null;
  readonly event: 'server.stopping';
  readonly reason: string;
  readonly at: string | null;
}

/** A job's status changed: the event is named for the status it has just entered. */
export interface JobChangeEvent {
  readonly id: number;
  readonly event:
    | 'job.queued'
    | 'job.running'
    | 'job.done'
    | 'job.failed'
    | 'job.cancelled'
    | 'job.interrupted'
    | 'job.removed';
  readonly at: string;
  readonly jobId: string;
  readonly type: string;
  readonly model: string | null;
  readonly client: string | null;
  readonly clientRef: string | null;
  readonly status: JobState;
  /** `job.queued`: its place in the line, or null when it went straight to the lane. */
  readonly position: number | null;
  /** `job.queued`: true when it waits in the line; null on the other events. */
  readonly waiting: boolean | null;
  /** `job.running`: when it started. */
  readonly started: string | null;
  /** `job.done`: the artifacts to fetch. */
  readonly artifacts: readonly string[] | null;
  /** `job.failed`: why. */
  readonly error: JobFailure | null;
  /** `job.interrupted`: when the server was found stopped under it. */
  readonly interruptedAt: string | null;
  /** `job.removed`: why it left the line without running. */
  readonly removal: RemovedData | null;
}

/** A running job's progress: at most one a second per job, and the latest always arrives. */
export interface JobProgressEvent {
  readonly id: number;
  readonly event: 'job.progress';
  readonly at: string;
  readonly jobId: string;
  readonly fraction: number;
  readonly message: string | null;
}

/** The waiting line changed, as `GET /v1/queue/events` says it. */
export interface QueueChangeEvent {
  readonly id: number;
  readonly event: 'queue.added' | 'queue.moved' | 'queue.started' | 'queue.removed' | 'queue.waiting';
  readonly at: string;
  readonly jobId: string;
  /** How many wait after this change. */
  readonly depth: number;
  readonly kind: 'job' | 'call' | 'session';
  /** `queue.added` and `queue.moved`: its place. */
  readonly position: number | null;
  /** `queue.started`: how long it waited. */
  readonly waitedS: number | null;
  /** `queue.removed`: why (`refused` for one refused at the front, with `error`). */
  readonly reason: string | null;
  /** `queue.waiting`: the item at the front waits for an accelerator held by someone else. */
  readonly waiting: { readonly code: string; readonly message: string } | null;
  /** Everything the server put on the frame (`type`, `model`, `client`, `message`, `error`, …). */
  readonly data: Readonly<Record<string, unknown>>;
}

/** A queue session changed: the same events its own stream sends, named `session.<event>`. */
export interface SessionChangeEvent {
  readonly id: number;
  readonly event:
    | 'session.queued'
    | 'session.moved'
    | 'session.opened'
    | 'session.closed'
    | 'session.removed'
    | 'session.waiting';
  readonly at: string;
  readonly sessionId: string;
  readonly client: string | null;
  readonly act: string;
  /** `session.queued` and `session.moved`: its place in the line. */
  readonly position: QueuePosition | null;
  /** `session.closed` and `session.removed`: why it ended. */
  readonly reason: string | null;
  readonly message: string | null;
  /** `session.waiting`: its opening load waits for an accelerator held by someone else. */
  readonly waiting: CardWaitData | null;
  /** Everything the server put on the frame (`items_run`, `held_s`, `model`, `error`, …). */
  readonly data: Readonly<Record<string, unknown>>;
}

/** What is resident changed. */
export interface CardEvent {
  readonly id: number;
  readonly event:
    | 'card.warming'
    | 'card.warming_ended'
    | 'card.loaded'
    | 'card.unloading'
    | 'card.unloaded';
  readonly at: string;
  /** The model, voice or other subject id. */
  readonly subject: string;
  /** `llm`, `tts`, `align`, `denoise`, `image`, `audio`, `segment` or `video`. */
  readonly kind: string | null;
  /** The engine's name; null while warming, and for an aligner or a separator. */
  readonly engine: string | null;
  /** `card.loaded`: what it is estimated to hold. */
  readonly memoryBytesEstimate: number | null;
  /** `card.loaded`, `card.unloading`, `card.unloaded`. */
  readonly since: string | null;
  /** `card.unloading` and `card.unloaded`: its processes. */
  readonly pids: readonly number[] | null;
}

/** How many chats and decisions are being answered: coalesced, at most one a second. */
export interface ChatInFlightEvent {
  readonly id: number;
  readonly event: 'chat.in_flight';
  readonly at: string;
  readonly inFlight: number;
  readonly byModel: Readonly<Record<string, number>>;
}

/** A task's state changed: the task as `GET /v1/tasks/{id}` shows it. */
export interface TaskChangeEvent {
  readonly id: number;
  readonly event: 'task.running' | 'task.done' | 'task.failed' | 'task.cancelled';
  readonly at: string;
  readonly task: TaskStatus;
}

/** A task began a step. */
export interface TaskStepEvent {
  readonly id: number;
  readonly event: 'task.step';
  readonly at: string;
  readonly taskId: string;
  readonly step: TaskStepData;
}

/** A task's progress: a line of output, or bytes of a download. */
export interface TaskProgressEvent {
  readonly id: number;
  readonly event: 'task.progress';
  readonly at: string;
  readonly taskId: string;
  readonly progress: TaskProgressData;
}

/** `PUT /v1/settings` wrote: read `GET /v1/settings` for the new values. */
export interface SettingsWrittenEvent {
  readonly id: number;
  readonly event: 'settings.written';
  readonly at: string;
  readonly act: string | null;
  readonly client: string | null;
  readonly changed: readonly string[];
}

/** An event this build does not know, carried rather than refused. */
export interface UnknownServerEvent {
  readonly id: number;
  readonly event: 'unknown';
  /** The event name the server actually sent. */
  readonly kind: string;
  readonly data: Readonly<Record<string, unknown>>;
}

/** One event from `GET /v1/events` (docs/EVENTS.md). */
export type ServerEvent =
  | ServerSnapshot
  | ServerOverflow
  | ServerStopping
  | JobChangeEvent
  | JobProgressEvent
  | QueueChangeEvent
  | SessionChangeEvent
  | CardEvent
  | ChatInFlightEvent
  | TaskChangeEvent
  | TaskStepEvent
  | TaskProgressEvent
  | SettingsWrittenEvent
  | UnknownServerEvent;
