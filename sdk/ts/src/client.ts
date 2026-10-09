import { encodeBase64 } from './base64.js';
import type { PendingPairing } from './connect.js';
import {
  ACCELERATOR_UNREADABLE,
  CAPABILITY_ROUTE_MISSING,
  CAPABILITY_ROUTE_UNKNOWN,
  CAPABILITY_UNDECIDED,
  CrucibleAcceleratorUnreadable,
  CrucibleAuthError,
  CrucibleCapabilityUndecided,
  CrucibleConfigError,
  CrucibleError,
  CrucibleNotACrucible,
  CrucibleProtocolError,
  CrucibleBusy,
  CrucibleCardHeld,
  CrucibleRefused,
  CrucibleSessionClosed,
  CrucibleSessionHeld,
  SERVER_BUSY,
  SERVER_UPDATING,
  SESSION_CLOSED,
  SESSION_OPEN,
  UNKNOWN_QUEUE_SESSION,
  CrucibleServerError,
  CrucibleUpdating,
  CrucibleUnreachable,
  CrucibleVersionError,
  UPSTREAM_TEST_REFUSALS,
  VOICES_NEEDS_REFERENCE_MISSING,
  VOICES_NEEDS_REFERENCE_UNKNOWN,
} from './errors.js';
import {
  arrayField,
  asArray,
  asObject,
  bool,
  field,
  num,
  nullableArray,
  nullableBool,
  nullableNum,
  nullableObject,
  nullableStr,
  nullableStrArray,
  objectField,
  oneOf,
  optBool,
  optNum,
  optObject,
  optStr,
  optStrArray,
  str,
  strArray,
  type Json,
} from './shape.js';
import { loadNodeBuiltins, requireFunctions } from './node-builtins.js';
import { queuePayload, requireSeconds, type QueuePayload } from './queue.js';
import { readSseFrames, type SseFrame } from './sse.js';
import { openTtsStream, type StreamOptions, type TtsStreamSession } from './stream.js';
import {
  API_VERSION,
  TASK_TERMINAL_STATES,
  TERMINAL_EVENTS,
  type AcceleratorHolder,
  type AcceleratorResident,
  type AcceleratorState,
  type Activity,
  type ActivityChat,
  type ActivityJob,
  type ActivityStreaming,
  type AlignItem,
  type Alignment,
  type AlignOptions,
  type ImageOptions,
  type ImageResult,
  type AudioOptions,
  type AudioResult,
  type SegmentOptions,
  type SegmentPoint,
  type SegmentResult,
  type VideoOptions,
  type VideoResult,
  type AlignWindowResult,
  type JobInput,
  type ArtifactHold,
  type ArtifactWrite,
  type AsrOptions,
  type CancelResult,
  type Capability,
  type CapabilityRecord,
  type CapabilityRow,
  type CapabilitySizing,
  type CapabilityWork,
  type ContextCeiling,
  type CatalogRow,
  type ChatMessage,
  type ChatOptions,
  type ChatResponse,
  type ChunkData,
  type PauseCut,
  type DecideAnswer,
  type DecideChoiceAnswer,
  type DecideChoiceQuestion,
  type DecideItemsRequest,
  type DecideItemsResponse,
  type DecideCallTiming,
  type DecideOptions,
  type DecideQuestion,
  type DecideRequest,
  type DecideResponse,
  type DoneData,
  type EngineOwner,
  type EngineRef,
  type Health,
  type JobEvent,
  type JobFailure,
  type JobRequest,
  type JobState,
  type JobStatus,
  type LoadModelOptions,
  type LoadVoiceOptions,
  type ModelDescriptor,
  type ModelInfo,
  type PagesEngine,
  type Ping,
  type PlaygroundField,
  type PlaygroundPage,
  type PlaygroundPreset,
  type ProgressData,
  type QueueChoice,
  type QueueEvent,
  type QueueItem,
  type QueueWaitingFor,
  type CardWaitData,
  type QueueList,
  type QueuePosition,
  type QueueRemoved,
  type QueueSessionEnd,
  type QueueSessionState,
  type RemovedData,
  type ServerEvent,
  type ServerEventsOptions,
  type SessionOptions,
  type Provenance,
  type RenderChunk,
  type RenderFailure,
  type RenderOptions,
  type RenderResult,
  type Resumable,
  type ResumableDiscarded,
  type RouteSetting,
  type CrucibleRole,
  type ServerInfo,
  type ServerNetwork,
  type ServerSetup,
  type SettingsDocument,
  type SettingsPatch,
  type Stopping,
  type SubjectKind,
  type TaskCancelResult,
  type TaskEvent,
  type TaskProgressData,
  type TaskRequest,
  type TaskState,
  type UnreadableRow,
  type TaskStatus,
  type UnmetNeed,
  type TaskStepData,
  type UploadResult,
  type UpstreamName,
  type UpstreamSetting,
  type UpstreamTestResult,
  type VoiceChunkGap,
  type VoiceEdgeFade,
  type VoiceInfo,
  type VoicePace,
  type VoiceSampling,
  type VoiceServing,
  type WrittenArtifact,
} from './types.js';
import { SDK_VERSION } from './version.js';

const API_HEADER = 'X-Crucible-Api';
const CLIENT_NAME_HEADER = 'X-Crucible-Client';
const JOB_STATES: readonly JobState[] = [
  'queued', 'running', 'done', 'failed', 'cancelled', 'interrupted', 'removed',
];
const SESSION_HEADER = 'X-Crucible-Session';
const QUEUE_SESSION_STATUSES = ['queued', 'open', 'closed'] as const;
const QUEUE_KINDS = ['job', 'call', 'session'] as const;
const SERVER_STOPPING = 'server.stopping';
/** The first wait before reconnecting a dropped event stream; it doubles up to the ceiling. */
const RECONNECT_FIRST_MS = 250;
const RECONNECT_CEILING_MS = 5_000;
const CHAT_ROLES = ['system', 'user', 'assistant'] as const;
const DONE_SENTINEL = '[DONE]';
const EVENT_NAMES = [
  'queued',
  'started',
  'removed',
  'warming',
  'waiting',
  'progress',
  'chunk',
  'artifact',
  'done',
  'failed',
  'cancelled',
] as const;

const DEFAULT_ARTIFACT_CONCURRENCY = 4;

/** Everything `new CrucibleClient(...)` needs. */
export interface CrucibleClientOptions {
  /** The server's base URL **without** `/v1`, e.g. `http://127.0.0.1:7100`. */
  url: string;
  /** The bearer token `crucible init` minted. */
  token: string;
  /** Who is calling. */
  clientName: string;
  /** A deadline on every call, in milliseconds; a per-call `signal` replaces it. */
  timeoutMs?: number;
  /**
   * How every request of this client that can wait (`submit` and every job helper, `chat`,
   * `chatStream`, `decide`, `decideItems`, `stream`) waits in the server's line while it is busy.
   * Left out, they wait: the server holds them up to its default (an hour). `{maxWaitS}` changes
   * the wait; `false` makes them refuse at once (`server_busy`, `model_not_resident`,
   * `chat_queue_full`, `session_open`) instead. A request's own `queue` wins.
   */
  queue?: QueueChoice;
}

/** Options for {@link CrucibleClient.events}. */
export interface EventsOptions {
  /**
   * Resume after this event id: the server replays everything with a higher id and then follows
   * live.
   */
  lastEventId?: number;
}

/** Options for {@link CrucibleClient.writeArtifactsTo}. */
export interface WriteArtifactsOptions extends EventsOptions {
  /** How many artifacts to fetch at once. */
  concurrency?: number;
}

export interface ProbeOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

function probeInit(options: ProbeOptions): RequestInit {
  const init: RequestInit = { method: 'GET' };
  const { signal, timeoutMs } = options;
  if (timeoutMs !== undefined) {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new CrucibleConfigError(
        'timeoutMs',
        `timeoutMs is ${String(timeoutMs)}; a probe's clock is a positive ` +
          'number of milliseconds. Omit it for no deadline at all.',
      );
    }
    const clock = AbortSignal.timeout(timeoutMs);
    init.signal = signal === undefined ? clock : AbortSignal.any([signal, clock]);
  } else if (signal !== undefined) {
    init.signal = signal;
  }
  return init;
}

export class CrucibleClient {
  /** The server base URL, normalised: no trailing slash, no `/v1`. */
  readonly url: string;
  /** The API contract version this client speaks. */
  readonly apiVersion = API_VERSION;
  /** This SDK's build version, as it appears in `User-Agent`. */
  readonly sdkVersion = SDK_VERSION;

  readonly #token: string;
  readonly #userAgent: string;
  readonly #clientName: string;
  readonly #timeoutMs: number | null;
  /** The client's own {@link CrucibleClientOptions.queue}; undefined waits the server's default. */
  readonly #queue: QueueChoice | undefined;
  /** What this client was made with, so a {@link session} is the same client plus its header. */
  readonly #options: CrucibleClientOptions;
  /** The queue session every request of this client is an item of; null for a plain client. */
  #session: SessionBinding | null = null;
  /** `info().features`, read once by {@link has}. */
  #features: Promise<ReadonlySet<string>> | null = null;

  constructor(options: CrucibleClientOptions) {
    const given = options as Partial<CrucibleClientOptions> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError('options', 'new CrucibleClient(...) needs {url, token, clientName}');
    }
    this.url = normaliseUrl(requireText(given.url, 'url'));
    this.#token = requireText(given.token, 'token');
    const clientName = requireText(given.clientName, 'clientName');
    this.#userAgent = `${clientName} crucible-client/${SDK_VERSION}`;
    this.#clientName = clientName;
    this.#queue = given.queue;
    queuePayload(this.#queue);
    if (given.timeoutMs !== undefined) {
      if (!Number.isFinite(given.timeoutMs) || given.timeoutMs <= 0) {
        throw new CrucibleConfigError(
          'timeoutMs',
          `timeoutMs is ${String(given.timeoutMs)}; a deadline is a positive ` +
            'number of milliseconds. Omit it to wait as long as the platform ' +
            'waits.',
        );
      }
      this.#timeoutMs = given.timeoutMs;
    } else {
      this.#timeoutMs = null;
    }
    this.#options = {
      url: this.url,
      token: this.#token,
      clientName,
      ...(this.#queue === undefined ? {} : { queue: this.#queue }),
      ...(this.#timeoutMs === null ? {} : { timeoutMs: this.#timeoutMs }),
    };
  }

  /** `GET /v1/ping`, unauthenticated. */
  async ping(options: ProbeOptions = {}): Promise<Ping> {
    const response = await this.#fetch('/v1/ping', probeInit(options), false);
    const text = await response.text();
    if (!response.ok) {
      throw new CrucibleNotACrucible(this.url, `HTTP ${response.status}: ${excerpt(text)}`);
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new CrucibleNotACrucible(this.url, excerpt(text));
    }
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      throw new CrucibleNotACrucible(this.url, excerpt(text));
    }
    const body = parsed as Json;
    if (body['crucible'] !== true) {
      throw new CrucibleNotACrucible(this.url, excerpt(text));
    }
    return {
      crucible: true,
      name: str(body, 'name', 'ping'),
      apiVersion: num(body, 'api_version', 'ping'),
    };
  }

  /** `GET /v1/info` — who this server is, what it runs on, what it can serve. */
  async info(options: ProbeOptions = {}): Promise<ServerInfo> {
    const body = await this.#json('/v1/info', probeInit(options), 'info');
    const server = objectField(body, 'server', 'info');
    const host = objectField(body, 'host', 'info');
    const gpu = objectField(host, 'gpu', 'info.host');
    const capabilities = asArray(field(body, 'capabilities', 'info'), 'info.capabilities').map(
      (entry, index) => asObject(entry, `info.capabilities[${index}]`),
    );
    const role = readRole(body);
    return {
      server: {
        name: str(server, 'name', 'info.server'),
        version: str(server, 'version', 'info.server'),
        apiVersion: num(server, 'api_version', 'info.server'),
      },
      host: {
        platform: str(host, 'platform', 'info.host'),
        arch: str(host, 'arch', 'info.host'),
        backend: str(host, 'backend', 'info.host'),
        gpu: {
          vendor: str(gpu, 'vendor', 'info.host.gpu'),
          name: str(gpu, 'name', 'info.host.gpu'),
          vramBytes: num(gpu, 'vram_bytes', 'info.host.gpu'),
        },
      },
      jobTypes: strArray(body, 'job_types', 'info'),
      features: strArray(body, 'features', 'info'),
      capabilities: capabilities.map((entry, index) => readCapability(entry, index)),
      ...role,
      pagesEngine: readPagesEngine(body, role.role),
    };
  }

  /** `GET /v1/capability` — what this server can hold, per capability class, and why not. */
  async capability(
    options: ProbeOptions = {},
    sizing?: CapabilitySizing,
  ): Promise<CapabilityRecord> {
    const body = await this.#json(
      '/v1/capability' + capabilityQuery(sizing),
      probeInit(options),
      'capability',
    );
    return readCapabilityRecord(body);
  }

  /** `GET /v1/settings` — where each class's work runs, and which upstreams are configured. */
  async settings(): Promise<SettingsDocument> {
    const body = await this.#json('/v1/settings', { method: 'GET' }, 'settings');
    return readSettings(body);
  }

  /** Pending connection approvals, visible only to an already trusted app. */
  async listPairingRequests(): Promise<PendingPairing[]> {
    const body = await this.#json('/v1/pairing/requests', { method: 'GET' }, 'listPairingRequests');
    return asArray(field(body, 'requests', 'pairing'), 'pairing.requests').map((value, index) => {
      const where = `pairing.requests[${index}]`;
      const row = asObject(value, where);
      return {
        id: str(row, 'id', where),
        userCode: str(row, 'user_code', where),
        clientName: str(row, 'client_name', where),
        address: str(row, 'address', where),
        expiresIn: num(row, 'expires_in', where),
      };
    });
  }

  /** Approve or deny the request whose displayed code the user has checked. */
  async decidePairing(id: string, userCode: string, allow: boolean): Promise<{ status: 'approved' | 'denied' }> {
    const body = await this.#json('/v1/pairing/decision', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, user_code: userCode, allow }),
    }, 'decidePairing');
    return { status: oneOf(str(body, 'status', 'pairing'), ['approved', 'denied'] as const, 'pairing.status') };
  }

  /** `PUT /v1/settings` — a PARTIAL patch, applied whole or not at all. */
  async putSettings(patch: SettingsPatch): Promise<SettingsDocument> {
    const body = await this.#json(
      '/v1/settings',
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settingsPayload(patch)),
      },
      'putSettings',
    );
    return readSettings(body);
  }

  /** `POST /v1/settings/upstreams/{name}/test` — ask an upstream what it serves. */
  async testUpstream(
    name: UpstreamName,
    probe?: { key?: string; url?: string },
  ): Promise<UpstreamTestResult> {
    const path = `/v1/settings/upstreams/${encodeURIComponent(name)}/test`;
    try {
      const body = await this.#json(
        path,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(probe ?? {}),
        },
        'testUpstream',
      );
      return { ok: true, models: strArray(body, 'models', 'testUpstream') };
    } catch (error) {
      const code = upstreamTestRefusal(error);
      if (code === null) throw error;
      const said = (error as { serverMessage?: unknown }).serverMessage;
      return {
        ok: false,
        code,
        message:
          typeof said === 'string' && said !== ''
            ? said
            : (error as CrucibleError).message,
      };
    }
  }

  /** `GET /v1/health` — is the lane free, and how deep is the queue. */
  async health(): Promise<Health> {
    const body = await this.#json('/v1/health', { method: 'GET' }, 'health');
    return {
      status: str(body, 'status', 'health'),
      queueDepth: num(body, 'queue_depth', 'health'),
      residentModels: strArray(body, 'resident_models', 'health'),
      residentKind: nullableStr(body, 'resident_kind', 'health'),
      stopping: readStopping(body, 'health'),
    };
  }

  /** `GET /v1/activity` — what is on this server and how far along, in one read. */
  async activity(options?: { acceleratorProbe?: boolean }): Promise<Activity> {
    const probe = options?.acceleratorProbe === true;
    const path = probe ? '/v1/activity?accelerator_probe=true' : '/v1/activity';
    return readActivity(await this.#json(path, { method: 'GET' }, 'activity'));
  }

  /** Whether this server's `GET /v1/info` lists `feature` (`queue.sessions`, `events`, …); read once. */
  async has(feature: string): Promise<boolean> {
    const name = requireText(feature, 'feature');
    let features = this.#features;
    if (features === null) {
      const reading = this.info().then((info) => new Set(info.features) as ReadonlySet<string>);
      features = reading;
      this.#features = reading;
      // A failed read is not remembered: the next call asks again.
      reading.catch(() => {
        if (this.#features === reading) this.#features = null;
      });
    }
    return (await features).has(name);
  }

  /** `POST /v1/uploads` — park bytes on the server and get a blob id to name as a job input. */
  async upload(data: Uint8Array | Blob, options: { filename: string }): Promise<UploadResult> {
    const filename = requireText(options?.filename, 'filename');
    const blob = data instanceof Uint8Array ? new Blob([data]) : data;
    const form = new FormData();
    form.append('file', blob, filename);
    const body = await this.#json('/v1/uploads', { method: 'POST', body: form }, 'upload');
    return {
      blobId: str(body, 'blob_id', 'upload'),
      bytes: num(body, 'bytes', 'upload'),
      sha256: str(body, 'sha256', 'upload'),
    };
  }

  /**
   * `POST /v1/jobs` — submit a job. A busy server holds it in its line (status `queued`) unless
   * the request's `queue` (else the client's) says `false`. Inside a {@link session} only the
   * request's own `queue` is sent: a session's item waits ahead of the line, up to a day.
   */
  async submit(request: JobRequest, options: { signal?: AbortSignal } = {}): Promise<string> {
    const type = requireText(request?.type, 'type');
    const inputs: Record<
      string,
      { blob_id: string } | { inline_base64: string } | { artifact: { job_id: string; name: string } }
    > = {};
    for (const [name, input] of Object.entries(request.inputs)) {
      if ('artifact' in input) {
        inputs[name] = {
          artifact: {
            job_id: requireText(input.artifact?.jobId, `inputs[${name}].artifact.jobId`),
            name: requireText(input.artifact?.name, `inputs[${name}].artifact.name`),
          },
        };
      } else if ('blobId' in input) {
        inputs[name] = { blob_id: requireText(input.blobId, `inputs[${name}].blobId`) };
      } else if ('inline' in input) {
        if (!(input.inline instanceof Uint8Array)) {
          throw new CrucibleConfigError(
            `inputs[${name}].inline`,
            'must be a Uint8Array of the bytes to send',
          );
        }
        inputs[name] = { inline_base64: encodeBase64(input.inline) };
      } else {
        throw new CrucibleConfigError(
          `inputs[${name}]`,
          'must be one of {blobId}, {inline} or {artifact}; it is none of them',
        );
      }
    }

    const payload: Record<string, unknown> = {
      type,
      params: request.params,
      inputs,
    };
    if (request.model !== undefined) payload['model'] = request.model;
    if (request.clientRef !== undefined) payload['client_ref'] = request.clientRef;
    if (request.hold !== undefined) payload['hold'] = requireBool(request.hold, 'hold');
    const choice =
      request.queue !== undefined ? request.queue : this.#session === null ? this.#queue : undefined;
    const queue = queuePayload(choice);
    if (queue !== null) payload['queue'] = queue;

    const init: RequestInit = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    };
    if (options.signal !== undefined) init.signal = options.signal;
    const body = await this.#json('/v1/jobs', init, 'submit');
    return str(body, 'job_id', 'submit');
  }

  /** `GET /v1/queue` — the jobs waiting for the lane, in the order they will get it. */
  async queue(): Promise<QueueList> {
    const body = await this.#json('/v1/queue', { method: 'GET' }, 'queue');
    const limits = objectField(body, 'limits', 'queue');
    const wait = objectField(limits, 'max_wait_s', 'queue.limits');
    return {
      items: arrayField(body, 'items', 'queue').map((row, index) =>
        readQueueItem(asObject(row, `queue.items[${index}]`), `queue.items[${index}]`),
      ),
      depth: num(body, 'depth', 'queue'),
      limits: {
        perClient: num(limits, 'per_client', 'queue.limits'),
        total: num(limits, 'total', 'queue.limits'),
        maxWaitS: {
          default: num(wait, 'default', 'queue.limits.max_wait_s'),
          min: num(wait, 'min', 'queue.limits.max_wait_s'),
          max: num(wait, 'max', 'queue.limits.max_wait_s'),
        },
        abandonAfterS: num(limits, 'abandon_after_s', 'queue.limits'),
      },
    };
  }

  /**
   * `DELETE /v1/queue/sessions/{id}` — close a session this client opened, by its id (reason
   * `client`). For an app that restarted and recorded the id of a session it no longer holds a
   * {@link CrucibleSession} for; a live session closes with its own `close()`. Closing one that
   * is already closed answers its state again.
   */
  async closeSession(sessionId: string): Promise<QueueSessionState> {
    const id = requireText(sessionId, 'sessionId');
    const body = await this.#json(
      `/v1/queue/sessions/${encodeURIComponent(id)}`,
      { method: 'DELETE' },
      'closeSession',
    );
    return readQueueSession(body, 'closeSession');
  }

  /**
   * `DELETE /v1/queue/{id}` — take a waiting job, call or session out of the queue (reason
   * `operator`), or, given the open queue session's id, end it (`status: 'closed'`). A job that
   * has started is cancelled with {@link cancel} instead.
   */
  async removeFromQueue(jobId: string): Promise<QueueRemoved> {
    const id = requireText(jobId, 'jobId');
    const body = await this.#json(
      `/v1/queue/${encodeURIComponent(id)}`,
      { method: 'DELETE' },
      'removeFromQueue',
    );
    return {
      jobId: str(body, 'job_id', 'removeFromQueue'),
      status: oneOf(
        str(body, 'status', 'removeFromQueue'),
        ['removed', 'closed'] as const,
        'removeFromQueue.status',
      ),
      reason: oneOf(str(body, 'reason', 'removeFromQueue'), ['operator'], 'removeFromQueue.reason'),
    };
  }

  /**
   * `POST /v1/queue/{id}/heartbeat` — I am still waiting for this job. Only needed while you
   * neither follow its {@link events} (or another job's) nor poll {@link job}: the server removes
   * a queued job nobody has asked about for five minutes.
   */
  async queueHeartbeat(jobId: string): Promise<{ position: number | null; expiresAt: string }> {
    const id = requireText(jobId, 'jobId');
    const body = await this.#json(
      `/v1/queue/${encodeURIComponent(id)}/heartbeat`,
      { method: 'POST' },
      'queueHeartbeat',
    );
    return {
      position: nullableNum(body, 'position', 'queueHeartbeat'),
      expiresAt: str(body, 'expires_at', 'queueHeartbeat'),
    };
  }

  /**
   * `GET /v1/queue/events` — every change to the server's queue, for a dashboard: a `snapshot`
   * first, then `added`, `moved`, `started` and `removed`. It never ends by itself; break out of
   * the loop to stop.
   */
  async *queueEvents(): AsyncGenerator<QueueEvent, void, undefined> {
    yield* this.#follow('/v1/queue/events', 'the queue', {}, readQueueEvent, []);
  }

  /** `GET /v1/jobs/{id}`. */
  async job(jobId: string): Promise<JobStatus> {
    const id = requireText(jobId, 'jobId');
    const body = await this.#json(`/v1/jobs/${encodeURIComponent(id)}`, { method: 'GET' }, 'job');
    return {
      jobId: str(body, 'job_id', 'job'),
      type: str(body, 'type', 'job'),
      model: nullableStr(body, 'model', 'job'),
      status: oneOf(str(body, 'status', 'job'), JOB_STATES, 'job.status'),
      progress: num(body, 'progress', 'job'),
      position: nullableNum(body, 'position', 'job'),
      error: readFailureOrNull(nullableObject(body, 'error', 'job'), 'job.error'),
      artifacts: strArray(body, 'artifacts', 'job'),
      created: str(body, 'created', 'job'),
      started: nullableStr(body, 'started', 'job'),
      finished: nullableStr(body, 'finished', 'job'),
      clientRef: nullableStr(body, 'client_ref', 'job'),
      interruptedAt: nullableStr(body, 'interrupted_at', 'job'),
      heldBy: nullableStr(body, 'held_by', 'job'),
      heldSince: nullableStr(body, 'held_since', 'job'),
      chunksDone: readChunksDone(body),
      chunksTotal: nullableNum(body, 'chunks_total', 'job'),
      chunkAt: nullableStr(body, 'chunk_at', 'job'),
      resumeId: nullableStr(body, 'resume_id', 'job'),
      resumed: bool(body, 'resumed', 'job'),
      removal: readRemovalOrNull(optObject(body, 'removal', 'job'), 'job.removal'),
    };
  }

  /** `GET /v1/resumable` — every resume journal the server keeps, newest first. */
  async resumable(): Promise<Resumable[]> {
    const body = await this.#json('/v1/resumable', { method: 'GET' }, 'resumable');
    const rows = asArray(field(body, 'resumable', 'resumable'), 'resumable');
    return rows.map((row, index) => readResumable(asObject(row, `resumable[${index}]`)));
  }

  /** `GET /v1/resumable/{id}` — one journal. */
  async resumableEntry(resumeId: string): Promise<Resumable> {
    const id = requireText(resumeId, 'resumeId');
    const body = await this.#json(
      `/v1/resumable/${encodeURIComponent(id)}`,
      { method: 'GET' },
      'resumable',
    );
    return readResumable(body);
  }

  /** `DELETE /v1/resumable/{id}` — discard a journal and its finished work. */
  async discardResumable(resumeId: string): Promise<ResumableDiscarded> {
    const id = requireText(resumeId, 'resumeId');
    const body = await this.#json(
      `/v1/resumable/${encodeURIComponent(id)}`,
      { method: 'DELETE' },
      'resumable',
    );
    return {
      resumeId: str(body, 'resume_id', 'resumable'),
      discarded: bool(body, 'discarded', 'resumable'),
      unitsDone: num(body, 'units_done', 'resumable'),
      unitsTotal: nullableNum(body, 'units_total', 'resumable'),
    };
  }

  /** `GET /v1/jobs/{id}/events` — the job's SSE stream, as typed events. */
  events(jobId: string, options?: EventsOptions): AsyncGenerator<JobEvent, void, undefined>;
  /**
   * `GET /v1/events` — every change on the server as typed events, instead of polling
   * `/v1/activity`, `/v1/tasks`, `/v1/queue` or `/v1/health`. A `snapshot` comes first (and again,
   * with `gap: true`, when a resume asks for history the server no longer has), then one event per
   * change. A dropped connection is reconnected with `Last-Event-ID`; `overflow` is yielded and
   * reconnected at once; `server.stopping` is yielded, then the stream is reconnected with backoff
   * until the server is back. It ends only when `signal` aborts or you break out of the loop.
   */
  events(options?: ServerEventsOptions): AsyncGenerator<ServerEvent, void, undefined>;
  events(
    target?: string | ServerEventsOptions,
    options: EventsOptions = {},
  ): AsyncGenerator<JobEvent, void, undefined> | AsyncGenerator<ServerEvent, void, undefined> {
    if (typeof target === 'string') return this.#jobEvents(target, options);
    return this.#serverEvents(target === undefined ? {} : target);
  }

  async *#jobEvents(jobId: string, options: EventsOptions): AsyncGenerator<JobEvent, void, undefined> {
    const id = requireText(jobId, 'jobId');
    yield* this.#follow(
      `/v1/jobs/${encodeURIComponent(id)}/events`,
      `job ${id}`,
      options,
      readEvent,
      TERMINAL_EVENTS,
    );
  }

  async *#follow<Event extends { readonly id: number; readonly event: string }>(
    path: string,
    what: string,
    options: EventsOptions,
    read: (rawId: string | null, rawName: string | null, rawData: string) => Event,
    terminal: readonly string[],
  ): AsyncGenerator<Event, void, undefined> {
    const start = readLastEventId(options.lastEventId);
    const stream = await this.#openStream(path, start, undefined, what);

    let previousId = start === null ? 0 : start;
    try {
      for await (const frame of readSseFrames(stream)) {
        if (frame.event === SERVER_STOPPING) {
          throw new CrucibleUnreachable(
            this.url,
            `the server is stopping (${readStopReason(frame.data)}), so the event stream for ` +
              `${what} ended after event ${previousId}. Follow it again with lastEventId ` +
              `${previousId} once the server is back`,
          );
        }
        const event = read(frame.lastEventId, frame.event, frame.data);
        if (event.id <= previousId) {
          throw new CrucibleProtocolError(
            `event id ${event.id} does not follow ${previousId} on ${what}`,
          );
        }
        previousId = event.id;
        yield event;
        if (terminal.includes(event.event)) return;
      }
    } catch (cause) {
      if (cause instanceof CrucibleError) throw cause;
      // The connection broke mid-read - WebKit's "Load failed" when a phone locks, undici's
      // "terminated". That is the same weather a failed connect is, so it is raised the same
      // way: a caller that follows again on CrucibleUnreachable (with lastEventId) recovers
      // (B-Side on an iPhone, 2026-10-04).
      throw new CrucibleUnreachable(
        this.url,
        `the event stream for ${what} broke after event ${previousId} (${describeCause(cause)}). ` +
          `Follow it again with lastEventId ${previousId}`,
        cause,
      );
    } finally {
      await stream.cancel().catch(() => undefined);
    }

    throw new CrucibleUnreachable(
      this.url,
      `the event stream for ${what} ended after event ${previousId} without a ` +
        'terminal event (done, failed or cancelled)',
    );
  }

  /** One connection to an SSE route, resuming after `lastEventId` when there is one. */
  async #openStream(
    path: string,
    lastEventId: number | null,
    signal: AbortSignal | undefined,
    what: string,
  ): Promise<ReadableStream<Uint8Array>> {
    const headers: Record<string, string> = { Accept: 'text/event-stream' };
    if (lastEventId !== null) headers['Last-Event-ID'] = String(lastEventId);
    const init: RequestInit = { method: 'GET', headers };
    if (signal !== undefined) init.signal = signal;
    const response = await this.#fetch(path, init, true);
    if (!response.ok) throw await this.#failure(response);
    const stream = response.body;
    if (stream === null) {
      throw new CrucibleProtocolError(`the event stream for ${what} carried no body`);
    }
    return stream;
  }

  async *#serverEvents(options: ServerEventsOptions): AsyncGenerator<ServerEvent, void, undefined> {
    const path = `/v1/events${topicsQuery(options.topics)}`;
    let cursor = readLastEventId(options.lastEventId);
    // A signal of our own when the caller gave none: an SSE stream must never meet timeoutMs.
    const signal = options.signal === undefined ? new AbortController().signal : options.signal;
    const backoff = new Backoff();
    for (;;) {
      if (signal.aborted) return;
      let stream: ReadableStream<Uint8Array>;
      try {
        stream = await this.#openStream(path, cursor, signal, 'the server');
      } catch (error) {
        if (signal.aborted) return;
        if (!isWeather(error)) throw error;
        if (!(await backoff.wait(signal))) return;
        continue;
      }
      let overflowed = false;
      try {
        for await (const frame of readSseFrames(stream)) {
          const event = readServerEvent(frame);
          if (event.event === 'overflow') {
            cursor = event.lastEventId;
            overflowed = true;
            yield event;
            break;
          }
          if (event.id !== null) cursor = event.id;
          yield event;
          if (event.event === SERVER_STOPPING) break;
          backoff.reset();
        }
      } catch (error) {
        if (signal.aborted) return;
        if (error instanceof CrucibleError) throw error;
        // The connection broke mid-stream: weather. Reconnect below from the last id.
      } finally {
        await stream.cancel().catch(() => undefined);
      }
      if (signal.aborted) return;
      if (!overflowed && !(await backoff.wait(signal))) return;
    }
  }

  /**
   * `POST /v1/queue/sessions` — this app's turn holding the server, for a run of requests it
   * cannot know in advance. Resolves once the session is OPEN: while it waits in the line its own
   * stream is followed (which keeps it present) and `onQueue` hears every move. A session that ends
   * before it opens (`expired`, `operator`, `load_failed`, the server stopping) throws
   * {@link CrucibleSessionClosed} with its `reason`; aborting `signal` takes it out of the line.
   *
   * The returned {@link CrucibleSession} is this client plus the session's header: every method
   * sends `X-Crucible-Session`. Close it when the run is done:
   *
   * ```ts
   * const session = await crucible.session({ act: 'analysis', model: 'qwen3.5-9b' });
   * try { ... } finally { await session.close(); }
   * ```
   */
  async session(options: SessionOptions): Promise<CrucibleSession> {
    const bound = this.#session;
    if (bound !== null) {
      throw new CrucibleConfigError(
        'session',
        `this client is queue session ${bound.id} already; open another session from the ` +
          'client it came from, after this one closes (one is open at a time)',
      );
    }
    const given = options as Partial<SessionOptions> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError('options', 'session(...) needs {act}');
    }
    const act = requireText(given.act, 'act');
    const payload: Record<string, unknown> = { act };
    if (given.model !== undefined) payload['model'] = requireText(given.model, 'model');
    if (given.idleS !== undefined) payload['idle_s'] = requireSeconds(given.idleS, 'idleS', '300 s');
    if (given.maxWaitS !== undefined) {
      payload['max_wait_s'] = requireSeconds(given.maxWaitS, 'maxWaitS', 'an hour');
    }
    const onQueue = given.onQueue;
    if (onQueue !== undefined && typeof onQueue !== 'function') {
      throw new CrucibleConfigError('onQueue', `must be a function, got ${typeof onQueue}`);
    }
    const onWaiting = given.onWaiting;
    if (onWaiting !== undefined && typeof onWaiting !== 'function') {
      throw new CrucibleConfigError('onWaiting', `must be a function, got ${typeof onWaiting}`);
    }
    const signal = given.signal;
    signal?.throwIfAborted();

    const ticket = await this.#askForSession(payload, signal);
    const id = str(ticket, 'session_id', 'session');
    const status = oneOf(str(ticket, 'status', 'session'), QUEUE_SESSION_STATUSES, 'session.status');
    if (signal !== undefined && signal.aborted) {
      // Aborted the instant the ticket arrived: the session is not wanted, open or not.
      await this.#leaveTheLine(id);
      throw signal.reason;
    }
    const cursor = status === 'open' ? 0 : await this.#untilOpen(id, { onQueue, onWaiting }, signal);
    return this.#bind(id, act, cursor);
  }

  /**
   * `POST /v1/queue/sessions`, abortable without leaving anything behind. The request itself is
   * never cut off: the server may already have made the session (and opened it, on an idle
   * server), and only its ticket names it. So an abort rejects at once, and when the ticket
   * arrives anyway the session it names is taken out of the line (or closed, if it opened).
   */
  async #askForSession(payload: Record<string, unknown>, signal: AbortSignal | undefined): Promise<Json> {
    const asking = this.#json(
      '/v1/queue/sessions',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      },
      'session',
    );
    if (signal === undefined) return asking;
    return new Promise<Json>((resolve, reject) => {
      const abandon = (): void => {
        reject(signal.reason);
        void asking
          .then((ticket) => this.#leaveTheLine(str(ticket, 'session_id', 'session')))
          .catch(() => undefined);
      };
      signal.addEventListener('abort', abandon, { once: true });
      asking.then(
        (ticket) => {
          signal.removeEventListener('abort', abandon);
          resolve(ticket);
        },
        (error: unknown) => {
          signal.removeEventListener('abort', abandon);
          reject(error);
        },
      );
    });
  }

  /** Follow a session waiting in the line until it opens; the id of its `opened` event. */
  async #untilOpen(
    sessionId: string,
    watch: LineWatch,
    signal: AbortSignal | undefined,
  ): Promise<number> {
    const following = signal === undefined ? new AbortController().signal : signal;
    let settled = false;
    try {
      for await (const event of this.#sessionFeed(sessionId, 0, following)) {
        switch (event.kind) {
          case 'queued':
          case 'moved':
            watch.onQueue?.(event.position);
            break;
          case 'waiting':
            watch.onWaiting?.(event.waiting);
            break;
          case 'opened':
            settled = true;
            return event.id;
          case 'removed':
            settled = true;
            throw sessionClosed(sessionId, event.reason, event.message, event.error);
          case 'closed':
            settled = true;
            throw sessionClosed(sessionId, event.end.reason, event.end.message, null);
          case 'stopping':
            settled = true;
            throw sessionClosed(
              sessionId,
              'server_restart',
              `the server stopped (${event.reason}) while queue session ${sessionId} waited in ` +
                'its line; ask again once it is back',
              null,
            );
          case 'forgotten':
            settled = true;
            throw sessionClosed(sessionId, 'server_restart', event.message, null);
          case 'unknown':
            break;
        }
      }
      // The feed ends without saying how only when the wait was aborted.
      throw following.reason;
    } finally {
      if (!settled) await this.#leaveTheLine(sessionId);
    }
  }

  /** Take a session that will not be waited for out of the line (reason `client`). */
  async #leaveTheLine(sessionId: string): Promise<void> {
    try {
      await this.#json(
        `/v1/queue/sessions/${encodeURIComponent(sessionId)}`,
        { method: 'DELETE' },
        'session.close',
      );
    } catch {
      // The reconciler owns this one: the server removes a waiting session nobody follows
      // (`expired`) within five minutes, and the error that ended the wait is the one to throw.
    }
  }

  /** The client a session's items go through: this one, plus the session's header. */
  #bind(sessionId: string, act: string, cursor: number): CrucibleSession {
    const watch = new AbortController();
    let resolve: (end: QueueSessionEnd) => void = () => undefined;
    const closed = new Promise<QueueSessionEnd>((settle) => {
      resolve = settle;
    });
    const binding: SessionBinding = {
      id: sessionId,
      end: null,
      watchFailure: null,
      settle(end: QueueSessionEnd): void {
        if (binding.end !== null) return;
        binding.end = end;
        resolve(end);
        watch.abort();
      },
    };
    const session = new CrucibleSession(
      this.#options,
      sessionId,
      act,
      closed,
      this.#sessionHooks(binding),
    );
    session.#session = binding;
    void this.#watch(binding, cursor, watch.signal);
    return session;
  }

  /** A session's own routes, sent as its owner: no session header, never refused locally as ended. */
  #sessionHooks(binding: SessionBinding): SessionHooks {
    const path = `/v1/queue/sessions/${encodeURIComponent(binding.id)}`;
    return {
      binding,
      touch: async (): Promise<void> => {
        if (binding.end !== null) throw endedSession(binding.id, binding.end);
        try {
          await this.#json(`${path}/touch`, { method: 'POST' }, 'session.touch');
        } catch (error) {
          noteSessionEnd(binding, error);
          throw error;
        }
      },
      state: async (): Promise<QueueSessionState> => {
        const state = readQueueSession(
          await this.#json(path, { method: 'GET' }, 'session.state'),
          'session.state',
        );
        if (state.status === 'closed') binding.settle(endOf(state, 'session.state'));
        return state;
      },
      close: async (): Promise<QueueSessionEnd> => {
        if (binding.end !== null) return binding.end;
        try {
          const state = readQueueSession(
            await this.#json(path, { method: 'DELETE' }, 'session.close'),
            'session.close',
          );
          binding.settle(endOf(state, 'session.close'));
        } catch (error) {
          if (!(error instanceof CrucibleRefused) || error.code !== UNKNOWN_QUEUE_SESSION) throw error;
          binding.settle({
            reason: 'server_restart',
            message: error.serverMessage,
            itemsRun: null,
            heldS: null,
          });
        }
        return settledEnd(binding);
      },
    };
  }

  /** Follow an open session's stream in the background until it says the session ended. */
  async #watch(binding: SessionBinding, cursor: number, signal: AbortSignal): Promise<void> {
    try {
      for await (const event of this.#sessionFeed(binding.id, cursor, signal)) {
        if (event.kind === 'closed') {
          binding.settle(event.end);
        } else if (event.kind === 'removed') {
          binding.settle({ reason: event.reason, message: event.message, itemsRun: null, heldS: null });
        } else if (event.kind === 'stopping') {
          binding.settle({
            reason: 'server_restart',
            message:
              `the server is stopping (${event.reason}); a stopping server closes the session ` +
              'it holds. Open a new session once it is back',
            itemsRun: null,
            heldS: null,
          });
        } else if (event.kind === 'forgotten') {
          binding.settle({ reason: 'server_restart', message: event.message, itemsRun: null, heldS: null });
        }
      }
    } catch (error) {
      // Not weather (that is reconnected inside the feed): a refusal or a malformed frame. The
      // session's items meet the same fault by name; this records why `closed` can no longer
      // resolve by itself.
      binding.watchFailure = error;
    }
  }

  /**
   * A queue session's own stream (`GET /v1/queue/sessions/{id}/events`) after `cursor`,
   * reconnected with `Last-Event-ID` after any drop, until it says how the session ended or
   * `signal` aborts.
   */
  async *#sessionFeed(
    sessionId: string,
    cursor: number,
    signal: AbortSignal,
  ): AsyncGenerator<SessionFeedEvent, void, undefined> {
    const path = `/v1/queue/sessions/${encodeURIComponent(sessionId)}/events`;
    const what = `queue session ${sessionId}`;
    const backoff = new Backoff();
    let delivered = cursor;
    for (;;) {
      if (signal.aborted) return;
      let stream: ReadableStream<Uint8Array>;
      try {
        stream = await this.#openStream(path, delivered === 0 ? null : delivered, signal, what);
      } catch (error) {
        if (signal.aborted) return;
        if (error instanceof CrucibleRefused && error.code === UNKNOWN_QUEUE_SESSION) {
          yield { id: delivered, kind: 'forgotten', message: error.serverMessage };
          return;
        }
        if (!isWeather(error)) throw error;
        if (!(await backoff.wait(signal))) return;
        continue;
      }
      try {
        for await (const frame of readSseFrames(stream)) {
          const event = readSessionFrame(frame, what, delivered);
          if (event.kind !== 'stopping') delivered = event.id;
          backoff.reset();
          yield event;
          if (event.kind === 'closed' || event.kind === 'removed' || event.kind === 'stopping') {
            return;
          }
        }
      } catch (error) {
        if (signal.aborted) return;
        if (error instanceof CrucibleError) throw error;
        // The connection broke mid-stream: weather. Reconnect below from the last id.
      } finally {
        await stream.cancel().catch(() => undefined);
      }
      if (!(await backoff.wait(signal))) return;
    }
  }

  /** `GET /v1/jobs/{id}/artifacts/{name}` — the artifact's bytes. */
  async artifact(jobId: string, name: string): Promise<Uint8Array> {
    const id = requireText(jobId, 'jobId');
    const member = requireText(name, 'name');
    const response = await this.#fetch(
      `/v1/jobs/${encodeURIComponent(id)}/artifacts/${encodeURIComponent(member)}`,
      { method: 'GET' },
      true,
    );
    if (!response.ok) throw await this.#failure(response);
    return new Uint8Array(await response.arrayBuffer());
  }

  /** The artifact's provenance sidecar, parsed. */
  async provenance(jobId: string, name: string): Promise<Provenance> {
    const member = requireText(name, 'name');
    const bytes = await this.artifact(jobId, `${member}.provenance.json`);
    const text = new TextDecoder('utf-8').decode(bytes);
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new CrucibleProtocolError(
        `${member}.provenance.json is not JSON: ${excerpt(text)}`,
      );
    }
    return readProvenance(asObject(parsed, `${member}.provenance.json`), member);
  }

  /** `DELETE /v1/jobs/{id}` — cancel. */
  async cancel(jobId: string): Promise<CancelResult> {
    const id = requireText(jobId, 'jobId');
    const body = await this.#json(
      `/v1/jobs/${encodeURIComponent(id)}`,
      { method: 'DELETE' },
      'cancel',
    );
    return {
      jobId: str(body, 'job_id', 'cancel'),
      status: oneOf(
        str(body, 'status', 'cancel'),
        ['cancelled', 'cancelling', 'removed'],
        'cancel.status',
      ),
    };
  }

  /**
   * `GET /v1/models` — every model this server has a manifest for, with support, installation,
   * residency and loadability.
   */
  async models(): Promise<ModelInfo[]> {
    const body = await this.#jsonValue('/v1/models', { method: 'GET' }, 'models');
    const entries = asArray(body, 'models');
    return entries.map((entry, index) =>
      readModelInfo(asObject(entry, `models[${index}]`), `models[${index}]`),
    );
  }

  /** Queue a `load-model` job and return its id. */
  async loadModel(model: string, options?: LoadModelOptions): Promise<string> {
    return this.submit({
      type: 'load-model',
      model: requireText(model, 'model'),
      params: options?.context === undefined ? {} : { context: options.context },
      inputs: {},
    });
  }

  /** Queue an `unload-model` job and return its id. */
  async unloadModel(model: string): Promise<string> {
    return this.submit({
      type: 'unload-model',
      model: requireText(model, 'model'),
      params: {},
      inputs: {},
    });
  }

  /** `POST /v1/openai/chat/completions` — one completion from the resident model. */
  async chat(options: ChatOptions): Promise<ChatResponse> {
    const init = this.#chatRequest(options, false);
    const body = await this.#json('/v1/openai/chat/completions', init, 'chat');
    return readChatResponse(body);
  }

  /**
   * The `queue` member a chat or decision sends: its own choice, else the client's, else none (the
   * server holds it in its line). Inside a session too: a session's call that must wait (its model
   * not resident, every slot taken) waits ahead of the line.
   */
  #callQueue(choice: QueueChoice | undefined): QueuePayload | null {
    return queuePayload(choice === undefined ? this.#queue : choice);
  }

  /** `POST /v1/decide` with the client's (or the request's) `queue`. */
  async #postDecide(
    payload: Record<string, unknown>,
    options: DecideOptions,
    what: string,
  ): Promise<Json> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (options.act !== undefined) headers['X-Crucible-Act'] = requireText(options.act, 'act');
    const queue = this.#callQueue(options.queue);
    const body = queue === null ? payload : { ...payload, queue };
    const init: RequestInit = { method: 'POST', headers, body: JSON.stringify(body) };
    if (options.signal !== undefined) init.signal = options.signal;
    return this.#json('/v1/decide', init, what);
  }

  /** The same completion as {@link chat}, streamed as content deltas. */
  async *chatStream(options: ChatOptions): AsyncGenerator<string, void, undefined> {
    const init = this.#chatRequest(options, true);
    const headers = new Headers(init.headers);
    headers.set('Accept', 'text/event-stream');
    const response = await this.#fetch('/v1/openai/chat/completions', { ...init, headers }, true);
    if (!response.ok) throw await this.#failure(response);
    const stream = response.body;
    if (stream === null) {
      throw new CrucibleProtocolError('the chat completion stream carried no body');
    }

    try {
      for await (const frame of readSseFrames(stream)) {
        if (frame.data === DONE_SENTINEL) return;
        const delta = readChatDelta(frame.data);
        if (delta !== null) yield delta;
      }
    } catch (cause) {
      if (cause instanceof CrucibleError) throw cause;
      // The caller's own abort is theirs to see as it is, as #fetch does.
      if (options.signal !== undefined && options.signal.aborted) throw cause;
      throw new CrucibleUnreachable(
        this.url,
        `the chat completion stream broke mid-answer (${describeCause(cause)}); the answer is truncated`,
        cause,
      );
    } finally {
      await stream.cancel().catch(() => undefined);
    }

    throw new CrucibleUnreachable(
      this.url,
      "the chat completion stream ended without OpenAI's [DONE] terminator; the " +
        'answer is truncated',
    );
  }

  #chatRequest(options: ChatOptions, stream: boolean): RequestInit {
    const given = options as Partial<ChatOptions> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError('options', 'chat(...) needs {model, messages}');
    }
    const payload: Record<string, unknown> = {
      model: requireText(given.model, 'model'),
      messages: readChatMessages(given.messages),
      stream,
    };
    if (given.temperature !== undefined) {
      payload['temperature'] = requireFinite(given.temperature, 'temperature');
    }
    if (given.topP !== undefined) payload['top_p'] = requireFinite(given.topP, 'topP');
    if (given.maxTokens !== undefined) {
      const maxTokens = requireFinite(given.maxTokens, 'maxTokens');
      if (!Number.isInteger(maxTokens) || maxTokens < 1) {
        throw new CrucibleConfigError('maxTokens', `must be a positive integer, got ${maxTokens}`);
      }
      payload['max_tokens'] = maxTokens;
    }
    if (given.stop !== undefined) payload['stop'] = requireStrings(given.stop, 'stop');
    if (given.seed !== undefined) {
      const seed = requireFinite(given.seed, 'seed');
      if (!Number.isInteger(seed)) {
        throw new CrucibleConfigError('seed', `must be an integer, got ${seed}`);
      }
      payload['seed'] = seed;
    }
    if (given.responseFormat !== undefined) {
      payload['response_format'] = readResponseFormat(given.responseFormat);
    }
    if (given.thinking !== undefined) {
      if (typeof given.thinking !== 'boolean') {
        throw new CrucibleConfigError(
          'thinking',
          `must be a boolean, got ${typeof given.thinking}`,
        );
      }
      payload['chat_template_kwargs'] = { enable_thinking: given.thinking };
    }
    const queue = this.#callQueue(given.queue);
    if (queue !== null) payload['queue'] = queue;
    if (given.contextTokens !== undefined) {
      const contextTokens = requireFinite(given.contextTokens, 'contextTokens');
      if (!Number.isInteger(contextTokens) || contextTokens < 1) {
        throw new CrucibleConfigError(
          'contextTokens',
          `must be a positive integer, got ${contextTokens}`,
        );
      }
      payload['context_tokens'] = contextTokens;
    }

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (given.act !== undefined) {
      headers['X-Crucible-Act'] = requireText(given.act, 'act');
    }

    const init: RequestInit = {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    };
    if (given.signal !== undefined) init.signal = given.signal;
    return init;
  }

  /**
   * `POST /v1/decide` — a probability distribution over each question's fixed answers, from one
   * forward pass of the resident model.
   */
  async decide(request: DecideRequest, options: DecideOptions = {}): Promise<DecideResponse> {
    const given = request as Partial<DecideRequest> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError('request', 'decide(...) needs {state, questions}');
    }
    if (!('state' in given) || given.state === undefined) {
      throw new CrucibleConfigError('state', 'is required and was not given');
    }
    const questions = readDecideQuestions(given.questions);
    const payload: Record<string, unknown> = {};
    if (given.model !== undefined) payload['model'] = requireText(given.model, 'model');
    payload['state'] = given.state;
    if (given.images !== undefined) payload['images'] = requireStrings(given.images, 'images');
    payload['questions'] = questions;
    let report = false;
    if (given.missing !== undefined) {
      if (given.missing !== 'refuse' && given.missing !== 'report') {
        throw new CrucibleConfigError(
          'missing',
          `must be 'refuse' or 'report', got ${JSON.stringify(given.missing)}`,
        );
      }
      payload['missing'] = given.missing;
      report = given.missing === 'report';
    }

    const body = await this.#postDecide(payload, options, 'decide');
    return readDecideResponse(body, questions, report);
  }

  /** `POST /v1/decide` with `items`: one choice answer per item about one state, in item order. */
  async decideItems(request: DecideItemsRequest, options: DecideOptions = {}): Promise<DecideItemsResponse> {
    const given = request as Partial<DecideItemsRequest> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError('request', 'decideItems(...) needs {state, items}');
    }
    if (!('state' in given) || given.state === undefined) {
      throw new CrucibleConfigError('state', 'is required and was not given');
    }
    const shared = given.options === undefined ? undefined : readOptionMap(given.options, 'options');
    const items = readDecideItems(given.items, shared);
    const payload: Record<string, unknown> = {};
    if (given.model !== undefined) payload['model'] = requireText(given.model, 'model');
    payload['state'] = given.state;
    if (given.images !== undefined) payload['images'] = requireStrings(given.images, 'images');
    if (given.instructions !== undefined) payload['instructions'] = requireText(given.instructions, 'instructions');
    if (shared !== undefined) payload['options'] = shared;
    payload['items'] = items.map((item) => (item.own ? { text: item.text, options: item.options } : { text: item.text }));
    const report = readMissingMode(given.missing, payload);

    const body = await this.#postDecide(payload, options, 'decideItems');
    return readDecideItemsResponse(body, items, report);
  }

  /**
   * `GET /v1/voices` — every voice this server has a manifest for, with support, installation,
   * residency and loadability.
   */
  async voices(): Promise<VoiceInfo[]> {
    const body = await this.#jsonValue('/v1/voices', { method: 'GET' }, 'voices');
    const entries = asArray(body, 'voices');
    return entries.map((entry, index) =>
      readVoiceInfo(asObject(entry, `voices[${index}]`), `voices[${index}]`),
    );
  }

  /** Queue a `load-voice` job and return its id. */
  async loadVoice(voice: string, options?: LoadVoiceOptions): Promise<string> {
    const reference = options?.reference;
    return this.submit({
      type: 'load-voice',
      model: requireText(voice, 'voice'),
      params: reference === undefined ? {} : {
        reference: {
          data: requireText(reference.data, 'reference.data'),
          transcript: requireText(reference.transcript, 'reference.transcript'),
          ...(reference.name === undefined ? {} : {name: reference.name}),
        },
      },
      inputs: {},
    });
  }

  /** Queue an `unload-voice` job and return its id. */
  async unloadVoice(voice: string): Promise<string> {
    return this.submit({
      type: 'unload-voice',
      model: requireText(voice, 'voice'),
      params: {},
      inputs: {},
    });
  }

  /** Queue a `tts` render job — text in, one `<index>.flac` per chunk out — and return its id. */
  async render(options: RenderOptions): Promise<string> {
    const given = options as Partial<RenderOptions> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError(
        'options',
        'render(...) needs {voice, language, take, chunks}',
      );
    }
    const submission: { signal?: AbortSignal } = {};
    if (given.signal !== undefined) submission.signal = given.signal;
    return this.submit(
      {
        type: 'tts',
        model: requireText(given.voice, 'voice'),
        params: {
          language: requireText(given.language, 'language'),
          take: requireIndex(given.take, 'take'),
          chunks: readRenderChunks(given.chunks),
          ...(given.retake === undefined ? {} : { retake: given.retake }),
          ...(given.band === undefined ? {} : { band: given.band }),
          ...(given.width === undefined ? {} : { width: given.width }),
        },
        inputs: {},
        ...(given.hold === undefined ? {} : { hold: requireBool(given.hold, 'hold') }),
      },
      submission,
    );
  }

  /**
   * {@link events}, plus writing each artifact and its provenance sidecar into `dir` atomically as
   * it lands.
   */
  async *writeArtifactsTo(
    jobId: string,
    dir: string,
    options: WriteArtifactsOptions = {},
  ): AsyncGenerator<ArtifactWrite, void, undefined> {
    const id = requireText(jobId, 'jobId');
    const directory = requireText(dir, 'dir');
    const limit = options.concurrency === undefined
      ? DEFAULT_ARTIFACT_CONCURRENCY
      : requireIndex(options.concurrency, 'concurrency');
    if (limit < 1) {
      throw new CrucibleConfigError('concurrency', `must be at least 1, got ${limit}`);
    }
    const node = await loadNodeFileApis();
    await node.fs.mkdir(directory, { recursive: true });

    const active = new Map<string, Promise<void>>();
    const landed: WrittenArtifact[] = [];
    const started = new Set<string>();
    let failure: unknown = null;

    const begin = (name: string): void => {
      if (started.has(name)) return;
      started.add(name);
      const task = (async () => {
        try {
          landed.push(await this.#writeArtifact(node, id, name, directory));
        } catch (error) {
          if (failure === null) failure = error;
        } finally {
          active.delete(name);
        }
      })();
      active.set(name, task);
    };

    const taken = (): WrittenArtifact[] => landed.splice(0, landed.length);

    let announced: readonly string[] | undefined;
    const events = options.lastEventId === undefined
      ? this.events(id)
      : this.events(id, { lastEventId: options.lastEventId });

    try {
      for await (const event of events) {
        if (event.event === 'artifact') {
          while (active.size >= limit) await Promise.race(active.values());
          begin(event.data.name);
        }
        if (event.event === 'done') announced = event.data.artifacts;
        yield { kind: 'event', event };
        for (const written of taken()) yield { kind: 'written', written };
        if (failure !== null) throw failure;
      }

      if (announced !== undefined && options.lastEventId === undefined) {
        for (const name of announced) {
          while (active.size >= limit) await Promise.race(active.values());
          begin(name);
          for (const written of taken()) yield { kind: 'written', written };
          if (failure !== null) throw failure;
        }
      }

      while (active.size > 0) {
        await Promise.race(active.values());
        for (const written of taken()) yield { kind: 'written', written };
        if (failure !== null) throw failure;
      }
    } finally {
      await Promise.all(active.values());
    }
  }

  async #writeArtifact(
    node: NodeFileApis,
    jobId: string,
    name: string,
    dir: string,
  ): Promise<WrittenArtifact> {
    refuseUnsafeMemberName(name);
    const sidecarName = `${name}.provenance.json`;
    const [bytes, sidecarBytes] = await Promise.all([
      this.artifact(jobId, name),
      this.artifact(jobId, sidecarName),
    ]);
    if (bytes.length === 0) {
      throw new CrucibleProtocolError(
        `artifact ${name} of job ${jobId} is zero bytes; the server announced a ` +
          'file it did not write',
      );
    }

    const text = new TextDecoder('utf-8').decode(sidecarBytes);
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new CrucibleProtocolError(`${sidecarName} is not JSON: ${excerpt(text)}`);
    }
    const provenance = readProvenance(asObject(parsed, sidecarName), name);

    const path = node.path.join(dir, name);
    const provenancePath = node.path.join(dir, sidecarName);
    await writeAtomically(node, provenancePath, sidecarBytes);
    await writeAtomically(node, path, bytes);
    return { name, path, bytes: bytes.length, provenancePath, provenance };
  }

  /** `GET /v1/accelerator` — what is on the card right now, and which of it is Crucible's own. */
  async accelerator(): Promise<AcceleratorState> {
    const body = await this.#json('/v1/accelerator', { method: 'GET' }, 'accelerator');
    return readAcceleratorState(body);
  }

  /** Queue an `asr` job — one audio file in, one transcript out — and return its id. */
  async asr(options: AsrOptions): Promise<string> {
    const given = options as Partial<AsrOptions> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError(
        'options',
        'asr(...) needs {model, audio, filename, language, vadFilter, wordTimestamps}',
      );
    }
    const audio = given.audio;
    if (audio === undefined || audio === null) {
      throw new CrucibleConfigError('audio', 'is required and was not given');
    }
    return this.submit({
      type: 'asr',
      model: requireText(given.model, 'model'),
      params: {
        language: requireText(given.language, 'language'),
        vad_filter: requireBool(given.vadFilter, 'vadFilter'),
        word_timestamps: requireBool(given.wordTimestamps, 'wordTimestamps'),
        ...(given.initialPrompt !== undefined
          ? { initial_prompt: readInitialPrompt(given.initialPrompt) }
          : {}),
        ...(given.context !== undefined
          ? { context: readContext(given.context) }
          : {}),
        ...(given.speechOnly !== undefined
          ? { speech_only: requireBool(given.speechOnly, 'speechOnly') }
          : {}),
        ...readSpeechKnob(given.speechThreshold, 'speechThreshold', 'speech_threshold'),
        ...readSpeechKnob(given.speechPadS, 'speechPadS', 'speech_pad_s'),
        ...readSpeechKnob(given.speechMinGapS, 'speechMinGapS', 'speech_min_gap_s'),
        ...(given.resume !== undefined && given.resume !== null
          ? { resume: requireText(given.resume, 'resume') }
          : {}),
      },
      inputs: { [requireText(given.filename, 'filename')]: audio },
    });
  }

  /** A previous job's artifact on this server as an input, without sending its bytes again. */
  artifactRef(jobId: string, name: string): JobInput {
    return { artifact: { jobId: requireText(jobId, 'jobId'), name: requireText(name, 'name') } };
  }

  /** `POST /v1/jobs/{id}/hold` — keep a done job's artifacts for the rest of a chain. */
  async holdArtifacts(jobId: string): Promise<ArtifactHold> {
    const id = requireText(jobId, 'jobId');
    const body = await this.#json(`/v1/jobs/${encodeURIComponent(id)}/hold`, { method: 'POST' }, 'hold');
    return {
      jobId: str(body, 'job_id', 'hold'),
      status: str(body, 'status', 'hold'),
      held: bool(body, 'held', 'hold'),
      heldBy: nullableStr(body, 'held_by', 'hold'),
      heldSince: str(body, 'held_since', 'hold'),
      gcAt: nullableStr(body, 'gc_at', 'hold'),
      artifacts: strArray(body, 'artifacts', 'hold'),
    };
  }

  /** `DELETE /v1/jobs/{id}/hold` — release the hold and remove the job's directory. */
  async releaseArtifacts(jobId: string): Promise<void> {
    const id = requireText(jobId, 'jobId');
    const response = await this.#fetch(`/v1/jobs/${encodeURIComponent(id)}/hold`, { method: 'DELETE' }, true);
    if (!response.ok) throw await this.#failure(response);
  }

  /**
   * Queue an `align` job — windows of audio, each with the text spoken in it, placed in time — and
   * return its id.
   */
  async align(options: AlignOptions): Promise<string> {
    const given = options as Partial<AlignOptions> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError('options', 'align(...) needs {model, language, windows}');
    }
    const windows = given.windows;
    if (!Array.isArray(windows) || windows.length === 0) {
      throw new CrucibleConfigError('windows', 'needs at least one {index, text, audio, extension}');
    }
    const chunks: { index: number; text: string }[] = [];
    const inputs: Record<string, JobInput> = {};
    windows.forEach((window, position) => {
      const where = `windows[${position}]`;
      const index = window?.index;
      if (typeof index !== 'number' || !Number.isInteger(index) || index < 0) {
        throw new CrucibleConfigError(`${where}.index`, 'must be a non-negative integer');
      }
      if (window.audio === undefined || window.audio === null) {
        throw new CrucibleConfigError(`${where}.audio`, 'is required and was not given');
      }
      const extension = requireText(window.extension, `${where}.extension`).replace(/^\./, '');
      const name = `${index}.${extension}`;
      if (name in inputs) {
        throw new CrucibleConfigError(`${where}.index`, `${index} appears more than once`);
      }
      chunks.push({ index, text: requireText(window.text, `${where}.text`) });
      inputs[name] = window.audio;
    });
    return this.submit({
      type: 'align',
      model: requireText(given.model, 'model'),
      params: { language: requireText(given.language, 'language'), chunks },
      inputs,
    });
  }

  /** Queue an `image` job (one PNG, `image.png`) and return its id. */
  async image(options: ImageOptions): Promise<string> {
    const given = options as Partial<ImageOptions> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError('options', 'image(...) needs {model, prompt}');
    }
    const strength = given.imageStrength ?? null;
    const picture = given.image ?? null;
    const mask = given.mask ?? null;
    const imageName = given.imageName ?? 'input.png';
    const maskName = given.maskName ?? 'mask.png';
    if (mask !== null) {
      if (picture === null) {
        throw new CrucibleConfigError('image', 'a mask marks a region of an image; send the image too');
      }
      if (maskName === imageName) {
        throw new CrucibleConfigError('maskName', `the image and the mask are both named ${JSON.stringify(maskName)}`);
      }
    } else if ((strength === null) !== (picture === null)) {
      throw new CrucibleConfigError(
        strength === null ? 'imageStrength' : 'image',
        'image-to-image needs both an image and an imageStrength between 0 and 1 (or a mask, to regenerate one region)',
      );
    }
    if (mask === null && given.maskBlur !== undefined) {
      throw new CrucibleConfigError('maskBlur', 'maskBlur softens the edge of a mask; send mask too');
    }
    const params: Record<string, unknown> = { prompt: requireText(given.prompt, 'prompt') };
    const optional: [keyof ImageOptions, string][] = [
      ['negativePrompt', 'negative_prompt'],
      ['width', 'width'],
      ['height', 'height'],
      ['seed', 'seed'],
      ['steps', 'steps'],
      ['guidance', 'guidance'],
      ['imageStrength', 'image_strength'],
      ['maskBlur', 'mask_blur'],
    ];
    for (const [key, wire] of optional) {
      const value = given[key];
      if (value !== undefined && value !== null) params[wire] = value;
    }
    const inputs: Record<string, JobInput> = {};
    if (picture !== null) inputs[imageName] = picture;
    if (mask !== null) {
      params.mask = maskName;
      inputs[maskName] = mask;
    }
    return this.submit({
      type: 'image',
      model: requireText(given.model, 'model'),
      params,
      inputs,
    });
  }

  /** Queue a `load-image` job (warm the image model up before the first prompt) and return its id. */
  async loadImage(model: string): Promise<string> {
    return this.submit({
      type: 'load-image',
      model: requireText(model, 'model'),
      params: {},
      inputs: {},
    });
  }

  /** Queue an `audio` job (one `audio.flac`, `audio.wav` or `audio.mp3`, plus `score.abc` for a song) and return its id. */
  async audio(options: AudioOptions): Promise<string> {
    const given = options as Partial<AudioOptions> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError('options', 'audio(...) needs {model, prompt} or {model, tags, lyrics}');
    }
    if (given.prompt === undefined && given.tags === undefined) {
      throw new CrucibleConfigError(
        'prompt',
        'audio(...) needs a prompt (sound effects, music) or tags and lyrics (songs)',
      );
    }
    const params: Record<string, unknown> = {};
    const optional: [keyof AudioOptions, string][] = [
      ['prompt', 'prompt'],
      ['tags', 'tags'],
      ['lyrics', 'lyrics'],
      ['negativePrompt', 'negative_prompt'],
      ['durationS', 'duration_s'],
      ['seed', 'seed'],
      ['steps', 'steps'],
      ['cfg', 'cfg'],
      ['format', 'format'],
      ['instrumental', 'instrumental'],
    ];
    for (const [key, wire] of optional) {
      const value = given[key];
      if (value !== undefined && value !== null) params[wire] = value;
    }
    return this.submit({
      type: 'audio',
      model: requireText(given.model, 'model'),
      params,
      inputs: {},
    });
  }

  /**
   * `GET /v1/playground` — one page per image, video and audio model this server's build
   * declares: its form (defaults and limits from the model's manifest, a song model's tag
   * suggestions and conflicts) and its standing here.
   */
  async playground(): Promise<PlaygroundPage[]> {
    const body = await this.#json('/v1/playground', { method: 'GET' }, 'playground');
    return asArray(field(body, 'pages', 'playground'), 'playground.pages').map((value, index) =>
      readPlaygroundPage(asObject(value, `playground.pages[${index}]`), `playground.pages[${index}]`));
  }

  /** `GET /v1/playground/presets/{model}` — the presets saved for `model` on this server. */
  async playgroundPresets(model: string): Promise<PlaygroundPreset[]> {
    const path = `/v1/playground/presets/${encodeURIComponent(requireText(model, 'model'))}`;
    const body = await this.#json(path, { method: 'GET' }, 'playgroundPresets');
    return asArray(field(body, 'presets', 'presets'), 'presets.presets').map((value, index) =>
      readPlaygroundPreset(asObject(value, `presets[${index}]`), `presets[${index}]`));
  }

  /**
   * `PUT /v1/playground/presets/{model}/{name}` — save (or replace) a preset: the form's own
   * params as text, numbers and true/false. A seed is refused by name (a preset is a sound,
   * not one take of it).
   */
  async savePlaygroundPreset(
    model: string,
    name: string,
    params: Readonly<Record<string, string | number | boolean>>,
  ): Promise<PlaygroundPreset> {
    const path = `/v1/playground/presets/${encodeURIComponent(requireText(model, 'model'))}/${encodeURIComponent(requireText(name, 'name'))}`;
    const body = await this.#json(path, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params }),
    }, 'savePlaygroundPreset');
    return readPlaygroundPreset(objectField(body, 'preset', 'savePlaygroundPreset'), 'savePlaygroundPreset.preset');
  }

  /** `DELETE /v1/playground/presets/{model}/{name}` — 404 `preset_not_found` when there is none. */
  async deletePlaygroundPreset(model: string, name: string): Promise<void> {
    const path = `/v1/playground/presets/${encodeURIComponent(requireText(model, 'model'))}/${encodeURIComponent(requireText(name, 'name'))}`;
    await this.#json(path, { method: 'DELETE' }, 'deletePlaygroundPreset');
  }

  /** Queue a `load-audio` job (warm an audio model up before the first request) and return its id. */
  async loadAudio(model: string): Promise<string> {
    return this.submit({
      type: 'load-audio',
      model: requireText(model, 'model'),
      params: {},
      inputs: {},
    });
  }

  /**
   * Queue a `segment` job (`mask.png` and `cutout.png`, both at the picture's size) and return
   * its id. `birefnet` cuts out the main subject by itself; `sam2.1-hiera-large` selects what
   * `points` and/or `box` point at.
   */
  async segment(options: SegmentOptions): Promise<string> {
    const given = options as Partial<SegmentOptions> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError('options', 'segment(...) needs {model, image}');
    }
    if (given.image === undefined || given.image === null) {
      throw new CrucibleConfigError('image', 'segment(...) needs the picture to cut from, as image');
    }
    const params: Record<string, unknown> = {};
    if (given.points !== undefined && given.points !== null) {
      params.points = given.points.map((point) => ({ x: point.x, y: point.y, label: point.label }));
    }
    if (given.box !== undefined && given.box !== null) {
      params.box = [...given.box];
    }
    return this.submit({
      type: 'segment',
      model: requireText(given.model, 'model'),
      params,
      inputs: { [given.imageName ?? 'input.png']: given.image },
    });
  }

  /** Queue a `load-segment` job (warm a segment model up, e.g. when a selection tool opens) and return its id. */
  async loadSegment(model: string): Promise<string> {
    return this.submit({
      type: 'load-segment',
      model: requireText(model, 'model'),
      params: {},
      inputs: {},
    });
  }

  /** Queue a `video` job (one `video.mp4`, H.264 with AAC sound) and return its id. */
  async video(options: VideoOptions): Promise<string> {
    const given = options as Partial<VideoOptions> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError('options', 'video(...) needs {model, prompt}');
    }
    if (given.durationS !== undefined && given.numFrames !== undefined) {
      throw new CrucibleConfigError('numFrames', 'video(...) takes durationS or numFrames, not both');
    }
    if ((given.width === undefined) !== (given.height === undefined)) {
      throw new CrucibleConfigError(
        given.width === undefined ? 'width' : 'height',
        'video(...) takes width and height together, or neither for the default size',
      );
    }
    const params: Record<string, unknown> = { prompt: requireText(given.prompt, 'prompt') };
    const optional: [keyof VideoOptions, string][] = [
      ['width', 'width'],
      ['height', 'height'],
      ['durationS', 'duration_s'],
      ['numFrames', 'num_frames'],
      ['fps', 'fps'],
      ['seed', 'seed'],
      ['steps', 'steps'],
      ['audio', 'audio'],
    ];
    for (const [key, wire] of optional) {
      const value = given[key];
      if (value !== undefined && value !== null) params[wire] = value;
    }
    const picture = given.image ?? null;
    return this.submit({
      type: 'video',
      model: requireText(given.model, 'model'),
      params,
      inputs: picture === null ? {} : { [given.imageName ?? 'start.png']: picture },
    });
  }

  /** Queue a `load-video` job (warm the video model up before a batch) and return its id. */
  async loadVideo(model: string): Promise<string> {
    return this.submit({
      type: 'load-video',
      model: requireText(model, 'model'),
      params: {},
      inputs: {},
    });
  }

  async #json(path: string, init: RequestInit, where: string): Promise<Json> {
    return asObject(await this.#jsonValue(path, init, where), where);
  }

  async #jsonValue(path: string, init: RequestInit, where: string): Promise<unknown> {
    const response = await this.#fetch(path, init, true);
    if (!response.ok) throw await this.#failure(response);
    const text = await response.text();
    try {
      return JSON.parse(text) as unknown;
    } catch {
      throw new CrucibleProtocolError(`${where} did not return JSON: ${excerpt(text)}`);
    }
  }

  /**
   * One request, waiting out a deploy's restart: a `503 server_updating` admitted nothing, so it is
   * sent again once the new server answers - never before, so a request is not sent to a server
   * that is going away - within {@link UPDATE_WAIT_MS}. A caller sees the refusal only when the
   * update outlasted that.
   */
  async #fetch(path: string, init: RequestInit, authenticated: boolean): Promise<Response> {
    const deadline = Date.now() + UPDATE_WAIT_MS;
    for (;;) {
      const answered = await this.#attempt(path, init, authenticated);
      if (answered.status !== 503) return answered;
      // Read once and handed on whole, so the caller's #failure reads the same refusal.
      const text = await answered.text();
      const response = new Response(text, {
        status: answered.status,
        statusText: answered.statusText,
        headers: answered.headers,
      });
      const retryAfterS = updatingRetryAfter(text, response.headers);
      if (retryAfterS === null || Date.now() >= deadline) return response;
      const signal = init.signal ?? undefined;
      await pause(Math.min(retryAfterS * 1000, Math.max(0, deadline - Date.now())), signal ?? NEVER);
      if (signal !== undefined && signal.aborted) return response;
      await this.#untilAnswering(deadline, signal);
    }
  }

  /** Wait until something answers `GET /v1/ping` (the restarted server), or the deadline. */
  async #untilAnswering(deadline: number, signal: AbortSignal | undefined): Promise<void> {
    while (Date.now() < deadline && !(signal !== undefined && signal.aborted)) {
      try {
        const probe = AbortSignal.timeout(UPDATE_PROBE_MS);
        await fetch(`${this.url}/v1/ping`, {
          method: 'GET',
          signal: signal === undefined ? probe : AbortSignal.any([signal, probe]),
        });
        return;
      } catch {
        // Not up yet: the old server has stopped and the new one is starting.
      }
      await pause(UPDATE_PROBE_MS, signal ?? NEVER);
    }
  }

  async #attempt(path: string, init: RequestInit, authenticated: boolean): Promise<Response> {
    const bound = this.#session;
    if (bound !== null && bound.end !== null) throw endedSession(bound.id, bound.end);
    const headers = new Headers(init.headers);
    headers.set('User-Agent', this.#userAgent);
    headers.set(CLIENT_NAME_HEADER, this.#clientName);
    if (bound !== null) headers.set(SESSION_HEADER, bound.id);
    if (authenticated) {
      headers.set('Authorization', `Bearer ${this.#token}`);
      headers.set(API_HEADER, String(API_VERSION));
    }
    const target = `${this.url}${path}`;
    const retryable = isSafeMethod(init.method);
    for (let attempt = 0; ; attempt += 1) {
      const timed =
        this.#timeoutMs !== null && init.signal === undefined
          ? { ...init, headers, signal: AbortSignal.timeout(this.#timeoutMs) }
          : { ...init, headers };
      try {
        return await fetch(target, timed);
      } catch (cause) {
        const signal = timed.signal;
        if (signal !== undefined && signal !== null && signal.aborted) throw cause;
        if (attempt === 0 && retryable && isStaleConnection(cause)) continue;
        throw new CrucibleUnreachable(this.url, describeCause(cause), cause);
      }
    }
  }

  async #failure(response: Response): Promise<CrucibleError> {
    const text = await response.text();
    let envelope: Json;
    try {
      envelope = objectField(asObject(JSON.parse(text), 'error response'), 'error', 'error response');
    } catch {
      return new CrucibleProtocolError(
        `HTTP ${response.status} from ${this.url} is not a crucible error ` +
          `({"error": {"code", "message"}}); it said: ${excerpt(text)}`,
      );
    }
    const code = str(envelope, 'code', 'error');
    const message = str(envelope, 'message', 'error');

    if (response.status === 401) return new CrucibleAuthError(code, message);
    if (response.status === 426) {
      const details = objectField(envelope, 'details', 'error');
      const serverApiVersion = nullableNum(details, 'server_api_version', 'error.details');
      return new CrucibleVersionError(code, message, serverApiVersion, API_VERSION);
    }
    const details = 'details' in envelope ? envelope['details'] : null;
    if (response.status >= 500) {
      if (code === SERVER_UPDATING) {
        return new CrucibleUpdating(response.status, code, message, details);
      }
      if (code === ACCELERATOR_UNREADABLE) {
        return new CrucibleAcceleratorUnreadable(response.status, code, message, details);
      }
      if (code === CAPABILITY_UNDECIDED) {
        return new CrucibleCapabilityUndecided(response.status, code, message, details);
      }
      return new CrucibleServerError(response.status, code, message, details);
    }
    if (response.status >= 400) {
      if (code === SERVER_BUSY) {
        if (heldAtTheOperatorDoor(details)) return heldRefusal(response.status, code, message, details);
        if (heldBySession(details)) return sessionRefusal(response.status, code, message, details);
        return busyRefusal(response.status, code, message, details);
      }
      if (code === SESSION_OPEN) return sessionRefusal(response.status, code, message, details);
      if (code === SESSION_CLOSED) {
        const refusal = closedRefusal(response.status, message, details);
        const bound = this.#session;
        if (bound !== null) noteSessionEnd(bound, refusal);
        return refusal;
      }
      return new CrucibleRefused(response.status, code, message, details);
    }
    return new CrucibleProtocolError(
      `HTTP ${response.status} from ${this.url} is neither a success nor a refusal`,
    );
  }

  /**
   * `POST /v1/tts/stream` — open a live TTS stream. It runs inside a queue session: this client's
   * own (a {@link CrucibleSession}'s, or the open one this client holds), else one opened for it,
   * which waits in the line and closes with the stream. Resolves once that session is open and the
   * voice is resident. Given `onQueue`, it reports that session's place in the line while it waits,
   * as {@link session} does.
   */
  async stream(options: StreamOptions): Promise<TtsStreamSession> {
    const given = options as Partial<StreamOptions> | undefined;
    const withQueue =
      given !== undefined && given !== null && given.queue === undefined && this.#queue !== undefined
        ? { ...options, queue: this.#queue }
        : options;
    return openTtsStream(
      {
        url: this.url,
        fetch: (path, init, authenticated) => this.#fetch(path, init, authenticated),
        failure: (response) => this.#failure(response),
        json: (path, init, where) => this.#json(path, init, where),
        untilOpen: async (queueSessionId, watch, signal) => {
          await this.#untilOpen(queueSessionId, watch, signal);
        },
        leaveTheLine: (queueSessionId) => this.#leaveTheLine(queueSessionId),
      },
      withQueue,
    );
  }

  /**
   * `GET /v1/setup` — everything another app needs to connect to this server, including its pairing
   * lines.
   */
  async setup(): Promise<ServerSetup> {
    const body = await this.#json('/v1/setup', { method: 'GET' }, 'setup');
    return {
      name: str(body, 'name', 'setup'),
      version: str(body, 'version', 'setup'),
      backend: str(body, 'backend', 'setup'),
      bind: str(body, 'bind', 'setup'),
      urls: strArray(body, 'urls', 'setup'),
      token: str(body, 'token', 'setup'),
      pairing: strArray(body, 'pairing', 'setup'),
      jobTypes: strArray(body, 'job_types', 'setup'),
      configPath: str(body, 'config_path', 'setup'),
      network: readNetwork(optObject(body, 'network', 'setup')),
    };
  }

  /** `GET /v1/catalog` — every subject this server's backend can hold, installed or not. */
  async catalog(): Promise<CatalogRow[]> {
    const body = await this.#json('/v1/catalog', { method: 'GET' }, 'catalog');
    const rows = asArray(field(body, 'rows', 'catalog'), 'catalog.rows');
    return rows.map((row, index) =>
      readCatalogRow(asObject(row, `catalog.rows[${index}]`), `catalog.rows[${index}]`),
    );
  }

  /** `DELETE /v1/catalog/{kind}/{id}` — remove an installed subject's files. */
  async removeSubject(kind: SubjectKind, id: string): Promise<void> {
    const path =
      `/v1/catalog/${encodeURIComponent(kind)}/${encodeURIComponent(id)}`;
    const response = await this.#fetch(path, { method: 'DELETE' }, true);
    if (!response.ok) throw await this.#failure(response);
    await response.text();
  }

  /** `POST /v1/tasks` — pull a subject, install a job type, or post a module. */
  async submitTask(request: TaskRequest): Promise<string> {
    const body = await this.#json(
      '/v1/tasks',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(taskPayload(request)),
      },
      'submitTask',
    );
    return str(body, 'task_id', 'submitTask');
  }

  /** `GET /v1/tasks/{id}`. */
  async task(taskId: string): Promise<TaskStatus> {
    const id = requireText(taskId, 'taskId');
    const body = await this.#json(
      `/v1/tasks/${encodeURIComponent(id)}`,
      { method: 'GET' },
      'task',
    );
    return readTaskStatus(body, 'task');
  }

  /** The last few tasks this server remembers, newest first. */
  async tasks(): Promise<TaskStatus[]> {
    const body = await this.#json('/v1/tasks', { method: 'GET' }, 'tasks');
    const rows = asArray(field(body, 'tasks', 'tasks'), 'tasks.tasks');
    return rows.map((row, index) =>
      readTaskStatus(asObject(row, `tasks[${index}]`), `tasks[${index}]`),
    );
  }

  /** `GET /v1/tasks/{id}/events` — the task's SSE stream, as typed events. */
  async *taskEvents(
    taskId: string,
    options: EventsOptions = {},
  ): AsyncGenerator<TaskEvent, void, undefined> {
    const id = requireText(taskId, 'taskId');
    yield* this.#follow(
      `/v1/tasks/${encodeURIComponent(id)}/events`,
      `task ${id}`,
      options,
      readTaskEvent,
      TASK_TERMINAL_STATES,
    );
  }

  /** `DELETE /v1/tasks/{id}` — cancel. */
  async cancelTask(taskId: string): Promise<TaskCancelResult> {
    const id = requireText(taskId, 'taskId');
    const body = await this.#json(
      `/v1/tasks/${encodeURIComponent(id)}`,
      { method: 'DELETE' },
      'cancelTask',
    );
    return {
      taskId: str(body, 'task_id', 'cancelTask'),
      status: oneOf(str(body, 'status', 'cancelTask'), ['cancelling'], 'cancelTask.status'),
    };
  }
}

/**
 * An app's queue session: its turn holding the server for a run of requests (docs/QUEUE.md). Made
 * by {@link CrucibleClient.session}, open when you get it.
 *
 * It IS a {@link CrucibleClient} — `chat`, `chatStream`, `decide`, `decideItems`, `submit`,
 * `render`, `asr`, `align`, `image`, `audio`, `segment`, `video`, every load and unload, `events`,
 * `artifact`, `stream`, … — and every request it makes carries `X-Crucible-Session`. Its helpers
 * send no `queue`: a session's items go ahead of the line. Once the session has ended, every one of
 * them throws {@link CrucibleSessionClosed} without asking the server; fetch what its jobs left
 * through the client it came from.
 *
 * End it with {@link close} in a `finally`; until then nothing from any other client runs.
 */
export class CrucibleSession extends CrucibleClient {
  /** The session's id (`ses-…`). */
  readonly id: string;
  /** The capability class it was opened for. */
  readonly act: string;
  /**
   * How the session ended, once it has: closed by {@link close}, idle, by an operator, at the
   * server's maximum hold, or the server stopping. Never rejects. A background follow of the
   * session's own stream is what resolves it; that follow ends with the session.
   */
  readonly closed: Promise<QueueSessionEnd>;

  readonly #hooks: SessionHooks;

  /** Made by {@link CrucibleClient.session}; an app never constructs one. */
  constructor(
    options: CrucibleClientOptions,
    id: string,
    act: string,
    closed: Promise<QueueSessionEnd>,
    hooks: SessionHooks,
  ) {
    super(options);
    this.id = id;
    this.act = act;
    this.closed = closed;
    this.#hooks = hooks;
  }

  /**
   * `POST /v1/queue/sessions/{id}/touch` — still here. Anything the server is running or answering
   * for the session already counts; this is for a long gap on the app's side with nothing in
   * flight there (a cloud call, a file copy) that would otherwise outlast `idleS`.
   */
  touch(): Promise<void> {
    return this.#hooks.touch();
  }

  /** `GET /v1/queue/sessions/{id}` — where it stands: what it has run and has in flight. */
  state(): Promise<QueueSessionState> {
    return this.#hooks.state();
  }

  /**
   * `DELETE /v1/queue/sessions/{id}` — end it (reason `client`); the server settles the card
   * before it answers. Resolves {@link closed}. Calling it again, or after the server ended it,
   * answers how it ended.
   */
  close(): Promise<QueueSessionEnd> {
    return this.#hooks.close();
  }

  /** How it ended, when this client knows it has; null while it is open. */
  get ended(): QueueSessionEnd | null {
    return this.#hooks.binding.end;
  }

  /**
   * Why the background follow of the session's stream stopped before the session ended (a refusal
   * or a malformed frame, never a dropped connection, which is reconnected), or null. While it is
   * set, {@link closed} resolves only through {@link close} or an item the server refuses as
   * `session_closed`.
   */
  get watchFailure(): unknown {
    return this.#hooks.binding.watchFailure;
  }
}

/** What a session's client knows about the session it sends items for. */
interface SessionBinding {
  readonly id: string;
  /** How it ended; null while it is open. */
  end: QueueSessionEnd | null;
  watchFailure: unknown;
  /** Record how it ended, once; the first word wins. */
  settle(end: QueueSessionEnd): void;
}

/** A session's own routes, which its owner sends. */
interface SessionHooks {
  readonly binding: SessionBinding;
  touch(): Promise<void>;
  state(): Promise<QueueSessionState>;
  close(): Promise<QueueSessionEnd>;
}

/** One event of a queue session's own stream, read. */
/** What a wait in the line reports while it follows a session's own stream. */
export interface LineWatch {
  readonly onQueue?: ((position: QueuePosition) => void) | undefined;
  readonly onWaiting?: ((waiting: CardWaitData) => void) | undefined;
}

type SessionFeedEvent =
  | { readonly id: number; readonly kind: 'queued' | 'moved'; readonly position: QueuePosition }
  | { readonly id: number; readonly kind: 'opened' }
  | { readonly id: number; readonly kind: 'waiting'; readonly waiting: CardWaitData }
  | { readonly id: number; readonly kind: 'closed'; readonly end: QueueSessionEnd }
  | {
      readonly id: number;
      readonly kind: 'removed';
      readonly reason: string;
      readonly message: string;
      readonly error: Json | null;
    }
  /** The server is stopping; it closes or removes the session as it does. */
  | { readonly id: number; readonly kind: 'stopping'; readonly reason: string }
  /** The server does not know the session: it restarted since (or forgot it long after it ended). */
  | { readonly id: number; readonly kind: 'forgotten'; readonly message: string }
  | { readonly id: number; readonly kind: 'unknown' };

/** Waits between reconnects: doubling from a quarter second to five, reset by a delivered event. */
class Backoff {
  #next = RECONNECT_FIRST_MS;

  reset(): void {
    this.#next = RECONNECT_FIRST_MS;
  }

  /** Wait before the next attempt; false when `signal` aborted the wait. */
  async wait(signal: AbortSignal): Promise<boolean> {
    const ms = this.#next;
    this.#next = Math.min(this.#next * 2, RECONNECT_CEILING_MS);
    await pause(ms, signal);
    return !signal.aborted;
  }
}

/** How long a request waits out a deploy's restart before the refusal reaches its caller. */
const UPDATE_WAIT_MS = 10 * 60_000;
/** How often a waiting request asks whether the restarted server answers yet. */
const UPDATE_PROBE_MS = 2_000;
const NEVER = new AbortController().signal;

/** The seconds a 503's body asks a client to wait when it is `server_updating`, else null. */
function updatingRetryAfter(text: string, headers: Headers): number | null {
  let code: unknown;
  try {
    code = (JSON.parse(text) as { error?: { code?: unknown } }).error?.code;
  } catch {
    return null;
  }
  if (code !== SERVER_UPDATING) return null;
  const said = Number(headers.get('Retry-After'));
  return Number.isFinite(said) && said > 0 ? said : UPDATE_PROBE_MS / 1000;
}

function pause(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve();
      return;
    }
    const done = (): void => {
      clearTimeout(timer);
      signal.removeEventListener('abort', done);
      resolve();
    };
    const timer = setTimeout(done, ms);
    signal.addEventListener('abort', done, { once: true });
  });
}

/**
 * Weather, not misconfiguration: the server could not be reached, or a proxy or a stopping server
 * answered for it. A follow reconnects after these and refuses everything else by name.
 */
function isWeather(error: unknown): boolean {
  if (error instanceof CrucibleUnreachable) return true;
  return (
    error instanceof CrucibleServerError &&
    (error.status === 502 || error.status === 503 || error.status === 504)
  );
}

function readLastEventId(value: number | undefined): number | null {
  if (value === undefined) return null;
  if (!Number.isInteger(value) || value < 0) {
    throw new CrucibleConfigError('lastEventId', `must be a non-negative integer, got ${String(value)}`);
  }
  return value;
}

const SERVER_EVENT_TOPICS: readonly string[] = [
  'job', 'queue', 'session', 'card', 'chat', 'task', 'settings', 'server',
];

function topicsQuery(topics: readonly string[] | undefined): string {
  if (topics === undefined) return '';
  if (!Array.isArray(topics) || topics.length === 0) {
    throw new CrucibleConfigError(
      'topics',
      'must be a non-empty array of topics; leave it out for all of them',
    );
  }
  for (const topic of topics) oneOfTopics(topic);
  return `?topics=${encodeURIComponent(topics.join(','))}`;
}

function oneOfTopics(topic: unknown): void {
  if (typeof topic !== 'string' || !SERVER_EVENT_TOPICS.includes(topic)) {
    throw new CrucibleConfigError(
      'topics',
      `${JSON.stringify(topic)} is not a topic of GET /v1/events; they are ` +
        SERVER_EVENT_TOPICS.join(', '),
    );
  }
}

function frameData(frame: SseFrame, where: string): Json {
  let parsed: unknown;
  try {
    parsed = JSON.parse(frame.data);
  } catch {
    throw new CrucibleProtocolError(`${where} has non-JSON data: ${excerpt(frame.data)}`);
  }
  return asObject(parsed, where);
}

function frameId(frame: SseFrame, where: string): number {
  const raw = frame.lastEventId;
  const id = Number(raw);
  if (raw === null || raw === '' || !Number.isInteger(id) || id < 1) {
    throw new CrucibleProtocolError(`${where} carried no usable id: ${JSON.stringify(raw)}`);
  }
  return id;
}

function readStopReason(data: string): string {
  return str(frameData({ lastEventId: null, event: SERVER_STOPPING, data }, SERVER_STOPPING), 'reason', SERVER_STOPPING);
}

function readSessionFrame(frame: SseFrame, what: string, previous: number): SessionFeedEvent {
  if (frame.event === SERVER_STOPPING) {
    return { id: previous, kind: 'stopping', reason: readStopReason(frame.data) };
  }
  const where = `${what}'s ${String(frame.event)} event`;
  const id = frameId(frame, where);
  if (id <= previous) {
    throw new CrucibleProtocolError(`event id ${id} does not follow ${previous} on ${what}`);
  }
  const data = frameData(frame, where);
  switch (frame.event) {
    case 'queued':
    case 'moved':
      return {
        id,
        kind: frame.event,
        position: { position: num(data, 'position', where), of: num(data, 'of', where) },
      };
    case 'opened':
      return { id, kind: 'opened' };
    case 'waiting':
      return { id, kind: 'waiting', waiting: readCardWait(data, where) };
    case 'closed':
      return {
        id,
        kind: 'closed',
        end: {
          reason: str(data, 'reason', where),
          message: str(data, 'message', where),
          itemsRun: num(data, 'items_run', where),
          heldS: nullableNum(data, 'held_s', where),
        },
      };
    case 'removed':
      return {
        id,
        kind: 'removed',
        reason: str(data, 'reason', where),
        message: str(data, 'message', where),
        error: optObject(data, 'error', where),
      };
    default:
      return { id, kind: 'unknown' };
  }
}

/** How a closed session's state says it ended. */
function endOf(state: QueueSessionState, where: string): QueueSessionEnd {
  if (state.status !== 'closed' || state.reason === null || state.message === null) {
    throw new CrucibleProtocolError(
      `${where} answered status ${state.status} with reason ${String(state.reason)}; a session ` +
        'that has ended says closed, and why',
    );
  }
  return {
    reason: state.reason,
    message: state.message,
    itemsRun: state.itemsRun,
    heldS:
      state.openedAt === null || state.closedAt === null
        ? null
        : (Date.parse(state.closedAt) - Date.parse(state.openedAt)) / 1000,
  };
}

function settledEnd(binding: SessionBinding): QueueSessionEnd {
  if (binding.end === null) {
    throw new CrucibleProtocolError(`queue session ${binding.id} was closed and says no end`);
  }
  return binding.end;
}

/** A session that ended before it opened, or whose end a client already knows. */
function sessionClosed(
  sessionId: string,
  reason: string,
  message: string,
  error: Json | null,
): CrucibleSessionClosed {
  return new CrucibleSessionClosed(
    409,
    message,
    { session_id: sessionId, reason, ...(error === null ? {} : { error }) },
    { sessionId, reason },
  );
}

function endedSession(sessionId: string, end: QueueSessionEnd): CrucibleSessionClosed {
  return sessionClosed(
    sessionId,
    end.reason,
    `queue session ${sessionId} ended (${end.reason}): ${end.message}. Nothing more runs in ` +
      'it; open a new one with session(...), and fetch what its jobs left through the client ' +
      'it came from',
    null,
  );
}

/** A refusal saying the session ended is how its client learns, when its follow has not yet. */
function noteSessionEnd(binding: SessionBinding, error: unknown): void {
  if (!(error instanceof CrucibleSessionClosed) || error.sessionId !== binding.id) return;
  binding.settle({ reason: error.reason, message: error.serverMessage, itemsRun: null, heldS: null });
}

/** `GET /v1/queue/sessions/{id}`, and `/v1/activity`'s `session`. */
function readQueueSession(data: Json, where: string): QueueSessionState {
  return {
    sessionId: str(data, 'session_id', where),
    status: oneOf(str(data, 'status', where), QUEUE_SESSION_STATUSES, `${where}.status`),
    act: str(data, 'act', where),
    client: nullableStr(data, 'client', where),
    model: nullableStr(data, 'model', where),
    position: nullableNum(data, 'position', where),
    idleS: num(data, 'idle_s', where),
    maxWaitS: num(data, 'max_wait_s', where),
    created: str(data, 'created', where),
    openedAt: nullableStr(data, 'opened_at', where),
    idleDeadline: nullableStr(data, 'idle_deadline', where),
    maxHoldDeadline: nullableStr(data, 'max_hold_deadline', where),
    itemsRun: num(data, 'items_run', where),
    inFlight: arrayField(data, 'in_flight', where).map((entry, index) =>
      asObject(entry, `${where}.in_flight[${index}]`),
    ),
    streamSession: nullableObject(data, 'stream_session', where),
    loadJob: nullableStr(data, 'load_job', where),
    closedAt: nullableStr(data, 'closed_at', where),
    reason: nullableStr(data, 'reason', where),
    message: nullableStr(data, 'message', where),
    error: nullableObject(data, 'error', where),
  };
}

const JOB_CHANGES = [
  'job.queued', 'job.running', 'job.done', 'job.failed', 'job.cancelled', 'job.interrupted',
  'job.removed',
] as const;
const QUEUE_CHANGES = [
  'queue.added', 'queue.moved', 'queue.started', 'queue.removed', 'queue.waiting',
] as const;
const SESSION_CHANGES = [
  'session.queued', 'session.moved', 'session.opened', 'session.closed', 'session.removed',
  'session.waiting',
] as const;
const CARD_CHANGES = [
  'card.warming', 'card.warming_ended', 'card.loaded', 'card.unloading', 'card.unloaded',
] as const;
const TASK_CHANGES = ['task.running', 'task.done', 'task.failed', 'task.cancelled'] as const;

function isOneOf<T extends string>(value: string, names: readonly T[]): value is T {
  return (names as readonly string[]).includes(value);
}

/** One frame of `GET /v1/events`, read (docs/EVENTS.md). */
function readServerEvent(frame: SseFrame): ServerEvent {
  const name = frame.event;
  if (name === null) {
    throw new CrucibleProtocolError(`a server event carried no event name: ${excerpt(frame.data)}`);
  }
  if (name === SERVER_STOPPING) {
    const where = 'the server.stopping event';
    const data = frameData(frame, where);
    return {
      id: frame.lastEventId === null ? null : frameId(frame, where),
      event: SERVER_STOPPING,
      reason: str(data, 'reason', where),
      at: optStr(data, 'at', where),
    };
  }
  const where = `server event ${String(frame.lastEventId)} (${name})`;
  const id = frameId(frame, where);
  const data = frameData(frame, where);
  if (name === 'snapshot') {
    const queue = objectField(data, 'queue', where);
    return {
      id,
      event: 'snapshot',
      gap: bool(data, 'gap', where),
      topics: strArray(data, 'topics', where),
      activity: readActivity(objectField(data, 'activity', where)),
      queue: {
        items: arrayField(queue, 'items', `${where}.queue`).map((row, index) =>
          readQueueItem(asObject(row, `${where}.queue.items[${index}]`), `${where}.queue.items[${index}]`),
        ),
        depth: num(queue, 'depth', `${where}.queue`),
      },
      tasks: arrayField(data, 'tasks', where).map((row, index) =>
        readTaskStatus(asObject(row, `${where}.tasks[${index}]`), `${where}.tasks[${index}]`),
      ),
    };
  }
  if (name === 'overflow') {
    return {
      id,
      event: 'overflow',
      lastEventId: num(data, 'last_event_id', where),
      limit: num(data, 'limit', where),
      message: str(data, 'message', where),
    };
  }
  const at = str(data, 'at', where);
  if (name === 'job.progress') {
    return {
      id,
      event: name,
      at,
      jobId: str(data, 'job_id', where),
      fraction: num(data, 'fraction', where),
      message: nullableStr(data, 'message', where),
    };
  }
  if (isOneOf(name, JOB_CHANGES)) {
    const error = name === 'job.failed' ? nullableObject(data, 'error', where) : null;
    const removal = name === 'job.removed' ? nullableObject(data, 'removal', where) : null;
    return {
      id,
      event: name,
      at,
      jobId: str(data, 'job_id', where),
      type: str(data, 'type', where),
      model: nullableStr(data, 'model', where),
      client: nullableStr(data, 'client', where),
      clientRef: nullableStr(data, 'client_ref', where),
      status: oneOf(str(data, 'status', where), JOB_STATES, `${where}.status`),
      position: name === 'job.queued' ? nullableNum(data, 'position', where) : null,
      waiting: name === 'job.queued' ? bool(data, 'waiting', where) : null,
      started: name === 'job.running' ? nullableStr(data, 'started', where) : null,
      artifacts: name === 'job.done' ? strArray(data, 'artifacts', where) : null,
      error: error === null ? null : readFailure(error, `${where}.error`),
      interruptedAt: name === 'job.interrupted' ? nullableStr(data, 'interrupted_at', where) : null,
      removal: readRemovalOrNull(removal, `${where}.removal`),
    };
  }
  if (isOneOf(name, QUEUE_CHANGES)) {
    const placed = name === 'queue.added' || name === 'queue.moved';
    return {
      id,
      event: name,
      at,
      jobId: str(data, 'job_id', where),
      depth: num(data, 'depth', where),
      kind: oneOf(str(data, 'kind', where), QUEUE_KINDS, `${where}.kind`),
      position: placed ? num(data, 'position', where) : null,
      waitedS: name === 'queue.started' ? num(data, 'waited_s', where) : null,
      reason: name === 'queue.removed' ? str(data, 'reason', where) : null,
      waiting:
        name === 'queue.waiting'
          ? { code: str(data, 'code', where), message: str(data, 'message', where) }
          : null,
      data,
    };
  }
  if (isOneOf(name, SESSION_CHANGES)) {
    const placed = name === 'session.queued' || name === 'session.moved';
    const ended = name === 'session.closed' || name === 'session.removed';
    return {
      id,
      event: name,
      at,
      sessionId: str(data, 'session_id', where),
      client: nullableStr(data, 'client', where),
      act: str(data, 'act', where),
      position: placed
        ? { position: num(data, 'position', where), of: num(data, 'of', where) }
        : null,
      reason: ended ? str(data, 'reason', where) : null,
      message: ended ? str(data, 'message', where) : null,
      waiting: name === 'session.waiting' ? readCardWait(data, where) : null,
      data,
    };
  }
  if (isOneOf(name, CARD_CHANGES)) {
    const loaded = name === 'card.loaded';
    const leaving = name === 'card.unloading' || name === 'card.unloaded';
    return {
      id,
      event: name,
      at,
      subject: str(data, 'subject', where),
      kind: nullableStr(data, 'kind', where),
      engine: nullableStr(data, 'engine', where),
      memoryBytesEstimate: loaded ? num(data, 'memory_bytes_estimate', where) : null,
      since: loaded || leaving ? nullableStr(data, 'since', where) : null,
      pids: leaving ? readNumbers(arrayField(data, 'pids', where), `${where}.pids`) : null,
    };
  }
  if (name === 'chat.in_flight') {
    return {
      id,
      event: name,
      at,
      inFlight: num(data, 'in_flight', where),
      byModel: readSampling(objectField(data, 'by_model', where), `${where}.by_model`),
    };
  }
  if (isOneOf(name, TASK_CHANGES)) {
    return { id, event: name, at, task: readTaskStatus(data, where) };
  }
  if (name === 'task.step') {
    return { id, event: name, at, taskId: str(data, 'task_id', where), step: readTaskStep(data, where) };
  }
  if (name === 'task.progress') {
    return {
      id,
      event: name,
      at,
      taskId: str(data, 'task_id', where),
      progress: readTaskProgress(data, where),
    };
  }
  if (name === 'settings.written') {
    return {
      id,
      event: name,
      at,
      act: nullableStr(data, 'act', where),
      client: nullableStr(data, 'client', where),
      changed: strArray(data, 'changed', where),
    };
  }
  return { id, event: 'unknown', kind: name, data };
}

function readResumable(body: Json): Resumable {
  const where = 'resumable';
  const model = objectField(body, 'model', where);
  const inputs = arrayField(body, 'inputs', where);
  return {
    resumeId: str(body, 'resume_id', where),
    jobType: str(body, 'job_type', where),
    model: {
      id: nullableStr(model, 'id', `${where}.model`),
      revision: nullableStr(model, 'revision', `${where}.model`),
    },
    formatVersion: num(body, 'format_version', where),
    inputs: inputs.map((raw, index) => {
      const input = asObject(raw, `${where}.inputs[${index}]`);
      return {
        name: str(input, 'name', `${where}.inputs[${index}]`),
        sha256: str(input, 'sha256', `${where}.inputs[${index}]`),
        bytes: num(input, 'bytes', `${where}.inputs[${index}]`),
      };
    }),
    params: objectField(body, 'params', where),
    unitsDone: num(body, 'units_done', where),
    unitsTotal: nullableNum(body, 'units_total', where),
    progress: str(body, 'progress', where),
    created: str(body, 'created', where),
    lastSaved: str(body, 'last_saved', where),
    expiresAt: str(body, 'expires_at', where),
    jobId: str(body, 'job_id', where),
    lastJobId: str(body, 'last_job_id', where),
    state: str(body, 'state', where),
  };
}

function readRole(body: Json): Pick<ServerInfo, 'role' | 'managedBy' | 'engine'> {
  const role: CrucibleRole = oneOf(
    str(body, 'role', 'info'),
    ['engine', 'orchestrator'] as const,
    'info.role',
  );
  if (role === 'engine') {
    const managed = nullableObject(body, 'managed_by', 'info');
    return {
      role,
      managedBy:
        managed === null
          ? null
          : {
              name: str(managed, 'name', 'info.managed_by'),
              url: str(managed, 'url', 'info.managed_by'),
            },
      engine: null,
    };
  }
  const engine = nullableObject(body, 'engine', 'info');
  return {
    role,
    managedBy: null,
    engine:
      engine === null
        ? null
        : {
            name: nullableStr(engine, 'name', 'info.engine'),
            url: str(engine, 'url', 'info.engine'),
            backend: nullableStr(engine, 'backend', 'info.engine'),
            owner: str(engine, 'owner', 'info.engine') as EngineOwner,
          },
  };
}

function readPagesEngine(body: Json, role: CrucibleRole): PagesEngine | null {
  if (role === 'orchestrator') return null;
  const block = objectField(body, 'pages_engine', 'info');
  const request = objectField(block, 'request', 'info.pages_engine');
  return {
    engine: nullableStr(block, 'engine', 'info.pages_engine'),
    installed: bool(block, 'installed', 'info.pages_engine'),
    detail: str(block, 'detail', 'info.pages_engine'),
    request: {
      model: str(request, 'model', 'info.pages_engine.request'),
      dpi: num(request, 'dpi', 'info.pages_engine.request'),
      maxPixels: num(request, 'max_pixels', 'info.pages_engine.request'),
      maxTokens: num(request, 'max_tokens', 'info.pages_engine.request'),
      temperature: num(request, 'temperature', 'info.pages_engine.request'),
      prompt: str(request, 'prompt', 'info.pages_engine.request'),
      dialect: str(request, 'dialect', 'info.pages_engine.request'),
      concurrency: num(request, 'concurrency', 'info.pages_engine.request'),
      truncatedFinishReason: str(
        request,
        'truncated_finish_reason',
        'info.pages_engine.request',
      ),
    },
  };
}

/** Where to send work, given an `info()`. */
export function engineOf(info: ServerInfo): EngineRef | null {
  if (info.role === 'engine') {
    return null;
  }
  if (info.engine === null) {
    throw new CrucibleProtocolError(
      `orchestrator_has_no_engine: ${info.server.name} is an orchestrator and manages ` +
        'no engine, so there is nothing here to send work to. Install one from its console.',
    );
  }
  return info.engine;
}

function readCapability(entry: Json, index: number): Capability {
  const where = `info.capabilities[${index}]`;
  const jobType = str(entry, 'job_type', where);
  const models = asArray(field(entry, 'models', where), `${where}.models`);
  if (jobType === 'llm') {
    const { rows, unreadableRows } = readRows(models, `${where}.models`, (row, at) =>
      readModelInfo(row, at),
    );
    return { jobType, models: rows, unreadableRows };
  }
  if (jobType === 'tts') {
    const { rows, unreadableRows } = readRows(models, `${where}.models`, (row, at) =>
      readVoiceInfo(row, at),
    );
    return { jobType, models: rows, unreadableRows };
  }
  try {
    return {
      jobType,
      models: models.map((model, at) =>
        readModel(asObject(model, `${where}.models[${at}]`), `${where}.models[${at}]`),
      ),
    };
  } catch (error) {
    return {
      jobType,
      models: models.map((model, at) =>
        asObject(model, `${where}.models[${at}]`),
      ),
      unreadable: error instanceof Error ? error.message : String(error),
    };
  }
}

function readRows<T>(
  models: readonly unknown[],
  where: string,
  read: (row: Json, at: string) => T,
): { rows: T[]; unreadableRows: UnreadableRow[] } {
  const rows: T[] = [];
  const unreadableRows: UnreadableRow[] = [];
  models.forEach((model, index) => {
    const at = `${where}[${index}]`;
    try {
      rows.push(read(asObject(model, at), at));
    } catch (error) {
      if (!(error instanceof CrucibleProtocolError)) throw error;
      const id =
        typeof model === 'object' && model !== null && !Array.isArray(model)
          ? (model as Json)['id']
          : undefined;
      unreadableRows.push({
        index,
        id: typeof id === 'string' ? id : null,
        raw: model,
        unreadable: error.message,
      });
    }
  });
  return { rows, unreadableRows };
}

function capabilityQuery(sizing: CapabilitySizing | undefined): string {
  if (sizing === undefined) return '';
  const params = new URLSearchParams({ class: requireText(sizing.class, 'class') });
  for (const [name, value] of [
    ['context_tokens', sizing.contextTokens],
    ['concurrency', sizing.concurrency],
  ] as const) {
    if (value === undefined) continue;
    if (!Number.isInteger(value)) {
      throw new CrucibleConfigError(
        name,
        `${name} is ${String(value)}; it is a whole number, and the server ` +
          'refuses anything below 1 by name.',
      );
    }
    params.set(name, String(value));
  }
  return `?${params.toString()}`;
}

function readCapabilityRecord(body: Json): CapabilityRecord {
  const where = 'capability';
  const classes = asArray(field(body, 'classes', where), `${where}.classes`);
  const rows = classes.map((entry, index) =>
    asObject(entry, `${where}.classes[${index}]`),
  );
  return {
    backendKind: str(body, 'backend_kind', where),
    totalBytes: num(body, 'total_bytes', where),
    desktopAllowanceBytes: num(body, 'desktop_allowance_bytes', where),
    classes: rows.map((row, index) =>
      readCapabilityRow(row, `${where}.classes[${index}]`),
    ),
  };
}

function readCapabilityRow(entry: Json, where: string): CapabilityRow {
  const raw = entry['route'];
  if (raw === undefined) {
    throw new CrucibleProtocolError(
      `${CAPABILITY_ROUTE_MISSING}: ${where} has no "route". Where a class runs ` +
        'decides whether its work costs GPU-minutes or money, so this is not ' +
        `something a client may fill in for ${str(entry, 'capability', where)}.`,
    );
  }
  if (typeof raw !== 'string' || !ROUTES.includes(raw as 'local')) {
    throw new CrucibleProtocolError(
      `${CAPABILITY_ROUTE_UNKNOWN}: ${where}.route is ${JSON.stringify(raw)}, ` +
        `which is not one of ${ROUTES.join(', ')}`,
    );
  }
  const route = raw as 'local' | 'upstream';
  return {
    capability: str(entry, 'capability', where),
    enabled: bool(entry, 'enabled', where),
    selected: str(entry, 'selected', where),
    reason: str(entry, 'reason', where),
    shortfallBytes: num(entry, 'shortfall_bytes', where),
    route,
    work: readCapabilityWork(nullableObject(entry, 'work', where), `${where}.work`),
    contextCeilings: readContextCeilings(entry, where),
  };
}

function readCapabilityWork(entry: Json | null, where: string): CapabilityWork | null {
  if (entry === null) return null;
  return {
    tokens: num(entry, 'tokens', where),
    concurrency: num(entry, 'concurrency', where),
    source: str(entry, 'source', where),
    from: str(entry, 'from', where),
  };
}

function readContextCeilings(entry: Json, where: string): ContextCeiling[] | null {
  const raw = nullableArray(entry, 'context_ceilings', where);
  if (raw === null) return null;
  return raw.map((item, index) => {
    const at = `${where}.context_ceilings[${index}]`;
    const ceiling = asObject(item, at);
    return {
      model: str(ceiling, 'model', at),
      tokens: num(ceiling, 'tokens', at),
      boundBy: str(ceiling, 'bound_by', at),
      servedContext: num(ceiling, 'served_context', at),
      memoryContext: nullableNum(ceiling, 'memory_context', at),
      concurrency: num(ceiling, 'concurrency', at),
    };
  });
}

const ROUTES = ['local', 'upstream'] as const;
const UPSTREAM_NAMES = ['anthropic', 'openai', 'ollama'] as const;

function readSettings(body: Json): SettingsDocument {
  const where = 'settings';
  const routesRaw = objectField(body, 'routes', where);
  const routes: Record<string, RouteSetting> = {};
  for (const name of Object.keys(routesRaw)) {
    const at = `${where}.routes.${name}`;
    const entry = objectField(routesRaw, name, `${where}.routes`);
    routes[name] = {
      route: oneOf(str(entry, 'route', at), ROUTES, `${at}.route`),
      model: nullableStr(entry, 'model', at),
    };
  }
  const upstreamsRaw = objectField(body, 'upstreams', where);
  const upstreams = {} as Record<UpstreamName, UpstreamSetting>;
  for (const name of UPSTREAM_NAMES) {
    const at = `${where}.upstreams.${name}`;
    const entry = objectField(upstreamsRaw, name, `${where}.upstreams`);
    const setting: {
      configured: boolean;
      keyHint?: string | null;
      url?: string | null;
    } = { configured: bool(entry, 'configured', at) };
    if ('key_hint' in entry) setting.keyHint = nullableStr(entry, 'key_hint', at);
    if ('url' in entry) setting.url = nullableStr(entry, 'url', at);
    upstreams[name] = setting;
  }
  const localModelsRaw = objectField(body, 'local_models', where);
  const localModels = Object.fromEntries(
    Object.keys(localModelsRaw).map((name) => [
      name,
      nullableStr(localModelsRaw, name, `${where}.local_models`),
    ]),
  );
  const choiceRows = objectField(body, 'local_model_choices', where);
  const localModelChoices = Object.fromEntries(
    Object.keys(choiceRows).map((name) => [
      name,
      asArray(
        field(choiceRows, name, `${where}.local_model_choices`),
        `${where}.local_model_choices.${name}`,
      ).map((raw, index) => {
        const at = `${where}.local_model_choices.${name}[${index}]`;
        const choice = asObject(raw, at);
        return {
          id: str(choice, 'id', at),
          memoryBytesEstimate: num(choice, 'memory_bytes_estimate', at),
          fits: bool(choice, 'fits', at),
          installed: bool(choice, 'installed', at),
        };
      }),
    ]),
  );
  return {
    localModels,
    localModelChoices,
    routes,
    upstreams,
    desktopAllowanceBytes: num(body, 'desktop_allowance_bytes', where),
    backendKind: str(body, 'backend_kind', where),
  };
}

function settingsPayload(patch: SettingsPatch): Json {
  const body: Json = {};
  if (patch.localModels !== undefined) body.local_models = { ...patch.localModels };
  if (patch.routes !== undefined) body.routes = { ...patch.routes };
  if (patch.upstreams !== undefined) body.upstreams = { ...patch.upstreams };
  if (patch.desktopAllowanceBytes !== undefined) {
    body.desktop_allowance_bytes = patch.desktopAllowanceBytes;
  }
  return body;
}

function upstreamTestRefusal(
  error: unknown,
): 'upstream_unreachable' | 'upstream_rejected' | 'upstream_unconfigured' | null {
  if (!(error instanceof CrucibleError)) return null;
  const code = (error as { code?: unknown }).code;
  if (typeof code !== 'string') return null;
  return (UPSTREAM_TEST_REFUSALS as readonly string[]).includes(code)
    ? (code as 'upstream_unreachable' | 'upstream_rejected' | 'upstream_unconfigured')
    : null;
}

function readModel(entry: Json, where: string): ModelDescriptor {
  return {
    id: str(entry, 'id', where),
    revision: str(entry, 'revision', where),
    source: str(entry, 'source', where),
    installed: bool(entry, 'installed', where),
    resident: bool(entry, 'resident', where),
    vramBytes: num(entry, 'vram_bytes', where),
  };
}

function readFailure(value: unknown, where: string): JobFailure {
  const entry = asObject(value, where);
  return { code: str(entry, 'code', where), message: str(entry, 'message', where) };
}

function readFailureOrNull(value: unknown, where: string): JobFailure | null {
  return value === null ? null : readFailure(value, where);
}

function readRemoval(entry: Json, where: string): RemovedData {
  return {
    reason: str(entry, 'reason', where),
    message: str(entry, 'message', where),
    waitedS: optNum(entry, 'waited_s', where),
    at: optStr(entry, 'at', where) ?? '',
  };
}

function readRemovalOrNull(entry: Json | null, where: string): RemovedData | null {
  return entry === null ? null : readRemoval(entry, where);
}

function readQueueItem(row: Json, where: string): QueueItem {
  return {
    position: num(row, 'position', where),
    jobId: str(row, 'job_id', where),
    type: str(row, 'type', where),
    model: nullableStr(row, 'model', where),
    client: nullableStr(row, 'client', where),
    clientRef: nullableStr(row, 'client_ref', where),
    submitted: str(row, 'submitted', where),
    waitedS: num(row, 'waited_s', where),
    maxWaitS: num(row, 'max_wait_s', where),
    expiresAt: str(row, 'expires_at', where),
    session: nullableStr(row, 'session', where),
    kind: oneOf(str(row, 'kind', where), QUEUE_KINDS, `${where}.kind`),
    waitingFor: readWaitingFor(row, where),
  };
}

/** A `waiting` event's data, flat; `next_check_at` is absent from a 1.0.82 server. */
function readCardWait(data: Json, where: string): CardWaitData {
  return {
    code: str(data, 'code', where),
    message: str(data, 'message', where),
    details: nullableObject(data, 'details', where),
    since: str(data, 'since', where),
    nextCheckAt: optStr(data, 'next_check_at', where),
  };
}

function readWaitingFor(row: Json, where: string): QueueWaitingFor | null {
  if (!('waiting_for' in row)) return null; // an older server never says
  const found = nullableObject(row, 'waiting_for', where);
  if (found === null) return null;
  const at = `${where}.waiting_for`;
  return {
    code: str(found, 'code', at),
    message: str(found, 'message', at),
    details: nullableObject(found, 'details', at),
    since: str(found, 'since', at),
    nextCheckAt: str(found, 'next_check_at', at),
  };
}

function readQueueEvent(rawId: string | null, rawName: string | null, rawData: string): QueueEvent {
  const id = Number(rawId);
  if (rawId === null || !Number.isInteger(id) || id < 1) {
    throw new CrucibleProtocolError(`a queue event carried no usable id: ${JSON.stringify(rawId)}`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawData);
  } catch {
    throw new CrucibleProtocolError(`queue event ${id} has non-JSON data: ${excerpt(rawData)}`);
  }
  const where = `queue event ${id} (${String(rawName)})`;
  const data = asObject(parsed, where);
  if (rawName === 'snapshot') {
    return {
      id,
      event: 'snapshot',
      items: arrayField(data, 'items', where).map((row, index) =>
        readQueueItem(asObject(row, `${where}.items[${index}]`), `${where}.items[${index}]`),
      ),
      depth: num(data, 'depth', where),
    };
  }
  if (rawName === 'added' || rawName === 'moved' || rawName === 'started' || rawName === 'removed') {
    return { id, event: rawName, jobId: str(data, 'job_id', where), depth: num(data, 'depth', where), data };
  }
  return { id, event: 'unknown', kind: String(rawName), data };
}

function readChunksDone(body: Json): number[] {
  return arrayField(body, 'chunks_done', 'job').map((entry, index) => {
    if (typeof entry !== 'number' || !Number.isInteger(entry)) {
      throw new CrucibleProtocolError(
        `job.chunks_done[${index}] is not an integer chunk index; it is ` +
          'what a resume differences against, so a rounded one would ' +
          're-render a chunk that is already on disk',
      );
    }
    return entry;
  });
}

function readProvenance(entry: Json, member: string): Provenance {
  const where = `${member}.provenance.json`;
  const server = objectField(entry, 'server', where);
  const model = nullableObject(entry, 'model', where);
  return {
    server: {
      name: str(server, 'name', `${where}.server`),
      version: str(server, 'version', `${where}.server`),
    },
    backend: str(entry, 'backend', where),
    job_type: str(entry, 'job_type', where),
    model: model === null ? null : readProvenanceModel(model, where),
    params: objectField(entry, 'params', where),
    started: nullableStr(entry, 'started', where),
    finished: str(entry, 'finished', where),
  };
}

function readProvenanceModel(
  model: Json,
  where: string,
): NonNullable<Provenance['model']> {
  return {
    id: str(model, 'id', `${where}.model`),
    revision: nullableStr(model, 'revision', `${where}.model`),
    fingerprint: nullableStr(model, 'fingerprint', `${where}.model`),
  };
}

function readEvent(rawId: string | null, rawName: string | null, rawData: string): JobEvent {
  if (rawId === null) {
    throw new CrucibleProtocolError(`an SSE frame carried no id: ${excerpt(rawData)}`);
  }
  const id = Number(rawId);
  if (!Number.isInteger(id) || id < 1) {
    throw new CrucibleProtocolError(`SSE frame id ${JSON.stringify(rawId)} is not a positive integer`);
  }
  if (rawName === null) {
    throw new CrucibleProtocolError(`SSE frame ${id} carried no event name`);
  }
  const name = rawName;
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawData);
  } catch {
    throw new CrucibleProtocolError(`SSE frame ${id} (${name}) has non-JSON data: ${excerpt(rawData)}`);
  }
  const data = asObject(parsed, `event ${id} (${name}) data`);
  const where = `event ${id} (${name})`;

  if (!(EVENT_NAMES as readonly string[]).includes(name)) {
    return { id, event: 'unknown', kind: name, data };
  }
  const known = name as (typeof EVENT_NAMES)[number];

  switch (known) {
    case 'queued':
      return {
        id,
        event: 'queued',
        data: { position: num(data, 'position', where), of: optNum(data, 'of', where) },
      };
    case 'started':
      return { id, event: 'started', data: { waitedS: num(data, 'waited_s', where) } };
    case 'removed':
      return { id, event: 'removed', data: readRemoval(data, where) };
    case 'warming':
      return { id, event: 'warming', data: { message: str(data, 'message', where) } };
    case 'progress':
      return { id, event: 'progress', data: readProgress(data, where) };
    case 'chunk':
      return { id, event: 'chunk', data: readChunk(data, where) };
    case 'waiting':
      return { id, event: 'waiting', data: readCardWait(data, where) };
    case 'artifact':
      return { id, event: 'artifact', data: { name: str(data, 'name', where) } };
    case 'done':
      return { id, event: 'done', data: readDone(data, where) };
    case 'failed':
      return { id, event: 'failed', data: { error: readFailure(field(data, 'error', where), `${where}.error`) } };
    case 'cancelled':
      return {
        id,
        event: 'cancelled',
        data: { status: oneOf(str(data, 'status', where), ['cancelled'], `${where}.status`) },
      };
  }
}

function readProgress(data: Json, where: string): ProgressData {
  const fraction = num(data, 'fraction', where);
  const message = str(data, 'message', where);
  const extra: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(data)) {
    if (key === 'fraction' || key === 'message') continue;
    extra[key] = value;
  }
  return { fraction, message, extra };
}

function readChunk(data: Json, where: string): ChunkData {
  return {
    index: num(data, 'index', where),
    seconds: num(data, 'seconds', where),
    chars: num(data, 'chars', where),
    charsPerSec: num(data, 'chars_per_sec', where),
    tokens: nullableNum(data, 'tokens', where),
    capped: nullableBool(data, 'capped', where),
    take: num(data, 'take', where),
    guard: nullableObject(data, 'guard', where),
    pauseCuts: readPauseCuts(data, where),
  };
}

function readPauseCuts(data: Json, where: string): PauseCut[] | null {
  if (!('pause_cuts' in data)) return null; // a server before 1.0.80 never says
  const cuts = nullableArray(data, 'pause_cuts', where);
  if (cuts === null) return null;
  return cuts.map((entry, index) => {
    const at = `${where}.pause_cuts[${index}]`;
    const cut = asObject(entry, at);
    return { atS: num(cut, 'at_s', at), fromS: num(cut, 'from_s', at), toS: num(cut, 'to_s', at) };
  });
}

function readDone(data: Json, where: string): DoneData {
  const hasArtifacts = 'artifacts' in data;
  const hasResident = 'resident' in data;
  if (!hasArtifacts && !hasResident) {
    throw new CrucibleProtocolError(
      `${where} carries neither "artifacts" nor "resident"; a done event has to ` +
        'say what finished',
    );
  }
  const extra: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(data)) {
    if (key === 'artifacts' || key === 'resident') continue;
    extra[key] = value;
  }
  const done: {
    artifacts?: readonly string[];
    resident?: string | null;
    extra: Readonly<Record<string, unknown>>;
  } = { extra };
  if (hasArtifacts) done.artifacts = strArray(data, 'artifacts', where);
  if (hasResident) done.resident = nullableStr(data, 'resident', where);
  return done;
}

/**
 * An `align` job's `alignment.json`, read: one result per window, in the order the job listed them.
 */
export function readAlignment(bytes: Uint8Array): Alignment {
  const where = 'alignment.json';
  let raw: unknown;
  try {
    raw = JSON.parse(new TextDecoder().decode(bytes));
  } catch (err) {
    throw new CrucibleProtocolError(`${where} is not JSON: ${(err as Error).message}`);
  }
  const body = asObject(raw, where);
  const windows: AlignWindowResult[] = asArray(field(body, 'chunks', where), `${where}.chunks`).map(
    (entry, position) => {
      const at = `${where}.chunks[${position}]`;
      const row = asObject(entry, at);
      const index = num(row, 'index', at);
      const hasItems = 'items' in row;
      const hasError = 'error' in row;
      if (hasItems === hasError) {
        throw new CrucibleProtocolError(
          `${at} (window ${index}) must carry exactly one of "items" and "error"; ` +
            'it is how a caller tells a placed window from a failed one',
        );
      }
      if (hasError) return { index, items: null, error: str(row, 'error', at) };
      const items: AlignItem[] = asArray(row['items'], `${at}.items`).map((item, i) => {
        const it = asObject(item, `${at}.items[${i}]`);
        return {
          text: str(it, 'text', `${at}.items[${i}]`),
          start: num(it, 'start', `${at}.items[${i}]`),
          end: num(it, 'end', `${at}.items[${i}]`),
        };
      });
      return { index, items, error: null };
    },
  );
  return { model: str(body, 'model', where), windows };
}

function numberMap(entry: Json | null, where: string): Record<string, number> {
  return entry === null ? {} : readSampling(entry, where);
}

/** A finished `image` job's effective parameters, read out of its `done` frame so the picture can be made again. */
export function readImageResult(done: DoneData): ImageResult {
  const where = 'the image done event';
  const image = objectField(done.extra as Json, 'image', where);
  const at = `${where}.image`;
  return {
    model: str(image, 'model', at),
    hfRepo: str(image, 'hf_repo', at),
    revision: str(image, 'revision', at),
    backend: str(image, 'backend', at),
    engine: str(image, 'engine', at),
    dtype: str(image, 'dtype', at),
    prompt: str(image, 'prompt', at),
    negativePrompt: nullableStr(image, 'negative_prompt', at),
    width: num(image, 'width', at),
    height: num(image, 'height', at),
    seed: num(image, 'seed', at),
    steps: num(image, 'steps', at),
    guidance: num(image, 'guidance', at),
    imageStrength: nullableNum(image, 'image_strength', at),
    input: nullableStr(image, 'input', at),
    mask: optStr(image, 'mask', at),
    maskBlur: optNum(image, 'mask_blur', at),
    maskCoverage: optNum(image, 'mask_coverage', at),
    maskBlendSteps: optNum(image, 'mask_blend_steps', at),
    maskOutsideDrift: optNum(image, 'mask_outside_drift', at),
    seconds: nullableNum(image, 'seconds', at),
    stageSeconds: numberMap(nullableObject(image, 'stage_seconds', at), `${at}.stage_seconds`),
    peakBytes: nullableNum(image, 'peak_bytes', at),
    stagePeakBytes: numberMap(nullableObject(image, 'stage_peak_bytes', at), `${at}.stage_peak_bytes`),
    memoryBytesEstimate: num(image, 'memory_bytes_estimate', at),
    memoryBasis: str(image, 'memory_basis', at),
    artifacts: done.artifacts ?? [],
    promptCache: readPromptCache(image, at),
  };
}

/** A finished `audio` job's effective parameters, read out of its `done` frame so the sound can be made again. */
export function readAudioResult(done: DoneData): AudioResult {
  const where = 'the audio done event';
  const audio = objectField(done.extra as Json, 'audio', where);
  const at = `${where}.audio`;
  return {
    model: str(audio, 'model', at),
    kind: oneOf(str(audio, 'kind', at), ['sfx', 'music', 'song'] as const, `${at}.kind`),
    hfRepo: str(audio, 'hf_repo', at),
    revision: str(audio, 'revision', at),
    backend: str(audio, 'backend', at),
    engine: str(audio, 'engine', at),
    dtype: str(audio, 'dtype', at),
    prompt: nullableStr(audio, 'prompt', at),
    tags: nullableStr(audio, 'tags', at),
    lyrics: nullableStr(audio, 'lyrics', at),
    durationS: nullableNum(audio, 'duration_s', at),
    seed: num(audio, 'seed', at),
    steps: nullableNum(audio, 'steps', at),
    cfg: nullableNum(audio, 'cfg', at),
    format: oneOf(str(audio, 'format', at), ['flac', 'wav', 'mp3'] as const, `${at}.format`),
    instrumental: optBool(audio, 'instrumental', at),
    artifact: str(audio, 'artifact', at),
    score: nullableStr(audio, 'score', at),
    audioSeconds: nullableNum(audio, 'audio_seconds', at),
    sampleRate: nullableNum(audio, 'sample_rate', at),
    channels: nullableNum(audio, 'channels', at),
    seconds: nullableNum(audio, 'seconds', at),
    stageSeconds: numberMap(nullableObject(audio, 'stage_seconds', at), `${at}.stage_seconds`),
    peakBytes: nullableNum(audio, 'peak_bytes', at),
    stagePeakBytes: numberMap(nullableObject(audio, 'stage_peak_bytes', at), `${at}.stage_peak_bytes`),
    memoryBytesEstimate: num(audio, 'memory_bytes_estimate', at),
    memoryBasis: str(audio, 'memory_basis', at),
    artifacts: done.artifacts ?? [],
  };
}

function readPlaygroundPage(page: Json, where: string): PlaygroundPage {
  return {
    jobType: str(page, 'job_type', where),
    id: str(page, 'id', where),
    name: str(page, 'name', where),
    media: str(page, 'media', where),
    kind: str(page, 'kind', where),
    makes: str(page, 'makes', where),
    standing: oneOf(str(page, 'standing', where), ['ready', 'download', 'unavailable'] as const, `${where}.standing`),
    available: bool(page, 'available', where),
    reason: nullableStr(page, 'reason', where),
    downloadBytes: nullableNum(page, 'download_bytes', where),
    fields: arrayField(page, 'fields', where).map((value, index) =>
      readPlaygroundField(asObject(value, `${where}.fields[${index}]`), `${where}.fields[${index}]`)),
  };
}

function readPlaygroundField(raw: Json, where: string): PlaygroundField {
  const suggestions = optRawArray(raw, 'suggestions', where);
  const conflicts = optObject(raw, 'conflicts', where);
  return {
    name: str(raw, 'name', where),
    label: str(raw, 'label', where),
    kind: str(raw, 'kind', where),
    required: bool(raw, 'required', where),
    default: raw['default'] ?? null,
    placeholder: optStr(raw, 'placeholder', where),
    hint: optStr(raw, 'hint', where),
    min: optNum(raw, 'min', where),
    max: optNum(raw, 'max', where),
    step: optNum(raw, 'step', where),
    options: optRawArray(raw, 'options', where),
    suggestions: suggestions === null ? null : suggestions.map((value, index) => {
      const at = `${where}.suggestions[${index}]`;
      const group = asObject(value, at);
      return { group: str(group, 'group', at), tags: strArray(group, 'tags', at) };
    }),
    conflicts: conflicts === null ? null : Object.fromEntries(
      Object.entries(conflicts).map(([tag, rules]) => [
        tag,
        asArray(rules, `${where}.conflicts.${tag}`).map((rule, index) => {
          const at = `${where}.conflicts.${tag}[${index}]`;
          const row = asObject(rule, at);
          return { tag: str(row, 'tag', at), why: str(row, 'why', at) };
        }),
      ]),
    ),
  };
}

/** An array that may be absent (not every field kind has it), else exactly an array. */
function optRawArray(object: Json, key: string, where: string): unknown[] | null {
  const value = object[key];
  if (value === undefined || value === null) return null;
  return asArray(value, `${where}.${key}`);
}

function readPlaygroundPreset(preset: Json, where: string): PlaygroundPreset {
  const params = objectField(preset, 'params', where);
  for (const [key, value] of Object.entries(params)) {
    if (typeof value !== 'string' && typeof value !== 'number' && typeof value !== 'boolean') {
      throw new CrucibleProtocolError(`${where}.params.${key} is not text, a number or true/false`);
    }
  }
  return {
    name: str(preset, 'name', where),
    params: params as Record<string, string | number | boolean>,
    savedAt: str(preset, 'saved_at', where),
  };
}

/** A finished `segment` job's effective parameters, read out of its `done` frame. */
export function readSegmentResult(done: DoneData): SegmentResult {
  const where = 'the segment done event';
  const segment = objectField(done.extra as Json, 'segment', where);
  const at = `${where}.segment`;
  return {
    model: str(segment, 'model', at),
    kind: oneOf(str(segment, 'kind', at), ['cutout', 'select'] as const, `${at}.kind`),
    hfRepo: str(segment, 'hf_repo', at),
    revision: str(segment, 'revision', at),
    backend: str(segment, 'backend', at),
    engine: str(segment, 'engine', at),
    dtype: str(segment, 'dtype', at),
    input: str(segment, 'input', at),
    width: num(segment, 'width', at),
    height: num(segment, 'height', at),
    points: readSegmentPoints(nullableArray(segment, 'points', at), `${at}.points`),
    box: readNumbers(nullableArray(segment, 'box', at), `${at}.box`),
    mask: str(segment, 'mask', at),
    cutout: str(segment, 'cutout', at),
    score: nullableNum(segment, 'score', at),
    multimask: nullableBool(segment, 'multimask', at),
    coverage: nullableNum(segment, 'coverage', at),
    seconds: nullableNum(segment, 'seconds', at),
    stageSeconds: numberMap(nullableObject(segment, 'stage_seconds', at), `${at}.stage_seconds`),
    peakBytes: nullableNum(segment, 'peak_bytes', at),
    stagePeakBytes: numberMap(nullableObject(segment, 'stage_peak_bytes', at), `${at}.stage_peak_bytes`),
    memoryBytesEstimate: num(segment, 'memory_bytes_estimate', at),
    memoryBasis: str(segment, 'memory_basis', at),
    artifacts: done.artifacts ?? [],
  };
}

/** A finished `video` job's effective parameters, read out of its `done` frame so the clip can be made again. */
export function readVideoResult(done: DoneData): VideoResult {
  const where = 'the video done event';
  const video = objectField(done.extra as Json, 'video', where);
  const at = `${where}.video`;
  const transformer = nullableObject(video, 'transformer', at);
  return {
    model: str(video, 'model', at),
    hfRepo: str(video, 'hf_repo', at),
    revision: str(video, 'revision', at),
    transformer:
      transformer === null
        ? null
        : {
            hfRepo: str(transformer, 'hf_repo', `${at}.transformer`),
            revision: str(transformer, 'revision', `${at}.transformer`),
            file: str(transformer, 'file', `${at}.transformer`),
            sha256: str(transformer, 'sha256', `${at}.transformer`),
          },
    backend: str(video, 'backend', at),
    engine: str(video, 'engine', at),
    dtype: str(video, 'dtype', at),
    mode: oneOf(str(video, 'mode', at), ['text-to-video', 'image-to-video'] as const, `${at}.mode`),
    prompt: str(video, 'prompt', at),
    input: nullableStr(video, 'input', at),
    width: num(video, 'width', at),
    height: num(video, 'height', at),
    numFrames: num(video, 'num_frames', at),
    fps: num(video, 'fps', at),
    durationS: num(video, 'duration_s', at),
    videoTokens: num(video, 'video_tokens', at),
    seed: num(video, 'seed', at),
    steps: num(video, 'steps', at),
    refineSteps: optNum(video, 'refine_steps', at),
    audio: bool(video, 'audio', at),
    audioSeconds: nullableNum(video, 'audio_seconds', at),
    audioSampleRate: nullableNum(video, 'audio_sample_rate', at),
    audioChannels: nullableNum(video, 'audio_channels', at),
    artifact: str(video, 'artifact', at),
    bytes: nullableNum(video, 'bytes', at),
    encoder: nullableStr(video, 'encoder', at),
    seconds: nullableNum(video, 'seconds', at),
    stageSeconds: numberMap(nullableObject(video, 'stage_seconds', at), `${at}.stage_seconds`),
    peakBytes: nullableNum(video, 'peak_bytes', at),
    stagePeakBytes: numberMap(nullableObject(video, 'stage_peak_bytes', at), `${at}.stage_peak_bytes`),
    memoryBytesEstimate: num(video, 'memory_bytes_estimate', at),
    memoryBasis: str(video, 'memory_basis', at),
    stageMemoryBytes: numberMap(nullableObject(video, 'stage_memory_bytes', at), `${at}.stage_memory_bytes`),
    promptCache: readPromptCache(video, at),
    artifacts: done.artifacts ?? [],
  };
}

function readSegmentPoints(entries: unknown[] | null, where: string): SegmentPoint[] | null {
  if (entries === null) return null;
  return entries.map((entry, index) => {
    const at = `${where}[${index}]`;
    const point = asObject(entry, at);
    const label = num(point, 'label', at);
    if (label !== 0 && label !== 1) {
      throw new CrucibleProtocolError(`${at}.label is ${label}, not 0 or 1`);
    }
    return { x: num(point, 'x', at), y: num(point, 'y', at), label: label as 0 | 1 };
  });
}

function readNumbers(entries: unknown[] | null, where: string): number[] | null {
  if (entries === null) return null;
  return entries.map((entry, index) => {
    if (typeof entry !== 'number' || !Number.isFinite(entry)) {
      throw new CrucibleProtocolError(`${where}[${index}] is not a number`);
    }
    return entry;
  });
}

function readPromptCache(image: Json, where: string): 'hit' | 'miss' | null {
  const said = optStr(image, 'prompt_cache', where);
  if (said === null || said === 'hit' || said === 'miss') return said;
  throw new CrucibleProtocolError(`${where}.prompt_cache is ${JSON.stringify(said)}, not "hit" or "miss"`);
}

/** A finished `tts` job's terminal news, read out of its `done` frame. */
export function readRenderResult(done: DoneData): RenderResult {
  const where = 'the tts done event';
  const extra = done.extra as Json;
  const artifacts = done.artifacts;
  if (artifacts === undefined) {
    throw new CrucibleProtocolError(
      `${where} carries no "artifacts"; a render publishes one <index>.flac per ` +
        'rendered chunk, so the list is how a caller knows what to collect',
    );
  }
  const failed = asArray(field(extra, 'failed', where), `${where}.failed`);
  const sampling = objectField(extra, 'sampling', where);
  const voice = objectField(extra, 'voice', where);
  return {
    rendered: num(extra, 'rendered', where),
    failed: failed.map((entry, index) =>
      readRenderFailure(asObject(entry, `${where}.failed[${index}]`), `${where}.failed[${index}]`),
    ),
    take: num(extra, 'take', where),
    sampling: readSampling(sampling, `${where}.sampling`),
    voice: readRenderVoice(voice, `${where}.voice`),
    width: nullableNum(extra, 'width', where),
    sampleRate: num(extra, 'sample_rate', where),
    artifacts,
  };
}

function readSampling(entry: Json, where: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const key of Object.keys(entry as Record<string, unknown>)) {
    out[key] = num(entry, key, where);
  }
  return out;
}

function readRenderVoice(
  entry: Json,
  where: string,
): { id: string; identity: string; identityBasis: string } {
  return {
    id: str(entry, 'id', where),
    identity: str(entry, 'identity', where),
    identityBasis: str(entry, 'identity_basis', where),
  };
}

function readRenderFailure(entry: Json, where: string): RenderFailure {
  return {
    index: num(entry, 'index', where),
    message: str(entry, 'message', where),
  };
}

function readModelInfo(entry: Json, where: string): ModelInfo {
  return {
    id: str(entry, 'id', where),
    family: str(entry, 'family', where),
    paramsB: num(entry, 'params_b', where),
    revision: nullableStr(entry, 'revision', where),
    fingerprint: nullableStr(entry, 'fingerprint', where),
    modalities: strArray(entry, 'modalities', where),
    backendSupported: bool(entry, 'backend_supported', where),
    installed: bool(entry, 'installed', where),
    weightsOf: nullableStr(entry, 'weights_of', where),
    resident: bool(entry, 'resident', where),
    loadable: bool(entry, 'loadable', where),
    reason: optStr(entry, 'reason', where),
    memoryBytesEstimate: nullableNum(entry, 'memory_bytes_estimate', where),
    contextDefault: num(entry, 'context_default', where),
    maxModelLen: nullableNum(entry, 'max_model_len', where),
  };
}

function readVoiceInfo(entry: Json, where: string): VoiceInfo {
  const rawNeedsReference = entry['needs_reference'];
  if (rawNeedsReference === undefined) {
    throw new CrucibleProtocolError(
      `${VOICES_NEEDS_REFERENCE_MISSING}: ${where} has no "needs_reference", so ` +
        `nothing says whether a load of ${str(entry, 'id', where)} must carry a clip.`,
    );
  }
  if (typeof rawNeedsReference !== 'boolean') {
    throw new CrucibleProtocolError(
      `${VOICES_NEEDS_REFERENCE_UNKNOWN}: ${where}.needs_reference is ` +
        `${JSON.stringify(rawNeedsReference)}, which is not a boolean`,
    );
  }
  const needsReference = rawNeedsReference;
  return {
    id: str(entry, 'id', where),
    display: str(entry, 'display', where),
    kind: nullableStr(entry, 'kind', where),
    orphan: bool(entry, 'orphan', where),
    language: nullableStr(entry, 'language', where),
    narratorEngine: nullableStr(entry, 'narrator_engine', where),
    backendSupported: bool(entry, 'backend_supported', where),
    installed: bool(entry, 'installed', where),
    resident: bool(entry, 'resident', where),
    loadable: bool(entry, 'loadable', where),
    reason: nullableStr(entry, 'reason', where),
    revision: nullableStr(entry, 'revision', where),
    fingerprint: nullableStr(entry, 'fingerprint', where),
    memoryBytesEstimate: nullableNum(entry, 'memory_bytes_estimate', where),
    estimateBasis: nullableStr(entry, 'estimate_basis', where),
    maxChars: nullableNum(entry, 'max_chars', where),
    sampleRate: nullableNum(entry, 'sample_rate', where),
    takes: num(entry, 'takes', where),
    serving: readVoiceServing(entry, where),
    needsReference,
    pace: readNullableVoicePace(entry, where),
    ref: optStr(entry, 'ref', where),
    latestRevision: optStr(entry, 'latest_revision', where),
    updateAvailable: optBool(entry, 'update_available', where) ?? false,
    updateCheckedAt: optStr(entry, 'update_checked_at', where),
    updateError: optStr(entry, 'update_error', where),
    sampling: readVoiceSampling(entry, where),
    edgeFadeMs: readVoiceEdgeFade(entry, where),
    chunkGap: readVoiceChunkGap(entry, where),
    referenceSecondsCap: optNum(entry, 'reference_seconds_cap', where),
    allowedControls: optStrArray(entry, 'allowed_controls', where),
  };
}

function readVoiceSampling(entry: Json, where: string): VoiceSampling | null {
  const block = optObject(entry, 'sampling', where);
  if (block === null) return null;
  const at = `${where}.sampling`;
  return {
    temperature: num(block, 'temperature', at),
    topP: num(block, 'top_p', at),
    topK: num(block, 'top_k', at),
  };
}

function readVoiceEdgeFade(entry: Json, where: string): VoiceEdgeFade | null {
  const block = optObject(entry, 'edge_fade_ms', where);
  if (block === null) return null;
  const at = `${where}.edge_fade_ms`;
  return { in: num(block, 'in', at), out: num(block, 'out', at) };
}

function readVoiceChunkGap(entry: Json, where: string): VoiceChunkGap | null {
  const block = optObject(entry, 'chunk_gap', where);
  if (block === null) return null;
  const at = `${where}.chunk_gap`;
  return {
    injectS: num(block, 'inject_s', at),
    targetJoinS: num(block, 'target_join_s', at),
    modelSelfTailS: num(block, 'model_self_tail_s', at),
    readerSentenceGapS: nullableNum(block, 'reader_sentence_gap_s', at),
    modelInternalGapS: nullableNum(block, 'model_internal_gap_s', at),
    rule: str(block, 'rule', at),
    method: str(block, 'method', at),
    source: str(block, 'source', at),
    measuredOn: str(block, 'measured_on', at),
  };
}

function readVoiceServing(entry: Json, where: string): VoiceServing | null {
  const block = nullableObject(entry, 'serving', where);
  if (block === null) return null;
  const at = `${where}.serving`;
  return {
    maxNumSeqs: num(block, 'max_num_seqs', at),
    maxNumSeqsNote: str(block, 'max_num_seqs_note', at),
    memFraction: nullableNum(block, 'mem_fraction', at),
    memFractionNote: nullableStr(block, 'mem_fraction_note', at),
    contextLength: nullableNum(block, 'context_length', at),
    contextLengthNote: nullableStr(block, 'context_length_note', at),
  };
}

function readNullableVoicePace(entry: Json, where: string): VoicePace | null {
  const block = nullableObject(entry, 'pace', where);
  return block === null ? null : readVoicePace(block, `${where}.pace`);
}

function readVoicePace(entry: Json, where: string): VoicePace {
  const paceCharsPerSec = nullableNum(entry, 'pace_chars_per_sec', where);
  const maxCharsPerSec = nullableNum(entry, 'max_chars_per_sec', where);
  const minCharsPerSec = nullableNum(entry, 'min_chars_per_sec', where);
  const stated = [paceCharsPerSec, maxCharsPerSec, minCharsPerSec]
    .filter((rate) => rate !== null).length;
  if (stated !== 0 && stated !== 3) {
    throw new CrucibleProtocolError(
      `${where} states ${stated} of its 3 rates. A band is a measured pace and the two ` +
        'edges derived from it, so a subset is a band nobody finished writing — and a ' +
        'client packing to an edge with no centre is the shape the group rule exists to ' +
        'prevent.',
    );
  }
  return {
    paceCharsPerSec,
    maxCharsPerSec,
    minCharsPerSec,
    targetChars: nullableNum(entry, 'target_chars', where),
    safeMinChars: nullableNum(entry, 'safe_min_chars', where),
    safeMaxChars: nullableNum(entry, 'safe_max_chars', where),
  };
}

function readAcceleratorState(body: Json): AcceleratorState {
  const where = 'accelerator';
  const gpu = objectField(body, 'gpu', where);
  const resident = nullableObject(body, 'resident', where);
  return {
    backend: str(body, 'backend', where),
    gpu: {
      vendor: str(gpu, 'vendor', 'accelerator.gpu'),
      name: str(gpu, 'name', 'accelerator.gpu'),
      totalBytes: num(gpu, 'total_bytes', 'accelerator.gpu'),
    },
    freeBytes: num(body, 'free_bytes', where),
    usedBytes: num(body, 'used_bytes', where),
    desktopAllowanceBytes: num(body, 'desktop_allowance_bytes', where),
    unattributedBytes: nullableNum(body, 'unattributed_bytes', where),
    resident: resident === null ? null : readAcceleratorResident(resident),
    holders: arrayField(body, 'holders', where).map((holder, index) =>
      readAcceleratorHolder(
        asObject(holder, `accelerator.holders[${index}]`),
        `accelerator.holders[${index}]`,
      ),
    ),
    detail: str(body, 'detail', where),
  };
}

function readAcceleratorResident(entry: Json): AcceleratorResident {
  const where = 'accelerator.resident';
  return {
    kind: str(entry, 'kind', where),
    id: str(entry, 'id', where),
    since: str(entry, 'since', where),
    memoryBytesEstimate: num(entry, 'memory_bytes_estimate', where),
  };
}

function readAcceleratorHolder(entry: Json, where: string): AcceleratorHolder {
  return {
    pid: num(entry, 'pid', where),
    name: str(entry, 'name', where),
    bytes: nullableNum(entry, 'bytes', where),
    ownedByCrucible: bool(entry, 'owned_by_crucible', where),
  };
}

function readChatMessages(messages: unknown): Array<{ role: string; content: string }> {
  if (messages === undefined || messages === null) {
    throw new CrucibleConfigError('messages', 'is required and was not given');
  }
  if (!Array.isArray(messages)) {
    throw new CrucibleConfigError('messages', `must be an array, got ${typeof messages}`);
  }
  if (messages.length === 0) {
    throw new CrucibleConfigError('messages', 'is required and was empty');
  }
  return messages.map((entry, index) => {
    if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
      throw new CrucibleConfigError(`messages[${index}]`, 'must be {role, content}');
    }
    const message = entry as Partial<ChatMessage>;
    const role = message.role;
    if (typeof role !== 'string' || !(CHAT_ROLES as readonly string[]).includes(role)) {
      throw new CrucibleConfigError(
        `messages[${index}].role`,
        `must be one of ${CHAT_ROLES.join(', ')}, got ${JSON.stringify(role)}`,
      );
    }
    const content = message.content;
    if (typeof content !== 'string') {
      throw new CrucibleConfigError(
        `messages[${index}].content`,
        `must be a string, got ${typeof content}`,
      );
    }
    return { role, content };
  });
}

function readChatResponse(body: Json): ChatResponse {
  const where = 'chat';
  const choices = asArray(field(body, 'choices', where), 'chat.choices');
  const first = choices[0];
  if (first === undefined) {
    throw new CrucibleProtocolError('chat.choices is empty; the engine returned no completion');
  }
  const choice = asObject(first, 'chat.choices[0]');
  const message = objectField(choice, 'message', 'chat.choices[0]');
  const usage = optObject(body, 'usage', where);
  const finishReason = str(choice, 'finish_reason', 'chat.choices[0]');
  refuseReasoningWithoutContent(message, finishReason);
  return {
    id: optStr(body, 'id', where),
    model: optStr(body, 'model', where),
    content: str(message, 'content', 'chat.choices[0].message'),
    finishReason,
    usage:
      usage === null
        ? null
        : {
            promptTokens: optNum(usage, 'prompt_tokens', 'chat.usage'),
            completionTokens: optNum(usage, 'completion_tokens', 'chat.usage'),
            totalTokens: optNum(usage, 'total_tokens', 'chat.usage'),
          },
  };
}

const DECIDE_TYPES = ['choice', 'score', 'yesno'] as const;

function readDecideQuestions(value: unknown): Record<string, DecideQuestion> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new CrucibleConfigError('questions', 'must be an object of question name -> question');
  }
  const out: Record<string, DecideQuestion> = {};
  for (const [name, raw] of Object.entries(value as Record<string, unknown>)) {
    const where = `questions.${name}`;
    if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
      throw new CrucibleConfigError(where, 'must be an object {type, instructions, ...}');
    }
    const question = raw as Record<string, unknown>;
    const type = question['type'];
    if (typeof type !== 'string' || !(DECIDE_TYPES as readonly string[]).includes(type)) {
      throw new CrucibleConfigError(
        `${where}.type`,
        `must be one of ${DECIDE_TYPES.join(', ')}, got ${JSON.stringify(type)}`,
      );
    }
    const known =
      type === 'choice'
        ? ['type', 'instructions', 'options']
        : type === 'score'
          ? ['type', 'instructions', 'levels']
          : ['type', 'instructions'];
    for (const key of Object.keys(question)) {
      if (!known.includes(key)) {
        throw new CrucibleConfigError(
          `${where}.${key}`,
          `is not a field of a ${type} question (it takes ${known.join(', ')})`,
        );
      }
    }
    const instructions = requireText(question['instructions'], `${where}.instructions`);
    if (type === 'choice') {
      const options = question['options'];
      if (typeof options !== 'object' || options === null || Array.isArray(options)) {
        throw new CrucibleConfigError(`${where}.options`, 'must be an object of option name -> description');
      }
      const read: Record<string, string> = {};
      for (const [option, description] of Object.entries(options as Record<string, unknown>)) {
        read[option] = requireText(description, `${where}.options.${option}`);
      }
      out[name] = { type, instructions, options: read };
    } else if (type === 'score') {
      out[name] = { type, instructions, levels: requireStrings(question['levels'], `${where}.levels`) };
    } else {
      out[name] = { type: 'yesno', instructions };
    }
  }
  return out;
}

function readOptionMap(value: unknown, where: string): Record<string, string> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new CrucibleConfigError(where, 'must be an object of option name -> description');
  }
  const read: Record<string, string> = {};
  for (const [option, description] of Object.entries(value as Record<string, unknown>)) {
    read[option] = requireText(description, `${where}.${option}`);
  }
  if (Object.keys(read).length < 2) {
    throw new CrucibleConfigError(where, 'needs at least 2 options');
  }
  return read;
}

interface ReadItem {
  readonly text: string;
  readonly options: Record<string, string>;
  readonly own: boolean;
}

function readDecideItems(value: unknown, shared: Record<string, string> | undefined): ReadItem[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new CrucibleConfigError('items', 'must be a non-empty array of {text, options?}');
  }
  return value.map((raw, index) => {
    const where = `items[${index}]`;
    if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
      throw new CrucibleConfigError(where, 'must be an object {text, options?}');
    }
    const item = raw as Record<string, unknown>;
    for (const key of Object.keys(item)) {
      if (key !== 'text' && key !== 'options') {
        throw new CrucibleConfigError(`${where}.${key}`, 'is not a field of an item (it takes text, options)');
      }
    }
    const text = requireText(item['text'], `${where}.text`);
    if (item['options'] !== undefined) {
      return { text, options: readOptionMap(item['options'], `${where}.options`), own: true };
    }
    if (shared === undefined) {
      throw new CrucibleConfigError(`${where}.options`, 'is not given and the request has no shared options');
    }
    return { text, options: shared, own: false };
  });
}

function readMissingMode(missing: unknown, payload: Record<string, unknown>): boolean {
  if (missing === undefined) return false;
  if (missing !== 'refuse' && missing !== 'report') {
    throw new CrucibleConfigError('missing', `must be 'refuse' or 'report', got ${JSON.stringify(missing)}`);
  }
  payload['missing'] = missing;
  return missing === 'report';
}

function readDecideItemsResponse(body: Json, items: readonly ReadItem[], report: boolean): DecideItemsResponse {
  const where = 'decideItems';
  const raw = body['answers'];
  if (!Array.isArray(raw) || raw.length !== items.length) {
    throw new CrucibleProtocolError(
      `${where}.answers is not a list of ${items.length} answers, one per item asked`,
    );
  }
  const answers = items.map((item, index) => {
    const asked: DecideChoiceQuestion = { type: 'choice', instructions: item.text, options: item.options };
    const entry = raw[index] as Json;
    if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
      throw new CrucibleProtocolError(`${where}.answers[${index}] is not an object`);
    }
    return readDecideAnswer(entry, asked, `${where}.answers[${index}]`, report) as DecideChoiceAnswer;
  });
  const timing = objectField(body, 'timing_ms', where);
  const tokens = objectField(body, 'tokens', where);
  const perItem = tokens['per_item'];
  if (!Array.isArray(perItem) || perItem.length !== items.length || !perItem.every((n) => typeof n === 'number')) {
    throw new CrucibleProtocolError(`${where}.tokens.per_item is not ${items.length} numbers`);
  }
  return {
    model: readDecideModel(objectField(body, 'model', where), `${where}.model`),
    engine: str(body, 'engine', where),
    answers,
    timingMs: { total: num(timing, 'total', `${where}.timing_ms`), engineRequests: num(timing, 'engine_requests', `${where}.timing_ms`) },
    tokens: {
      shared: nullableNum(tokens, 'shared', `${where}.tokens`),
      perItem: perItem as number[],
      images: num(tokens, 'images', `${where}.tokens`),
    },
  };
}

function readDecideResponse(
  body: Json,
  asked: Record<string, DecideQuestion>,
  report: boolean,
): DecideResponse {
  const where = 'decide';
  const names = Object.keys(asked);
  const answersBody = objectField(body, 'answers', where);
  sameKeys(Object.keys(answersBody), names, `${where}.answers`, 'the questions asked');
  const answers: Record<string, DecideAnswer> = {};
  for (const name of names) {
    answers[name] = readDecideAnswer(
      objectField(answersBody, name, `${where}.answers`),
      asked[name] as DecideQuestion,
      `${where}.answers.${name}`,
      report,
    );
  }
  return {
    model: readDecideModel(objectField(body, 'model', where), `${where}.model`),
    engine: str(body, 'engine', where),
    answers,
    timingMs: readDecideTiming(objectField(body, 'timing_ms', where), names, `${where}.timing_ms`),
    tokens: readDecideTokens(objectField(body, 'tokens', where), names, `${where}.tokens`),
  };
}

function readDecideModel(model: Json, where: string): DecideResponse['model'] {
  return {
    id: str(model, 'id', where),
    revision: str(model, 'revision', where),
    fingerprint: str(model, 'fingerprint', where),
  };
}

function readDecideTiming(
  timing: Json,
  names: readonly string[],
  where: string,
): DecideResponse['timingMs'] {
  const perQuestionRaw = objectField(timing, 'per_question', where);
  sameKeys(Object.keys(perQuestionRaw), names, `${where}.per_question`, 'the questions asked');
  const perQuestion: Record<string, DecideCallTiming> = {};
  for (const name of names) {
    perQuestion[name] = readDecideCallTiming(
      objectField(perQuestionRaw, name, `${where}.per_question`),
      `${where}.per_question.${name}`,
    );
  }
  const prime = nullableObject(timing, 'prime', where);
  return {
    total: num(timing, 'total', where),
    perQuestion,
    prime: prime === null ? null : readDecideCallTiming(prime, `${where}.prime`),
  };
}

function readDecideTokens(
  tokens: Json,
  names: readonly string[],
  where: string,
): DecideResponse['tokens'] {
  const perQuestionRaw = objectField(tokens, 'per_question', where);
  sameKeys(Object.keys(perQuestionRaw), names, `${where}.per_question`, 'the questions asked');
  const perQuestion: Record<string, number> = {};
  for (const name of names) {
    perQuestion[name] = num(perQuestionRaw, name, `${where}.per_question`);
  }
  return { perQuestion, images: num(tokens, 'images', where) };
}

function readDecideAnswer(
  entry: Json,
  question: DecideQuestion,
  where: string,
  report: boolean,
): DecideAnswer {
  const type = str(entry, 'type', where);
  if (type !== question.type) {
    throw new CrucibleProtocolError(
      `${where}.type is ${JSON.stringify(type)} but the question asked was a ${question.type}`,
    );
  }
  const labelMass = num(entry, 'label_mass', where);
  const labels =
    question.type === 'choice'
      ? Object.keys(question.options)
      : question.type === 'score'
        ? question.levels
        : ['Yes', 'No'];
  let missingLabels: string[] | undefined;
  if (report) {
    missingLabels = strArray(entry, 'missing_labels', where);
    for (const label of missingLabels) oneOf(label, labels, `${where}.missing_labels`);
  } else if ('missing_labels' in entry) {
    throw new CrucibleProtocolError(
      `${where}.missing_labels is present but the request did not ask for missing: 'report'`,
    );
  }
  const common = missingLabels === undefined ? { labelMass } : { labelMass, missingLabels };
  if (question.type === 'yesno') {
    return { type: 'yesno', p: num(entry, 'p', where), logprob: nullableNum(entry, 'logprob', where), ...common };
  }
  const missing = missingLabels === undefined ? [] : missingLabels;
  const probabilities = readDistribution(entry, 'probabilities', labels, missing, where);
  const logprobs = readDistribution(entry, 'logprobs', labels, missing, where);
  const confidence = num(entry, 'confidence', where);
  if (question.type === 'choice') {
    const choice = str(entry, 'choice', where);
    oneOf(choice, labels, `${where}.choice`);
    return { type: 'choice', choice, probabilities, logprobs, confidence, ...common };
  }
  const level = str(entry, 'level', where);
  oneOf(level, labels, `${where}.level`);
  return { type: 'score', score: num(entry, 'score', where), level, probabilities, logprobs, confidence, ...common };
}

function readDistribution(
  answer: Json,
  key: 'probabilities' | 'logprobs',
  labels: readonly string[],
  missing: readonly string[],
  where: string,
): Record<string, number | null> {
  const entry = objectField(answer, key, where);
  sameKeys(Object.keys(entry), labels, `${where}.${key}`, 'the options or levels asked');
  const out: Record<string, number | null> = {};
  for (const label of labels) {
    if (missing.includes(label)) {
      if (entry[label] !== null) {
        throw new CrucibleProtocolError(
          `${where}.${key}.${label} is ${JSON.stringify(entry[label])} but the answer names ` +
            `${JSON.stringify(label)} missing; a missing label's value is null`,
        );
      }
      out[label] = null;
    } else if (key === 'logprobs') {
      out[label] = nullableNum(entry, label, `${where}.${key}`);
    } else {
      out[label] = num(entry, label, `${where}.${key}`);
    }
  }
  return out;
}

function readDecideCallTiming(entry: Json, where: string): DecideCallTiming {
  return {
    wallMs: num(entry, 'wall_ms', where),
    promptTokens: num(entry, 'prompt_tokens', where),
    cachedTokens: nullableNum(entry, 'cached_tokens', where),
  };
}

function sameKeys(got: readonly string[], want: readonly string[], where: string, what: string): void {
  const missing = want.filter((key) => !got.includes(key));
  const extra = got.filter((key) => !want.includes(key));
  if (missing.length > 0 || extra.length > 0) {
    throw new CrucibleProtocolError(
      `${where} does not match ${what}` +
        (missing.length > 0 ? `; missing ${JSON.stringify(missing)}` : '') +
        (extra.length > 0 ? `; not asked ${JSON.stringify(extra)}` : ''),
    );
  }
}

function refuseReasoningWithoutContent(message: Json, finishReason: string): void {
  const content = 'content' in message ? message['content'] : null;
  if (typeof content === 'string') return;
  const reasoning = 'reasoning' in message ? message['reasoning'] : null;
  if (typeof reasoning !== 'string' || reasoning === '') return;
  const stopped =
    finishReason === 'length'
      ? 'and hit the token ceiling before it began the answer'
      : `and stopped with finish_reason ${JSON.stringify(finishReason)}`;
  throw new CrucibleProtocolError(
    'chat.choices[0].message has no "content": the model emitted ' +
      `${reasoning.length} characters of "reasoning" ${stopped}. This is a ` +
      'reasoning model thinking before it answers — raise maxTokens, or pass ' +
      'thinking: false to turn the thinking off.',
  );
}

function readChatDelta(raw: string): string | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new CrucibleProtocolError(`a chat completion chunk is not JSON: ${excerpt(raw)}`);
  }
  const chunk = asObject(parsed, 'chat chunk');
  const choices = asArray(field(chunk, 'choices', 'chat chunk'), 'chat chunk.choices');
  const first = choices[0];
  if (first === undefined) return null;
  const choice = asObject(first, 'chat chunk.choices[0]');
  const delta = objectField(choice, 'delta', 'chat chunk.choices[0]');
  if (!('content' in delta)) return null;
  const content = delta['content'];
  if (content === null) return null;
  if (typeof content !== 'string') {
    throw new CrucibleProtocolError(
      `chat chunk.choices[0].delta.content is neither a string nor null`,
    );
  }
  return content;
}

interface NodeFileApis {
  readonly fs: {
    mkdir(path: string, options: { recursive: true }): Promise<string | undefined>;
    writeFile(path: string, data: Uint8Array): Promise<void>;
    rename(from: string, to: string): Promise<void>;
    rm(path: string, options: { force: true }): Promise<void>;
  };
  readonly path: {
    join(...parts: string[]): string;
  };
}

async function loadNodeFileApis(): Promise<NodeFileApis> {
  const node = await loadNodeBuiltins(
    'writeArtifactsTo needs node:fs/promises and node:path, and this ' +
      'runtime has neither. It writes the artifacts to disk itself because ' +
      'there is no shared mount between a Crucible host and its client; in ' +
      'a browser, fetch each artifact with artifact(jobId, name) and put ' +
      'the bytes wherever that runtime keeps bytes.',
  );
  requireFunctions(
    node.fs,
    ['mkdir', 'writeFile', 'rename', 'rm'],
    (name) =>
      `node:fs/promises on this runtime has no ${name}(); writeArtifactsTo ` +
      'cannot write files atomically without it',
  );
  requireFunctions(node.path, ['join'], (name) => `node:path on this runtime has no ${name}()`);
  return {
    fs: node.fs as unknown as NodeFileApis['fs'],
    path: node.path as unknown as NodeFileApis['path'],
  };
}

let temporaryCounter = 0;

async function writeAtomically(
  node: NodeFileApis,
  target: string,
  bytes: Uint8Array,
): Promise<void> {
  temporaryCounter += 1;
  const temporary = `${target}.${Date.now().toString(36)}-${temporaryCounter}.part`;
  try {
    await node.fs.writeFile(temporary, bytes);
    await node.fs.rename(temporary, target);
  } catch (error) {
    await node.fs.rm(temporary, { force: true }).catch(() => undefined);
    throw error;
  }
}

function refuseUnsafeMemberName(name: string): void {
  const bad =
    name === '' ||
    name === '.' ||
    name === '..' ||
    name.startsWith('.') ||
    name.includes('/') ||
    name.includes('\\') ||
    name.includes('\0');
  if (bad) {
    throw new CrucibleProtocolError(
      `the server announced an artifact named ${JSON.stringify(name)}, which is ` +
        'not a single path member; writeArtifactsTo will not turn it into a path',
    );
  }
}

function readRenderChunks(chunks: unknown): Array<{ index: number; text: string }> {
  if (chunks === undefined || chunks === null) {
    throw new CrucibleConfigError('chunks', 'is required and was not given');
  }
  if (!Array.isArray(chunks)) {
    throw new CrucibleConfigError('chunks', `must be an array, got ${typeof chunks}`);
  }
  if (chunks.length === 0) {
    throw new CrucibleConfigError('chunks', 'is required and was empty');
  }
  const seen = new Map<number, number>();
  return chunks.map((entry, at) => {
    if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
      throw new CrucibleConfigError(`chunks[${at}]`, 'must be {index, text}');
    }
    const chunk = entry as Partial<RenderChunk>;
    const index = requireIndex(chunk.index, `chunks[${at}].index`);
    const first = seen.get(index);
    if (first !== undefined) {
      throw new CrucibleConfigError(
        `chunks[${at}].index`,
        `is ${index}, which chunks[${first}] already claimed. An index is an ` +
          'artifact name, so two chunks sharing one would be two renders writing ' +
          'the same <index>.flac',
      );
    }
    seen.set(index, at);
    const text = chunk.text;
    if (typeof text !== 'string') {
      throw new CrucibleConfigError(`chunks[${at}].text`, `must be a string, got ${typeof text}`);
    }
    if (text.trim() === '') {
      throw new CrucibleConfigError(
        `chunks[${at}].text`,
        'is blank, and narrator refuses an empty generate with a whole-request ' +
          'error that would end the batch',
      );
    }
    return { index, text };
  });
}

function requireIndex(value: unknown, option: string): number {
  if (value === undefined || value === null) {
    throw new CrucibleConfigError(option, 'is required and was not given');
  }
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new CrucibleConfigError(
      option,
      `must be a non-negative integer, got ${String(value)}`,
    );
  }
  return value;
}

function requireText(value: unknown, option: string): string {
  if (value === undefined || value === null) {
    throw new CrucibleConfigError(option, 'is required and was not given');
  }
  if (typeof value !== 'string') {
    throw new CrucibleConfigError(option, `must be a string, got ${typeof value}`);
  }
  if (value.trim() === '') {
    throw new CrucibleConfigError(option, 'is required and was empty');
  }
  return value;
}

function readInitialPrompt(value: unknown): string | null {
  if (value === null) return null;
  if (typeof value !== 'string') {
    throw new CrucibleConfigError(
      'initialPrompt',
      `must be a string or null, got ${typeof value}`,
    );
  }
  if (value.trim() === '') {
    throw new CrucibleConfigError('initialPrompt', 'is blank; send null for no prompt');
  }
  return value;
}

function readContext(value: unknown): string | null {
  if (value === null) return null;
  if (typeof value !== 'string') {
    throw new CrucibleConfigError('context', `must be a string or null, got ${typeof value}`);
  }
  if (value.trim() === '') {
    throw new CrucibleConfigError('context', 'is blank; send null for no context');
  }
  return value;
}

function requireBool(value: unknown, option: string): boolean {
  if (value === undefined || value === null) {
    throw new CrucibleConfigError(option, 'is required and was not given');
  }
  if (typeof value !== 'boolean') {
    throw new CrucibleConfigError(option, `must be a boolean, got ${typeof value}`);
  }
  return value;
}

function readSpeechKnob(value: unknown, option: string, key: string): Record<string, unknown> {
  if (value === undefined) return {};
  if (value === null) return { [key]: null };
  return { [key]: requireFinite(value, option) };
}

function requireFinite(value: unknown, option: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new CrucibleConfigError(option, `must be a finite number, got ${String(value)}`);
  }
  return value;
}

const RESPONSE_FORMAT_TYPES = ['text', 'json_object', 'json_schema'] as const;

function readResponseFormat(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new CrucibleConfigError('responseFormat', `must be an object, got ${typeof value}`);
  }
  const format = value as Record<string, unknown>;
  const type = format['type'];
  if (typeof type !== 'string' || !(RESPONSE_FORMAT_TYPES as readonly string[]).includes(type)) {
    throw new CrucibleConfigError(
      'responseFormat.type',
      `must be one of ${RESPONSE_FORMAT_TYPES.join(', ')}, got ${JSON.stringify(type)}`,
    );
  }
  if (type === 'json_schema') {
    const schema = format['json_schema'];
    if (typeof schema !== 'object' || schema === null || Array.isArray(schema)) {
      throw new CrucibleConfigError(
        'responseFormat.json_schema',
        'is required when type is "json_schema", and must be {name, schema}',
      );
    }
    const declared = schema as Record<string, unknown>;
    requireText(declared['name'], 'responseFormat.json_schema.name');
    const grammar = declared['schema'];
    if (typeof grammar !== 'object' || grammar === null || Array.isArray(grammar)) {
      throw new CrucibleConfigError(
        'responseFormat.json_schema.schema',
        'must be a JSON Schema object',
      );
    }
  }
  return format;
}

function requireStrings(value: unknown, option: string): string[] {
  if (!Array.isArray(value)) {
    throw new CrucibleConfigError(option, `must be an array of strings, got ${typeof value}`);
  }
  return value.map((entry, index) => {
    if (typeof entry !== 'string') {
      throw new CrucibleConfigError(`${option}[${index}]`, `must be a string, got ${typeof entry}`);
    }
    return entry;
  });
}

function normaliseUrl(url: string): string {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new CrucibleConfigError('url', `${JSON.stringify(url)} is not an absolute URL`);
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new CrucibleConfigError('url', `${JSON.stringify(url)} is not http or https`);
  }
  const trimmed = url.replace(/\/+$/, '');
  if (/\/v1$/.test(trimmed)) {
    throw new CrucibleConfigError(
      'url',
      `${JSON.stringify(url)} already ends in /v1; give the server's base URL ` +
        "(e.g. http://127.0.0.1:7100) and the client will add the version prefix",
    );
  }
  return trimmed;
}

function busyRefusal(
  status: number,
  code: string,
  message: string,
  details: unknown,
): CrucibleError {
  try {
    const body = asObject(details, 'error.details');
    return new CrucibleBusy(status, code, message, details, {
      holder: nullableStr(body, 'holder', 'error.details'),
      jobId: str(body, 'job_id', 'error.details'),
      jobType: str(body, 'type', 'error.details'),
      model: nullableStr(body, 'model', 'error.details'),
      jobStatus: str(body, 'status', 'error.details'),
      since: str(body, 'since', 'error.details'),
      progress: num(body, 'progress', 'error.details'),
      jobMessage: nullableStr(body, 'message', 'error.details'),
    });
  } catch (cause) {
    if (cause instanceof CrucibleProtocolError) return cause;
    throw cause;
  }
}

const OPERATOR_DOOR = 'operator';

function heldAtTheOperatorDoor(details: unknown): boolean {
  if (typeof details !== 'object' || details === null) return false;
  const door = (details as Record<string, unknown>)['door'];
  if (typeof door === 'string') return door === OPERATOR_DOOR;
  return 'fact' in details;
}

function heldRefusal(
  status: number,
  code: string,
  message: string,
  details: unknown,
): CrucibleError {
  try {
    const body = asObject(details, 'error.details');
    return new CrucibleCardHeld(status, code, message, details, {
      fact: str(body, 'fact', 'error.details'),
      who: str(body, 'who', 'error.details'),
    });
  } catch (cause) {
    if (cause instanceof CrucibleProtocolError) return cause;
    throw cause;
  }
}

const SESSION_DOOR = 'session';

/** `server_busy` because another client's queue session holds the server. */
function heldBySession(details: unknown): boolean {
  if (typeof details !== 'object' || details === null) return false;
  return (details as Record<string, unknown>)['door'] === SESSION_DOOR;
}

/** `server_busy` (door `session`) or `session_open`: another client's session holds the server. */
function sessionRefusal(
  status: number,
  code: string,
  message: string,
  details: unknown,
): CrucibleError {
  try {
    const body = asObject(details, 'error.details');
    return new CrucibleSessionHeld(status, code, message, details, {
      holder: nullableStr(body, 'holder', 'error.details'),
      sessionId: str(body, 'session_id', 'error.details'),
      act: str(body, 'act', 'error.details'),
      model: nullableStr(body, 'model', 'error.details'),
      sessionStatus: str(body, 'status', 'error.details'),
      since: str(body, 'since', 'error.details'),
    });
  } catch (cause) {
    if (cause instanceof CrucibleProtocolError) return cause;
    throw cause;
  }
}

/**
 * `session_closed`. An item names the session as `session_id`; a TTS stream whose session ended
 * before it opened names it as `queue_session_id`.
 */
function closedRefusal(status: number, message: string, details: unknown): CrucibleError {
  try {
    const body = asObject(details, 'error.details');
    const key = 'session_id' in body ? 'session_id' : 'queue_session_id';
    return new CrucibleSessionClosed(status, message, details, {
      sessionId: str(body, key, 'error.details'),
      reason: str(body, 'reason', 'error.details'),
    });
  } catch (cause) {
    if (cause instanceof CrucibleProtocolError) return cause;
    throw cause;
  }
}

function readStopping(body: Json, where: string): Stopping | null {
  const data = nullableObject(body, 'stopping', where);
  if (data === null) return null;
  const at = `${where}.stopping`;
  return {
    kind: str(data, 'kind', at),
    id: str(data, 'id', at),
    since: str(data, 'since', at),
    pids: asArray(field(data, 'pids', at), `${at}.pids`).map((entry, index) => {
      if (typeof entry !== 'number' || !Number.isInteger(entry)) {
        throw new CrucibleProtocolError(
          `${at}.pids[${index}] is not an integer pid; it is what an operator ` +
            'types into a kill command, so a rounded or absent one is unusable',
        );
      }
      return entry;
    }),
  };
}

function readHeldBy(
  resident: Json,
): { fact: string; who: string; details: Record<string, unknown> } | null {
  const held = nullableObject(resident, 'held_by', 'activity.resident');
  if (held === null) {
    return null;
  }
  const where = 'activity.resident.held_by';
  return {
    fact: str(held, 'fact', where),
    who: str(held, 'who', where),
    details: asObject(field(held, 'details', where), `${where}.details`) as Record<
      string,
      unknown
    >,
  };
}

/** `GET /v1/activity`'s body, read; also the `activity` of a server event stream's snapshot. */
function readActivity(body: Json): Activity {
  const server = objectField(body, 'server', 'activity');
  const resident = nullableObject(body, 'resident', 'activity');
  const claim = nullableObject(body, 'claim', 'activity');
  const streaming = nullableObject(body, 'streaming', 'activity');
  const session = nullableObject(body, 'session', 'activity');
  const chat = objectField(body, 'chat', 'activity');
  const slot = objectField(objectField(body, 'slots', 'activity'), 'accelerated', 'activity.slots');
  return {
    server: {
      name: str(server, 'name', 'activity.server'),
      version: str(server, 'version', 'activity.server'),
      apiVersion: num(server, 'api_version', 'activity.server'),
      backend: str(server, 'backend', 'activity.server'),
      uptimeS: num(server, 'uptime_s', 'activity.server'),
    },
    resident:
      resident === null
        ? null
        : {
            kind: str(resident, 'kind', 'activity.resident'),
            id: str(resident, 'id', 'activity.resident'),
            since: str(resident, 'since', 'activity.resident'),
            memoryBytesEstimate: num(
              resident,
              'memory_bytes_estimate',
              'activity.resident',
            ),
            heldBy: readHeldBy(resident),
            unclaimedSince: nullableStr(
              resident,
              'unclaimed_since',
              'activity.resident',
            ),
            engineExitCode: nullableNum(
              resident,
              'engine_exit_code',
              'activity.resident',
            ),
          },
    stopping: readStopping(body, 'activity'),
    warming: nullableStr(body, 'warming', 'activity'),
    claim: claim === null ? null : { heldBy: str(claim, 'held_by', 'activity.claim') },
    streaming: streaming === null ? null : readStreaming(streaming),
    session: session === null ? null : readQueueSession(session, 'activity.session'),
    chat: readActivityChat(chat),
    slots: {
      accelerated: {
        busy: num(slot, 'busy', 'activity.slots.accelerated'),
        of: num(slot, 'of', 'activity.slots.accelerated'),
        queueDepth: num(slot, 'queue_depth', 'activity.slots.accelerated'),
        acceptsWork: bool(slot, 'accepts_work', 'activity.slots.accelerated'),
      },
    },
    running: asArray(field(body, 'running', 'activity'), 'activity.running').map(
      (entry, index) => readActivityJob(asObject(entry, `activity.running[${index}]`), `activity.running[${index}]`),
    ),
    queued: asArray(field(body, 'queued', 'activity'), 'activity.queued').map(
      (entry, index) => readActivityJob(asObject(entry, `activity.queued[${index}]`), `activity.queued[${index}]`),
    ),
  };
}

function readActivityJob(data: Json, where: string): ActivityJob {
  return {
    jobId: str(data, 'job_id', where),
    type: str(data, 'type', where),
    model: nullableStr(data, 'model', where),
    status: str(data, 'status', where),
    position: nullableNum(data, 'position', where),
    progress: num(data, 'progress', where),
    message: nullableStr(data, 'message', where),
    created: str(data, 'created', where),
    started: nullableStr(data, 'started', where),
    client: nullableStr(data, 'client', where),
    waitedS: optNum(data, 'waited_s', where),
    maxWaitS: optNum(data, 'max_wait_s', where),
    waitingFor: readWaitingFor(data, where),
  };
}

function readActivityChat(chat: Json): Activity['chat'] {
  return {
    inFlight: num(chat, 'in_flight', 'activity.chat'),
    maxInFlight: nullableNum(chat, 'max_in_flight', 'activity.chat'),
    maxInFlightBasis: nullableStr(chat, 'max_in_flight_basis', 'activity.chat'),
    rows: arrayField(chat, 'rows', 'activity.chat').map((entry, index) => {
      const where = `activity.chat.rows[${index}]`;
      const row = asObject(entry, where);
      return {
        id: num(row, 'id', where),
        act: nullableStr(row, 'act', where),
        model: str(row, 'model', where),
        client: nullableStr(row, 'client', where),
        since: str(row, 'since', where),
      };
    }),
  };
}

function readStreaming(data: Json): ActivityStreaming {
  const where = 'activity.streaming';
  const progress = nullableNum(data, 'progress', where);
  if (progress !== null) {
    throw new CrucibleProtocolError(
      `${where}.progress is ${progress}, but a streaming session has no total to ` +
        'be a fraction of; only null is meaningful here',
    );
  }
  return {
    sessionId: str(data, 'session_id', where),
    voice: str(data, 'voice', where),
    language: str(data, 'language', where),
    narratorEngine: str(data, 'narrator_engine', where),
    since: str(data, 'since', where),
    client: nullableStr(data, 'client', where),
    progress: null,
    said: num(data, 'said', where),
    finished: num(data, 'finished', where),
    inFlight: num(data, 'in_flight', where),
    seconds: num(data, 'seconds', where),
    chars: num(data, 'chars', where),
  };
}

function isSafeMethod(method: string | undefined): boolean {
  const name = (method ?? 'GET').toUpperCase();
  return name === 'GET' || name === 'HEAD';
}

const STALE_CONNECTION_CODES = new Set(['ECONNRESET', 'UND_ERR_SOCKET']);

function isStaleConnection(cause: unknown): boolean {
  for (let error: unknown = cause, depth = 0; error instanceof Error && depth < 4; depth += 1) {
    const code = (error as Error & { code?: unknown }).code;
    if (typeof code === 'string' && STALE_CONNECTION_CODES.has(code)) return true;
    error = (error as Error & { cause?: unknown }).cause;
  }
  return false;
}

function readNetwork(body: Json | null): ServerNetwork | null {
  if (body === null) return null;
  const where = 'setup.network';
  return {
    reachable: bool(body, 'reachable', where),
    urls: strArray(body, 'urls', where),
    sentence: str(body, 'sentence', where),
    how: nullableStr(body, 'how', where),
    command: nullableStr(body, 'command', where),
    changes: nullableStr(body, 'changes', where),
  };
}

function describeCause(cause: unknown): string {
  if (cause instanceof Error) {
    const nested = (cause as Error & { cause?: unknown }).cause;
    const code = nested instanceof Error ? `${nested.message}` : undefined;
    return code === undefined ? cause.message : `${cause.message} (${code})`;
  }
  return String(cause);
}

function excerpt(text: string): string {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length > 200 ? `${flat.slice(0, 200)}…` : flat;
}

const ENGINE_TARGETS = ['wsl'] as const;

function readUnmet(body: Json, where: string): UnmetNeed[] {
  return arrayField(body, 'unmet', where).map((entry, index) => {
    const at = `${where}.unmet[${index}]`;
    const row = asObject(entry, at);
    return { class: str(row, 'class', at), reason: str(row, 'reason', at) };
  });
}

const TASK_STATES: readonly TaskState[] = ['running', 'done', 'failed', 'cancelled'];

const SUBJECT_KINDS: readonly SubjectKind[] = [
  'model',
  'voice',
  'rvc',
  'rvc-base',
  'denoise',
  'engine',
];

const TASK_EVENT_NAMES = [
  'started',
  'step',
  'progress',
  'skipped',
  'done',
  'failed',
  'cancelled',
] as const;

function readCatalogRow(row: Json, where: string): CatalogRow {
  return {
    kind: oneOf(str(row, 'kind', where), SUBJECT_KINDS, `${where}.kind`),
    id: str(row, 'id', where),
    name: nullableStr(row, 'name', where),
    jobType: str(row, 'job_type', where),
    installed: bool(row, 'installed', where),
    installedBytes: nullableNum(row, 'installed_bytes', where),
    expectedBytes: nullableNum(row, 'expected_bytes', where),
    sharesWeightsOf: nullableStr(row, 'shares_weights_of', where),
    missingFiles: nullableStrArray(row, 'missing_files', where),
    floors: strArray(row, 'floors', where),
    license: nullableStr(row, 'license', where),
    source: str(row, 'source', where),
    resident: bool(row, 'resident', where),
  };
}

function readTaskStatus(row: Json, where: string): TaskStatus {
  return {
    taskId: str(row, 'task_id', where),
    type: str(row, 'type', where),
    request: objectField(row, 'request', where),
    state: oneOf(str(row, 'state', where), TASK_STATES, `${where}.state`),
    error: readFailureOrNull(nullableObject(row, 'error', where), `${where}.error`),
    created: str(row, 'created', where),
    started: str(row, 'started', where),
    finished: nullableStr(row, 'finished', where),
    unmet: readUnmet(row, where),
    message: nullableStr(row, 'message', where),
  };
}

function taskPayload(request: TaskRequest): Record<string, unknown> {
  const given = request as {
    type?: unknown;
    kind?: unknown;
    id?: unknown;
    jobType?: unknown;
    narratorEngine?: unknown;
    module?: unknown;
    target?: unknown;
  };
  const type = requireText(given.type, 'type');
  if (type === 'pull') {
    return {
      type,
      kind: oneOf(requireText(given.kind, 'kind'), SUBJECT_KINDS, 'kind'),
      id: requireText(given.id, 'id'),
    };
  }
  if (type === 'install') {
    const payload: Record<string, unknown> = {
      type,
      job_type: requireText(given.jobType, 'jobType'),
    };
    if (given.narratorEngine !== undefined) {
      payload['narrator_engine'] = requireText(given.narratorEngine, 'narratorEngine');
    }
    return payload;
  }
  if (type === 'module') {
    if (given.module === undefined || given.module === null) {
      throw new CrucibleConfigError('module', 'a module task needs the module document');
    }
    return { type, module: given.module };
  }
  if (type === 'engine') {
    return {
      type,
      target: oneOf(requireText(given.target, 'target'), ENGINE_TARGETS, 'target'),
    };
  }
  throw new CrucibleConfigError(
    'type',
    `must be 'pull', 'install', 'module' or 'engine', got ${JSON.stringify(type)}`,
  );
}

function readTaskEvent(rawId: string | null, rawName: string | null, rawData: string): TaskEvent {
  if (rawId === null) {
    throw new CrucibleProtocolError(`a task SSE frame carried no id: ${excerpt(rawData)}`);
  }
  const id = Number(rawId);
  if (!Number.isInteger(id) || id < 1) {
    throw new CrucibleProtocolError(
      `task SSE frame id ${JSON.stringify(rawId)} is not a positive integer`,
    );
  }
  if (rawName === null) {
    throw new CrucibleProtocolError(`task SSE frame ${id} carried no event name`);
  }
  const name = rawName;
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawData);
  } catch {
    throw new CrucibleProtocolError(
      `task SSE frame ${id} (${name}) has non-JSON data: ${excerpt(rawData)}`,
    );
  }
  const data = asObject(parsed, `task event ${id} (${name}) data`);
  const where = `task event ${id} (${name})`;

  if (!(TASK_EVENT_NAMES as readonly string[]).includes(name)) {
    return { id, event: 'unknown', kind: name, data };
  }
  const known = name as (typeof TASK_EVENT_NAMES)[number];

  switch (known) {
    case 'started':
      return { id, event: 'started', data: { type: str(data, 'type', where) } };
    case 'step':
      return { id, event: 'step', data: readTaskStep(data, where) };
    case 'progress':
      return { id, event: 'progress', data: readTaskProgress(data, where) };
    case 'skipped':
      return { id, event: 'skipped', data: { reason: str(data, 'reason', where) } };
    case 'done':
      return { id, event: 'done', data };
    case 'failed':
      return { id, event: 'failed', data: readFailure(data, where) };
    case 'cancelled':
      return { id, event: 'cancelled', data };
  }
}

function readTaskStep(data: Json, where: string): TaskStepData {
  const step: {
    name: string;
    index: number;
    total: number;
    jobTypes?: readonly string[];
  } = {
    name: str(data, 'name', where),
    index: num(data, 'index', where),
    total: num(data, 'total', where),
  };
  if ('job_types' in data) step.jobTypes = strArray(data, 'job_types', where);
  return step;
}

function readTaskProgress(data: Json, where: string): TaskProgressData {
  if ('line' in data) return { line: str(data, 'line', where) };
  if ('bytes_done' in data) {
    return {
      bytesDone: num(data, 'bytes_done', where),
      bytesTotal: nullableNum(data, 'bytes_total', where),
      file: str(data, 'file', where),
    };
  }
  throw new CrucibleProtocolError(
    `${where} is neither a pull's progress ({bytes_done, bytes_total, file}) nor ` +
      "an install's ({line})",
  );
}
