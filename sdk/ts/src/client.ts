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
  CrucibleLeased,
  CrucibleRefused,
  LEASED,
  SERVER_BUSY,
  CrucibleServerError,
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
  optNum,
  optObject,
  optStr,
  str,
  strArray,
  type Json,
} from './shape.js';
import { loadNodeBuiltins, requireFunctions } from './node-builtins.js';
import { readSseFrames } from './sse.js';
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
  type ActivityLease,
  type ActivityStreaming,
  type AlignItem,
  type Alignment,
  type AlignOptions,
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
  type Lease,
  type LeaseOnLoad,
  type LoadModelOptions,
  type LoadVoiceOptions,
  type ModelDescriptor,
  type ModelInfo,
  type PagesEngine,
  type Ping,
  type ProgressData,
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
  type VoiceInfo,
  type VoicePace,
  type VoiceServing,
  type WrittenArtifact,
} from './types.js';
import { SDK_VERSION } from './version.js';

const API_HEADER = 'X-Crucible-Api';
const CLIENT_NAME_HEADER = 'X-Crucible-Client';
const JOB_STATES: readonly JobState[] = [
  'queued', 'running', 'done', 'failed', 'cancelled', 'interrupted',
];
const CHAT_ROLES = ['system', 'user', 'assistant'] as const;
const DONE_SENTINEL = '[DONE]';
const EVENT_NAMES = [
  'queued',
  'warming',
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
    const body = await this.#json(path, { method: 'GET' }, 'activity');
    const server = objectField(body, 'server', 'activity');
    const resident = nullableObject(body, 'resident', 'activity');
    const claim = nullableObject(body, 'claim', 'activity');
    const streaming = nullableObject(body, 'streaming', 'activity');
    const lease = nullableObject(body, 'lease', 'activity');
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
      lease: lease === null ? null : readLease(lease, 'activity.lease'),
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

  /**
   * `POST /v1/models/{subject}/lease` — keep the resident model, voice or aligner on the card
   * during a run.
   */
  async lease(
    subject: string,
    options: { act: string; ttlSeconds: number },
  ): Promise<Lease> {
    const id = requireText(subject, 'subject');
    const act = requireText(options?.act, 'act');
    const ttlSeconds = options?.ttlSeconds;
    if (typeof ttlSeconds !== 'number' || !Number.isInteger(ttlSeconds)) {
      throw new CrucibleConfigError(
        'ttlSeconds',
        'is required and must be a whole number of seconds; the server states ' +
          'its own accepted range if this one is outside it',
      );
    }
    const body = await this.#json(
      `/v1/models/${encodeURIComponent(id)}/lease`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ act, ttl_seconds: ttlSeconds }),
      },
      'lease',
    );
    return { ...readLease(body, 'lease'), subject: str(body, 'subject', 'lease') };
  }

  /** `POST /v1/leases/{id}/heartbeat` — I am still here. */
  async heartbeat(leaseId: string): Promise<string> {
    const id = requireText(leaseId, 'leaseId');
    const body = await this.#json(
      `/v1/leases/${encodeURIComponent(id)}/heartbeat`,
      { method: 'POST' },
      'heartbeat',
    );
    return str(body, 'expires_at', 'heartbeat');
  }

  /** `DELETE /v1/leases/{id}` — give the card back. */
  async release(leaseId: string): Promise<void> {
    const id = requireText(leaseId, 'leaseId');
    const response = await this.#fetch(
      `/v1/leases/${encodeURIComponent(id)}`,
      { method: 'DELETE' },
      true,
    );
    if (!response.ok) throw await this.#failure(response);
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

  /** `POST /v1/jobs` — queue a job. */
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

    const init: RequestInit = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    };
    if (options.signal !== undefined) init.signal = options.signal;
    const body = await this.#json('/v1/jobs', init, 'submit');
    return str(body, 'job_id', 'submit');
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
      leaseId: optStr(body, 'lease_id', 'job'),
      clientRef: nullableStr(body, 'client_ref', 'job'),
      interruptedAt: nullableStr(body, 'interrupted_at', 'job'),
      heldBy: nullableStr(body, 'held_by', 'job'),
      heldSince: nullableStr(body, 'held_since', 'job'),
      chunksDone: readChunksDone(body),
      chunksTotal: nullableNum(body, 'chunks_total', 'job'),
      chunkAt: nullableStr(body, 'chunk_at', 'job'),
      resumeId: nullableStr(body, 'resume_id', 'job'),
      resumed: bool(body, 'resumed', 'job'),
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
  async *events(jobId: string, options: EventsOptions = {}): AsyncGenerator<JobEvent, void, undefined> {
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
    const headers: Record<string, string> = { Accept: 'text/event-stream' };
    if (options.lastEventId !== undefined) {
      if (!Number.isInteger(options.lastEventId) || options.lastEventId < 0) {
        throw new CrucibleConfigError(
          'lastEventId',
          `must be a non-negative integer, got ${String(options.lastEventId)}`,
        );
      }
      headers['Last-Event-ID'] = String(options.lastEventId);
    }

    const response = await this.#fetch(path, { method: 'GET', headers }, true);
    if (!response.ok) throw await this.#failure(response);
    const stream = response.body;
    if (stream === null) {
      throw new CrucibleProtocolError(`the event stream for ${what} carried no body`);
    }

    let previousId = options.lastEventId === undefined ? 0 : options.lastEventId;
    try {
      for await (const frame of readSseFrames(stream)) {
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
    } finally {
      await stream.cancel().catch(() => undefined);
    }

    throw new CrucibleUnreachable(
      this.url,
      `the event stream for ${what} ended after event ${previousId} without a ` +
        'terminal event (done, failed or cancelled)',
    );
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
      status: oneOf(str(body, 'status', 'cancel'), ['cancelled', 'cancelling'], 'cancel.status'),
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
      params: {
        ...leaseParams(options?.lease),
        ...(options?.context === undefined ? {} : { context: options.context }),
      },
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
      throw new CrucibleConfigError('request', 'decide(...) needs {model, state, questions}');
    }
    if (!('state' in given) || given.state === undefined) {
      throw new CrucibleConfigError('state', 'is required and was not given');
    }
    const questions = readDecideQuestions(given.questions);
    const payload: Record<string, unknown> = {
      model: requireText(given.model, 'model'),
      state: given.state,
    };
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

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (options.act !== undefined) headers['X-Crucible-Act'] = requireText(options.act, 'act');
    const init: RequestInit = { method: 'POST', headers, body: JSON.stringify(payload) };
    if (options.signal !== undefined) init.signal = options.signal;

    const body = await this.#json('/v1/decide', init, 'decide');
    return readDecideResponse(body, questions, report);
  }

  /** `POST /v1/decide` with `items`: one choice answer per item about one state, in item order. */
  async decideItems(request: DecideItemsRequest, options: DecideOptions = {}): Promise<DecideItemsResponse> {
    const given = request as Partial<DecideItemsRequest> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError('request', 'decideItems(...) needs {model, state, items}');
    }
    if (!('state' in given) || given.state === undefined) {
      throw new CrucibleConfigError('state', 'is required and was not given');
    }
    const shared = given.options === undefined ? undefined : readOptionMap(given.options, 'options');
    const items = readDecideItems(given.items, shared);
    const payload: Record<string, unknown> = {
      model: requireText(given.model, 'model'),
      state: given.state,
    };
    if (given.images !== undefined) payload['images'] = requireStrings(given.images, 'images');
    if (given.instructions !== undefined) payload['instructions'] = requireText(given.instructions, 'instructions');
    if (shared !== undefined) payload['options'] = shared;
    payload['items'] = items.map((item) => (item.own ? { text: item.text, options: item.options } : { text: item.text }));
    const report = readMissingMode(given.missing, payload);

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (options.act !== undefined) headers['X-Crucible-Act'] = requireText(options.act, 'act');
    const init: RequestInit = { method: 'POST', headers, body: JSON.stringify(payload) };
    if (options.signal !== undefined) init.signal = options.signal;

    const body = await this.#json('/v1/decide', init, 'decideItems');
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
      params: {
        ...leaseParams(options?.lease),
        ...(reference === undefined ? {} : {
          reference: {
            data: requireText(reference.data, 'reference.data'),
            transcript: requireText(reference.transcript, 'reference.transcript'),
            ...(reference.name === undefined ? {} : {name: reference.name}),
          },
        }),
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

  async #fetch(path: string, init: RequestInit, authenticated: boolean): Promise<Response> {
    const headers = new Headers(init.headers);
    headers.set('User-Agent', this.#userAgent);
    headers.set(CLIENT_NAME_HEADER, this.#clientName);
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
        return heldAtTheOperatorDoor(details)
          ? heldRefusal(response.status, code, message, details)
          : busyRefusal(response.status, code, message, details);
      }
      if (code === LEASED) return leasedRefusal(response.status, code, message, details);
      return new CrucibleRefused(response.status, code, message, details);
    }
    return new CrucibleProtocolError(
      `HTTP ${response.status} from ${this.url} is neither a success nor a refusal`,
    );
  }

  /** `POST /v1/tts/stream` — open a live TTS session on the resident voice. */
  async stream(options: StreamOptions): Promise<TtsStreamSession> {
    return openTtsStream(
      {
        url: this.url,
        fetch: (path, init, authenticated) => this.#fetch(path, init, authenticated),
        failure: (response) => this.#failure(response),
        json: (path, init, where) => this.#json(path, init, where),
      },
      options,
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
      return { id, event: 'queued', data: { position: num(data, 'position', where) } };
    case 'warming':
      return { id, event: 'warming', data: { message: str(data, 'message', where) } };
    case 'progress':
      return { id, event: 'progress', data: readProgress(data, where) };
    case 'chunk':
      return { id, event: 'chunk', data: readChunk(data, where) };
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
  };
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

function leasedRefusal(
  status: number,
  code: string,
  message: string,
  details: unknown,
): CrucibleError {
  try {
    const body = asObject(details, 'error.details');
    return new CrucibleLeased(status, code, message, details, {
      leaseId: str(body, 'lease_id', 'error.details'),
      kind: str(body, 'kind', 'error.details'),
      holder: nullableStr(body, 'client', 'error.details'),
      act: str(body, 'act', 'error.details'),
      since: str(body, 'since', 'error.details'),
      expiresAt: str(body, 'expires_at', 'error.details'),
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

function leaseParams(lease: LeaseOnLoad | undefined): Record<string, unknown> {
  if (lease === undefined) {
    return {};
  }
  return {
    lease: {
      act: requireText(lease.act, 'lease.act'),
      ttl_seconds: lease.ttlSeconds,
    },
  };
}

function readLease(data: Json, where: string): ActivityLease {
  return {
    leaseId: str(data, 'lease_id', where),
    kind: str(data, 'kind', where),
    client: nullableStr(data, 'client', where),
    act: str(data, 'act', where),
    since: str(data, 'since', where),
    expiresAt: str(data, 'expires_at', where),
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
