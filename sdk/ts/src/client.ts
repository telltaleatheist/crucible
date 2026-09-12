/**
 * `CrucibleClient` — the whole of API v1, and nothing else.
 *
 * The client always speaks HTTP, even to a server it just started on localhost
 * (DESIGN.md section 1). It has no runtime dependencies: `fetch`,
 * `ReadableStream`, `TextDecoder`, `FormData` and `Blob` are globals in Node 20,
 * bun and the Electron main process.
 */

import { encodeBase64 } from './base64.js';
import {
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
import {
  asArray,
  asObject,
  bool,
  field,
  num,
  nullableNum,
  nullableStr,
  objectField,
  oneOf,
  str,
  strArray,
  type Json,
} from './shape.js';
import { readSseFrames } from './sse.js';
import {
  API_VERSION,
  TERMINAL_EVENTS,
  type CancelResult,
  type Capability,
  type Health,
  type JobEvent,
  type JobFailure,
  type JobRequest,
  type JobState,
  type JobStatus,
  type ModelDescriptor,
  type Ping,
  type Provenance,
  type ServerInfo,
  type UploadResult,
} from './types.js';
import { SDK_VERSION } from './version.js';

const API_HEADER = 'X-Crucible-Api';
const JOB_STATES: readonly JobState[] = ['queued', 'running', 'done', 'failed', 'cancelled'];
const HEALTH_STATES = ['ok', 'warming', 'busy'] as const;
const EVENT_NAMES = [
  'queued',
  'warming',
  'progress',
  'artifact',
  'done',
  'failed',
  'cancelled',
] as const;

/** Everything `new CrucibleClient(...)` needs. There are no optional fields. */
export interface CrucibleClientOptions {
  /**
   * The server's base URL **without** `/v1`, e.g. `http://127.0.0.1:7100`. The
   * client appends the version prefix itself; passing one is refused, because
   * two different `/v1`s in one URL is a misconfiguration, not a preference.
   */
  url: string;
  /** The bearer token `crucible init` minted. Sent on every route except `ping`. */
  token: string;
  /**
   * Who is calling. Goes into `User-Agent` as
   * `<clientName> crucible-client/<sdk version>`, so a server's log says which
   * app queued a job.
   */
  clientName: string;
}

/** Options for {@link CrucibleClient.events}. */
export interface EventsOptions {
  /**
   * Resume after this event id: the server replays everything with a higher id
   * and then follows live. Omit to start from the beginning of the job.
   */
  lastEventId?: number;
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

  constructor(options: CrucibleClientOptions) {
    const given = options as Partial<CrucibleClientOptions> | undefined;
    if (given === undefined || given === null) {
      throw new CrucibleConfigError('options', 'new CrucibleClient(...) needs {url, token, clientName}');
    }
    this.url = normaliseUrl(requireText(given.url, 'url'));
    this.#token = requireText(given.token, 'token');
    const clientName = requireText(given.clientName, 'clientName');
    this.#userAgent = `${clientName} crucible-client/${SDK_VERSION}`;
  }

  // ------------------------------------------------------------------- ping

  /**
   * `GET /v1/ping`, unauthenticated. Answers "is there a Crucible here at all",
   * separately from "is my token right": a responder that is not a Crucible
   * throws {@link CrucibleNotACrucible}, not an auth error.
   */
  async ping(): Promise<Ping> {
    const response = await this.#fetch('/v1/ping', { method: 'GET' }, false);
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

  // ------------------------------------------------------------------- info

  /** `GET /v1/info` — who this server is, what it runs on, what it can serve. */
  async info(): Promise<ServerInfo> {
    const body = await this.#json('/v1/info', { method: 'GET' }, 'info');
    const server = objectField(body, 'server', 'info');
    const host = objectField(body, 'host', 'info');
    const gpu = objectField(host, 'gpu', 'info.host');
    const capabilities = asArray(field(body, 'capabilities', 'info'), 'info.capabilities');
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
      capabilities: capabilities.map((entry, index) =>
        readCapability(asObject(entry, `info.capabilities[${index}]`), index),
      ),
    };
  }

  /** `GET /v1/health` — is the lane free, and how deep is the queue. */
  async health(): Promise<Health> {
    const body = await this.#json('/v1/health', { method: 'GET' }, 'health');
    return {
      status: oneOf(str(body, 'status', 'health'), HEALTH_STATES, 'health.status'),
      queueDepth: num(body, 'queue_depth', 'health'),
      residentModels: strArray(body, 'resident_models', 'health'),
    };
  }

  // ---------------------------------------------------------------- uploads

  /**
   * `POST /v1/uploads` — park bytes on the server and get a blob id to name as a
   * job input. Use this for anything too big to carry inline.
   */
  async upload(data: Uint8Array | Blob, options: { filename: string }): Promise<UploadResult> {
    const filename = requireText(options?.filename, 'filename');
    const blob = data instanceof Uint8Array ? new Blob([data]) : data;
    const form = new FormData();
    // The server's parameter is named `file`; python-multipart needs the
    // filename to treat the part as an upload rather than a plain field.
    form.append('file', blob, filename);
    const body = await this.#json('/v1/uploads', { method: 'POST', body: form }, 'upload');
    return {
      blobId: str(body, 'blob_id', 'upload'),
      bytes: num(body, 'bytes', 'upload'),
      sha256: str(body, 'sha256', 'upload'),
    };
  }

  // ------------------------------------------------------------------- jobs

  /**
   * `POST /v1/jobs` — queue a job. Returns its id. The server refuses an unknown
   * type, a disabled type, or a model the type does not serve, by name.
   */
  async submit(request: JobRequest): Promise<string> {
    const type = requireText(request?.type, 'type');
    const inputs: Record<string, { blob_id: string } | { inline_base64: string }> = {};
    for (const [name, input] of Object.entries(request.inputs)) {
      if ('blobId' in input) {
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
          'must be either {blobId} or {inline}; it is neither',
        );
      }
    }

    const payload: Record<string, unknown> = {
      type,
      params: request.params,
      inputs,
    };
    if (request.model !== undefined) payload['model'] = request.model;

    const body = await this.#json(
      '/v1/jobs',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      },
      'submit',
    );
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
      error: readFailureOrNull(field(body, 'error', 'job'), 'job.error'),
      artifacts: strArray(body, 'artifacts', 'job'),
      created: str(body, 'created', 'job'),
      started: nullableStr(body, 'started', 'job'),
      finished: nullableStr(body, 'finished', 'job'),
    };
  }

  /**
   * `GET /v1/jobs/{id}/events` — the job's SSE stream, as typed events.
   *
   * The iterator ends after the first terminal event (`done`, `failed`,
   * `cancelled`). If the stream closes before one arrives, that is a dead
   * connection and it throws {@link CrucibleUnreachable} rather than ending
   * quietly, because "the job finished" and "the socket died" must never look
   * the same to a caller.
   *
   * Event ids are the server's monotonic counter; pass the last one you saw as
   * `lastEventId` to resume without a gap. Ids that do not increase are a
   * protocol violation and throw.
   */
  async *events(jobId: string, options: EventsOptions = {}): AsyncGenerator<JobEvent, void, undefined> {
    const id = requireText(jobId, 'jobId');
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

    const response = await this.#fetch(
      `/v1/jobs/${encodeURIComponent(id)}/events`,
      { method: 'GET', headers },
      true,
    );
    if (!response.ok) throw await this.#failure(response);
    const stream = response.body;
    if (stream === null) {
      throw new CrucibleProtocolError(`the event stream for job ${id} carried no body`);
    }

    let previousId = options.lastEventId === undefined ? 0 : options.lastEventId;
    try {
      for await (const frame of readSseFrames(stream)) {
        const event = readEvent(frame.lastEventId, frame.event, frame.data);
        if (event.id <= previousId) {
          throw new CrucibleProtocolError(
            `event id ${event.id} does not follow ${previousId} on job ${id}`,
          );
        }
        previousId = event.id;
        yield event;
        if ((TERMINAL_EVENTS as readonly string[]).includes(event.event)) return;
      }
    } finally {
      // Closing the socket is cleanup; a failure to close does not change what
      // the stream already delivered, so it must not mask the real outcome.
      await stream.cancel().catch(() => undefined);
    }

    // Reached only when the stream ended without a terminal event.
    throw new CrucibleUnreachable(
      this.url,
      `the event stream for job ${id} ended after event ${previousId} without a ` +
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

  /**
   * The artifact's provenance sidecar, parsed. Every artifact has one
   * (DESIGN.md section 7); persist it with the output.
   */
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

  /**
   * `DELETE /v1/jobs/{id}` — cancel. A queued job is cancelled at once
   * (`cancelled`); a running one is told to stop and ends at its next
   * checkpoint (`cancelling`). Watch {@link events} for the `cancelled` event.
   */
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

  // ---------------------------------------------------------------- plumbing

  async #json(path: string, init: RequestInit, where: string): Promise<Json> {
    const response = await this.#fetch(path, init, true);
    if (!response.ok) throw await this.#failure(response);
    const text = await response.text();
    try {
      return asObject(JSON.parse(text), where);
    } catch (cause) {
      if (cause instanceof CrucibleError) throw cause;
      throw new CrucibleProtocolError(`${where} did not return JSON: ${excerpt(text)}`);
    }
  }

  async #fetch(path: string, init: RequestInit, authenticated: boolean): Promise<Response> {
    const headers = new Headers(init.headers);
    headers.set('User-Agent', this.#userAgent);
    if (authenticated) {
      headers.set('Authorization', `Bearer ${this.#token}`);
      headers.set(API_HEADER, String(API_VERSION));
    }
    const target = `${this.url}${path}`;
    try {
      return await fetch(target, { ...init, headers });
    } catch (cause) {
      // fetch rejects only for a transport failure; every HTTP status resolves.
      throw new CrucibleUnreachable(this.url, describeCause(cause), cause);
    }
  }

  /** Map a non-2xx response onto the one error type that describes it. */
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
    if (response.status >= 500) return new CrucibleServerError(response.status, code, message);
    if (response.status >= 400) {
      const details = 'details' in envelope ? envelope['details'] : null;
      return new CrucibleRefused(response.status, code, message, details);
    }
    return new CrucibleProtocolError(
      `HTTP ${response.status} from ${this.url} is neither a success nor a refusal`,
    );
  }
}

// ------------------------------------------------------------------ readers

function readCapability(entry: Json, index: number): Capability {
  const where = `info.capabilities[${index}]`;
  const models = asArray(field(entry, 'models', where), `${where}.models`);
  return {
    jobType: str(entry, 'job_type', where),
    models: models.map((model, at) =>
      readModel(asObject(model, `${where}.models[${at}]`), `${where}.models[${at}]`),
    ),
  };
}

function readModel(entry: Json, where: string): ModelDescriptor {
  return {
    id: str(entry, 'id', where),
    revision: str(entry, 'revision', where),
    source: str(entry, 'source', where),
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

function readProvenance(entry: Json, member: string): Provenance {
  const where = `${member}.provenance.json`;
  const server = objectField(entry, 'server', where);
  const model = field(entry, 'model', where);
  return {
    server: {
      name: str(server, 'name', `${where}.server`),
      version: str(server, 'version', `${where}.server`),
    },
    backend: str(entry, 'backend', where),
    job_type: str(entry, 'job_type', where),
    model:
      model === null
        ? null
        : {
            id: str(asObject(model, `${where}.model`), 'id', `${where}.model`),
            revision: nullableStr(asObject(model, `${where}.model`), 'revision', `${where}.model`),
          },
    params: asObject(field(entry, 'params', where), `${where}.params`),
    started: nullableStr(entry, 'started', where),
    finished: str(entry, 'finished', where),
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
  const name = oneOf(rawName, EVENT_NAMES, `SSE frame ${id} event name`);
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawData);
  } catch {
    throw new CrucibleProtocolError(`SSE frame ${id} (${name}) has non-JSON data: ${excerpt(rawData)}`);
  }
  const data = asObject(parsed, `event ${id} (${name}) data`);
  const where = `event ${id} (${name})`;

  switch (name) {
    case 'queued':
      return { id, event: 'queued', data: { position: nullableNum(data, 'position', where) } };
    case 'warming':
      return { id, event: 'warming', data };
    case 'progress':
      return {
        id,
        event: 'progress',
        data: { fraction: num(data, 'fraction', where), message: str(data, 'message', where) },
      };
    case 'artifact':
      return { id, event: 'artifact', data: { name: str(data, 'name', where) } };
    case 'done':
      return { id, event: 'done', data: { artifacts: strArray(data, 'artifacts', where) } };
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

// ------------------------------------------------------------------ helpers

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
