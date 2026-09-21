/**
 * `CrucibleClient` — the whole of API v1, and nothing else.
 *
 * The client always speaks HTTP, even to a server it just started on localhost
 * (DESIGN.md section 1). It has no runtime dependencies: `fetch`,
 * `ReadableStream`, `TextDecoder`, `FormData` and `Blob` are globals in Node 20,
 * bun and the Electron main process.
 */

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
  asArray,
  asObject,
  bool,
  field,
  num,
  nullableBool,
  nullableNum,
  nullableObject,
  nullableStr,
  objectField,
  oneOf,
  str,
  strArray,
  type Json,
} from './shape.js';
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
  type ArtifactWrite,
  type AsrOptions,
  type CancelResult,
  type Capability,
  type CapabilityRecord,
  type CapabilityRow,
  type CatalogRow,
  type ChatMessage,
  type ChatOptions,
  type ChatResponse,
  type ChunkData,
  type DoneData,
  type EngineOwner,
  type EngineRef,
  type EstimateBasis,
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
  type TaskStatus,
  type UnmetNeed,
  type TaskStepData,
  type UploadResult,
  type UpstreamName,
  type UpstreamSetting,
  type UpstreamTestResult,
  type VoiceInfo,
  type VoiceKind,
  type VoicePace,
  type VoiceServing,
  type WrittenArtifact,
} from './types.js';
import { SDK_VERSION } from './version.js';

const API_HEADER = 'X-Crucible-Api';
/**
 * The header a client NAMES ITSELF in — `crucible/__init__.py`'s
 * `CLIENT_NAME_HEADER`, and it must keep that spelling.
 *
 * `User-Agent` is a FORBIDDEN HEADER NAME in a browser: `fetch` silently drops
 * it, so an extension's `clientName` never reached the server and its jobs were
 * recorded under `Mozilla/5.0 (Macintosh; …) Chrome/154.0.0.0 Safari/537.36`.
 * The BookForge Reader popup printed that 120-character string back at Owen as
 * if it were an error. A bench's "held by" column is the worst place for one.
 *
 * Sent ALONGSIDE the User-Agent rather than instead of it: outside a browser the
 * UA still arrives and is still what curl and the CLI are read by, and the
 * server prefers this only when it is present and valid.
 */
const CLIENT_NAME_HEADER = 'X-Crucible-Client';
const JOB_STATES: readonly JobState[] = ['queued', 'running', 'done', 'failed', 'cancelled'];
const HEALTH_STATES = ['ok', 'warming', 'busy'] as const;
const CHAT_ROLES = ['system', 'user', 'assistant'] as const;
/** PHASE3-TTS.md section 2. The voice loader refuses any other word. */
const VOICE_KINDS: readonly VoiceKind[] = ['checkpoint', 'zeroshot', 'token'];
/** PHASE3-TTS.md section 2, difference 4. Also closed by the voice loader. */
const ESTIMATE_BASES: readonly EstimateBasis[] = ['measured', 'declared'];
/** OpenAI's stream terminator, sent as a bare `data:` line with no JSON. */
const DONE_SENTINEL = '[DONE]';
const EVENT_NAMES = [
  'queued',
  'warming',
  'progress',
  // PHASE3-TTS.md section 6's addition, and the reason `api_version` did not
  // move for it: a client that does not know the kind still sees every
  // `progress`, `artifact` and `done` it saw before. This client now knows it.
  'chunk',
  'artifact',
  'done',
  'failed',
  'cancelled',
] as const;

/**
 * How many artifacts {@link CrucibleClient.writeArtifactsTo} fetches at once.
 *
 * Not a throughput knob — a ceiling. `events()` replays a job's whole history
 * before it follows live, so attaching to a nearly-finished 1,400-chunk render
 * delivers 1,400 `artifact` frames in one burst; unbounded, that is 2,800
 * sockets opened in a tick (each artifact has a sidecar). Four keeps the fetch
 * of one chunk overlapped with the generation of the next, which is the whole
 * point, without turning a reconnect into a denial of service against the
 * server that is still rendering.
 */
const DEFAULT_ARTIFACT_CONCURRENCY = 4;

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
  /**
   * A deadline on EVERY call this client makes, in milliseconds.
   *
   * Optional, and there is no default: a client with no clock waits as long
   * as the platform waits, which is what `fetch` does and what this package
   * did before. An app that would rather give up says how long — BookForge's
   * old seam aborted at 60 s and had to keep its own timer to do it.
   *
   * **A per-call `signal` REPLACES it rather than composing with it.** The
   * two are different statements: the constructor's is "this app's patience",
   * the call's is "this caller owns this request", and a caller that brought
   * its own cancel has already decided. `timeoutMs` on a probe composes with
   * it, because that one is explicitly a clock and not an owner.
   */
  timeoutMs?: number;
}

/** Options for {@link CrucibleClient.events}. */
export interface EventsOptions {
  /**
   * Resume after this event id: the server replays everything with a higher id
   * and then follows live. Omit to start from the beginning of the job.
   */
  lastEventId?: number;
}

/** Options for {@link CrucibleClient.writeArtifactsTo}. */
export interface WriteArtifactsOptions extends EventsOptions {
  /**
   * How many artifacts to fetch at once. Defaults to
   * {@link DEFAULT_ARTIFACT_CONCURRENCY}; see there for why there is a ceiling
   * at all. A value below 1 is refused rather than rounded up.
   */
  concurrency?: number;
}

/**
 * A clock on a PROBE — `ping`, `info`, `capability`.
 *
 * The three calls an app makes to ask "is this server there, and what can it
 * do", often about a machine that may be asleep. Foundry draws a tooltip from
 * `capability()` and puts a 3 s clock on it, because a Mac Studio that has
 * suspended its network answers nothing at all and a fetch with no deadline
 * hangs until the OS gives up — minutes, behind a tooltip.
 *
 * Both fields are OPTIONAL and neither has a default: this client never puts a
 * deadline on a call the caller did not put one on. Waiting forever is the
 * platform's behaviour, a caller who wants otherwise says how long, and a
 * number invented here would cancel somebody's slow-but-working probe.
 *
 * A timeout aborts with the platform's own `TimeoutError` DOMException and a
 * caller's `signal` aborts with whatever reason they gave, so
 * `error.name === 'TimeoutError'` and `'AbortError'` both stay true — the two
 * are told apart by the caller, not merged here into one "it did not answer".
 */
export interface ProbeOptions {
  /** The caller's own cancel. Composed with `timeoutMs` when both are given. */
  signal?: AbortSignal;
  /** Milliseconds before this probe is abandoned. Must be > 0. */
  timeoutMs?: number;
}

/**
 * `ProbeOptions` as a `RequestInit`. One place, so the three probes cannot
 * come to compose a signal differently.
 */
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
  /** The constructor's clock, or null: wait as long as the platform does. */
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
    // The bare name, for `X-Crucible-Client`. The server validates 1-80
    // characters with no control characters and falls back to the User-Agent
    // when that fails, so the full `<name> crucible-client/<version>` string
    // would be rejected by length on a long client name and silently lose the
    // whole point. `requireText` has already refused an empty one.
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

  // ------------------------------------------------------------------- ping

  /**
   * `GET /v1/ping`, unauthenticated. Answers "is there a Crucible here at all",
   * separately from "is my token right": a responder that is not a Crucible
   * throws {@link CrucibleNotACrucible}, not an auth error.
   *
   * Takes a {@link ProbeOptions} clock, like the other two probes: this is the
   * call an app makes about a machine that may be asleep.
   */
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

  // ------------------------------------------------------------------- info

  /**
   * `GET /v1/info` — who this server is, what it runs on, what it can serve.
   *
   * Takes a {@link ProbeOptions} clock (PHASE15-HOST.md 3.8).
   */
  async info(options: ProbeOptions = {}): Promise<ServerInfo> {
    const body = await this.#json('/v1/info', probeInit(options), 'info');
    const server = objectField(body, 'server', 'info');
    const host = objectField(body, 'host', 'info');
    const gpu = objectField(host, 'gpu', 'info.host');
    const capabilities = asArray(field(body, 'capabilities', 'info'), 'info.capabilities').map(
      (entry, index) => asObject(entry, `info.capabilities[${index}]`),
    );
    // THE VINTAGE IS READ ONCE, FOR THE WHOLE DOCUMENT — PHASE15-HOST.md
    // section 3.3's client reading rule, applied to `needs_reference` exactly
    // as `capability()` applies it to `route`. The `tts` capability's rows ARE
    // `/v1/voices`' rows, so the question is asked of every voice row this
    // document carries, wherever it sits, and never of a row on its own. See
    // {@link readVoiceInfo}.
    const statesNeedsReference = capabilities.some(
      (entry) =>
        entry['job_type'] === 'tts' &&
        Array.isArray(entry['models']) &&
        anyRowStates(entry['models'], 'needs_reference'),
    );
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
      // What to POST, which is a different list from what the server can serve:
      // `llm` is a capability, `load-model` and `unload-model` are the job types
      // that operate it.
      jobTypes: strArray(body, 'job_types', 'info'),
      capabilities: capabilities.map((entry, index) =>
        readCapability(entry, index, statesNeedsReference),
      ),
      ...readRole(body),
      pagesEngine: readPagesEngine(body),
    };
  }

  /**
   * `GET /v1/capability` — what this server can hold, per capability class,
   * and why not. PHASE9-CAPABILITY.md.
   *
   * The read to make before deciding what to ask for. A class is what a client
   * wants — "a translate-class model", "a voice" — and the row is the server's
   * answer: the concrete id it picked on this card, or `enabled: false` with the
   * shortfall that decided it. A disabled class **is an answer**, not an error;
   * render it as "this machine cannot do that", never as a fault.
   *
   * A server that has decided nothing — a config written before
   * `crucible capability` ran — throws {@link CrucibleCapabilityUndecided}
   * (503 `capability_undecided`) rather than answering with empty rows, which
   * would read as "probed, and nothing fit". That is the operator's to fix
   * (`crucible capability --write`), and {@link info} still says what the
   * server offers meanwhile.
   */
  async capability(options: ProbeOptions = {}): Promise<CapabilityRecord> {
    const body = await this.#json('/v1/capability', probeInit(options), 'capability');
    return readCapabilityRecord(body);
  }

  // -------------------------------------------------------------- settings
  //
  // PHASE15-HOST.md sections 3.1, 3.2 and 3.8. Owen, 2026-09-14: *"Bookforge
  // and foundry setup/settings should be able to configure crucible settings.
  // If the user enters an anthropic api key, it should pass through to
  // crucible."* These three methods are the whole of that pass-through, and an
  // app that uses them holds no key, no route and no cloud model list.

  /**
   * `GET /v1/settings` — where each class's work runs, and which upstreams are
   * configured.
   *
   * **No key comes back, ever.** {@link UpstreamSetting.keyHint} is `…` plus
   * four characters, which is enough to recognise which of two accounts is in
   * there and nothing else. Render it verbatim; the ellipsis is the server's.
   */
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

  /**
   * `PUT /v1/settings` — a PARTIAL patch, applied whole or not at all.
   *
   * Returns the document AFTER the write, so a window draws what it is handed
   * and never guesses what took. **A refusal applies nothing**: configure an
   * upstream and route a class to it in ONE call and either both happen or
   * neither does, which is what makes the "paste a key" step in an app's AI
   * settings a single action rather than two that can half-fail.
   *
   * Every refusal carries `details.field`, the dotted path — `routes.translate`,
   * `upstreams.anthropic.key` — so a window highlights the control that caused
   * it instead of showing a banner. `upstream_in_use` additionally carries
   * `details.classes`.
   */
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

  /**
   * `POST /v1/settings/upstreams/{name}/test` — ask an upstream what it serves.
   *
   * With a `probe` the key or url is tested BEFORE it is saved, which is the
   * order a person works in: paste, check, save. Without one, the stored
   * credential is used.
   *
   * **This does not throw for the three test refusals.** `upstream_unreachable`,
   * `upstream_rejected` and `upstream_unconfigured` are ordinary answers to
   * "does this work" — a person pasting a key expects to be told no, not to
   * have an exception raised at their settings page — so they come back as
   * `{ok: false, code, message}` with the provider's own words. Everything
   * else (a bad bearer, a version mismatch, a server that is not there) throws
   * exactly as every other method does.
   */
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
      // THE SERVER'S OWN SENTENCE, not `error.message`, which every error
      // class here prefixes with "crucible refused/failed the request". A
      // settings window renders this beside the key field, and "crucible
      // refused the request" about a key ANTHROPIC rejected names the wrong
      // party — measured by BookForge. `serverMessage` is the sentence the
      // server wrote, and the server writes it saying who did what.
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
      status: oneOf(str(body, 'status', 'health'), HEALTH_STATES, 'health.status'),
      queueDepth: num(body, 'queue_depth', 'health'),
      residentModels: strArray(body, 'resident_models', 'health'),
      // Read as a plain nullable string, not narrowed to `llm | tts`: the set of
      // kinds grows with the job types (PHASE4's aligner is next), and a client
      // that threw a protocol error on a kind it had not heard of would be
      // broken by the server that added one.
      residentKind: nullableStr(body, 'resident_kind', 'health'),
      stopping: readStopping(body, 'health'),
    };
  }

  /**
   * `GET /v1/activity` — what is on this server and how far along, in one read,
   * with no job id.
   *
   * The question a bench widget asks, which is not the question a job's event
   * stream answers. The step that owns a job reads the stream; a widget that
   * owns no job and may never own one reads this. The two do not compete.
   *
   * **It reports and nothing else.** It does not admit, reserve, claim or lock.
   * Reading `acceptsWork: true` and submitting is racing every other client, and
   * that race is settled at the door: `POST /v1/jobs` admits one and refuses the
   * other by name. The loser has lost a round trip and nothing else, because it
   * never gave up ownership of its own queue.
   *
   * `accelerator` is opt-in and costs an `nvidia-smi` per call; a bench polling
   * three servers every few seconds should not ask for it.
   */
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
              memoryBytesEstimate: nullableNum(
                resident,
                'memory_bytes_estimate',
                'activity.resident',
              ),
              // THE STRANDED CARD, and the two fields that let a client see it.
              // `heldBy` null with `resident` set means nothing is coming back
              // for what is on the card. `details` is passed through as the
              // holding fact's own shape rather than reshaped here: a job's is
              // the `server_busy` body a client already parses, and rebuilding
              // it would be this SDK inventing a second vocabulary for a
              // document the server already speaks.
              heldBy: readHeldBy(resident),
              unclaimedSince: nullableStr(
                resident,
                'unclaimed_since',
                'activity.resident',
              ),
            },
      stopping: readStopping(body, 'activity'),
      warming: nullableStr(body, 'warming', 'activity'),
      claim: claim === null ? null : { heldBy: str(claim, 'held_by', 'activity.claim') },
      streaming: streaming === null ? null : readStreaming(streaming),
      lease: lease === null ? null : readLease(lease, 'activity.lease'),
      chat: {
        inFlight: num(chat, 'in_flight', 'activity.chat'),
        // What this engine admits at once. Null for an engine that states no
        // concurrency AND for an empty card — never "unlimited".
        maxInFlight: nullableNum(chat, 'max_in_flight', 'activity.chat'),
        maxInFlightBasis: nullableStr(chat, 'max_in_flight_basis', 'activity.chat'),
        rows: asArray(field(chat, 'rows', 'activity.chat'), 'activity.chat.rows').map(
          (entry, index) => {
            const where = `activity.chat.rows[${index}]`;
            const row = asObject(entry, where);
            return {
              id: num(row, 'id', where),
              act: nullableStr(row, 'act', where),
              model: str(row, 'model', where),
              client: nullableStr(row, 'client', where),
              since: str(row, 'since', where),
            };
          },
        ),
      },
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

  // ----------------------------------------------------------------- leases

  /**
   * `POST /v1/models/{subject}/lease` — say that a run against the resident
   * thing is in progress, so nothing takes it off the card underneath.
   *
   * **Why this exists.** A chat completion holds nothing on a Crucible: no lane,
   * no job, no claim — deliberately, because a vLLM engine batches. That is
   * right for one chat and wrong for two thousand. A book translated block by
   * block leaves the server idle by every published measure between any two
   * blocks, and a `load-voice` submitted in one of those gaps evicts the
   * translator mid-run. Only the client knows the run exists, so the client
   * says so.
   *
   * ```ts
   * const lease = await crucible.lease('qwen3.8-27b-4bit', { act: 'translate', ttlSeconds: 120 });
   * const beat = setInterval(() => void crucible.heartbeat(lease.leaseId), 60_000);
   * try { await translateTheBook(); } finally {
   *   clearInterval(beat);
   *   await crucible.release(lease.leaseId);
   * }
   * ```
   *
   * **It may name a model, a voice or an aligner.** The card holds one thing, so
   * the id alone identifies it and the server supplies `kind` — there is nothing
   * for the caller to state and nothing to get wrong. That is what a book
   * rendered CHAPTER BY CHAPTER needs: without a lease on its voice each chapter
   * loads narrator again, because the card is cleared the moment nothing holds
   * it. Same for a book aligned chapter by chapter, and for a re-roll after a
   * render.
   *
   * ```ts
   * await crucible.job({ type: 'load-voice', model: 'mistborn' }).wait();
   * const held = await crucible.lease('mistborn', { act: 'tts', ttlSeconds: 300 });
   * for (const chapter of chapters) await crucible.render(chapter); // one load
   * await crucible.release(held.leaseId);
   * ```
   *
   * **It must already be resident** — a lease promises not to move what is on
   * the card and never loads anything, so anything else is refused
   * `not_resident` naming what IS resident, of whatever kind. **One lease at a
   * time, per server**: a second is refused {@link CrucibleLeased}, naming the
   * holder, exactly as a loader is.
   *
   * `ttlSeconds` is how long the lease outlives silence, not how long the run
   * is: heartbeat a short one rather than asking for a long one. The server
   * states its own range in the refusal (`invalid_ttl`).
   *
   * **Release it.** Expiry is the backstop for a client that died, not the way a
   * finished run ends — a `finally` that releases frees the next client at once
   * instead of after the whole ttl.
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

  /**
   * `POST /v1/leases/{id}/heartbeat` — I am still here. Returns the new
   * `expiresAt`.
   *
   * **A rejection here is not a log line.** `unknown_lease` means this run is no
   * longer protected — the lease expired, or was released — and the card may
   * move at any moment. The server's `details.reason` says which of the two.
   */
  async heartbeat(leaseId: string): Promise<string> {
    const id = requireText(leaseId, 'leaseId');
    const body = await this.#json(
      `/v1/leases/${encodeURIComponent(id)}/heartbeat`,
      { method: 'POST' },
      'heartbeat',
    );
    return str(body, 'expires_at', 'heartbeat');
  }

  /**
   * `DELETE /v1/leases/{id}` — give the card back.
   *
   * Answers 204 with no body, so there is nothing to return. Releasing a lease
   * that is already gone is a refusal rather than a shrug: a client that thinks
   * it still holds one has to be told it does not.
   */
  async release(leaseId: string): Promise<void> {
    const id = requireText(leaseId, 'leaseId');
    const response = await this.#fetch(
      `/v1/leases/${encodeURIComponent(id)}`,
      { method: 'DELETE' },
      true,
    );
    if (!response.ok) throw await this.#failure(response);
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
   *
   * `options.signal` aborts the POST itself and nothing else. Once the server
   * has answered with a job id the job exists and is queued, and abandoning the
   * socket would not stop it — {@link cancel} is what stops a job. The
   * distinction matters here more than on a chat because a `tts` body can be a
   * whole book's text.
   */
  async submit(request: JobRequest, options: { signal?: AbortSignal } = {}): Promise<string> {
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

    const init: RequestInit = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    };
    // `exactOptionalPropertyTypes` forbids writing `signal: undefined`, and
    // fetch would reject it anyway. Same shape as `#chatRequest`.
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
      error: readFailureOrNull(field(body, 'error', 'job'), 'job.error'),
      artifacts: strArray(body, 'artifacts', 'job'),
      created: str(body, 'created', 'job'),
      started: nullableStr(body, 'started', 'job'),
      finished: nullableStr(body, 'finished', 'job'),
      // ABSENT IS A FACT HERE, unlike everywhere else in this file. A loader
      // always states `lease_id` — null when it was asked to hold nothing — so
      // for those two job types the strict reading applies and "no lease" is
      // distinguishable from "old server". Every OTHER job type cannot hold a
      // lease at all and carries no such key: a `tts` render takes the lane
      // instead. Demanding the key from them would be demanding a field about
      // a question they are not asked.
      leaseId: 'lease_id' in body ? nullableStr(body, 'lease_id', 'job') : null,
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

  // -------------------------------------------------------------------- llm

  /**
   * `GET /v1/models` - every model this server has a manifest for, and the four
   * separate facts about each: whether this host's backend is supported,
   * whether the weights are installed, whether it is resident right now, and
   * whether it could be loaded right now (which also depends on the accelerator
   * guard, so a model can be installed and still not loadable).
   *
   * A model that is not loadable always carries the server's `reason`.
   */
  async models(): Promise<ModelInfo[]> {
    const body = await this.#jsonValue('/v1/models', { method: 'GET' }, 'models');
    const entries = asArray(body, 'models');
    return entries.map((entry, index) =>
      readModelInfo(asObject(entry, `models[${index}]`), `models[${index}]`),
    );
  }

  /**
   * Queue a `load-model` job and return its id. Watch it with {@link events}:
   * `queued`, then a `warming {message}` per line of the engine's readiness,
   * then `done {resident}`.
   *
   * Nothing is loaded implicitly anywhere else - {@link chat} on a model that is
   * not resident is refused, never satisfied by a silent load - and the server
   * refuses this call by name before queuing if the model is unknown, not
   * installed, unsupported on this backend, too big for the free VRAM, or the
   * card is busy with someone else's work.
   */
  async loadModel(model: string, options?: LoadModelOptions): Promise<string> {
    return this.submit({
      type: 'load-model',
      model: requireText(model, 'model'),
      params: leaseParams(options?.lease),
      inputs: {},
    });
  }

  /**
   * Queue an `unload-model` job and return its id. `done` when the engine has
   * exited and the card is back. `model_not_resident` if it was not loaded.
   */
  async unloadModel(model: string): Promise<string> {
    return this.submit({
      type: 'unload-model',
      model: requireText(model, 'model'),
      params: {},
      inputs: {},
    });
  }

  /**
   * `POST /v1/openai/chat/completions` - one completion from the resident
   * engine, read down to the parts a caller uses.
   *
   * `model` must name the resident model. If it does not, the server answers
   * 409 and this throws {@link CrucibleRefused} with
   * `code === "model_not_resident"`, its message naming what is resident
   * instead. It never loads a model to satisfy a chat.
   *
   * `finishReason` comes back as the engine said it. `length` means the answer
   * is truncated — check it before you use `content`, and check it before you
   * parse `content` as JSON under a `responseFormat`, because a truncated
   * document and a malformed one are different problems.
   *
   * Passing an already-aborted `signal`, or aborting during the call, rejects
   * with the DOM `AbortError` itself - the caller's own cancellation is not a
   * dead server and is not reported as one.
   */
  async chat(options: ChatOptions): Promise<ChatResponse> {
    const init = this.#chatRequest(options, false);
    const body = await this.#json('/v1/openai/chat/completions', init, 'chat');
    return readChatResponse(body);
  }

  /**
   * The same completion, streamed: an async iterable of the content deltas, in
   * order, as OpenAI's `chat.completion.chunk` frames arrive. Concatenating
   * everything it yields gives the text {@link chat} would have returned.
   *
   * The iterator ends on OpenAI's `data: [DONE]` terminator. A stream that ends
   * *without* one throws {@link CrucibleUnreachable}: a truncated answer and a
   * finished answer must never look the same, exactly as with {@link events}.
   *
   * Aborting `signal` mid-stream throws the DOM `AbortError` out of the
   * iterator, so a `for await` rejects rather than ending quietly. See the
   * README.
   */
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
      // Closing the socket is cleanup; a failure to close must not mask what
      // the stream already delivered.
      await stream.cancel().catch(() => undefined);
    }

    // Reached only when the body ended without [DONE].
    throw new CrucibleUnreachable(
      this.url,
      "the chat completion stream ended without OpenAI's [DONE] terminator; the " +
        'answer is truncated',
    );
  }

  /** The one request both chat calls make, validated once. */
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
      // Checked for the shape the engines agree on and then forwarded as it
      // stands. The `schema` inside is the caller's grammar: this client does
      // not read it, and an engine that will not compile it says so itself in a
      // 400 the proxy relays untouched.
      payload['response_format'] = readResponseFormat(given.responseFormat);
    }
    if (given.thinking !== undefined) {
      if (typeof given.thinking !== 'boolean') {
        throw new CrucibleConfigError(
          'thinking',
          `must be a boolean, got ${typeof given.thinking}`,
        );
      }
      // The engines take this per request: mlx-lm reads `chat_template_kwargs`
      // off the body and merges it into the template arguments, and vLLM
      // honours the same field. Crucible proxies the body verbatim, so it
      // reaches the engine as written.
      payload['chat_template_kwargs'] = { enable_thinking: given.thinking };
    }

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (given.act !== undefined) {
      // A HEADER AND NOT A BODY FIELD, because the body is OpenAI's and is
      // proxied to the engine verbatim; the act is Crucible's and stops at the
      // proxy. Sent only when the caller says one: an absent header is how the
      // server records "it did not say", and a default here would put a name
      // nobody chose on somebody's bench. The name itself is the SERVER's
      // vocabulary — it answers 400 `unknown_act` listing the capability
      // classes, before the completion runs — so this client checks that a
      // string was given and no more.
      headers['X-Crucible-Act'] = requireText(given.act, 'act');
    }

    const init: RequestInit = {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    };
    // `exactOptionalPropertyTypes` forbids writing `signal: undefined`, and
    // fetch would reject it anyway.
    if (given.signal !== undefined) init.signal = given.signal;
    return init;
  }

  // -------------------------------------------------------------------- tts

  /**
   * `GET /v1/voices` — every voice this server has a manifest for, and the four
   * separate facts about each, exactly as {@link models} reports them for a
   * model: backend support, installation, residency, and whether it could be
   * loaded right now.
   *
   * These are the same rows the `tts` capability carries in {@link info}, from
   * the same producer, so a voice has one description wherever you find it.
   *
   * A voice that is not loadable always carries the server's `reason`. Unlike a
   * model row, a *loadable* voice row carries `reason: null` rather than omitting
   * the key — the two routes differ there, and this client reads each as it is
   * rather than making them look alike.
   *
   * Refused with `job_type_disabled` on a server where `[jobs] enable_tts` is
   * false, the same way `/v1/models` is for `llm`.
   *
   * A document in which NO row carries `needs_reference` comes from a server
   * that predates the field and every voice on it is read as
   * `needsReference: false` — PHASE15-HOST.md section 3.3's client reading
   * rule, all or nothing, exactly as {@link capability} reads `route`.
   */
  async voices(): Promise<VoiceInfo[]> {
    const body = await this.#jsonValue('/v1/voices', { method: 'GET' }, 'voices');
    const entries = asArray(body, 'voices');
    const statesNeedsReference = anyRowStates(entries, 'needs_reference');
    return entries.map((entry, index) =>
      readVoiceInfo(
        asObject(entry, `voices[${index}]`),
        `voices[${index}]`,
        statesNeedsReference,
      ),
    );
  }

  /**
   * Queue a `load-voice` job and return its id. Watch it with {@link events},
   * exactly as you watch {@link loadModel}: `queued`, then a `warming {message}`
   * per line of narrator's readiness, then `done {resident}`.
   *
   * One card holds one thing, and since PHASE3-TTS.md section 5 that thing may
   * be a voice or a model (see {@link Health.residentKind}). So loading a voice
   * is refused for the same reasons loading a model is — unknown, not installed,
   * unsupported on this backend, too big for the free VRAM, the card busy with
   * someone else's work — plus one of its own: the tts env for this voice's
   * narrator engine is not installed (`env_missing`).
   *
   * Nothing loads a voice implicitly anywhere else.
   *
   * **A zero-shot voice takes its reference clip here** (PHASE3-TTS.md section
   * 5). It is the base weights plus somebody's recording, and the recording is
   * yours rather than the server's — so pass
   * {@link LoadVoiceOptions.reference} whenever the voice's row says
   * {@link VoiceInfo.needsReference}. Three more refusals come with it, all
   * before the job is queued: `reference_required` (a zero-shot load with no
   * clip — the engine would otherwise come up in the model's OWN voice under
   * this id), `reference_not_allowed` (a clip on a checkpoint, whose voice is
   * in its weights) and `reference_malformed` (not base64, not a readable WAV,
   * no transcript, or over narrator's 30-second budget).
   *
   * Once it is loaded, a zero-shot voice is the resident voice and nothing
   * downstream knows it was cloned: {@link render} and {@link stream} name it
   * like any other. Which clip is resident is on the server's `/v1/activity`,
   * as the clip's `name` and a sha256 of its audio.
   */
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
            // Omitted rather than sent as null when there is none: the load door
            // forbids unknown keys and a null label is not a label.
            ...(reference.name === undefined ? {} : {name: reference.name}),
          },
        }),
      },
      inputs: {},
    });
  }

  /**
   * Queue an `unload-voice` job and return its id. `done {resident: null}` when
   * narrator has exited and the card is back. `voice_not_resident` if it was not
   * loaded — including when a *model* holds the card, which is a different
   * refusal than "nothing is loaded" and says so.
   */
  async unloadVoice(voice: string): Promise<string> {
    return this.submit({
      type: 'unload-voice',
      model: requireText(voice, 'voice'),
      params: {},
      inputs: {},
    });
  }

  /**
   * Queue a `tts` render job — text in, one `<index>.flac` per chunk out — and
   * return its id (PHASE3-TTS.md section 6).
   *
   * It returns a job id and nothing else, because that is what every other
   * queueing call in this client returns and because there is only one way to
   * watch a job. Drive it with {@link events} exactly as you drive
   * {@link loadModel}, or with {@link writeArtifactsTo}, which is
   * {@link events} plus the batch writer:
   *
   * ```ts
   * const jobId = await crucible.render({voice, language: 'en', take: 0, chunks});
   * for await (const event of crucible.events(jobId)) {
   *   if (event.event === 'chunk' && event.data.guard !== null) {
   *     record(event.data.index, event.data.guard);   // what the engine decided
   *   }
   * }
   * ```
   *
   * **Do not retake on what you read here.** That example used to be
   * `if (event.data.capped === true) retake(event.data.index)`, and it was the
   * superseded model in one line: a client re-deciding something the engine had
   * already decided, without the frame cap, the seed or the book's running pace
   * in front of it. Owen ruled on 2026-09-13 that the guard and the retake
   * decision belong to the model and its inference
   * (`docs/PHASE6-REMOTE-RENDER.md`), so a chunk that arrives has already been
   * through the ladder and been accepted. {@link ChunkData.guard} says what
   * happened to it, for your records and your eye — `null` when narrator sent no
   * verdict, which is never to be read as "it was clean".
   *
   * **The voice need not be resident.** A render job owns the exclusive lane for
   * its whole duration and is an operator's explicit order, so it loads its own
   * voice if it has to, emitting `warming` as `loadVoice` does. That is the one
   * asymmetry with `llm`, where a fine-grained unattended chat never loads.
   *
   * **A failed chunk is reported and the run continues.** One bad sentence never
   * sinks the other 1,399: the failure is named in a `progress` line when it
   * happens and again in `done`, where {@link readRenderResult} reads the
   * authoritative list. No `chunk` event and no artifact is produced for a row
   * that rendered nothing.
   *
   * Refused before the job is queued, by name: `unknown_model`,
   * `invalid_params`, `ffmpeg_missing`, `backend_unsupported`, `env_missing`,
   * `voice_not_installed`, `accelerator_busy`, `insufficient_memory`,
   * `voice_kind_unsupported` (a zero-shot voice that is not already resident:
   * a render job loads its own voice, and a zero-shot load needs the clip only
   * {@link loadVoice} carries), `retake_without_band`, `band_malformed` and
   * `width_over_serving`.
   *
   * **`chunk_too_long` and `unknown_take` are retired** (2026-09-19). The
   * server no longer refuses a chunk by length at all, and a take past the end
   * of the ladder is a seed lane at the voice's own sampling rather than an
   * error. Pack against the row you read from {@link voices} because that is
   * still where a voice's cap and band are published; nothing in this file
   * keeps a copy of either, for the reason {@link asr} keeps no copy of
   * faster-whisper's language list.
   */
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
        // For `tts` the model IS the voice: a Higgs v3 voice is the merged
        // checkpoint the engine runs, so the wire needs no second word for it.
        model: requireText(given.voice, 'voice'),
        params: {
          language: requireText(given.language, 'language'),
          take: requireIndex(given.take, 'take'),
          chunks: readRenderChunks(given.chunks),
          // ABSENT STAYS ABSENT for all three. `retake` omitted is the bare
          // arm, a `band` nobody stated is a band nobody can be held to, and
          // `width` omitted is a render that says nothing about width, which
          // the server passes on as silence so the engine keeps the width it
          // was STARTED at. Each is a real state the server names, so sending
          // a key this client invented would be answering a question the
          // caller did not.
          ...(given.retake === undefined ? {} : { retake: given.retake }),
          ...(given.band === undefined ? {} : { band: given.band }),
          ...(given.width === undefined ? {} : { width: given.width }),
        },
        inputs: {},
      },
      submission,
    );
  }

  /**
   * {@link events}, plus the batch writer: every artifact this job publishes is
   * fetched and written into `dir` as `<name>`, with its provenance sidecar
   * beside it, while the job is still running.
   *
   * **Why the bytes cross the wire at all.** There is no shared mount, ever. The
   * library lives on `Z:` (`\\TITAN\iO`), WSL cannot mount a network drive, and
   * that single fact is why whole-m4b alignment cannot run on Owen's PC today.
   * So Crucible writes files on its own host, the client fetches them over HTTP
   * — even from a server on localhost — and writes them where assembly and
   * resume already look.
   *
   * Three properties, each of which is the point rather than a nicety:
   *
   * - **Each artifact is fetched as its `artifact` event lands**, overlapped with
   *   the next chunk still generating. A writer that waited for `done` would turn
   *   a streaming server back into a batch one — an hour of finished FLACs
   *   sitting on the server while the last sentence renders.
   * - **Each file is written atomically**: bytes to a temporary name in the same
   *   directory, then a rename. BookForge's resume test is "the file exists and
   *   exceeds 1024 bytes", so a half-written FLAC left behind by a killed run
   *   reads as a finished chunk and that sentence is silently missing from the
   *   book. A rename within one directory is atomic on NTFS and on ext4, so
   *   `<index>.flac` only ever exists complete.
   * - **The sidecar is written first**, then the artifact. DESIGN.md section 7
   *   says clients must persist provenance beside the output, and doing it in
   *   this order means the existence of `<index>.flac` implies the existence of
   *   `<index>.flac.provenance.json`. The other order can leave a finished
   *   chunk that cannot say which voice, which revision or which server made it,
   *   and resume would never ask for it again.
   *
   * It yields the job's own events unchanged, interleaved with what it has
   * written ({@link ArtifactWrite}) — the writer reports progress without
   * swallowing anything. The iterator ends after the terminal event and after
   * every outstanding write has landed, so when it returns, the directory is
   * complete.
   *
   * A write that fails throws out of the iterator. It is not reported as a
   * `kind` and the run does not continue: a failed *chunk* is the job's ordinary
   * news and the server already reports it, but a failed *write* means this
   * client cannot do the one thing it was asked to do, and a caller that learned
   * about it from a yielded value would be free to ignore it.
   *
   * **Resuming with `lastEventId` writes only what arrives after it.** The
   * server replays events above that id and no further back, so artifacts
   * announced before it are the caller's own — they are what the caller already
   * had when it recorded that id. Omit it and the whole history replays, and
   * this reconciles against `done`'s authoritative `artifacts` list as well, so
   * an artifact whose event was somehow missed is still written.
   *
   * It needs a Node-like runtime, and gets one lazily; see
   * {@link loadNodeFileApis} for how that is kept inside the SDK's
   * zero-dependency rule.
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

    /** Writes still in flight, by artifact name. None of these ever rejects. */
    const active = new Map<string, Promise<void>>();
    /** Landed writes waiting to be yielded at the next boundary. */
    const landed: WrittenArtifact[] = [];
    /** Names already started, so a replayed event never writes a file twice. */
    const started = new Set<string>();
    let failure: unknown = null;

    const begin = (name: string): void => {
      if (started.has(name)) return;
      started.add(name);
      // The task deletes itself from `active` in its own `finally`, before the
      // promise settles, so a `Promise.race` over `active.values()` can never
      // observe an entry that has already finished.
      const task = (async () => {
        try {
          landed.push(await this.#writeArtifact(node, id, name, directory));
        } catch (error) {
          // The first failure is the one that explains the rest; a full disk
          // fails every write after it and the ninth message says nothing.
          if (failure === null) failure = error;
        } finally {
          active.delete(name);
        }
      })();
      active.set(name, task);
    };

    /** Yield-ready writes, taken as a batch so the array is never mutated mid-loop. */
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

      // `done` lists what the job published, and it is the authority. With the
      // whole history replayed, a name here that produced no `artifact` frame is
      // a gap, and writing it is the difference between a complete directory and
      // a book with a hole in it. Skipped when the caller resumed from an id:
      // the prefix they chose not to replay is theirs, and re-fetching a
      // finished book's worth of FLACs on every reconnect would be a worse bug
      // than the one it guards against.
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
      // A caller that breaks out early, and any throw above, leaves writes in
      // flight. Settling them is cleanup: it keeps a half-written temporary from
      // outliving this call, and because no task ever rejects it cannot mask the
      // error that is already on its way out.
      await Promise.all(active.values());
    }
  }

  /**
   * One artifact and its sidecar: fetched together, checked, then written
   * sidecar-first so that the artifact's existence implies the sidecar's.
   */
  async #writeArtifact(
    node: NodeFileApis,
    jobId: string,
    name: string,
    dir: string,
  ): Promise<WrittenArtifact> {
    // The name comes off the wire and is about to become a path on the caller's
    // disk. The server validates it as a single member and would not send a
    // traversal — but "the server would not" is not a property of this machine's
    // filesystem, and the whole point of the batch writer is that it writes into
    // a real library directory.
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

    // Parsed to prove it is the document DESIGN.md section 7 describes — but the
    // BYTES that go to disk are the server's own, not a re-serialisation of this
    // reading. `Provenance` keeps the server's key spelling precisely so the
    // file round-trips, and writing back what was read is stronger still.
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

  // ------------------------------------------------------------ accelerator

  /**
   * `GET /v1/accelerator` — what is on the card right now, and which of it is
   * Crucible's own (PHASE4-AUDIO.md section 5).
   *
   * This is the `nvidia-smi --query-compute-apps` the load guard runs, plus the
   * free and total figures, plus what Crucible has resident, plus a flag per
   * holder saying whether that pid is one of this server's engines. It is the
   * one call that answers what a client's own GPU arbitration is otherwise
   * guessing at.
   *
   * **It reports and it never evicts.** Nothing here asks anybody to leave.
   *
   * Three things to read carefully before deciding the card is free:
   *
   * - {@link AcceleratorHolder.bytes} is `number | null`, and the null is the
   *   driver declining to say (WDDM, permissions) — **not zero**. A queue shown
   *   0 for a process holding 8 GB concludes the card is idle.
   * - an **empty** `holders` is not an idle card either. Under WSL2 the driver
   *   shim answers the compute-app query with an empty list while a process
   *   inside that VM holds 17 GB, which is why
   *   {@link AcceleratorState.unattributedBytes} exists.
   * - a probe that cannot read the card throws
   *   {@link CrucibleAcceleratorUnreadable} (503 `accelerator_unreadable`) rather
   *   than returning zeroes. That is the distinction the whole route is for:
   *   "ask again", never "it is free".
   */
  async accelerator(): Promise<AcceleratorState> {
    const body = await this.#json('/v1/accelerator', { method: 'GET' }, 'accelerator');
    return readAcceleratorState(body);
  }

  // -------------------------------------------------------------------- asr

  /**
   * Queue an `asr` job — one audio file in, one transcript out — and return its
   * id. Watch it with {@link events}; the transcript arrives as the artifact
   * `transcript.json`, which you fetch with {@link artifact}.
   *
   * Every one of `model`, `language`, `vadFilter` and `wordTimestamps` is
   * required, and this client supplies none of them. The server refuses a
   * missing one, and papering over that would be worse than the refusal: a
   * transcript produced under rules the caller did not choose looks exactly like
   * one produced under the rules they did. **There is no default model** for the
   * same reason — an ASR pass at the wrong size is a transcript that looks fine,
   * is worse, and has nothing in it to say so.
   *
   * `progress` events carry `{stage, processed_s, total_s, cues}` in
   * {@link ProgressData.extra} alongside the fraction, because the decode drives
   * no fraction at all and an eighteen-hour book spends its first minutes with
   * the percentage rounding to zero.
   *
   * The job fails, naming every bad window, rather than publishing a transcript
   * with a hole in it.
   */
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
        // Sent as the server names them. `language` is not checked against a
        // code list here: faster-whisper's own list is the authority, the server
        // checks against it before the job is queued and refuses naming the
        // code, and a second copy of a hundred codes in this file is a second
        // thing to drift.
        language: requireText(given.language, 'language'),
        vad_filter: requireBool(given.vadFilter, 'vadFilter'),
        word_timestamps: requireBool(given.wordTimestamps, 'wordTimestamps'),
      },
      // The input's NAME becomes the file's name on the server's disk, and
      // ffmpeg reads the container off the extension — so the caller names the
      // file and this client does not invent one.
      inputs: { [requireText(given.filename, 'filename')]: audio },
    });
  }

  // ---------------------------------------------------------------- plumbing

  async #json(path: string, init: RequestInit, where: string): Promise<Json> {
    return asObject(await this.#jsonValue(path, init, where), where);
  }

  /**
   * The same, for the one authed route whose body is a JSON array rather than
   * an object (`GET /v1/models`, PHASE2-LLM.md section 5).
   */
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
    // THE NAME, IN A HEADER A BROWSER CAN ACTUALLY SET. Unauthenticated too:
    // the pairing doors record who is asking, and a browser pairing itself
    // should not be `Mozilla/5.0 …` there either.
    headers.set(CLIENT_NAME_HEADER, this.#clientName);
    if (authenticated) {
      headers.set('Authorization', `Bearer ${this.#token}`);
      headers.set(API_HEADER, String(API_VERSION));
    }
    const target = `${this.url}${path}`;
    // A STALE POOLED SOCKET IS RETRIED ONCE, AND ONLY ON A SAFE METHOD.
    // See `isStaleConnection` below for what that means and `retryable` for
    // which methods qualify. The loop runs at most twice.
    const retryable = isSafeMethod(init.method);
    for (let attempt = 0; ; attempt += 1) {
      // ONE PLACE, so no door can be built that forgets the clock. A call that
      // brought its own `signal` keeps it untouched: the caller owning a
      // request has already decided when it ends, and quietly ANDing a second
      // deadline onto their cancel would end a stream they were still reading.
      //
      // BUILT PER ATTEMPT, because `AbortSignal.timeout` starts counting when
      // it is created: reusing the first attempt's signal would give the retry
      // whatever was left of a clock the first attempt already spent.
      const timed =
        this.#timeoutMs !== null && init.signal === undefined
          ? { ...init, headers, signal: AbortSignal.timeout(this.#timeoutMs) }
          : { ...init, headers };
      try {
        return await fetch(target, timed);
      } catch (cause) {
        // The caller cancelling is not the server dying. When the signal they
        // handed us is the reason the fetch rejected, the rejection is theirs and
        // travels back untouched (a DOM `AbortError`, or whatever reason they
        // passed to `abort`), so `error.name === 'AbortError'` still holds.
        const signal = timed.signal;
        if (signal !== undefined && signal !== null && signal.aborted) throw cause;
        if (attempt === 0 && retryable && isStaleConnection(cause)) continue;
        // Otherwise fetch rejects only for a transport failure; every HTTP status
        // resolves.
        throw new CrucibleUnreachable(this.url, describeCause(cause), cause);
      }
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
    if (response.status >= 500) {
      // One 5xx gets its own type, and only because one conclusion must never
      // be drawn from it: an unreadable accelerator probe is not an idle card
      // (PHASE4-AUDIO.md section 5). The subclass is still a
      // CrucibleServerError, so nothing that already handles 5xx changes.
      if (code === ACCELERATOR_UNREADABLE) {
        return new CrucibleAcceleratorUnreadable(response.status, code, message);
      }
      // The second, for the same reason: a host that has decided NOTHING is
      // not a host that can do nothing, and a client must be able to tell the
      // two apart without reading the message (PHASE9-CAPABILITY.md).
      if (code === CAPABILITY_UNDECIDED) {
        return new CrucibleCapabilityUndecided(response.status, code, message);
      }
      return new CrucibleServerError(response.status, code, message);
    }
    if (response.status >= 400) {
      const details = 'details' in envelope ? envelope['details'] : null;
      // One 4xx gets its own type, for the reason one 5xx does: the body is not
      // decoration. `crucible/jobs/queue.py` answers server_busy with the
      // holder, the job, what it is doing and how far along — everything a bench
      // needs to say "GPU busy: foundry" and everything a `waitFor: "any"` walk
      // needs to decide to try the next machine. Read once here rather than
      // re-parsed identically in every client.
      // TWO SHAPES, ONE CODE, discriminated on `details.fact`. The job door
      // answers `server_busy` about the lane, where the holder is always a
      // job; the operator door (PHASE13-OPERATOR.md 3.3) answers it about the
      // CARD, where the holder is one of four kinds and says which in `fact`.
      // Read as a job, a lease-shaped body would come back a protocol error —
      // a page told its server sent nonsense when it sent the contract.
      if (code === SERVER_BUSY) {
        return isHeldByAFact(details)
          ? heldRefusal(response.status, code, message, details)
          : busyRefusal(response.status, code, message, details);
      }
      // The second 4xx with a body worth reading, for the first one's reason.
      // A caller shown "leased" has to be able to say who is mid-run, doing
      // what, and until when — and both doors that emit this code (a second
      // lease, and a loader that would evict) send the same shape.
      if (code === LEASED) return leasedRefusal(response.status, code, message, details);
      return new CrucibleRefused(response.status, code, message, details);
    }
    return new CrucibleProtocolError(
      `HTTP ${response.status} from ${this.url} is neither a success nor a refusal`,
    );
  }

  // -------------------------------------------------------- tts streaming

  /**
   * `POST /v1/tts/stream` — open a live TTS session. PHASE3-TTS.md section 7.
   *
   * The Listen path and the browser extension, as against {@link render}, which
   * is the book. The two are different in kind rather than in buffer size:
   * sub-sentence audio emitted while a row is still generating, rows retiring
   * out of order, and a cancel that aborts work in flight.
   *
   * ```ts
   * const session = await crucible.stream({ voice: 'deathstalker', language: 'en' });
   * void session.say('r1', 'He had been walking for some time.');
   * for await (const event of session) {
   *   if (event.kind === 'audio') speaker.write(event.pcm);
   * }
   * ```
   *
   * **The voice must already be resident.** This door never loads one — it
   * behaves like chat, not like a render job, and refuses `voice_not_resident`
   * naming what is resident instead. Post a `load-voice` job first
   * ({@link loadVoice}).
   *
   * **One session at a time, per server.** A session holds the resident voice's
   * whole attention; a second is refused as `stream_session_open`, and a `tts`
   * render job submitted while one is open is refused as `engine_in_use`.
   *
   * **It resolves attached.** The session's event stream is opened and the
   * server's `ready` frame read and checked before this returns, so the
   * example above is exactly the order it may be called in: the server's
   * `stream_not_attached` refusal — a row said into a session nobody is
   * listening to — is one a caller of this client cannot meet. (Until
   * 2026-09-14 the stream attached lazily on the first iteration, and a `say`
   * before it was refused.)
   *
   * The session is its own `AsyncIterable` and it reattaches across a dropped
   * connection on its own — see `src/stream.ts` for why that differs from
   * {@link events}, which never reconnects.
   */
  async stream(options: StreamOptions): Promise<TtsStreamSession> {
    // The session is handed exactly the four things it needs and nothing else.
    // Arrow functions rather than bound methods because `#fetch`, `#failure`
    // and `#json` are private to this class and stay that way: the streaming
    // module is a consumer of this client, not a second one.
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

  // -------------------------------------------------------- the operator door
  //
  // PHASE13-OPERATOR.md section 3.6. Six methods, and every one of them exists
  // because a person now administers a Crucible from a page it serves itself
  // rather than from a shell on that machine. They are on this client and not
  // in a second package for the reason there is one client at all: an app that
  // can render a book on a server should not need a different object to ask
  // that server what it has got.

  /**
   * `GET /v1/setup` — name, version, backend, every URL this server is
   * reachable on, the token, and a `crucible://` pairing line per URL.
   *
   * **It returns the token**, which reveals nothing: only a caller that
   * already has it can reach this route. What it buys is that nobody types a
   * secret twice — hand {@link ServerSetup.pairing}`[0]` to another app's
   * connect door and it fills in all three fields through
   * {@link parsePairing}.
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

  /**
   * `GET /v1/catalog` — every subject this server's backend can hold,
   * installed or not, with what the manifests already say about it.
   *
   * A subject with no block for this backend is ABSENT rather than listed as
   * unsupported, so a row here is always something this machine could really
   * have. Pull one with {@link submitTask}.
   */
  async catalog(): Promise<CatalogRow[]> {
    const body = await this.#json('/v1/catalog', { method: 'GET' }, 'catalog');
    const rows = asArray(field(body, 'rows', 'catalog'), 'catalog.rows');
    return rows.map((row, index) =>
      readCatalogRow(asObject(row, `catalog.rows[${index}]`), `catalog.rows[${index}]`),
    );
  }

  /**
   * `DELETE /v1/catalog/{kind}/{id}` — remove an installed subject's files.
   *
   * PHASE15-HOST.md 3.5a. The door the weights rule needs: a subject is never
   * stored twice on one machine, so when a second engine has its own copy the
   * first one's goes — and whoever asks must never reach into the server's
   * layout to do it. Answers nothing on success (204).
   *
   * **An app does not call this on a user's behalf without saying so on
   * screen.** That condition is the durable half and is unchanged.
   *
   * WHAT CHANGED, 2026-09-16: this comment used to go on to say that 3.5a is
   * explicit that NEITHER BookForge NOR Foundry calls it in this phase — the
   * host does, and an operator does from the page. Owen withdrew that:
   * *"they sohuld have a way to delete models from crucible, too. probably
   * through bookforge/foundry settings"* (docs/MODEL-CHOICE.md section 7). So
   * an app calls it now, from a settings page, behind a confirm that names the
   * SIZE — deciding about 17.3 GB is a different decision from deciding about
   * "a file" — with keep as the default.
   *
   * The sentence above survives the reversal intact, which is why it is stated
   * separately from the division it used to be attached to. A reader who found
   * the old division still written here would conclude the app's own delete
   * button was a mistake.
   *
   * Refused by name, and each name is a different thing to do about it:
   * `subject_unknown` (404), `subject_not_installed` (409),
   * `subject_in_use` (409, `details.who` says what is holding it) and
   * `subject_remove_failed` (500, `details.path` says which file would not
   * go). None is retried here.
   */
  async removeSubject(kind: SubjectKind, id: string): Promise<void> {
    const path =
      `/v1/catalog/${encodeURIComponent(kind)}/${encodeURIComponent(id)}`;
    const response = await this.#fetch(path, { method: 'DELETE' }, true);
    if (!response.ok) throw await this.#failure(response);
    // 204, and reading the body is what makes the connection reusable.
    await response.text();
  }

  /**
   * `POST /v1/tasks` — pull a subject, install a job type, or post a module.
   * Returns the task id; watch it with {@link taskEvents}.
   *
   * **One task at a time per server.** A second is refused `task_busy` naming
   * the running one, and an `install` is additionally refused `server_busy`
   * while anything holds the card, because it ends by reloading this server's
   * job registry. Neither is retried here: queues belong to clients
   * (ARCHITECTURE.md R5), and this client's contribution to backing off is
   * telling the caller exactly what is in the way.
   *
   * A `pull` of something already installed is REFUSED (`already_installed`)
   * rather than skipped; the same subject inside a `module` is SKIPPED. That
   * asymmetry is deliberate and is written into section 3.3: a single pull is
   * a person asking for one specific thing, and a module is an app saying what
   * must be true.
   */
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
    return {
      taskId: str(body, 'task_id', 'task'),
      type: str(body, 'type', 'task'),
      request: asObject(field(body, 'request', 'task'), 'task.request'),
      state: oneOf(str(body, 'state', 'task'), TASK_STATES, 'task.state'),
      error: readFailureOrNull(field(body, 'error', 'task'), 'task.error'),
      created: str(body, 'created', 'task'),
      started: str(body, 'started', 'task'),
      finished: nullableStr(body, 'finished', 'task'),
      unmet: readUnmet(body, 'task'),
    };
  }

  /**
   * `GET /v1/tasks/{id}` for the last few tasks this server remembers, newest
   * first. In memory, capped at fifty; a restart forgets them.
   */
  async tasks(): Promise<TaskStatus[]> {
    const body = await this.#json('/v1/tasks', { method: 'GET' }, 'tasks');
    const rows = asArray(field(body, 'tasks', 'tasks'), 'tasks.tasks');
    return rows.map((row, index) =>
      readTaskStatus(asObject(row, `tasks[${index}]`), `tasks[${index}]`),
    );
  }

  /**
   * `GET /v1/tasks/{id}/events` — the task's SSE stream, as typed events.
   *
   * {@link CrucibleClient.events}' contract exactly: the iterator ends after
   * the first terminal event, a stream that closes without one throws
   * {@link CrucibleUnreachable} rather than ending quietly, and `lastEventId`
   * resumes without a gap.
   */
  async *taskEvents(
    taskId: string,
    options: EventsOptions = {},
  ): AsyncGenerator<TaskEvent, void, undefined> {
    const id = requireText(taskId, 'taskId');
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
      `/v1/tasks/${encodeURIComponent(id)}/events`,
      { method: 'GET', headers },
      true,
    );
    if (!response.ok) throw await this.#failure(response);
    const stream = response.body;
    if (stream === null) {
      throw new CrucibleProtocolError(`the event stream for task ${id} carried no body`);
    }

    let previousId = options.lastEventId === undefined ? 0 : options.lastEventId;
    try {
      for await (const frame of readSseFrames(stream)) {
        const event = readTaskEvent(frame.lastEventId, frame.event, frame.data);
        if (event.id <= previousId) {
          throw new CrucibleProtocolError(
            `event id ${event.id} does not follow ${previousId} on task ${id}`,
          );
        }
        previousId = event.id;
        yield event;
        if ((TASK_TERMINAL_STATES as readonly string[]).includes(event.event)) return;
      }
    } finally {
      await stream.cancel().catch(() => undefined);
    }

    throw new CrucibleUnreachable(
      this.url,
      `the event stream for task ${id} ended after event ${previousId} without a ` +
        'terminal event (done, failed or cancelled)',
    );
  }

  /**
   * `DELETE /v1/tasks/{id}` — cancel. Answers `cancelling`, never `cancelled`.
   *
   * A pull stops at its next chunk and its partial directory is removed; an
   * install is SIGTERMed and its half-built env is left for a `--force`
   * rebuild. Watch {@link taskEvents} for the `cancelled` event: a caller told
   * "cancelled" before the download thread had stopped would be told a
   * "maybe" (ARCHITECTURE.md R3).
   */
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

// ------------------------------------------------------------------ readers

/**
 * `role`, `managed_by` and `engine` out of an `/v1/info` document.
 * PHASE17-ORCHESTRATOR.md 3.3.
 *
 * ALL-OR-NOTHING BY VINTAGE, which is PHASE15 3.3's rule applied a second
 * time and for the reason it was written the first time. A document with NO
 * `role` comes from a server that predates Phase 17, and such a server IS an
 * engine that nobody manages: a fact the document states by what it is, not a
 * default this client invents. A document that HAS `role` and is then missing
 * the field its role owes is a defect and is refused by name, because a
 * half-new document is the one thing a vintage rule cannot read.
 */
function readRole(body: Json): Pick<ServerInfo, 'role' | 'managedBy' | 'engine'> {
  if (!('role' in body)) {
    return { role: 'engine', managedBy: null, engine: null };
  }
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
            // NOT `oneOf`. The owner set can grow, and a client that threw a
            // protocol error on a word it had not heard of would break on the
            // server that added one — `Health.residentKind`'s rule, for
            // `Health.residentKind`'s reason.
            owner: str(engine, 'owner', 'info.engine') as EngineOwner,
          },
  };
}

/**
 * `pages_engine` out of an `/v1/info` document. PHASE15-HOST.md 3.10 fact 7.
 *
 * VINTAGE FIRST, EVERYTHING AFTER IT STRICT — `readRole`'s rule, for
 * `readRole`'s reason. A document with NO `pages_engine` comes from a server
 * that predates the block, and `null` is that fact; a document that HAS the
 * block and is then missing a field of it is a defect and is refused by name,
 * because a half-new document is the one thing a vintage rule cannot read.
 *
 * NOTHING INSIDE IS FILLED IN. The prompt, the dpi, the pixel budget and the
 * ceiling are facts about the weights and this client owns none of them: the
 * reason the block is on the wire is that clients were pinning their own
 * copies, and an SDK that substituted one when the server went quiet would be
 * the third owner all over again.
 *
 * The field names are the Python's, verbatim — `crucible/pages.py`'s
 * `request_shape()` and `engine_block()`, and `tests/test_pages_request.py` is what
 * holds the two spellings to each other.
 */
function readPagesEngine(body: Json): PagesEngine | null {
  if (!('pages_engine' in body)) {
    return null;
  }
  const block = objectField(body, 'pages_engine', 'info');
  const request = objectField(block, 'request', 'info.pages_engine');
  return {
    // NOT `oneOf`. The engine set grows — `vllm`, `llama-server`, `mlx-vlm`
    // and whatever reads a page next — and this field is for an operator to
    // look at, never for a client to branch on.
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

/**
 * Where to send work, given an `info()`. PHASE17-ORCHESTRATOR.md section 6.
 *
 * Three answers and no fourth:
 *
 * * `null` — **this IS the engine; talk to the address you already have.**
 *   The overwhelmingly common case, and every pre-Phase-17 server.
 * * an {@link EngineRef} — this is an orchestrator; **follow `engine.url`
 *   ONCE, with the SAME token**, and talk to the engine for everything after.
 * * it throws `orchestrator_has_no_engine` — this machine's orchestrator
 *   manages nothing, so there is nothing here to ask. That is a fact to show
 *   a person, next to the button that installs one; it is not a fault.
 *
 * ONCE, AND NEVER A CHAIN. An app follows one hop and no more. An
 * orchestrator whose `engine.url` named another orchestrator would be a
 * misconfiguration, and a client that followed it would loop; the caller
 * checks the second document's `role` and refuses anything but `engine`
 * rather than following it again.
 *
 * @example
 * const here = await client.info();
 * const engine = engineOf(here);
 * const work = engine === null ? client : new CrucibleClient({ url: engine.url, token });
 */
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

/**
 * One capability from `info()`.
 *
 * Two capabilities carry the rows of the route that lists them rather than
 * DESIGN.md section 4's descriptor, because the contract says the same shape
 * from the same producer: `llm`'s rows are `GET /v1/models`' (PHASE2-LLM.md
 * section 5) and `tts`'s are `GET /v1/voices`' (PHASE3-TTS.md section 8). So
 * each is read with that route's reader. Every other capability keeps the
 * descriptor.
 *
 * `documentStatesNeedsReference` is the whole info document's vintage, asked
 * once by {@link CrucibleClient.info} and passed down rather than re-derived
 * per capability — see {@link readVoiceInfo}.
 */
function readCapability(
  entry: Json,
  index: number,
  documentStatesNeedsReference: boolean,
): Capability {
  const where = `info.capabilities[${index}]`;
  const jobType = str(entry, 'job_type', where);
  const models = asArray(field(entry, 'models', where), `${where}.models`);
  if (jobType === 'llm') {
    return {
      jobType,
      models: models.map((model, at) =>
        readModelInfo(asObject(model, `${where}.models[${at}]`), `${where}.models[${at}]`),
      ),
    };
  }
  if (jobType === 'tts') {
    return {
      jobType,
      models: models.map((voice, at) =>
        readVoiceInfo(
          asObject(voice, `${where}.models[${at}]`),
          `${where}.models[${at}]`,
          documentStatesNeedsReference,
        ),
      ),
    };
  }
  // Every other capability: try the descriptor shape, and carry the rows raw if
  // they do not fit rather than losing the whole `info()` call. `llm` and `tts`
  // above are the two shapes this client claims and stays strict about; this is
  // the one it makes no claim about.
  //
  // Not hypothetical. A v0.2.0 client read EVERY capability with the descriptor,
  // so the day `tts` shipped — whose rows are voices — `info()` began throwing
  // `models[0] has no field "source"` against any server with tts enabled, and
  // the client could no longer ask what it was talking to. Measured against a
  // real server on 2026-09-13. The next capability to grow richer rows would do
  // it again, to this build, which is why the fix is general rather than a third
  // `if`.
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

/**
 * `GET /v1/capability`'s record — `CapabilityRecord.to_dict()` in
 * `crucible/config.py`. The rows arrive under `classes`, which is the wire's
 * name for them and stays the member's: they ARE the capability classes.
 */
function readCapabilityRecord(body: Json): CapabilityRecord {
  const where = 'capability';
  const classes = asArray(field(body, 'classes', where), `${where}.classes`);
  const rows = classes.map((entry, index) =>
    asObject(entry, `${where}.classes[${index}]`),
  );
  // THE VINTAGE IS READ ONCE, FOR THE WHOLE DOCUMENT, and that is the whole
  // content of PHASE15-HOST.md 3.3's last bullet: a document where NO row
  // carries `route` comes from a server that predates phase 15, and every
  // class on such a server IS local. That is a fact the document states by
  // being what it is, not a value this client picks when one is missing —
  // which is why the question is asked of the document and never of a row.
  const anyRoute = rows.some((row) => row['route'] !== undefined);
  return {
    backendKind: str(body, 'backend_kind', where),
    totalBytes: num(body, 'total_bytes', where),
    desktopAllowanceBytes: num(body, 'desktop_allowance_bytes', where),
    classes: rows.map((row, index) =>
      readCapabilityRow(row, `${where}.classes[${index}]`, anyRoute),
    ),
  };
}

function readCapabilityRow(
  entry: Json,
  where: string,
  documentHasRoutes: boolean,
): CapabilityRow {
  const raw = entry['route'];
  let route: 'local' | 'upstream';
  if (raw === undefined) {
    /*
     * A MISSING `route` IS A REFUSAL NOW, WHOLE DOCUMENT OR HALF.
     *
     * The half-routed case always refused: some rows have it and this one does
     * not, so the document cannot say where this class runs, and reading it as
     * `local` would invent the one thing a routed server is about.
     *
     * The WHOLE-document case used to answer `local` instead, on the reading
     * that no row carrying it means a server predating phase 15, where every
     * class really was local. That was Crucible-version tolerance, and Owen
     * ended the population it served on 2026-09-16: nothing is released, so
     * there is no pre-phase-15 server for anyone but us to point at. It was a
     * SHIM by the test Foundry proposed the same evening — it made a wrong
     * version work and said nothing — and the rule is that shims go while named
     * refusals stay.
     *
     * So the one sentence covers both, and neither invents a route. What made
     * the old arm dangerous is worth keeping in view: `local` and `upstream`
     * decide whether a run costs GPU-minutes or money, and that is not a fact
     * a client may fill in.
     */
    throw new CrucibleProtocolError(
      `${CAPABILITY_ROUTE_MISSING}: ${where} has no "route"` +
        (documentHasRoutes
          ? ', but other rows in the same capability document do'
          : ', and neither does any other row in the document') +
        '. Every server states it on every row; where a class runs decides ' +
        'whether its work costs GPU-minutes or money, so this is not something ' +
        `a client may fill in for ${str(entry, 'capability', where)}.`,
    );
  } else if (typeof raw !== 'string' || !ROUTES.includes(raw as 'local')) {
    throw new CrucibleProtocolError(
      `${CAPABILITY_ROUTE_UNKNOWN}: ${where}.route is ${JSON.stringify(raw)}, ` +
        `which is not one of ${ROUTES.join(', ')}`,
    );
  } else {
    route = raw as 'local' | 'upstream';
  }
  return {
    capability: str(entry, 'capability', where),
    enabled: bool(entry, 'enabled', where),
    selected: str(entry, 'selected', where),
    reason: str(entry, 'reason', where),
    shortfallBytes: num(entry, 'shortfall_bytes', where),
    route,
  };
}

// ---------------------------------------------------------------- settings

const ROUTES = ['local', 'upstream'] as const;
const UPSTREAM_NAMES = ['anthropic', 'openai', 'ollama'] as const;

/**
 * `GET /v1/settings`, read whole. Every field the contract promises is
 * REQUIRED here: a document missing one is a protocol error rather than an
 * `undefined` handed to a settings page, which is this client's rule
 * everywhere and matters most on the page that writes a key.
 */
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
    // The two shapes differ by ONE key, and which one is present is decided
    // by the upstream rather than by this client: `ollama` is reached by
    // address and has no secret, the other two are the reverse. Read what is
    // there rather than demanding both, so neither card is drawn with a field
    // its provider does not have.
    const setting: {
      configured: boolean;
      keyHint?: string | null;
      url?: string | null;
    } = { configured: bool(entry, 'configured', at) };
    if ('key_hint' in entry) setting.keyHint = nullableStr(entry, 'key_hint', at);
    if ('url' in entry) setting.url = nullableStr(entry, 'url', at);
    upstreams[name] = setting;
  }
  /*
   * BOTH KEYS ARE REQUIRED, and they were optional until 2026-09-16.
   *
   * They were optional to let this SDK read an engine older than the fields —
   * absent meaning "this engine does not answer the question", which is a
   * different claim from "it answers with nothing". Owen ended that:
   * *"I won't be releasing any of this until it's completely done, so we don't
   * need to worry about legacy functionality at all right now. Nothing is
   * legacy because nothing exists publicly. There will be no person trying to
   * access the system with an older version of crucible other than us."*
   *
   * So the population the optionality served has exactly zero members. Reading
   * them as required means an engine that does not send them fails HERE, by
   * name and with the field path, instead of every caller carrying a branch for
   * a vintage that will never arrive.
   */
  const localModels = Object.fromEntries(
    Object.keys(objectField(body, 'local_models', where)).map(name =>
      [name, nullableStr(objectField(body, 'local_models', where), name, `${where}.local_models`)]));
  const choiceRows = objectField(body, 'local_model_choices', where);
  const localModelChoices = Object.fromEntries(Object.keys(choiceRows).map(name => [name,
    asArray(field(choiceRows, name, `${where}.local_model_choices`), `${where}.local_model_choices.${name}`).map((raw, index) => {
      const at = `${where}.local_model_choices.${name}[${index}]`;
      const choice = asObject(raw, at);
      return { id: str(choice, 'id', at), memoryBytesEstimate: num(choice, 'memory_bytes_estimate', at),
        fits: bool(choice, 'fits', at), installed: bool(choice, 'installed', at) };
    }),
  ]));
  return {
    localModels,
    localModelChoices,
    routes,
    upstreams,
    desktopAllowanceBytes: num(body, 'desktop_allowance_bytes', where),
    backendKind: str(body, 'backend_kind', where),
  };
}

/**
 * A patch, in the wire's spelling. Only what the caller stated travels: a
 * patch is partial, and a key written here as `undefined` would be sent as an
 * absent field anyway — but sending `{"routes": undefined}` through
 * `JSON.stringify` and sending nothing are the same bytes only by accident,
 * so the omission is deliberate rather than relied upon.
 */
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

/**
 * Is this error one of `testUpstream`'s three RESULTS, or a real failure?
 *
 * **Narrowed on the CODE, across every error type, and never on the status.**
 * `upstream_rejected` arrives as a 401 from this door, which `#failure` maps
 * to `CrucibleAuthError` like every other 401 — correctly, since it cannot
 * know whose credential was refused. Keying on the status here would have
 * turned an upstream's bad API key into "your Crucible token is wrong", shown
 * on a page whose Crucible token is demonstrably fine. Keying on the class
 * would have done the same thing one layer up.
 *
 * A real auth failure of THIS server still throws: its code is `unauthorized`,
 * which is not in the list, and a test pins that.
 */
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
    model: model === null ? null : readProvenanceModel(asObject(model, `${where}.model`), where),
    params: asObject(field(entry, 'params', where), `${where}.params`),
    started: nullableStr(entry, 'started', where),
    finished: str(entry, 'finished', where),
  };
}

/**
 * The `model` block of a provenance sidecar. `revision` and `fingerprint` are
 * nullable together: a model served on a backend whose block the manifest does
 * not carry has neither, and a fingerprint without a pin would read as one.
 */
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

  // A kind this build does not know is CARRIED, not refused. The server's event
  // vocabulary grows without moving `api_version` — `chunk` arrived for `tts` on
  // the stated ground that a client which does not know it still sees every
  // `progress`, `artifact` and `done` it saw before — and that argument only
  // holds if the client survives the frame. It did not: this narrowed against a
  // closed list and threw, so an 0.2.0 client watching ANY job on a newer server
  // lost the whole stream at the first `chunk`.
  //
  // Strict about what it claims to understand, tolerant of what it makes no
  // claim about. A `chunk` missing `capped` is still a protocol error below;
  // this is the other case entirely.
  if (!(EVENT_NAMES as readonly string[]).includes(name)) {
    return { id, event: 'unknown', kind: name, data };
  }
  // Narrowed by the check above, and narrowed rather than left a string so the
  // switch below stays exhaustive: adding a kind to EVENT_NAMES without giving
  // it a case is then a compile error, which is the half of the old strictness
  // worth keeping.
  const known = name as (typeof EVENT_NAMES)[number];

  switch (known) {
    case 'queued':
      return { id, event: 'queued', data: { position: nullableNum(data, 'position', where) } };
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

/**
 * A `progress` frame: the two fields API v1 pins, plus everything else the job
 * type put on it, carried rather than dropped.
 *
 * `JobContext.progress(fraction, message, **extra)` is open on purpose — `asr`
 * sends `{stage, processed_s, total_s, cues}` so a client can show a moving
 * position while the percentage still rounds to zero (PHASE4-AUDIO.md section
 * 3). Those keys are that job type's vocabulary rather than the API's, so they
 * travel verbatim in `extra` instead of being modelled here, where a second job
 * type's measurements would collide with them.
 */
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

/**
 * One `chunk` frame (PHASE3-TTS.md section 6, amended by
 * PHASE6-REMOTE-RENDER.md section 3): the server's measurements of one rendered
 * chunk, and the engine's verdict about it.
 *
 * `tokens`, `capped` and `guard` are read with the nullable readers and **the
 * key must be there**. That is the load-bearing part: `null` on this wire means
 * "narrator did not say", and an absent key would mean "this server does not
 * speak the field at all" — two different pieces of news, and only one of them
 * is something the server stated. A `chunk` frame that omits `capped` is
 * therefore a protocol error rather than a null, and the null that does arrive
 * travels to the caller as a null, never softened into `false`. A client that
 * read it as `false` would report every runaway as a long sentence.
 *
 * `guard` follows that precedent exactly rather than inventing a second one, and
 * adds nothing to it: {@link nullableObject} checks that it is an object and
 * reads nothing inside. The verdict's vocabulary is the retake ladder's, it is
 * free to grow, and a reader here that knew the words would be a second owner of
 * them — which is the one defect `docs/ARCHITECTURE.md` section 1 says every
 * other defect in this system turned out to be.
 */
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

/**
 * A `done` frame says what finished, and what that means depends on the job:
 * artifacts for a producing job, the resident model for `load-model`. One of
 * the two must be there — a `done` that says nothing is a protocol error, not
 * an empty result.
 *
 * Everything else on the frame is carried in `extra`, verbatim, for the reason
 * {@link readProgress} carries a progress frame's own measurements: the server
 * builds `done` as `{"artifacts": [...], **job.done_extra}`, and `done_extra` is
 * a job type's own terminal news. Reading only the two modelled keys threw away
 * `tts`'s `failed` list — the authoritative answer to "which indices do I have to
 * ask for again" — and `load-voice`'s `fingerprint`.
 */
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
  // `null` is the answer an unload gives: nothing is resident now.
  if (hasResident) done.resident = nullableStr(data, 'resident', where);
  return done;
}

/**
 * A finished `tts` job's terminal news, read strictly out of its `done` frame.
 *
 * This exists because a successful render can still have failed chunks. **A
 * failed chunk is reported and the run continues** (PHASE3-TTS.md section 6) —
 * one bad sentence never sinks the other 1,399, and a missing `<index>.flac` is
 * a file that is not there, which BookForge's resume already knows how to ask
 * for again. The `progress` line at the moment of each failure says the same
 * thing, so a client watching live already knows; this is for the one reading
 * only the terminal event, and it is the authoritative list.
 *
 * Nothing here is optional and nothing is defaulted. A `done` frame from a `tts`
 * job carries all four keys — `crucible/jobs/tts/render.py` writes them
 * unconditionally — so a missing one is a server that changed, and a `failed: []`
 * substituted for an absent key would read as a clean render.
 *
 * Pass it the `done` event of a `tts` job. Handing it any other job type's
 * `done` throws, naming the field that is not there, which is the correct answer
 * to asking a `load-model` how many chunks it rendered.
 */
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
  return {
    rendered: num(extra, 'rendered', where),
    failed: failed.map((entry, index) =>
      readRenderFailure(asObject(entry, `${where}.failed[${index}]`), `${where}.failed[${index}]`),
    ),
    take: num(extra, 'take', where),
    // THE TRIPLE THE ENGINE APPLIED, and the weights it applied them to. Read
    // strictly — a result that omits either is a protocol error and not a
    // null — because the whole point of pinning them is that a record which
    // cannot say what it ran at is comparable to nothing.
    sampling: readSampling(asObject(field(extra, 'sampling', where), `${where}.sampling`),
      `${where}.sampling`),
    voice: readRenderVoice(
      asObject(field(extra, 'voice', where), `${where}.voice`), `${where}.voice`),
    // `null` only for a voice whose manifest declares no serving table.
    width: nullableNum(extra, 'width', where),
    // The rate the voice was loaded at, which the load already reconciled
    // against the manifest — so it is both the engine's truth and the
    // manifest's, and the FLAC headers on disk say the same thing.
    sampleRate: num(extra, 'sample_rate', where),
    artifacts,
  };
}

/**
 * The sampling triple off a `done`, unopened beyond "every value is a number".
 *
 * NOT a fixed set of three keys. The levers a voice may state are the engine's
 * — `temperature`, `top_p`, `top_k`, and `repetition_penalty` on the arm that
 * has one — and a reader that named them would refuse a result the server
 * produced perfectly well the day a fifth arrives. This is `guard`'s discipline
 * one field over: carry it, do not interpret it.
 */
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
    // `verified` for a pin — the sha is what was fetched — `asserted` for a
    // directory somebody pointed at. Carried as the server's own word rather
    // than narrowed to a union here, for `readSampling`'s reason.
    identityBasis: str(entry, 'identity_basis', where),
  };
}

function readRenderFailure(entry: Json, where: string): RenderFailure {
  return {
    index: num(entry, 'index', where),
    // narrator's own words for why. Surfaced, never summarised: 'No audio
    // generated' and 'cancelled' call for different responses from the caller.
    message: str(entry, 'message', where),
  };
}

function readModelInfo(entry: Json, where: string): ModelInfo {
  const loadable = bool(entry, 'loadable', where);
  const common = {
    id: str(entry, 'id', where),
    family: str(entry, 'family', where),
    paramsB: num(entry, 'params_b', where),
    // Null on a model this backend cannot serve; a string everywhere else.
    revision: nullableStr(entry, 'revision', where),
    // `<id>@<revision>`, assembled by the server so that every client records
    // one spelling of it. Null exactly where `revision` is.
    fingerprint: nullableStr(entry, 'fingerprint', where),
    // Never null, on any host: unlike `revision` this is not a per-host fact but
    // a statement of what the model is offered FOR, and it is the same answer on
    // a host whose backend cannot serve it (PHASE3-VLM.md section 2).
    modalities: strArray(entry, 'modalities', where),
    backendSupported: bool(entry, 'backend_supported', where),
    installed: bool(entry, 'installed', where),
    resident: bool(entry, 'resident', where),
    loadable,
    // Null on a model this backend cannot serve, exactly like `revision`: both
    // figures live in the backend block this manifest does not have.
    memoryBytesEstimate: nullableNum(entry, 'memory_bytes_estimate', where),
    contextDefault: num(entry, 'context_default', where),
    // What is being served right now, which is the number to size a request
    // against. Null on a model this backend cannot serve, like `revision`.
    maxModelLen: nullableNum(entry, 'max_model_len', where),
  };
  if (!loadable) {
    // A refusal with no reason is unusable: the operator cannot tell whether to
    // pull weights, free the card, or go to the other host.
    return { ...common, reason: str(entry, 'reason', where) };
  }
  // `reason` is optional only in this direction, and null reads as absent.
  const reason = 'reason' in entry ? entry['reason'] : null;
  if (reason === null) return common;
  if (typeof reason !== 'string') {
    throw new CrucibleProtocolError(
      `${where}.reason is neither a string nor null on a loadable model`,
    );
  }
  return { ...common, reason };
}

/**
 * Does this document STATE a field, anywhere in it?
 *
 * The one question PHASE15-HOST.md section 3.3's reading rule asks, and it is
 * asked of the DOCUMENT: an absent field is a statement about the server's
 * vintage only when no row in the whole document carries it. Rows that are not
 * objects are left to the reader that will refuse them by name.
 */
function anyRowStates(rows: readonly unknown[], key: string): boolean {
  return rows.some(
    (row) =>
      typeof row === 'object' &&
      row !== null &&
      !Array.isArray(row) &&
      (row as Json)[key] !== undefined,
  );
}

/**
 * One `/v1/voices` row.
 *
 * The nullability differs from {@link readModelInfo} in exactly one place and it
 * is deliberate on the server's side: a voice row always carries `reason`, null
 * when the voice is loadable, where a model row omits the key. So this reads it
 * as a nullable field and keeps the rule that matters — a voice that cannot be
 * loaded and does not say why is unusable, because the operator cannot tell
 * whether to pull weights, install an env, free the card, or go to the other
 * host.
 *
 * **`needs_reference` is read the way `route` is** — PHASE15-HOST.md section
 * 3.3's client reading rule, and `documentStatesNeedsReference` is the answer
 * the caller already got from the WHOLE document. A document in which no voice
 * row carries the field comes from a server built before PHASE3-TTS.md section
 * 2 added it (before commit 743dc1a), and on such a server every voice IS a
 * checkpoint whose voice is in its weights: `needsReference` reads `false`
 * because the document's vintage says so, not because this client picked a
 * default. Some rows carrying it and one not is a defect
 * (`voices_needs_reference_missing`), and so is a value that is not a boolean
 * (`voices_needs_reference_unknown`).
 *
 * Not hypothetical: Foundry pointed 0.6.0 at a server one commit older and
 * BOTH its first reads threw — `info.capabilities[6].models[0] has no field
 * "needs_reference"` and the same from `voices()`. Measured 2026-09-14.
 */
function readVoiceInfo(
  entry: Json,
  where: string,
  documentStatesNeedsReference: boolean,
): VoiceInfo {
  const rawNeedsReference = entry['needs_reference'];
  let needsReference: boolean;
  if (rawNeedsReference === undefined) {
    if (documentStatesNeedsReference) {
      // Some rows say it and this one does not, so the document cannot say
      // whether a load of THIS voice must carry a clip. Reading it as `false`
      // would invent the answer to the one question the field exists for: a
      // zero-shot voice loaded with no clip comes up in the model's own voice
      // under this id.
      throw new CrucibleProtocolError(
        `${VOICES_NEEDS_REFERENCE_MISSING}: ${where} has no "needs_reference", but ` +
          'other voice rows in the same document do. A document either predates ' +
          'the field entirely (no row has it, and every voice is a checkpoint) or ' +
          'states it on every row; a half-stated document says nothing trustworthy ' +
          `about ${str(entry, 'id', where)}.`,
      );
    }
    needsReference = false;
  } else if (typeof rawNeedsReference !== 'boolean') {
    throw new CrucibleProtocolError(
      `${VOICES_NEEDS_REFERENCE_UNKNOWN}: ${where}.needs_reference is ` +
        `${JSON.stringify(rawNeedsReference)}, which is not a boolean`,
    );
  } else {
    needsReference = rawNeedsReference;
  }
  const loadable = bool(entry, 'loadable', where);
  const reason = nullableStr(entry, 'reason', where);
  if (!loadable && reason === null) {
    throw new CrucibleProtocolError(
      `${where} is not loadable and its "reason" is null; a refusal with no ` +
        'reason does not say whether to pull weights, install an env or free the card',
    );
  }
  const basis = nullableStr(entry, 'estimate_basis', where);
  return {
    id: str(entry, 'id', where),
    display: str(entry, 'display', where),
    kind: oneOf(str(entry, 'kind', where), VOICE_KINDS, `${where}.kind`),
    language: str(entry, 'language', where),
    narratorEngine: str(entry, 'narrator_engine', where),
    backendSupported: bool(entry, 'backend_supported', where),
    installed: bool(entry, 'installed', where),
    resident: bool(entry, 'resident', where),
    loadable,
    reason,
    // These five live in the backend block this host may not have, and are null
    // together when `backend_supported` is false — never 0, which would read as
    // "needs nothing", and never "", which would read as a pin.
    revision: nullableStr(entry, 'revision', where),
    fingerprint: nullableStr(entry, 'fingerprint', where),
    memoryBytesEstimate: nullableNum(entry, 'memory_bytes_estimate', where),
    estimateBasis:
      basis === null ? null : oneOf(basis, ESTIMATE_BASES, `${where}.estimate_basis`),
    maxChars: nullableNum(entry, 'max_chars', where),
    // These three are facts about the voice rather than about this host, and are
    // never null: a client writing FLACs cannot be handed a null sample rate,
    // and a client that packs cannot be handed half a pace block.
    sampleRate: num(entry, 'sample_rate', where),
    takes: num(entry, 'takes', where),
    // `null` for a voice with no serving table; read strictly otherwise, for
    // `pace`'s reason — a client picking a render width against half a
    // serving block would pick against a ceiling nobody stated.
    serving: readVoiceServing(entry, where),
    needsReference,
    pace: readVoicePace(objectField(entry, 'pace', where), `${where}.pace`),
  };
}

/**
 * `[voice.serving]` off the row, or null for a voice that declares none.
 *
 * The two optional levers are read with the nullable readers and **the null is
 * kept**: `null` means "this voice states none, so narrator's own launcher
 * default applies", which is a different fact from any number, exactly as
 * `capped: null` is a different fact from `false`.
 */
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

/**
 * A voice's pace block. The three rates are ALL THREE OR NONE — a voice with no
 * measured pace states none, and narrator derives the centre. The three that
 * describe the packing shape are nullable each on their own, and *which* of
 * them are null is how a client tells a band from a target from neither.
 *
 * They were required until 2026-09-18, and what that cost is why the group rule
 * is worth a paragraph: `higgs-default` and `zeroshot` are the base weights and
 * no ladder has been run on either, so both manifests satisfied the requirement
 * by copying narrator's own Higgs v3 constants back to it — a pace of 15.0 that
 * is the frame cap's divisor rather than a measured speaking rate, inside edges
 * written around a real book pace nearer 17.2. narrator re-centres a band's
 * RATIOS on the running median, so healthy chunks fell under the short edge and
 * went to the bottom of the retake ladder. A manifest now states what was
 * measured or states nothing, and NOTHING HERE FILLS IN THE NOTHING.
 *
 * A HALF-STATED TRIPLE IS REFUSED, which is the one invariant this reader does
 * check — because it is not the manifest's schema, it is the wire disagreeing
 * with itself. `min < pace < max` and "never both a band and a target" stay the
 * loader's to enforce: this client reads the wire and does not keep a second
 * copy of the server's rules to disagree with it.
 */
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

/** `GET /v1/accelerator`, read strictly — see {@link CrucibleClient.accelerator}. */
function readAcceleratorState(body: Json): AcceleratorState {
  const where = 'accelerator';
  const gpu = objectField(body, 'gpu', where);
  const resident = field(body, 'resident', where);
  const holders = asArray(field(body, 'holders', where), 'accelerator.holders');
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
    // Null on mlx-darwin, where the question cannot be asked. Not defaulted to
    // zero: "nobody unaccounted for" and "unanswerable" are different answers.
    unattributedBytes: nullableNum(body, 'unattributed_bytes', where),
    resident:
      resident === null
        ? null
        : readAcceleratorResident(asObject(resident, 'accelerator.resident')),
    holders: holders.map((holder, index) =>
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
    // A plain string, not narrowed: the kinds grow with the job types, and this
    // client must not be the thing that breaks when one is added.
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
    // `null` is the driver refusing to say, and it stays null all the way to the
    // caller. Substituting 0 here would turn "I do not know what this process
    // holds" into "this process holds nothing", which is how a queue decides a
    // busy card is free.
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
  const usage = objectField(body, 'usage', where);
  const finishReason = str(choice, 'finish_reason', 'chat.choices[0]');
  refuseReasoningWithoutContent(message, finishReason);
  return {
    id: str(body, 'id', where),
    model: str(body, 'model', where),
    content: str(message, 'content', 'chat.choices[0].message'),
    finishReason,
    usage: {
      promptTokens: num(usage, 'prompt_tokens', 'chat.usage'),
      completionTokens: num(usage, 'completion_tokens', 'chat.usage'),
      totalTokens: num(usage, 'total_tokens', 'chat.usage'),
    },
  };
}

/**
 * A reasoning model that runs out of budget mid-thought answers with
 * `reasoning` and no `content` at all.
 *
 * That is still a protocol error — this client promises a completion carries
 * text, and an answer that is not there must never read as an empty one — but
 * the *cause* belongs in the message, because the remedy is the caller's: raise
 * `maxTokens`, or pass `thinking: false`. Nothing is substituted; the strict
 * `content` rule below still runs for every other shape.
 */
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

/**
 * One `chat.completion.chunk`, reduced to the content it carries, or `null` for
 * a chunk that carries none: the opening frame holds only `role`, the closing
 * one only a `finish_reason`, and a usage-only frame no choice at all. Those
 * are real states of the stream, not missing fields — but a chunk whose
 * `content` is present and is not a string is a protocol error.
 */
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

// ------------------------------------------------------- the batch writer's fs

/**
 * The four filesystem calls and the one path call the batch writer makes.
 *
 * Declared here rather than imported from `node:fs/promises`, and that is the
 * whole trick — see {@link loadNodeFileApis} for why the module cannot be named
 * in an import statement, and what that costs.
 */
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

let nodeFileApis: Promise<NodeFileApis> | null = null;

/**
 * `node:fs/promises` and `node:path`, loaded the first time the batch writer
 * needs them and never before.
 *
 * **The SDK's one hard rule since phase 1 is zero runtime dependencies**, so
 * that Node 20, bun and the Electron main process can all import it. Node's own
 * builtins are not a dependency in that sense — nothing is installed to get them
 * — but a *static* `import ... from 'node:fs/promises'` at the top of this file
 * would still break the rule in practice, because it puts fs into the module
 * graph of `import {CrucibleClient} from '@crucible/client'` itself. A bundler
 * targeting a browser-ish runtime resolves that specifier at build time and
 * fails on it, and every caller pays for a method most of them never call.
 *
 * So the specifier is assembled at run time, out of reach of static analysis,
 * and the import happens inside the one method that writes files. `ping`,
 * `info`, `chat`, `render`, `events` — none of them loads fs at all, and a
 * browser bundle that never calls {@link CrucibleClient.writeArtifactsTo} never
 * resolves it. (webpack will warn about an expression as a dependency; that
 * warning is the mechanism working.)
 *
 * The cost of hiding the specifier is that `import()` hands back `any`, so
 * {@link NodeFileApis} declares the shapes and they are checked here, at the
 * seam, rather than trusted. A runtime with no `node:fs/promises` — a browser —
 * gets a {@link CrucibleError} that says what it is missing and why, not a
 * `TypeError` about `undefined`.
 */
async function loadNodeFileApis(): Promise<NodeFileApis> {
  if (nodeFileApis === null) {
    nodeFileApis = (async () => {
      // Built from parts so that no bundler can see a literal module specifier
      // here. This is the point of the function; do not inline it.
      const scheme = 'node:';
      let fs: unknown;
      let path: unknown;
      try {
        fs = await import(/* webpackIgnore: true */ `${scheme}fs/promises`);
        path = await import(/* webpackIgnore: true */ `${scheme}path`);
      } catch (cause) {
        throw new CrucibleError(
          "writeArtifactsTo needs node:fs/promises and node:path, and this " +
            'runtime has neither. It writes the artifacts to disk itself because ' +
            'there is no shared mount between a Crucible host and its client; in ' +
            'a browser, fetch each artifact with artifact(jobId, name) and put ' +
            'the bytes wherever that runtime keeps bytes.',
          { cause },
        );
      }
      return { fs: checkedFs(fs), path: checkedPath(path) };
    })();
  }
  return nodeFileApis;
}

/** Every call the writer makes, proven to exist before the writer makes it. */
function checkedFs(module: unknown): NodeFileApis['fs'] {
  const found = module as Record<string, unknown> | null;
  for (const name of ['mkdir', 'writeFile', 'rename', 'rm']) {
    if (found === null || typeof found[name] !== 'function') {
      throw new CrucibleError(
        `node:fs/promises on this runtime has no ${name}(); writeArtifactsTo ` +
          'cannot write files atomically without it',
      );
    }
  }
  return found as unknown as NodeFileApis['fs'];
}

function checkedPath(module: unknown): NodeFileApis['path'] {
  const found = module as Record<string, unknown> | null;
  if (found === null || typeof found['join'] !== 'function') {
    throw new CrucibleError('node:path on this runtime has no join()');
  }
  return found as unknown as NodeFileApis['path'];
}

/** Distinguishes two temporaries in one directory. Not a secret; no crypto needed. */
let temporaryCounter = 0;

/**
 * Bytes to `target`, via a temporary in the **same directory**, then a rename.
 *
 * BookForge's resume test is "the file exists and exceeds 1024 bytes". A FLAC
 * written in place and interrupted — a killed run, a full disk, a pulled plug —
 * passes that test while being half a sentence, so resume never asks for it
 * again and the book is quietly missing audio. A rename within one directory is
 * atomic on NTFS and on ext4, so `<index>.flac` either does not exist or is
 * whole. The temporary must be a sibling: a rename across filesystems is a copy,
 * and a copy is the thing being avoided.
 *
 * The `.part` suffix keeps a temporary from ever matching `<index>.flac`, so a
 * crash cannot leave something resume would count.
 *
 * Node's `rename` replaces an existing destination on Windows as well as on
 * POSIX (it passes `MOVEFILE_REPLACE_EXISTING`), so re-writing an artifact — a
 * retake, a resumed run — does not need an unlink first, which would be a window
 * where neither file exists.
 */
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
    // Cleanup, so a failed write does not leave a temporary behind. A failure to
    // clean up does not change what went wrong and must not replace it — the
    // same rule `events()` applies to closing its socket.
    await node.fs.rm(temporary, { force: true }).catch(() => undefined);
    throw error;
  }
}

/**
 * An artifact name is about to become a path on the caller's disk, so it is
 * checked here as well as on the server.
 *
 * The server validates artifact names as single members already
 * (`validate_member_name`), and this is not a guard against that server. It is a
 * guard against a *path* being built from a string this process did not choose,
 * in a method whose whole purpose is to write into a real library directory —
 * `Z:\books\...`, where a `..` would land somewhere nobody was looking.
 */
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

// ------------------------------------------------------------------ helpers

/**
 * A render's chunks, checked for the things that are facts about the request
 * rather than facts about the server.
 *
 * Deliberately **not** checked here: `maxChars`. That cap is per (voice,
 * backend), it is published on the voice row, and a copy of it in this file
 * would be a second thing to drift — the same reasoning that keeps
 * faster-whisper's language list out of {@link CrucibleClient.asr}.
 *
 * Deliberately checked here: duplicate indices. An index is an artifact name, so
 * two chunks sharing one are two renders writing the same FLAC with one of them
 * winning silently — and on this side of the wire they would also be two writes
 * racing for the same path in the caller's library.
 */
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
      // narrator answers an empty generate with a WHOLE-REQUEST error, which
      // would take the other 1,399 rows with it. The server refuses this too;
      // refusing it here names the chunk before a book's worth of text is sent.
      throw new CrucibleConfigError(
        `chunks[${at}].text`,
        'is blank, and narrator refuses an empty generate with a whole-request ' +
          'error that would end the batch',
      );
    }
    // Spelled `index`, which is the key `TtsChunk` declares and the only one it
    // accepts — the model forbids extras. narrator's own batch key is `i`, and
    // translating between the two is the server's business, not this client's.
    return { index, text };
  });
}

/** A non-negative integer the caller must state: an index, a take, a ceiling. */
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

/**
 * A boolean the caller must state. There is no coercion and no default: `asr`'s
 * two switches change what whisper is asked for, and a client that turned an
 * absent one into `true` would be choosing the rules a transcript was made
 * under on the caller's behalf.
 */
function requireBool(value: unknown, option: string): boolean {
  if (value === undefined || value === null) {
    throw new CrucibleConfigError(option, 'is required and was not given');
  }
  if (typeof value !== 'boolean') {
    throw new CrucibleConfigError(option, `must be a boolean, got ${typeof value}`);
  }
  return value;
}

function requireFinite(value: unknown, option: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new CrucibleConfigError(option, `must be a finite number, got ${String(value)}`);
  }
  return value;
}

const RESPONSE_FORMAT_TYPES = ['text', 'json_object', 'json_schema'] as const;

/**
 * `responseFormat`, checked only as far as the engines agree and no further.
 *
 * The parts that are checked are the ones a typo in makes the engine answer
 * something plausible and wrong: a `type` it does not know, or a `json_schema`
 * with no `name` or no `schema`. The `schema` itself is not read — it is a JSON
 * Schema document for the engine's guided-decoding backend, and which dialect of
 * it an engine supports is the engine's business to accept or refuse.
 */
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

/**
 * Read a 409 `server_busy` body into {@link CrucibleBusy}.
 *
 * A body that is not the v1 shape comes back as a {@link CrucibleProtocolError}
 * rather than quietly degrading to a plain {@link CrucibleRefused}. That is the
 * same call `#failure` already makes two branches up for an unparseable
 * envelope, and it is the right one: these fields are API v1's promise, a change
 * to them is a breaking change that arrives with a new `api_version`, and a
 * silent downgrade here would hide a broken wire behind an error that still
 * looks normal — a caller would see "busy" and never learn that the holder,
 * progress and job id it was about to display had gone missing.
 */
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

/** Is this `server_busy` the operator door's four-fact answer, or the lane's? */
function isHeldByAFact(details: unknown): boolean {
  return typeof details === 'object' && details !== null && 'fact' in details;
}

/**
 * Read the operator door's 409 `server_busy` into {@link CrucibleCardHeld}.
 *
 * A body that is not the v1 shape comes back a {@link CrucibleProtocolError}
 * rather than degrading to a plain refusal, for {@link busyRefusal}'s reason:
 * a caller would otherwise be shown "busy" and never learn that the holder it
 * was about to name had gone missing from the body.
 */
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

/**
 * Read a 409 `leased` body into {@link CrucibleLeased}.
 *
 * A body that is not the v1 shape comes back as a {@link CrucibleProtocolError}
 * rather than degrading to a plain {@link CrucibleRefused}, for the reason
 * {@link busyRefusal} does: a caller would otherwise see "leased" and never
 * learn that the holder and the deadline it was about to display had gone
 * missing.
 */
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

/**
 * `stopping`, wherever it appears — `/v1/health` and `/v1/activity` publish
 * the server's ONE `DyingResident`, so there is one reader for it here.
 *
 * The key must be present; `null` is the statement "nothing is stopping", and
 * a server that omitted it would be one whose answer about a wedged card is
 * unknown rather than negative. See {@link Stopping}.
 */
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

/** The six fields a lease carries wherever it appears. */
/**
 * `resident.held_by`, or null — **which is the stranded card, not an idle one.**
 *
 * `details` is handed through unchanged. It is the holding fact's OWN shape
 * (`crucible/settle.py`'s `Held`): a job's is the `server_busy` body this SDK
 * already types, a lease's is the lease receipt. Reshaping it here would make
 * this file a second owner of documents the server already speaks, which is
 * exactly the drift the field exists to avoid.
 */
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

/**
 * `params` for a load, carrying the lease when one was asked for.
 *
 * Snake_case on the wire because that is what the server's `LeaseOnLoad` model
 * declares, and it forbids unknown keys — a camelCase `ttlSeconds` would be a
 * 400 rather than a lease that quietly did nothing, which is the right failure
 * and still a failure this function exists to never cause.
 */
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

/**
 * One open streaming session.
 *
 * `progress` is read with {@link nullableNum} and then **required to be null**,
 * rather than simply not read. Reading it proves the key is on the wire — which
 * is what makes the null a statement rather than an absence — and asserting it
 * is null is the one place a server that started inventing a percentage for a
 * session would be caught. A session has no denominator; a number here would be
 * a fraction of the work that happened to have arrived so far, which falls as
 * more arrives.
 */
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

/**
 * The methods a stale-socket retry is allowed on: the ones with no side
 * effect to repeat, and no body to re-send.
 *
 * `undefined` is GET — `fetch` with no method is a GET, and every probe in
 * this file relies on that. A POST, PUT, PATCH or DELETE is NEVER retried
 * here, even though the failure looks identical from the outside: a reset
 * arrives with no way to know whether the server read the request first, and
 * a silently repeated `POST /v1/jobs` is a second render of somebody's book.
 * That risk belongs to the caller, who knows whether their call was idempotent.
 */
function isSafeMethod(method: string | undefined): boolean {
  const name = (method ?? 'GET').toUpperCase();
  return name === 'GET' || name === 'HEAD';
}

/**
 * Is this rejection a connection the server had already closed?
 *
 * WHY THIS EXISTS, in four occurrences. align's first `GET /v1/info` after a
 * render's last artifact fetch failed `read ECONNRESET` on 2026-09-18 and
 * 2026-09-19 against the PC and on 2026-09-20 at 00:57 against the Mac on
 * 127.0.0.1 — with the server up before and after each time (`serve.log` shows
 * no gap, `crucible api ping` answered). The cause was a race nobody can win
 * by timing: Node's `fetch` (undici) keeps an idle pooled connection about
 * **4 s** and uvicorn's default `timeout_keep_alive` is **5 s**, so a request
 * a few seconds after the last one is written onto a socket the server is
 * closing. Crucible now states `KEEP_ALIVE_SECONDS = 75` on both of its
 * uvicorn doors, which makes it rare rather than impossible — a restart, a
 * proxy in the middle or a dropped network can close a pooled connection at
 * any time — so this half must exist too.
 *
 * THE RULE, exactly: **an idempotent request is retried once, and only when
 * the connection failed before any response byte arrived.** `fetch` rejecting
 * IS that condition — once it resolves, a Response exists and a later failure
 * surfaces on the body stream, which this function never sees. Never a POST,
 * PUT, PATCH or DELETE ({@link isSafeMethod}); never more than once; never
 * after the caller's own `signal` aborted, which is checked first. An
 * unconditional retry loop would turn a server that is genuinely down into a
 * client that hangs twice as long for the same answer.
 *
 * **The apps cannot do this themselves.** This client takes no custom `fetch`,
 * so there is no seam outside this file to wrap.
 *
 * What is matched: undici raises `TypeError: fetch failed` whose `cause`
 * carries `code: 'ECONNRESET'` (the peer closed a socket we had written to) or
 * `code: 'UND_ERR_SOCKET'` (undici's own name for a socket that closed while a
 * request was on it). Both are read off the nested cause AND off the error
 * itself, because a non-undici `fetch` may raise the coded error directly.
 */
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

// ------------------------------------------------- the operator door's readers

/** Where an `engine` task can move this machine. 4.7: forward only. */
const ENGINE_TARGETS = ['wsl'] as const;

/**
 * `unmet` off a task document. PHASE15-HOST.md 5.3a.
 *
 * ABSENT reads as EMPTY, and that is a statement about vintage rather than a
 * default: a server that predates the field ran a module in which every
 * class was resolved or the module named none, because a server that could
 * leave one unmet is one that carries the field. A PRESENT `unmet` that is
 * not a list of `{class, reason}` is a protocol error, because a settings
 * window drawing "not on this engine" needs the reason and there is nothing
 * to fall back to.
 */
function readUnmet(body: Json, where: string): UnmetNeed[] {
  const raw = body['unmet'];
  if (raw === undefined) return [];
  const rows = asArray(raw, `${where}.unmet`);
  return rows.map((entry, index) => {
    const at = `${where}.unmet[${index}]`;
    const row = asObject(entry, at);
    return { class: str(row, 'class', at), reason: str(row, 'reason', at) };
  });
}

/** The vocabulary `TaskState` closes over, for `oneOf`. */
const TASK_STATES: readonly TaskState[] = ['running', 'done', 'failed', 'cancelled'];

/**
 * The six subject kinds. A seventh would be a contract change, not a surprise.
 *
 * `engine` joined them with PHASE15-HOST.md 3.10: on `llama-windows` the
 * llama.cpp binaries are pulled and reported exactly like weights, so the
 * page's Tasks and Catalog panels need no case for them.
 */
const SUBJECT_KINDS: readonly SubjectKind[] = [
  'model',
  'voice',
  'rvc',
  'rvc-base',
  'denoise',
  'engine',
];

/**
 * Task event names this build understands. Anything else arrives as
 * {@link UnknownEvent}, for `EVENT_NAMES`' reason: the vocabulary grows
 * without moving `api_version`, and a client that threw would lose the whole
 * stream rather than one frame.
 */
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
    request: asObject(field(row, 'request', where), `${where}.request`),
    state: oneOf(str(row, 'state', where), TASK_STATES, `${where}.state`),
    error: readFailureOrNull(field(row, 'error', where), `${where}.error`),
    created: str(row, 'created', where),
    started: str(row, 'started', where),
    finished: nullableStr(row, 'finished', where),
    unmet: readUnmet(row, where),
  };
}

/**
 * A `TaskRequest` in the server's spelling.
 *
 * Only the fields the type owns are sent, because the server refuses a `pull`
 * carrying a `job_type` — a request with another type's fields is a client
 * that has confused two requests, and the server would rather say so than run
 * the wrong one. `narratorEngine` is omitted when absent rather than sent as
 * `null`, since `null` would be a stated engine that is not one.
 */
function taskPayload(request: TaskRequest): Record<string, unknown> {
  // Widened to a bag of optional unknowns rather than an intersection of the
  // three arms: `Partial<Pull & Install & Module>` collapses to `never`,
  // because their `type` literals cannot all hold at once. What is wanted here
  // is "whatever the caller actually passed", which this says and that did not.
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
    // Sent as it was handed over. The document is a file the app vendors byte
    // for byte from the crucible repo's generator (PHASE13-OPERATOR.md 5.4);
    // reshaping it here would make this client a second author of it.
    return { type, module: given.module };
  }
  if (type === 'engine') {
    // 4.7. `wsl` and nothing else in this phase: moving BACK to Windows is
    // an explicit operator act (section 6), and a client that could ask for
    // it here would get a server's refusal for a request this package knew
    // was wrong before it left.
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
  const step: { name: string; index: number; total: number; jobTypes?: readonly string[] } = {
    name: str(data, 'name', where),
    index: num(data, 'index', where),
    total: num(data, 'total', where),
  };
  // Only the `reload` step carries it (section 3.4), so its absence is not a
  // missing field — it is a step that made nothing new reachable.
  if ('job_types' in data) step.jobTypes = strArray(data, 'job_types', where);
  return step;
}

/**
 * One `progress` frame, of the TWO shapes a task's progress takes.
 *
 * A pull counts bytes and an install relays the installer's lines, and the
 * server sends whichever is true rather than a merged shape with half its
 * fields null. Which arrived is decided on the keys that are there — the same
 * discrimination {@link isTaskBytesProgress} offers a caller — and a frame
 * that is neither is a protocol error, not an empty progress.
 */
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
