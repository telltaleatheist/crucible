/**
 * Every refusal this client surfaces has its own type.
 *
 * There is no error that means "something went wrong". If a call does not return
 * what the caller asked for, the thrown error says which of the seven possible
 * reasons it was, and carries the server's own `error.code` / `error.message`
 * whenever the server sent one (DESIGN.md section 4: errors are always
 * `{"error": {"code", "message", "details"?}}`).
 */

/** Base class for everything `@crucible/client` throws deliberately. */
export class CrucibleError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message);
    this.name = new.target.name;
    if (options !== undefined && 'cause' in options) {
      // `cause` is ES2022; assign it explicitly so the CJS build carries it too.
      (this as Error & { cause?: unknown }).cause = options.cause;
    }
  }
}

/**
 * A required piece of client configuration is missing or unusable. Names the
 * option. There is no default for anything the caller must supply.
 */
export class CrucibleConfigError extends CrucibleError {
  readonly option: string;

  constructor(option: string, message: string) {
    super(`${option}: ${message}`);
    this.option = option;
  }
}

/**
 * The server could not be reached at all: connection refused, DNS failure, TLS
 * failure, the socket died mid-response. No retry is attempted; a dead server is
 * reported as a dead server.
 */
export class CrucibleUnreachable extends CrucibleError {
  readonly url: string;

  constructor(url: string, message: string, cause?: unknown) {
    super(`crucible at ${url} is unreachable: ${message}`, { cause });
    this.url = url;
  }
}

/**
 * Something answered `GET /v1/ping` but it is not a Crucible: the body was not
 * JSON, or it did not say `{"crucible": true}`. Distinguishes "wrong address"
 * from "wrong token".
 */
export class CrucibleNotACrucible extends CrucibleError {
  readonly url: string;
  readonly body: string;

  constructor(url: string, body: string) {
    super(
      `${url} answered /v1/ping but did not identify as a crucible ` +
        `(no {"crucible": true}); it said: ${body}`,
    );
    this.url = url;
    this.body = body;
  }
}

/** 401: the bearer token is missing, malformed, or not this server's token. */
export class CrucibleAuthError extends CrucibleError {
  readonly status = 401;
  readonly code: string;
  readonly serverMessage: string;

  constructor(code: string, serverMessage: string) {
    super(`crucible refused the token (${code}): ${serverMessage}`);
    this.code = code;
    this.serverMessage = serverMessage;
  }
}

/**
 * 426: the server speaks a different major API version than this client. Both
 * versions are named, because only the caller can decide which side to upgrade.
 */
export class CrucibleVersionError extends CrucibleError {
  readonly status = 426;
  readonly code: string;
  readonly serverMessage: string;
  readonly serverApiVersion: number | null;
  readonly clientApiVersion: number;

  constructor(
    code: string,
    serverMessage: string,
    serverApiVersion: number | null,
    clientApiVersion: number,
  ) {
    super(
      `crucible speaks API version ${serverApiVersion ?? 'unknown'}, this client ` +
        `speaks ${clientApiVersion} (${code}): ${serverMessage}`,
    );
    this.code = code;
    this.serverMessage = serverMessage;
    this.serverApiVersion = serverApiVersion;
    this.clientApiVersion = clientApiVersion;
  }
}

/**
 * Any other 4xx: the server understood the request and refused it by name —
 * `unknown_job_type`, `job_type_disabled`, `unknown_model`, `unknown_blob`,
 * `unknown_job`, `job_not_cancellable`, `invalid_request`, ...
 */
export class CrucibleRefused extends CrucibleError {
  readonly status: number;
  readonly code: string;
  readonly serverMessage: string;
  readonly details: unknown;

  constructor(status: number, code: string, serverMessage: string, details: unknown) {
    super(`crucible refused the request (${status} ${code}): ${serverMessage}`);
    this.status = status;
    this.code = code;
    this.serverMessage = serverMessage;
    this.details = details;
  }
}

/**
 * The server's code for "one job at a time, and it is not yours".
 *
 * Exported for the same reason {@link ACCELERATOR_UNREADABLE} is: the mapping
 * turns exactly this code into a type, and a caller comparing `error.code`
 * should compare against one spelling of it.
 */
export const SERVER_BUSY = 'server_busy';

/**
 * 409 `server_busy`: Crucible admits one job at a time and **does not queue**.
 *
 * Its own type because of what the body carries and what every client would
 * otherwise re-derive from an untyped blob. `crucible/jobs/queue.py` deliberately
 * answers with the holder's name, job id, type, model, status, since, progress
 * and latest progress line — *"enough for a client to back off intelligently,
 * and exactly the 'GPU busy: foundry' line BookForge wants, for free."* Leaving
 * that in `details: unknown` means every bench re-implements the same parse of
 * the same shape, which is one fact with as many owners as there are clients
 * (ARCHITECTURE.md §1). It is read here, once, through the same shape helpers
 * everything else on this wire goes through.
 *
 * **`holder` is null when the busy job arrived without a User-Agent, and null
 * means "it did not say".** The server refuses to invent a name there, for the
 * reason a bench must never be confidently wrong about whose render is on the
 * card (PHASE7-LANES.md §5) — so a caller rendering this must say something like
 * "an unnamed client", never a guess.
 *
 * **This client does not retry it, and that is a design decision rather than an
 * omission.** ARCHITECTURE.md R5: *queues belong to clients.* A retry loop in
 * the SDK would be a queue — an invisible one, with a policy nobody chose, that
 * cannot know the caller has nineteen other chapters or that the operator went
 * to bed. The SDK's whole contribution to backing off is telling the caller
 * precisely what is in the way and how far along it is. What to do about it is
 * the caller's, every time.
 */
export class CrucibleBusy extends CrucibleRefused {
  /** The busy job's `client` — its recorded User-Agent. Null = it did not say. */
  readonly holder: string | null;
  readonly jobId: string;
  /** The busy job's type: `tts`, `llm`, `rvc`, ... */
  readonly jobType: string;
  /** The model it holds, or null for a job type that names none. */
  readonly model: string | null;
  /** `running` or `queued`. Names which clock {@link since} is on. */
  readonly jobStatus: string;
  /**
   * When it started (`running`) or was admitted (`queued`), as the server said
   * it. {@link jobStatus} is what makes this unambiguous: without it a caller
   * cannot tell a job that has been rendering for an hour from one admitted
   * 2 ms ago.
   */
  readonly since: string;
  /** 0..1. */
  readonly progress: number;
  /** The holder's latest progress line. Null until the job has said anything. */
  readonly jobMessage: string | null;

  constructor(
    status: number,
    code: string,
    serverMessage: string,
    details: unknown,
    fields: {
      holder: string | null;
      jobId: string;
      jobType: string;
      model: string | null;
      jobStatus: string;
      since: string;
      progress: number;
      jobMessage: string | null;
    },
  ) {
    super(status, code, serverMessage, details);
    this.holder = fields.holder;
    this.jobId = fields.jobId;
    this.jobType = fields.jobType;
    this.model = fields.model;
    this.jobStatus = fields.jobStatus;
    this.since = fields.since;
    this.progress = fields.progress;
    this.jobMessage = fields.jobMessage;
  }

  /**
   * "GPU busy: foundry, tts 62% done" — the one line a bench puts in front of a
   * human. Here rather than in each client for the same reason the fields are.
   */
  get busyLine(): string {
    const who = this.holder === null ? 'an unnamed client' : this.holder;
    const what = this.model === null ? this.jobType : `${this.jobType} ${this.model}`;
    const done = `${Math.round(this.progress * 100)}% done`;
    return this.jobMessage === null
      ? `busy: ${who}, ${what}, ${done}`
      : `busy: ${who}, ${what}, ${done} — ${this.jobMessage}`;
  }
}

/**
 * Would this refusal be different on a different server?
 *
 * The fact this answers has exactly one honest owner — the server that emits the
 * code — and it is the fact a `waitFor: "any"` client needs (PHASE7-LANES.md
 * §4.2.1: *"the first server that will take it, preferring rank order"*). Trying
 * the next server is right for a refusal that is about THIS machine's state, and
 * wrong for one that is about the request, which will be refused identically
 * everywhere and should be surfaced once instead of N times.
 *
 * It is a function over the code rather than a retry loop, for
 * {@link CrucibleBusy}'s reason: walking a registry is the client's job. This
 * answers the one question the client cannot answer for itself.
 *
 * Codes are listed explicitly and an unknown code answers **false**. That is the
 * conservative direction on purpose: a new refusal this build has never seen is
 * reported to the caller rather than silently swallowed by a walk that tries
 * four machines and reports the fourth machine's error. A code that should be
 * server-specific and is not yet listed costs one wasted opportunity; the
 * reverse costs a confusing error and three pointless round trips.
 */
export function isServerSpecificRefusal(code: string): boolean {
  return SERVER_SPECIFIC_REFUSALS.has(code);
}

/**
 * The refusals that are about a server's state rather than about the request.
 *
 * - `server_busy` — the lane is taken. Another server's may not be.
 * - `engine_in_use` — **narrator's wire is claimed, which is not the same as the
 *   lane.** A streaming session holds the resident engine without occupying the
 *   lane at all (`crucible/residency.py`'s `refuse_if_claimed`), so this is the
 *   refusal a render gets while the browser extension is reading on that machine.
 *   It travels as well as `server_busy` does and for the same reason.
 * - `job_type_disabled` — this server does not do this. Under PHASE9 the flag is
 *   set by what fits on the card at install, so it genuinely varies by machine:
 *   a 6 GB card has no `tts`, the 3090 Ti does.
 * - `model_not_resident` / `unknown_model` — residency is per server, and
 *   loading is the operator's act, not a job's (PHASE2). Another server may
 *   already hold it.
 * - `stream_session_open` — this server already has its one session. Another
 *   server's streaming door may be free.
 * - `env_missing` — the venv for this job type was never installed here.
 * - `accelerator_unreadable` is NOT here: it is a 5xx and arrives as
 *   {@link CrucibleAcceleratorUnreadable}, which must never be read as an answer
 *   about the card at all.
 *
 * Everything else — `invalid_request`, `unknown_job_type`, `unknown_blob`,
 * `unknown_job`, `job_not_cancellable` — is about what was asked or about state
 * that only exists on the server already spoken to, and travels no better.
 */
const SERVER_SPECIFIC_REFUSALS: ReadonlySet<string> = new Set([
  SERVER_BUSY,
  'engine_in_use',
  'stream_session_open',
  'job_type_disabled',
  'model_not_resident',
  'unknown_model',
  'env_missing',
]);

/** 5xx: the server broke. The client never retries one of these. */
export class CrucibleServerError extends CrucibleError {
  readonly status: number;
  readonly code: string;
  readonly serverMessage: string;

  constructor(status: number, code: string, serverMessage: string) {
    super(`crucible failed the request (${status} ${code}): ${serverMessage}`);
    this.status = status;
    this.code = code;
    this.serverMessage = serverMessage;
  }
}

/**
 * The server's code for "I cannot see my own accelerator at the moment".
 * Exported because the mapping below turns exactly this code into a type, and a
 * caller comparing `error.code` should compare against one spelling of it.
 */
export const ACCELERATOR_UNREADABLE = 'accelerator_unreadable';

/**
 * 503 `accelerator_unreadable`: the probe ran and could not read the card —
 * nvidia-smi missing, refusing, or timing out.
 *
 * Its own type because of the one conclusion it must never be confused with.
 * `GET /v1/accelerator` exists so a client can tell "the card is busy" from "the
 * card is free", and the server raises rather than returning zeroes precisely so
 * that an unreadable probe cannot be read as an idle card. A caller polling for
 * a free GPU treats this as **ask again**: it is not a refusal of anything it
 * asked for, and it is not an answer about the card.
 *
 * It is a {@link CrucibleServerError} — the status really is a 5xx and every
 * phase-2 handler that catches one still catches this — with a narrower name for
 * the callers that need to act differently.
 */
export class CrucibleAcceleratorUnreadable extends CrucibleServerError {}

/**
 * The server answered with a status the client accepts, but the payload is not
 * the shape API v1 promises: unparseable JSON, a missing field, or an SSE event
 * name that is not in the v1 vocabulary. A new event kind is a breaking change
 * and would come with a new `api_version`, so this is never absorbed quietly.
 */
export class CrucibleProtocolError extends CrucibleError {
  readonly detail: string;

  constructor(detail: string) {
    super(`crucible sent something API v1 does not describe: ${detail}`);
    this.detail = detail;
  }
}
