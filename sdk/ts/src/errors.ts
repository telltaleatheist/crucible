/** Base class for everything `@crucible/client` throws deliberately. */
export class CrucibleError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message);
    this.name = new.target.name;
    if (options !== undefined && 'cause' in options) {
      (this as Error & { cause?: unknown }).cause = options.cause;
    }
  }
}

/** A required piece of client configuration is missing or unusable. */
export class CrucibleConfigError extends CrucibleError {
  readonly option: string;

  constructor(option: string, message: string) {
    super(`${option}: ${message}`);
    this.option = option;
  }
}

/** The server could not be reached at all; never retried. */
export class CrucibleUnreachable extends CrucibleError {
  readonly url: string;

  constructor(url: string, message: string, cause?: unknown) {
    super(`crucible at ${url} is unreachable: ${message}`, { cause });
    this.url = url;
  }
}

/** Something answered `GET /v1/ping` but it is not a Crucible. */
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

/** 426: the server speaks a different major API version than this client. */
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

/** Any other 4xx: the server understood the request and refused it by name. */
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

/** The server's code for "one job at a time, and it is not yours". */
export const SERVER_BUSY = 'server_busy';

/**
 * 409 `server_busy`: Crucible admits one job at a time. A job submitted with `queue` waits in the
 * server's queue instead of meeting this; one submitted without it is refused.
 */
export class CrucibleBusy extends CrucibleRefused {
  /** The busy job's `client` (its recorded User-Agent), or null when it did not say. */
  readonly holder: string | null;
  readonly jobId: string;
  /** The busy job's type: `tts`, `llm`, `rvc`, ... */
  readonly jobType: string;
  /** The model it holds, or null for a job type that names none. */
  readonly model: string | null;
  /** `running` or `queued`. */
  readonly jobStatus: string;
  /** When it started (`running`) or was admitted (`queued`). */
  readonly since: string;
  /** The busy job's fraction done, 0..1. */
  readonly progress: number;
  /** The holder's latest progress line. */
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

  /** "GPU busy: foundry, tts 62% done" — one line for a bench. */
  get busyLine(): string {
    const who = this.holder === null ? 'an unnamed client' : this.holder;
    const what = this.model === null ? this.jobType : `${this.jobType} ${this.model}`;
    const parts = [who, what, `${Math.round(this.progress * 100)}% done`];
    const line = `busy: ${parts.join(', ')}`;
    return this.jobMessage === null ? line : `${line} — ${this.jobMessage}`;
  }
}

/**
 * 409 `server_busy` from the OPERATOR door: something holds the card and an install may not start.
 */
export class CrucibleCardHeld extends CrucibleRefused {
  /** `a job`, `a session`, `the claim` or `a chat`. */
  readonly fact: string;
  /** Who, in the server's own words. */
  readonly who: string;

  constructor(
    status: number,
    code: string,
    serverMessage: string,
    details: unknown,
    fields: { fact: string; who: string },
  ) {
    super(status, code, serverMessage, details);
    this.fact = fields.fact;
    this.who = fields.who;
  }

  /** "held by a session: session ses-… of 'foundry' for 'translate'" — one line, for a row. */
  get heldLine(): string {
    return `held by ${this.fact}: ${this.who}`;
  }
}

/** The server's code for "another client's queue session holds this server". */
export const SESSION_OPEN = 'session_open';
/** A session header (or a session's own route) naming a session that has ended. */
export const SESSION_CLOSED = 'session_closed';
/** A session header naming a session still waiting in the line: send its items after it opens. */
export const SESSION_NOT_OPEN = 'session_not_open';
/** A session header naming another client's session. */
export const SESSION_NOT_YOURS = 'session_not_yours';
/** A session id this server does not know: it never had it, or it restarted since. */
export const UNKNOWN_QUEUE_SESSION = 'unknown_queue_session';

/**
 * 409 `server_busy` (`details.door: "session"`) or `session_open`: another client's queue session
 * holds the server, and nothing but its own items runs until it closes. A job or call sent with
 * `queue` waits in the line instead of meeting this.
 */
export class CrucibleSessionHeld extends CrucibleRefused {
  /** Who holds it (its `X-Crucible-Client`, else `User-Agent`), or null when it did not say. */
  readonly holder: string | null;
  readonly sessionId: string;
  /** What the holder's run is: a capability class name. */
  readonly act: string;
  /** The model it opened with, or null. */
  readonly model: string | null;
  /** `open`, or `queued` for one about to open. */
  readonly sessionStatus: string;
  /** When it opened (or was asked for). */
  readonly since: string;

  constructor(
    status: number,
    code: string,
    serverMessage: string,
    details: unknown,
    fields: {
      holder: string | null;
      sessionId: string;
      act: string;
      model: string | null;
      sessionStatus: string;
      since: string;
    },
  ) {
    super(status, code, serverMessage, details);
    this.holder = fields.holder;
    this.sessionId = fields.sessionId;
    this.act = fields.act;
    this.model = fields.model;
    this.sessionStatus = fields.sessionStatus;
    this.since = fields.since;
  }

  /** "held: foundry's session for translate, since …" — one line for a bench. */
  get heldLine(): string {
    const who = this.holder === null ? 'an unnamed client' : this.holder;
    return `held: ${who}'s session for ${this.act}, since ${this.since}`;
  }
}

/**
 * 409 `session_closed`: the queue session ended, so nothing more runs in it. Also what a session
 * that never opened throws (`reason` `expired`, `operator`, `load_failed`, …), and what using a
 * {@link CrucibleSession} after it ended throws without asking the server.
 */
export class CrucibleSessionClosed extends CrucibleRefused {
  readonly sessionId: string;
  /** Why it ended: `client`, `idle`, `operator`, `max_hold`, `server_restart`, `expired`, `load_failed`, … */
  readonly reason: string;

  constructor(
    status: number,
    serverMessage: string,
    details: unknown,
    fields: { sessionId: string; reason: string },
  ) {
    super(status, SESSION_CLOSED, serverMessage, details);
    this.sessionId = fields.sessionId;
    this.reason = fields.reason;
  }
}

/** The code {@link CrucibleFleetUnavailable} carries. */
export const FLEET_UNAVAILABLE = 'fleet_unavailable';

/** Why one server of a fleet could not give a {@link fleetSession} its session. */
export interface FleetServerReason {
  /** The server's base URL. */
  readonly url: string;
  /**
   * `probe`: it could not serve the request at all (did not answer, no `queue.sessions`, the model
   * not in its catalogue). `line`: it was asked, and its session ended before opening, or was still
   * waiting when the fleet's `maxWaitS` ran out.
   */
  readonly stage: 'probe' | 'line';
  /** A sentence a person reads. */
  readonly reason: string;
}

/**
 * No server of a fleet gave {@link fleetSession} a session: none could serve the request, or every
 * one that could ended its session before it opened, or `maxWaitS` ran out first. `servers` names
 * each server and why, in the order the fleet was given.
 */
export class CrucibleFleetUnavailable extends CrucibleError {
  readonly code = FLEET_UNAVAILABLE;
  readonly servers: readonly FleetServerReason[];

  constructor(summary: string, servers: readonly FleetServerReason[]) {
    super(
      `${summary}: ` + servers.map((entry) => `${entry.url} (${entry.stage}): ${entry.reason}`).join('; '),
    );
    this.servers = servers;
  }
}

/** The code a malformed pairing line is refused with. */
export const INVALID_PAIRING = 'invalid_pairing';

/** A pasted `crucible://` line is not one. */
export class CruciblePairingError extends CrucibleError {
  readonly code = INVALID_PAIRING;
  /** The offending line, with everything after `#` replaced by `…`. */
  readonly line: string;
  /** What was wrong with its shape. */
  readonly detail: string;

  constructor(detail: string, line: string) {
    super(
      `that is not a Crucible pairing line (${INVALID_PAIRING}): ${detail}` +
        (line === '' ? '' : ` — got ${line}`),
    );
    this.detail = detail;
    this.line = line;
  }
}

/** The code a malformed pairing file is refused with. */
export const PAIRING_FILE_MALFORMED = 'pairing_file_malformed';

/** `<CRUCIBLE_HOME>/pairing` exists and is not what a writer of it writes. */
export class CruciblePairingFileError extends CrucibleError {
  readonly code = PAIRING_FILE_MALFORMED;
  /** Where the file is, so the sentence names what to delete. */
  readonly path: string;

  constructor(path: string, detail: string) {
    super(`${PAIRING_FILE_MALFORMED}: ${path} ${detail}`);
    this.path = path;
  }
}

/** Would this refusal be different on a different server? */
export function isServerSpecificRefusal(code: string): boolean {
  return SERVER_SPECIFIC_REFUSALS.has(code);
}

const SERVER_SPECIFIC_REFUSALS: ReadonlySet<string> = new Set([
  SERVER_BUSY,
  SESSION_OPEN,
  SESSION_CLOSED,
  SESSION_NOT_OPEN,
  SESSION_NOT_YOURS,
  UNKNOWN_QUEUE_SESSION,
  'engine_in_use',
  'stream_session_open',
  'job_type_disabled',
  'model_not_resident',
  'not_resident',
  'unknown_model',
  'env_missing',
  'task_busy',
  'already_installed',
  'job_type_installed',
]);

/** 5xx: the server broke; never retried. */
export class CrucibleServerError extends CrucibleError {
  readonly status: number;
  readonly code: string;
  readonly serverMessage: string;
  readonly details: unknown;

  constructor(status: number, code: string, serverMessage: string, details: unknown) {
    super(`crucible failed the request (${status} ${code}): ${serverMessage}`);
    this.status = status;
    this.code = code;
    this.serverMessage = serverMessage;
    this.details = details;
  }
}

/** The server's code for "a deploy is about to restart me; nothing was admitted". */
export const SERVER_UPDATING = 'server_updating';

/**
 * 503 `server_updating`: a deploy holds the server for a restart, so it admitted nothing. The
 * client waits these out by itself (it sends again once the new server answers), so a caller
 * sees one only when the wait outlasted its budget.
 */
export class CrucibleUpdating extends CrucibleServerError {
  /** The release being installed, when the deploy said. */
  readonly release: string | null;
  /** When the hold lapses by itself if nothing restarts the server. */
  readonly until: string | null;

  constructor(status: number, code: string, serverMessage: string, details: unknown) {
    super(status, code, serverMessage, details);
    const fields = typeof details === 'object' && details !== null ? (details as Record<string, unknown>) : {};
    this.release = typeof fields['release'] === 'string' ? fields['release'] : null;
    this.until = typeof fields['until'] === 'string' ? fields['until'] : null;
  }
}

/** The server's code for "I cannot see my own accelerator at the moment". */
export const ACCELERATOR_UNREADABLE = 'accelerator_unreadable';

/** 503 `accelerator_unreadable`: the probe could not read the card, which is never an idle card. */
export class CrucibleAcceleratorUnreadable extends CrucibleServerError {}

/** The server's code for "nothing has decided what this host can hold yet". */
export const CAPABILITY_UNDECIDED = 'capability_undecided';

/**
 * 503 `capability_undecided`: this server has not decided what it can hold (`crucible capability
 * --write`).
 */
export class CrucibleCapabilityUndecided extends CrucibleServerError {}

/** The server answered, but the payload is not the shape API v1 promises. */
export class CrucibleProtocolError extends CrucibleError {
  readonly detail: string;

  constructor(detail: string) {
    super(`crucible sent something API v1 does not describe: ${detail}`);
    this.detail = detail;
  }
}

/** A capability row with no `route`. */
export const CAPABILITY_ROUTE_MISSING = 'capability_route_missing';
/** A `route` that is neither `local` nor `upstream`. */
export const CAPABILITY_ROUTE_UNKNOWN = 'capability_route_unknown';

/** A voice row with no `needs_reference`. */
export const VOICES_NEEDS_REFERENCE_MISSING = 'voices_needs_reference_missing';
/** A `needs_reference` that is not a boolean. */
export const VOICES_NEEDS_REFERENCE_UNKNOWN = 'voices_needs_reference_unknown';

/** `DELETE /v1/catalog/{kind}/{id}`: no such kind, or no such id here (404). */
export const SUBJECT_UNKNOWN = 'subject_unknown';
/** It is a subject this server can hold and it does not hold it (409). */
export const SUBJECT_NOT_INSTALLED = 'subject_not_installed';
/** Resident, held by a session, or named by a running task (409). */
export const SUBJECT_IN_USE = 'subject_in_use';
/** The files would not go (500). */
export const SUBJECT_REMOVE_FAILED = 'subject_remove_failed';
export const WEIGHTS_SHARED = 'weights_shared';

/** `PUT /v1/settings`: a class that cannot run anywhere but this card. */
export const ROUTE_NOT_ROUTABLE = 'route_not_routable';
/** A route value (or a chat's `model`) that is not `<upstream>/<model>`. */
export const ROUTE_BAD_MODEL = 'route_bad_model';
/** A route whose upstream has no key/url. */
export const ROUTE_UPSTREAM_UNCONFIGURED = 'route_upstream_unconfigured';
/** Removing an upstream a route still names. */
export const UPSTREAM_IN_USE = 'upstream_in_use';
/** A name that is not one of the three. */
export const UNKNOWN_UPSTREAM = 'unknown_upstream';
/** An upstream handed the field it does not take (a `url` for `anthropic`). */
export const UPSTREAM_BAD_FIELD = 'upstream_bad_field';

/** A chat naming an upstream this server has no credential for (409). */
export const UPSTREAM_UNCONFIGURED = 'upstream_unconfigured';
/** The upstream refused. */
export const UPSTREAM_REJECTED = 'upstream_rejected';
/** The upstream did not answer. */
export const UPSTREAM_UNREACHABLE = 'upstream_unreachable';
/** The upstream rate-limited it, passed through with `Retry-After`. */
export const UPSTREAM_RATE_LIMITED = 'upstream_rate_limited';
/** A `load-model` or a session's `model` naming an upstream model, which is never resident. */
export const UPSTREAM_NEVER_RESIDENT = 'upstream_never_resident';

/** The three `testUpstream` answers that are results rather than exceptions. */
export const UPSTREAM_TEST_REFUSALS = [
  UPSTREAM_UNREACHABLE,
  UPSTREAM_REJECTED,
  UPSTREAM_UNCONFIGURED,
] as const;
