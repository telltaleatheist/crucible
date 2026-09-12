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
