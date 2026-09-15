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
 * 409 `server_busy` from the OPERATOR door: something holds the card and an
 * install may not start. PHASE13-OPERATOR.md section 3.3.
 *
 * **The same code as {@link CrucibleBusy} and a different shape**, which is not
 * an accident and is not drift. `POST /v1/jobs` asks one question — *is the
 * lane free?* — and there is exactly one kind of answer, a job. `POST /v1/tasks`
 * asks a different one — *is anything at all using the card?* — and there are
 * FOUR kinds of holder (`crucible/settle.py`): a job, a lease, a streaming
 * claim, a chat in flight. Flattening those into the job shape would mean
 * inventing a `job_id` for a lease, and giving them four codes would make a
 * client learn four words for "not now".
 *
 * So the body says which, in `details.fact`, and carries that fact's OWN
 * fields beside it — a job's are `POST /v1/jobs`' verbatim, a lease's are the
 * ones a `409 leased` and a lease receipt already carry. `fact` is what the
 * client discriminates on, and its presence is what tells this type from
 * {@link CrucibleBusy}.
 *
 * {@link who} is the sentence the server wrote, and an app's row is expected to
 * show it verbatim — *"held by foundry — translate, qwen3.8-27b-4bit"*. An
 * operator shown a dead button with no name concludes the button is broken.
 */
export class CrucibleCardHeld extends CrucibleRefused {
  /** `a job`, `a lease`, `the claim` or `a chat`. */
  readonly fact: string;
  /** Who, in the server's own words. Never null: a fact that holds has a holder. */
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

  /** "held by a lease: 'foundry/owens-pc' for 'translate'" — one line, for a row. */
  get heldLine(): string {
    return `held by ${this.fact}: ${this.who}`;
  }
}

/**
 * The server's code for "somebody has said they are mid-run on this".
 *
 * Exported for {@link SERVER_BUSY}'s reason: the mapping turns exactly this code
 * into a type, and a caller comparing `error.code` should compare against one
 * spelling of it.
 *
 * It is `leased` and not `model_leased`: since 2026-09-14 a lease names the
 * resident thing of any kind, so a code naming one kind would be false whenever
 * narrator or the aligner holds the card. {@link CrucibleLeased.kind} says which.
 */
export const LEASED = 'leased';

/**
 * 409 `leased`: a client holds a lease on the resident model, voice or aligner,
 * so anything that would take it off the card is refused until the lease is
 * released or expires.
 *
 * **Why a lease exists at all.** A chat completion holds nothing on a Crucible —
 * no lane, no job, no claim — which is right for one chat and wrong for two
 * thousand. A book translated block by block leaves the server looking idle
 * between any two blocks, and a `load-voice` submitted in one of those gaps
 * evicted the translator at block 400 of 2000 with nothing having gone wrong
 * anywhere. The client is the only thing that knows a run is in progress, so it
 * says so.
 *
 * **What it does NOT refuse.** Chats (they are what the lease protects), any job
 * that leaves the card's contents alone — `echo`, `asr`, `rvc`, `denoise`, and
 * the unloaders that can only unload some other kind — and, on a voice or
 * aligner lease, **the work the lease was taken for**: a `tts` render of the
 * leased voice and an `align` on the leased aligner reuse what is resident
 * instead of loading it, which is the whole reason to hold one. A lease is not a
 * reservation: the lane is still free and admission is still the door's.
 *
 * Its own type for {@link CrucibleBusy}'s reason — the body is not decoration. A
 * caller shown this has to be able to say WHO is in the way, doing WHAT, and
 * until when, without every client re-parsing the same shape.
 *
 * **This client does not wait it out**, exactly as it does not retry
 * {@link CrucibleBusy}: a sleep loop in the SDK would be a queue with a policy
 * nobody chose (ARCHITECTURE.md R5).
 */
export class CrucibleLeased extends CrucibleRefused {
  readonly leaseId: string;
  /**
   * Which resident kind is held: `llm`, `tts` or `align`.
   *
   * On the refusal and not only on the lease, because a refusal arrives with no
   * `resident` beside it and the kind is what says WHICH jobs this lease covers
   * — `leased` on a `load-voice` means something different when a 27B is held
   * than when narrator is.
   */
  readonly kind: string;
  /**
   * Who holds it — the lease's recorded `client`. Null = it did not say, and
   * never a guess, for the reason {@link CrucibleBusy.holder} is null.
   *
   * Named `holder` rather than `client` because on a refusal the question is
   * who is in the way; the `client` spelling belongs to the lease itself
   * (`Lease.client`), where the question is whose lease it is.
   */
  readonly holder: string | null;
  /** What the run IS: a capability class name. A lease must say, so never null. */
  readonly act: string;
  /** When the lease was taken. */
  readonly since: string;
  /** When it stops being open unless its holder heartbeats it. */
  readonly expiresAt: string;

  constructor(
    status: number,
    code: string,
    serverMessage: string,
    details: unknown,
    fields: {
      leaseId: string;
      kind: string;
      holder: string | null;
      act: string;
      since: string;
      expiresAt: string;
    },
  ) {
    super(status, code, serverMessage, details);
    this.leaseId = fields.leaseId;
    this.kind = fields.kind;
    this.holder = fields.holder;
    this.act = fields.act;
    this.since = fields.since;
    this.expiresAt = fields.expiresAt;
  }

  /**
   * "leased: foundry, translate, until 2026-09-14T03:12:00+00:00" — the one line
   * a bench puts in front of a human, here rather than in each client for the
   * reason {@link CrucibleBusy.busyLine} is.
   */
  get leasedLine(): string {
    const who = this.holder === null ? 'an unnamed client' : this.holder;
    return `leased: ${who}, ${this.act}, until ${this.expiresAt}`;
  }
}

/**
 * The code a malformed pairing line is refused with.
 *
 * Exported for {@link SERVER_BUSY}'s reason: a connect door comparing
 * `error.code` should compare against one spelling of it.
 */
export const INVALID_PAIRING = 'invalid_pairing';

/**
 * A pasted `crucible://` line is not one. PHASE13-OPERATOR.md sections 2.1, 5.1.
 *
 * **Not a {@link CrucibleRefused}**, because no server refused anything: this
 * is thrown by {@link parsePairing}, which is pure and runs before a client
 * exists. It is also not a {@link CrucibleConfigError}, whose contract is "a
 * required option of `new CrucibleClient` is missing" — a bad paste is a
 * person's typo in a field, and a connect door shows it beside that field
 * rather than in a configuration error.
 *
 * `line` is the line with its **fragment elided**, because the fragment is the
 * token and this error is exactly the kind an app logs.
 */
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

/** The pairing FILE's one refusal (PHASE15-HOST.md 3.6). */
export const PAIRING_FILE_MALFORMED = 'pairing_file_malformed';

/**
 * `<CRUCIBLE_HOME>/pairing` exists and is not what a writer of it writes.
 *
 * Separate from {@link CruciblePairingError}, which is about a LINE somebody
 * pasted: this one is about a FILE on this machine, and the two have different
 * answers. A bad pasted line means "check what you copied"; a bad file means
 * "something other than `crucible init` wrote to the engine's own state", and
 * the caller must not turn either into `null` — "there is no server here" would
 * send somebody to install a second Crucible over a running one.
 *
 * The writer (`crucible/pairing.py`) enforces exactly one line; this is what
 * the reader says when it finds more.
 */
export class CruciblePairingFileError extends CrucibleError {
  readonly code = PAIRING_FILE_MALFORMED;
  /** Where the file is, so the sentence names what to delete. */
  readonly path: string;

  constructor(path: string, detail: string) {
    super(`${PAIRING_FILE_MALFORMED}: ${path} ${detail}`);
    this.path = path;
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
 * - `model_not_resident` / `not_resident` / `unknown_model` — residency is per
 *   server, and loading is the operator's act, not a job's (PHASE2). Another
 *   server may already hold it. The bare `not_resident` is the lease door's,
 *   which takes an id of any resident kind and so cannot name one.
 * - `stream_session_open` — this server already has its one session. Another
 *   server's streaming door may be free.
 * - `leased` — a client is mid-run on THIS machine's resident model, voice or
 *   aligner. Another machine's card is not held by it, and a `waitFor: "any"`
 *   walk should try the next one rather than wait out somebody else's book.
 * - `env_missing` — the venv for this job type was never installed here.
 * - `accelerator_unreadable` is NOT here: it is a 5xx and arrives as
 *   {@link CrucibleAcceleratorUnreadable}, which must never be read as an answer
 *   about the card at all.
 *
 * Three more arrived with the operator door (PHASE13-OPERATOR.md section 3.3),
 * and each is about THIS machine:
 *
 * - `task_busy` — this server is already running an operator task. Another
 *   server's task lane is its own.
 * - `already_installed` / `job_type_installed` — the weights or the env are on
 *   THIS disk. A `waitFor: "any"` walk setting up a fleet should move to the
 *   next machine rather than stop, because the next machine may well need it.
 *
 * Everything else — `invalid_request`, `unknown_job_type`, `unknown_subject`,
 * `invalid_module`, `narrator_engine_required`, `narrator_engine_refused`,
 * `unknown_blob`, `unknown_job`, `unknown_task`, `job_not_cancellable`,
 * `not_running` — is about what was asked, or about state that only exists on
 * the server already spoken to, and travels no better. A misspelled subject id
 * is misspelled everywhere.
 */
const SERVER_SPECIFIC_REFUSALS: ReadonlySet<string> = new Set([
  SERVER_BUSY,
  LEASED,
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
 * The server's code for "nothing has decided what this host can hold yet".
 * Exported for the reason {@link ACCELERATOR_UNREADABLE} is: one spelling.
 */
export const CAPABILITY_UNDECIDED = 'capability_undecided';

/**
 * 503 `capability_undecided`: this server has no capability record — its config
 * was written before `crucible capability` ran, or by a build that predates it.
 *
 * Its own type for the same one-conclusion reason as
 * {@link CrucibleAcceleratorUnreadable}. Absent is its own answer and the
 * server refuses to dress it up as an empty decision: empty rows would read as
 * "probed, and nothing fit", which is the opposite news. A client that catches
 * this knows the host has decided NOTHING — not that it can do nothing — and
 * the fix is the operator's (`crucible capability --write`), not a retry.
 * `GET /v1/info` still says what the server offers meanwhile.
 *
 * It is a {@link CrucibleServerError} — the status really is a 5xx and every
 * handler that catches one still catches this — with a narrower name for the
 * callers that need to act differently.
 */
export class CrucibleCapabilityUndecided extends CrucibleServerError {}

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

// ------------------------------------------- the capability document's route
//
// PHASE15-HOST.md section 3.3's last bullet, which is a READING RULE and not a
// default: a capability document in which NO row carries `route` comes from a
// server that predates phase 15, and every class on such a server IS local —
// that is a fact the document states by its own vintage, not a value this
// client fills in. A document in which SOME rows carry it and one does not is
// a defect, and so is a `route` this vocabulary does not have.

/** A document where some rows say `route` and one does not. Names the row. */
export const CAPABILITY_ROUTE_MISSING = 'capability_route_missing';
/** A `route` that is neither `local` nor `upstream`. */
export const CAPABILITY_ROUTE_UNKNOWN = 'capability_route_unknown';

// ------------------------------------ the voice document's needs_reference
//
// The SAME reading rule (PHASE15-HOST.md 3.3), applied to the field
// PHASE3-TTS.md section 2 added: a document in which NO voice row carries
// `needs_reference` comes from a server that predates the field, and a voice
// on such a server IS a checkpoint whose voice is in its weights — so every
// row reads `needsReference: false`, because that is the document's vintage
// speaking and not a default this client fills in per row. The rule holds for
// both documents voice rows arrive in: `/v1/voices` and the `tts` capability
// of `/v1/info`.

/** A document where some voice rows say `needs_reference` and one does not. Names the row. */
export const VOICES_NEEDS_REFERENCE_MISSING = 'voices_needs_reference_missing';
/** A `needs_reference` that is not a boolean. */
export const VOICES_NEEDS_REFERENCE_UNKNOWN = 'voices_needs_reference_unknown';

// ------------------------------------------- the catalog's removal refusals
//
// PHASE15-HOST.md 3.5a. Four names because they are four different things to
// do about: fix the id, stop asking, close what is holding it, or look at the
// path that would not go.

/** `DELETE /v1/catalog/{kind}/{id}`: no such kind, or no such id here (404). */
export const SUBJECT_UNKNOWN = 'subject_unknown';
/** It is a subject this server can hold and it does not hold it (409). */
export const SUBJECT_NOT_INSTALLED = 'subject_not_installed';
/** Resident, leased, or named by a running task (409). `details.who` says. */
export const SUBJECT_IN_USE = 'subject_in_use';
/** The files would not go (500). `details.path` is the one that refused. */
export const SUBJECT_REMOVE_FAILED = 'subject_remove_failed';

// ---------------------------------------------- the settings door's refusals
//
// PHASE15-HOST.md sections 3.2 and 3.4, as constants so a caller switches on a
// name this package spells rather than on a string literal it typed. Every one
// of them arrives as a `CrucibleRefused` (4xx) or a `CrucibleServerError`
// (5xx) whose `code` is one of these; they are not subclasses, because nothing
// about handling them differs from handling any other named refusal — what a
// window needs is the name and `details.field`, and both are already there.

/** `PUT /v1/settings`: a class that cannot run anywhere but this card. */
export const ROUTE_NOT_ROUTABLE = 'route_not_routable';
/** A route value (or a chat's `model`) that is not `<upstream>/<model>`. */
export const ROUTE_BAD_MODEL = 'route_bad_model';
/** A route whose upstream has no key/url. The server never stores one. */
export const ROUTE_UPSTREAM_UNCONFIGURED = 'route_upstream_unconfigured';
/** Removing an upstream a route still names. `details.classes` says which. */
export const UPSTREAM_IN_USE = 'upstream_in_use';
/** A name that is not one of the three. */
export const UNKNOWN_UPSTREAM = 'unknown_upstream';
/** An upstream handed the field it does not take (a `url` for `anthropic`). */
export const UPSTREAM_BAD_FIELD = 'upstream_bad_field';

/** A chat naming an upstream this server has no credential for (409). */
export const UPSTREAM_UNCONFIGURED = 'upstream_unconfigured';
/** The upstream refused. The message is the provider's own words. */
export const UPSTREAM_REJECTED = 'upstream_rejected';
/** The upstream did not answer. */
export const UPSTREAM_UNREACHABLE = 'upstream_unreachable';
/**
 * The upstream rate-limited it, passed through with `Retry-After`. **The
 * caller waits**: Crucible never retries a request that may already be billed,
 * and neither does this client.
 */
export const UPSTREAM_RATE_LIMITED = 'upstream_rate_limited';
/** A lease or a `load-model` naming an upstream model. */
export const LEASE_NOT_NEEDED = 'lease_not_needed';

/** The three `testUpstream` answers that are results rather than exceptions. */
export const UPSTREAM_TEST_REFUSALS = [
  UPSTREAM_UNREACHABLE,
  UPSTREAM_REJECTED,
  UPSTREAM_UNCONFIGURED,
] as const;
