import { CrucibleClient, CrucibleSession } from './client.js';
import {
  CrucibleConfigError,
  CrucibleFleetUnavailable,
  CrucibleSessionClosed,
  CrucibleUnreachable,
  type FleetServerReason,
} from './errors.js';
import { requireSeconds } from './queue.js';
import type { QueuePosition, SessionOptions } from './types.js';

/** The feature a server lists in `GET /v1/info` when it has queue sessions. */
const QUEUE_SESSIONS_FEATURE = 'queue.sessions';
/** How long a server has to say what it can serve before it is left out, by default. */
const DEFAULT_PROBE_TIMEOUT_MS = 10_000;
/** A loser's close is retried this many times across weather before the server's idle ends it. */
const LOSER_CLOSE_ATTEMPTS = 3;
const LOSER_CLOSE_PAUSE_MS = 1_000;

/** Options for {@link fleetSession}. */
export interface FleetSessionOptions {
  /** The capability class the run is for, as `X-Crucible-Act` names it. Required. */
  readonly act: string;
  /**
   * A model to have resident when the session opens. Only servers whose `GET /v1/models` lists it,
   * supported on their backend and installed, are asked.
   */
  readonly model?: string;
  /** The session's idle close, as {@link SessionOptions.idleS}; every server is asked with it. */
  readonly idleS?: number;
  /**
   * How long the whole fleet may wait for a session to open: 10..86400 seconds. Every server is
   * asked with it, and the fleet stops waiting when it runs out (throwing
   * {@link CrucibleFleetUnavailable}). Leave it out for each server's own default (an hour).
   */
  readonly maxWaitS?: number;
  /** Called whenever a server reports its session's place in its line. */
  readonly onQueue?: (update: FleetQueueUpdate) => void;
  /** Aborts the wait on every server: each session asked for leaves its line, and the abort is thrown. */
  readonly signal?: AbortSignal;
  /**
   * How long each server has to answer `GET /v1/info` (and `GET /v1/models`, with `model`) before
   * it is left out as not answering, in milliseconds. Default 10 000.
   */
  readonly probeTimeoutMs?: number;
}

/** One server's place, as {@link FleetQueueUpdate.places} lists it. */
export interface FleetPlace {
  readonly client: CrucibleClient;
  /** Its session's place in that server's line; null until the server has said. */
  readonly position: QueuePosition | null;
}

/** What {@link FleetSessionOptions.onQueue} hears: the server that moved, and where all of them stand. */
export interface FleetQueueUpdate {
  /** The server whose line moved. */
  readonly client: CrucibleClient;
  readonly position: QueuePosition;
  /** Every server still waiting to open, in the order the fleet was given. */
  readonly places: readonly FleetPlace[];
}

/** A server that dropped out of the race, and why. */
export interface FleetDropout extends FleetServerReason {
  readonly client: CrucibleClient;
}

/** What {@link fleetSession} answers: the open session, and the server it is on. */
export interface FleetSessionResult {
  /** The open session, on the server that opened first. Close it when the run is done. */
  readonly session: CrucibleSession;
  /** The server it is on: one of the clients the fleet was given. */
  readonly client: CrucibleClient;
  /** That client's index in the list the fleet was given. */
  readonly index: number;
  /** The servers that dropped out before the session opened, and why. */
  readonly dropouts: readonly FleetDropout[];
}

/**
 * Ask a fleet of servers for one queue session, and take the first that opens.
 *
 * Every server that can serve the request (it lists `queue.sessions`, and with `model`, has that
 * model supported and installed) is asked at once; one that cannot, or does not answer within
 * `probeTimeoutMs`, is left out with its reason. The first session to OPEN wins. Every other one
 * is taken out of its line the moment that happens, and one that opened in the same instant is
 * closed at once, so a loser never holds its machine.
 *
 * `onQueue` hears each server's place as it moves, so an app can show "2nd on the PC, 1st on the
 * Mac". Aborting `signal` takes every session out of its line and throws the abort. When no server
 * can serve, every one that could ended its session before it opened, or `maxWaitS` ran out, it
 * throws {@link CrucibleFleetUnavailable} naming each server and why. There is no preference
 * between servers and nothing is pre-empted: whichever opens first wins.
 *
 * ```ts
 * const { session, client } = await fleetSession([pc, mac], { act: 'analysis', model: 'qwen3.5-9b' });
 * try { ... } finally { await session.close(); }
 * ```
 */
export async function fleetSession(
  clients: readonly CrucibleClient[],
  options: FleetSessionOptions,
): Promise<FleetSessionResult> {
  const fleet = readFleet(clients);
  const given = options as Partial<FleetSessionOptions> | undefined;
  if (given === undefined || given === null) {
    throw new CrucibleConfigError('options', 'fleetSession(clients, ...) needs {act}');
  }
  if (typeof given.act !== 'string' || given.act.trim() === '') {
    throw new CrucibleConfigError('act', 'fleetSession(...) needs {act}, the capability class the run is for');
  }
  if (given.model !== undefined && (typeof given.model !== 'string' || given.model.trim() === '')) {
    throw new CrucibleConfigError('model', `must be a model id, got ${String(given.model)}`);
  }
  if (given.idleS !== undefined) requireSeconds(given.idleS, 'idleS', '300 s');
  const maxWaitS =
    given.maxWaitS === undefined ? undefined : requireSeconds(given.maxWaitS, 'maxWaitS', 'an hour');
  if (given.onQueue !== undefined && typeof given.onQueue !== 'function') {
    throw new CrucibleConfigError('onQueue', `must be a function, got ${typeof given.onQueue}`);
  }
  const probeTimeoutMs = given.probeTimeoutMs ?? DEFAULT_PROBE_TIMEOUT_MS;
  if (!Number.isFinite(probeTimeoutMs) || probeTimeoutMs <= 0) {
    throw new CrucibleConfigError(
      'probeTimeoutMs',
      `is ${String(given.probeTimeoutMs)}; a probe's clock is a positive number of milliseconds`,
    );
  }
  const signal = given.signal;
  signal?.throwIfAborted();

  const ask: SessionOptions = {
    act: given.act,
    ...(given.model === undefined ? {} : { model: given.model }),
    ...(given.idleS === undefined ? {} : { idleS: given.idleS }),
    ...(maxWaitS === undefined ? {} : { maxWaitS }),
  };
  return new Race(fleet, ask, given.onQueue, probeTimeoutMs, maxWaitS, signal).run();
}

function readFleet(clients: readonly CrucibleClient[]): readonly CrucibleClient[] {
  if (!Array.isArray(clients) || clients.length === 0) {
    throw new CrucibleConfigError('clients', 'fleetSession needs at least one CrucibleClient');
  }
  const urls = new Set<string>();
  clients.forEach((client: unknown, index) => {
    if (!(client instanceof CrucibleClient)) {
      throw new CrucibleConfigError(`clients[${index}]`, 'is not a CrucibleClient');
    }
    if (client instanceof CrucibleSession) {
      throw new CrucibleConfigError(
        `clients[${index}]`,
        `is queue session ${client.id} already; give the fleet the plain clients`,
      );
    }
    if (urls.has(client.url)) {
      throw new CrucibleConfigError(
        `clients[${index}]`,
        `${client.url} is in the fleet twice; a server is asked once`,
      );
    }
    urls.add(client.url);
  });
  return clients;
}

type Entry =
  | { readonly state: 'probing' }
  | { readonly state: 'waiting'; readonly position: QueuePosition | null }
  | { readonly state: 'out' };

/** One call of {@link fleetSession}: its servers, and how it ends. */
class Race {
  /** Ends everything this race started: probes, and every session still waiting. */
  readonly #stop = new AbortController();
  readonly #entries: Entry[];
  readonly #dropouts: FleetDropout[] = [];
  #finished = false;
  #resolve: (result: FleetSessionResult) => void = () => undefined;
  #reject: (error: unknown) => void = () => undefined;
  #timer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    readonly fleet: readonly CrucibleClient[],
    readonly ask: SessionOptions,
    readonly onQueue: ((update: FleetQueueUpdate) => void) | undefined,
    readonly probeTimeoutMs: number,
    readonly maxWaitS: number | undefined,
    readonly signal: AbortSignal | undefined,
  ) {
    this.#entries = fleet.map(() => ({ state: 'probing' }));
  }

  run(): Promise<FleetSessionResult> {
    const result = new Promise<FleetSessionResult>((resolve, reject) => {
      this.#resolve = resolve;
      this.#reject = reject;
    });
    this.signal?.addEventListener('abort', this.#onAbort, { once: true });
    if (this.maxWaitS !== undefined) {
      this.#timer = setTimeout(() => this.#outOfTime(), this.maxWaitS * 1_000);
    }
    this.fleet.forEach((client, index) => void this.#enter(client, index));
    return result;
  }

  readonly #onAbort = (): void => {
    if (this.signal !== undefined) this.#fail(this.signal.reason);
  };

  async #enter(client: CrucibleClient, index: number): Promise<void> {
    let cannot: string | null;
    try {
      cannot = await this.#probe(client);
    } catch (error) {
      if (this.#finished) return;
      cannot = describe(error, this.probeTimeoutMs);
    }
    if (this.#finished) return;
    if (cannot !== null) {
      this.#drop(index, 'probe', cannot);
      return;
    }
    this.#entries[index] = { state: 'waiting', position: null };
    let session: CrucibleSession;
    try {
      session = await client.session({
        ...this.ask,
        signal: this.#stop.signal,
        onQueue: (position) => this.#moved(index, position),
      });
    } catch (error) {
      if (this.#finished) return; // the race is over; its own end took this one out of the line
      this.#drop(index, 'line', describe(error, this.probeTimeoutMs));
      return;
    }
    if (this.#finished) {
      void releaseLoser(session); // opened in the same instant as the winner, or after an abort
      return;
    }
    this.#finish(() =>
      this.#resolve({ session, client, index, dropouts: [...this.#dropouts] }),
    );
  }

  /** Why `client` cannot serve this race, or null when it can. */
  async #probe(client: CrucibleClient): Promise<string | null> {
    const clock = AbortSignal.any([this.#stop.signal, AbortSignal.timeout(this.probeTimeoutMs)]);
    const info = await client.info({ signal: clock });
    if (!info.features.includes(QUEUE_SESSIONS_FEATURE)) {
      return `it does not offer queue sessions (GET /v1/info features lacks ${QUEUE_SESSIONS_FEATURE})`;
    }
    const model = this.ask.model;
    if (model === undefined) return null;
    const models = await bounded(client.models(), clock);
    const row = models.find((entry) => entry.id === model);
    if (row === undefined) return `${model} is not in its catalogue (GET /v1/models)`;
    if (!row.backendSupported) {
      return `${model} is not supported on its backend${row.reason === null ? '' : ` (${row.reason})`}`;
    }
    if (!row.installed) return `${model} is not installed there`;
    return null;
  }

  #moved(index: number, position: QueuePosition): void {
    if (this.#finished) return;
    this.#entries[index] = { state: 'waiting', position };
    if (this.onQueue === undefined) return;
    const places: FleetPlace[] = [];
    this.#entries.forEach((entry, at) => {
      if (entry.state === 'waiting') places.push({ client: this.fleet[at]!, position: entry.position });
    });
    try {
      this.onQueue({ client: this.fleet[index]!, position, places });
    } catch (error) {
      this.#fail(error); // the app's own callback threw: that is its bug, surfaced by name
    }
  }

  #drop(index: number, stage: 'probe' | 'line', reason: string): void {
    this.#entries[index] = { state: 'out' };
    const client = this.fleet[index]!;
    this.#dropouts.push({ client, url: client.url, stage, reason });
    if (this.#entries.some((entry) => entry.state !== 'out')) return;
    const anyAsked = this.#dropouts.some((entry) => entry.stage === 'line');
    this.#fail(
      new CrucibleFleetUnavailable(
        anyAsked
          ? `no server in the fleet opened a queue session for ${this.#what()}`
          : `no server in the fleet can serve a queue session for ${this.#what()}`,
        this.#inFleetOrder(this.#dropouts),
      ),
    );
  }

  #outOfTime(): void {
    const still: FleetServerReason[] = [];
    this.#entries.forEach((entry, index) => {
      if (entry.state === 'out') return;
      still.push({
        url: this.fleet[index]!.url,
        stage: entry.state === 'probing' ? 'probe' : 'line',
        reason:
          entry.state === 'probing'
            ? `it had not said what it can serve when maxWaitS (${this.maxWaitS} s) ran out`
            : `its session was still waiting${placeOf(entry.position)} when maxWaitS (${this.maxWaitS} s) ran out`,
      });
    });
    this.#fail(
      new CrucibleFleetUnavailable(
        `no queue session opened for ${this.#what()} within maxWaitS (${this.maxWaitS} s)`,
        this.#inFleetOrder([...this.#dropouts, ...still]),
      ),
    );
  }

  #inFleetOrder(reasons: readonly FleetServerReason[]): FleetServerReason[] {
    const order = new Map(this.fleet.map((client, index) => [client.url, index]));
    return [...reasons]
      .sort((a, b) => order.get(a.url)! - order.get(b.url)!)
      .map(({ url, stage, reason }) => ({ url, stage, reason }));
  }

  #what(): string {
    return this.ask.model === undefined
      ? `act ${this.ask.act}`
      : `act ${this.ask.act} with ${this.ask.model}`;
  }

  #fail(error: unknown): void {
    this.#finish(() => this.#reject(error));
  }

  /** The race's one ending: stop every server still in it, then answer. */
  #finish(answer: () => void): void {
    if (this.#finished) return;
    this.#finished = true;
    if (this.#timer !== null) clearTimeout(this.#timer);
    this.signal?.removeEventListener('abort', this.#onAbort);
    // Every session still waiting leaves its line (the SDK's session() owns that DELETE, and the
    // one for a ticket still in flight); a probe still running is cut off.
    this.#stop.abort(new Error('the fleet session race is over'));
    answer();
  }
}

/** `promise`, or `signal`'s reason the moment it aborts. */
function bounded<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(signal.reason);
  return new Promise<T>((resolve, reject) => {
    const stop = (): void => reject(signal.reason);
    signal.addEventListener('abort', stop, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener('abort', stop);
        resolve(value);
      },
      (error: unknown) => {
        signal.removeEventListener('abort', stop);
        reject(error);
      },
    );
  });
}

/** A sentence for why a server dropped out. */
function describe(error: unknown, probeTimeoutMs: number): string {
  if (error instanceof DOMException && error.name === 'TimeoutError') {
    return `it did not answer within ${probeTimeoutMs} ms`;
  }
  if (error instanceof CrucibleSessionClosed) {
    return `its session ended before it opened (${error.reason}): ${error.serverMessage}`;
  }
  if (error instanceof Error) return error.message;
  return String(error);
}

function placeOf(position: QueuePosition | null): string {
  return position === null ? '' : ` at ${position.position} of ${position.of}`;
}

/**
 * Close a session that lost the race. Weather (the server not answering for a moment) is retried
 * within a small budget; past it, the server's own idle close (`idleS`) ends the session.
 */
async function releaseLoser(session: CrucibleSession): Promise<void> {
  for (let attempt = 1; ; attempt += 1) {
    try {
      await session.close();
      return;
    } catch (error) {
      if (!(error instanceof CrucibleUnreachable) || attempt >= LOSER_CLOSE_ATTEMPTS) return;
      await new Promise((resolve) => setTimeout(resolve, LOSER_CLOSE_PAUSE_MS));
    }
  }
}
