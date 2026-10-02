import { decodeBase64 } from './base64.js';
import {
  CrucibleError,
  CrucibleProtocolError,
  CrucibleRefused,
  CrucibleUnreachable,
} from './errors.js';
import { queuePayload, requireSeconds } from './queue.js';
import { readSseFrames } from './sse.js';
import { asObject, bool, nullableBool, nullableNum, num, str, type Json } from './shape.js';
import type { QueueChoice, QueuePosition } from './types.js';

/** Everything `stream(...)` needs. */
export interface StreamOptions {
  /** The voice to speak in. */
  voice: string;
  /** The language every row of this session is spoken in. */
  language: string;
  /**
   * A stream runs inside a queue session: the client's own when it holds one, else one opened for
   * the stream, which waits in the line like any session and closes with the stream. This is that
   * session's `idle_s` (10..86400, the server's default 900): no row being said, no op and no
   * touch for this long closes it and the stream. Ignored inside the client's own session.
   */
  idleS?: number;
  /**
   * How the stream's session waits in the line. Left out (and the client says nothing), it waits up
   * to an hour; `{maxWaitS}` changes the wait; `false` refuses instead (`session_open`,
   * `server_busy`) when the server is not free now.
   */
  queue?: QueueChoice;
  /**
   * Called with the stream's place in the line whenever its queue session joins or moves, while
   * it waits; never once the session is open. Given, the open asks the server for a ticket rather
   * than a held-open request when the session has to wait, follows the session's own stream
   * through the line, and opens the stream in it once it opens. A server older than 1.0.82 holds
   * the request open as before and this is never called; neither is it inside this client's own
   * session or on a free server, where there is no line to wait in.
   */
  onQueue?: (position: QueuePosition) => void;
  /**
   * Aborts the open, which is held until the session is open and the voice resident. A session
   * waiting in the line for the stream leaves it.
   */
  signal?: AbortSignal;
}

/** One sub-sentence chunk of audio, decoded. */
export interface StreamAudio {
  readonly kind: 'audio';
  /** The row's id — the caller's own, from {@link TtsStreamSession.say}. */
  readonly id: string;
  /** Which chunk of this row it is. */
  readonly seq: number;
  /** Mono PCM16 at the session's `sampleRate`, ready to play or to write. */
  readonly pcm: Int16Array;
  /** Seconds of audio in `pcm`, as the server measured it in the same bytes. */
  readonly seconds: number;
}

/** A row finishing, whether it spoke its whole text or was stopped. */
export interface StreamRowDone {
  readonly kind: 'done';
  readonly id: string;
  readonly done: true;
  /** Seconds of audio actually delivered for this row. */
  readonly seconds: number;
  /** The characters the server sent to the engine — its own count, not a reply's. */
  readonly chars: number;
  /** `chars / seconds`, or null for a row that delivered no audio at all. */
  readonly charsPerSec: number | null;
  /** Whether generation stopped at the frame cap; null means narrator did not say, never false. */
  readonly capped: boolean | null;
  /** Whether this row was stopped rather than finished. */
  readonly cancelled: boolean;
  /** The silence the caller inserts after this row, in seconds; null when the row was cancelled. */
  readonly gapSec: number | null;
}

/** A row starting again, and everything it already sent being void. */
export interface StreamRestart {
  readonly kind: 'restart';
  readonly id: string;
  readonly restart: true;
  /** Discard every chunk of this row below this seq. */
  readonly fromSeq: number;
  /** Why, in the server's words. */
  readonly reason: string;
}

/** One row failing on its own. */
export interface StreamRowError {
  readonly kind: 'error';
  readonly id: string;
  readonly code: string;
  readonly message: string;
}

/** A session frame this build does not know, carried rather than refused. */
export interface StreamUnknown {
  readonly kind: 'unknown';
  /** The event name the server actually sent. */
  readonly event: string;
  readonly data: Readonly<Record<string, unknown>>;
}

/** What iterating a session yields. */
export type StreamEvent =
  | StreamAudio
  | StreamRowDone
  | StreamRestart
  | StreamRowError
  | StreamUnknown;

/** What `cancel` did. */
export type CancelOutcome = 'dropped' | 'aborting_batch' | 'already_finished';

/** A live TTS session. */
export interface TtsStreamSession extends AsyncIterable<StreamEvent> {
  readonly sessionId: string;
  readonly voice: string;
  /** `<voice>@<revision>` — the merge that is speaking, not just its name. */
  readonly fingerprint: string;
  readonly sampleRate: number;
  /** The backend speaking. */
  readonly backend: string;
  /** The queue session the stream runs in. */
  readonly queueSessionId: string;
  /** True when that session was opened for this stream, so it closes with it. */
  readonly openedForStream: boolean;

  /** Speak one row and return its id; the audio comes out of the iterator. */
  say(id: string, text: string, take?: number): Promise<string>;

  /** Stop one row. */
  cancel(id: string): Promise<CancelOutcome>;

  /** Stop every row. */
  cancelAll(): Promise<number>;

  /** Close the stream; a queue session opened for it closes with it, one the client opened stays. */
  close(): Promise<void>;
}

/** What the session needs from the client, and nothing more. */
export interface StreamTransport {
  readonly url: string;
  fetch(path: string, init: RequestInit, authenticated: boolean): Promise<Response>;
  failure(response: Response): Promise<CrucibleError>;
  json(path: string, init: RequestInit, where: string): Promise<Json>;
  /**
   * Follow a queue session waiting in the line until it opens, calling `onQueue` on every move.
   * Throws when it ends before it opens; on an abort it leaves the line and throws the abort.
   */
  untilOpen(
    queueSessionId: string,
    onQueue: (position: QueuePosition) => void,
    signal: AbortSignal | undefined,
  ): Promise<void>;
  /** Take a queue session out of the line, or close it if it opened; never throws. */
  leaveTheLine(queueSessionId: string): Promise<void>;
}

/** Asks `POST /v1/tts/stream` for a `202` ticket instead of a held-open request while it waits. */
const QUEUE_TICKET_HEADER = 'X-Crucible-Queue-Ticket';

/** Names the queue session a request is an item of. */
const SESSION_HEADER = 'X-Crucible-Session';

const REATTACH_BUDGET_MS = 16_000;

const REATTACH_DELAY_MS = 250;

const JSON_HEADERS = { 'Content-Type': 'application/json' } as const;

export async function openTtsStream(
  transport: StreamTransport,
  options: StreamOptions,
): Promise<TtsStreamSession> {
  const given = options as Partial<StreamOptions> | undefined;
  if (given === undefined || given === null) {
    throw new CrucibleError('stream(...) needs {voice, language}');
  }
  const voice = requireOption(given.voice, 'voice');
  const language = requireOption(given.language, 'language');
  const payload: Record<string, unknown> = { voice, language };
  if (given.idleS !== undefined) payload['idle_s'] = requireSeconds(given.idleS, 'idleS', '900 s');
  const queue = queuePayload(given.queue);
  if (queue !== null) payload['queue'] = queue;
  const onQueue = given.onQueue;
  if (onQueue !== undefined && typeof onQueue !== 'function') {
    throw new CrucibleError(`onQueue must be a function, got ${typeof onQueue}`);
  }
  const signal = given.signal;
  const opening = JSON.stringify(payload);
  const open = (headers: Record<string, string>): RequestInit => {
    const init: RequestInit = { method: 'POST', headers: { ...JSON_HEADERS, ...headers }, body: opening };
    if (signal !== undefined) init.signal = signal;
    return init;
  };
  const body =
    onQueue === undefined
      ? await transport.json('/v1/tts/stream', open({}), 'stream')
      : await openThroughTheLine(transport, open, onQueue, signal);
  const session = new Session(transport, {
    sessionId: str(body, 'session_id', 'stream'),
    voice: str(body, 'voice', 'stream'),
    fingerprint: str(body, 'fingerprint', 'stream'),
    sampleRate: num(body, 'sample_rate', 'stream'),
    backend: str(body, 'backend', 'stream'),
    queueSessionId: str(body, 'queue_session_id', 'stream'),
    openedForStream: bool(body, 'queue_session_opened_for_stream', 'stream'),
  });
  try {
    await session.attach();
  } catch (cause) {
    await session.discard();
    throw cause;
  }
  return session;
}

/**
 * The open with a ticket: a `201` is the stream (the server was free, or an older server held the
 * request open); a `202` names the queue session opened for it, which is followed through the line
 * and then named by a second open, which claims it for the stream.
 */
async function openThroughTheLine(
  transport: StreamTransport,
  open: (headers: Record<string, string>) => RequestInit,
  onQueue: (position: QueuePosition) => void,
  signal: AbortSignal | undefined,
): Promise<Json> {
  const first = await answerOf(transport, open({ [QUEUE_TICKET_HEADER]: '1' }));
  if (first.status !== 202) return first.body;
  const queueSessionId = str(first.body, 'queue_session_id', 'stream ticket');
  await transport.untilOpen(queueSessionId, onQueue, signal);
  try {
    signal?.throwIfAborted();
    const second = await answerOf(transport, open({ [SESSION_HEADER]: queueSessionId }));
    if (second.status === 202) {
      throw new CrucibleProtocolError(
        `the server answered the open inside queue session ${queueSessionId} with another ` +
          'ticket; a stream opened inside an open session never waits for one',
      );
    }
    return second.body;
  } catch (cause) {
    // The session was opened for this stream, and no stream will claim it now.
    await transport.leaveTheLine(queueSessionId);
    throw cause;
  }
}

async function answerOf(
  transport: StreamTransport,
  init: RequestInit,
): Promise<{ status: number; body: Json }> {
  const response = await transport.fetch('/v1/tts/stream', init, true);
  if (!response.ok) throw await transport.failure(response);
  const text = await response.text();
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new CrucibleProtocolError(`stream did not return JSON: ${text.slice(0, 200)}`);
  }
  return { status: response.status, body: asObject(value, 'stream') };
}

type Pumped = StreamEvent | { readonly kind: 'ready' };

interface Identity {
  sessionId: string;
  voice: string;
  fingerprint: string;
  sampleRate: number;
  backend: string;
  queueSessionId: string;
  openedForStream: boolean;
}

class Session implements TtsStreamSession {
  readonly sessionId: string;
  readonly voice: string;
  readonly fingerprint: string;
  readonly sampleRate: number;
  readonly backend: string;
  readonly queueSessionId: string;
  readonly openedForStream: boolean;

  readonly #transport: StreamTransport;
  readonly #pump: AsyncGenerator<Pumped, void, undefined>;
  #closed = false;
  #ready = false;
  #closedReason: string | null = null;

  constructor(transport: StreamTransport, identity: Identity) {
    this.#transport = transport;
    this.sessionId = identity.sessionId;
    this.voice = identity.voice;
    this.fingerprint = identity.fingerprint;
    this.sampleRate = identity.sampleRate;
    this.backend = identity.backend;
    this.queueSessionId = identity.queueSessionId;
    this.openedForStream = identity.openedForStream;
    this.#pump = this.#run();
  }

  async attach(): Promise<void> {
    let step = await this.#pump.next();
    while (step.done !== true && step.value.kind === 'unknown') {
      step = await this.#pump.next();
    }
    if (step.done === true) {
      throw new CrucibleProtocolError(
        `session ${this.sessionId} closed before it was ready` +
          (this.#closedReason === null ? '' : `: ${this.#closedReason}`),
      );
    }
    if (step.value.kind !== 'ready') {
      throw new CrucibleProtocolError(
        `session ${this.sessionId}'s stream began with a ${step.value.kind} frame ` +
          'rather than ready',
      );
    }
  }

  async discard(): Promise<void> {
    await this.#pump.return(undefined).catch(() => undefined);
    await this.close().catch(() => undefined);
  }

  async say(id: string, text: string, take = 0): Promise<string> {
    const row = requireOption(id, 'id');
    const spoken = requireOption(text, 'text');
    if (!Number.isInteger(take) || take < 0) {
      throw new CrucibleError(`take must be a non-negative integer, got ${String(take)}`);
    }
    const body = await this.#op({ op: 'say', id: row, text: spoken, take }, 'say');
    return str(body, 'id', 'say');
  }

  async cancel(id: string): Promise<CancelOutcome> {
    const row = requireOption(id, 'id');
    const body = await this.#op({ op: 'cancel', id: row }, 'cancel');
    const outcome = str(body, 'outcome', 'cancel');
    if (outcome !== 'dropped' && outcome !== 'aborting_batch' && outcome !== 'already_finished') {
      throw new CrucibleProtocolError(
        `cancel answered with outcome "${outcome}", which this client does not know`,
      );
    }
    return outcome;
  }

  async cancelAll(): Promise<number> {
    const body = await this.#op({ op: 'cancel_all' }, 'cancelAll');
    return num(body, 'cancelled', 'cancelAll');
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    const response = await this.#transport.fetch(
      `/v1/tts/stream/${encodeURIComponent(this.sessionId)}`,
      { method: 'DELETE' },
      true,
    );
    if (!response.ok) {
      const failure = await this.#transport.failure(response);
      if (failure instanceof CrucibleRefused && failure.code === 'unknown_session') return;
      throw failure;
    }
    await response.text();
  }

  async #op(op: Json, where: string): Promise<Json> {
    return this.#transport.json(
      `/v1/tts/stream/${encodeURIComponent(this.sessionId)}`,
      { method: 'POST', headers: JSON_HEADERS, body: JSON.stringify(op) },
      where,
    );
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<StreamEvent, void, undefined> {
    for (;;) {
      const step = await this.#pump.next();
      if (step.done === true) return;
      if (step.value.kind === 'ready') {
        throw new CrucibleProtocolError(`session ${this.sessionId} sent ready twice`);
      }
      yield step.value;
    }
  }

  async *#run(): AsyncGenerator<Pumped, void, undefined> {
    let delivered = 0;
    let droppedAt: number | null = null;

    const reattachOrGiveUp = async (why: string): Promise<void> => {
      const now = Date.now();
      if (droppedAt === null) droppedAt = now;
      if (now - droppedAt > REATTACH_BUDGET_MS) {
        throw new CrucibleUnreachable(
          this.#transport.url,
          `the event stream for session ${this.sessionId} ${why} after event ` +
            `${delivered}, and could not be reattached within the ` +
            `${Math.round(REATTACH_BUDGET_MS / 1000)}s grace window`,
        );
      }
      await delay(REATTACH_DELAY_MS);
    };

    for (;;) {
      let stream: ReadableStream<Uint8Array>;
      try {
        stream = await this.#attach(delivered);
      } catch (cause) {
        if (!(cause instanceof CrucibleUnreachable)) throw cause;
        await reattachOrGiveUp('could not be reached');
        continue;
      }

      let readFailure: unknown = null;
      try {
        for await (const frame of readSseFrames(stream)) {
          delivered = readFrameId(frame.lastEventId, delivered);
          droppedAt = null;
          const event = frame.event ?? 'message';
          if (event === 'closed') {
            this.#closed = true;
            this.#closedReason = str(asObject(parse(frame.data, event), event), 'reason', event);
            return;
          }
          yield this.#read(event, frame.data);
        }
      } catch (cause) {
        if (cause instanceof CrucibleError) throw cause;
        readFailure = cause;
      } finally {
        await stream.cancel().catch(() => undefined);
      }
      if (readFailure !== null) {
        if (this.#closed) return;
        await reattachOrGiveUp('was cut off');
        continue;
      }

      if (this.#closed) return;
      await reattachOrGiveUp('ended without a closed frame');
    }
  }

  async #attach(delivered: number): Promise<ReadableStream<Uint8Array>> {
    const headers: Record<string, string> = { Accept: 'text/event-stream' };
    if (delivered > 0) headers['Last-Event-ID'] = String(delivered);
    const response = await this.#transport.fetch(
      `/v1/tts/stream/${encodeURIComponent(this.sessionId)}/events`,
      { method: 'GET', headers },
      true,
    );
    if (!response.ok) throw await this.#transport.failure(response);
    const stream = response.body;
    if (stream === null) {
      throw new CrucibleProtocolError(
        `the event stream for session ${this.sessionId} carried no body`,
      );
    }
    return stream;
  }

  #read(event: string, data: string): Pumped {
    const body = asObject(parse(data, event), event);
    if (event === 'ready') {
      if (this.#ready) {
        throw new CrucibleProtocolError(`session ${this.sessionId} sent ready twice`);
      }
      const said = {
        voice: str(body, 'voice', 'ready'),
        fingerprint: str(body, 'fingerprint', 'ready'),
        sampleRate: num(body, 'sample_rate', 'ready'),
        backend: str(body, 'backend', 'ready'),
      };
      const opened = {
        voice: this.voice,
        fingerprint: this.fingerprint,
        sampleRate: this.sampleRate,
        backend: this.backend,
      };
      for (const key of ['voice', 'fingerprint', 'sampleRate', 'backend'] as const) {
        if (said[key] !== opened[key]) {
          throw new CrucibleProtocolError(
            `session ${this.sessionId} was opened with ${key} ${String(opened[key])} ` +
              `and its stream's ready says ${String(said[key])}`,
          );
        }
      }
      this.#ready = true;
      return { kind: 'ready' };
    }
    if (event === 'audio') {
      return {
        kind: 'audio',
        id: str(body, 'id', 'audio'),
        seq: num(body, 'seq', 'audio'),
        pcm: pcm16(decodeBase64(str(body, 'pcm_base64', 'audio'))),
        seconds: num(body, 'seconds', 'audio'),
      };
    }
    if (event === 'done') {
      return {
        kind: 'done',
        id: str(body, 'id', 'done'),
        done: true,
        seconds: num(body, 'seconds', 'done'),
        chars: num(body, 'chars', 'done'),
        charsPerSec: nullableNum(body, 'chars_per_sec', 'done'),
        capped: nullableBool(body, 'capped', 'done'),
        cancelled: bool(body, 'cancelled', 'done'),
        gapSec: nullableNum(body, 'gap_sec', 'done'),
      };
    }
    if (event === 'restart') {
      return {
        kind: 'restart',
        id: str(body, 'id', 'restart'),
        restart: true,
        fromSeq: num(body, 'from_seq', 'restart'),
        reason: str(body, 'reason', 'restart'),
      };
    }
    if (event === 'error') {
      const id = body['id'];
      const code = str(body, 'code', 'error');
      const message = str(body, 'message', 'error');
      if (typeof id === 'string') return { kind: 'error', id, code, message };
      this.#closed = true;
      throw new CrucibleProtocolError(
        `session ${this.sessionId} failed: ${code}: ${message}`,
      );
    }
    return { kind: 'unknown', event, data: body };
  }
}

/** Narrow a {@link StreamEvent} to a chunk of audio. */
export function isStreamAudio(event: StreamEvent): event is StreamAudio {
  return event.kind === 'audio';
}

/** Narrow a {@link StreamEvent} to a row retiring. */
export function isStreamDone(event: StreamEvent): event is StreamRowDone {
  return event.kind === 'done';
}

/** Narrow a {@link StreamEvent} to a row starting again. */
export function isStreamRestart(event: StreamEvent): event is StreamRestart {
  return event.kind === 'restart';
}

/** Narrow a {@link StreamEvent} to a row failing. */
export function isStreamRowError(event: StreamEvent): event is StreamRowError {
  return event.kind === 'error';
}

function requireOption(value: unknown, name: string): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new CrucibleError(`${name} is required and must be a non-empty string`);
  }
  return value;
}

function parse(data: string, where: string): unknown {
  try {
    return JSON.parse(data);
  } catch {
    throw new CrucibleProtocolError(`the ${where} frame's data is not JSON: ${data.slice(0, 200)}`);
  }
}

function readFrameId(raw: string | null, previous: number): number {
  if (raw === null) {
    throw new CrucibleProtocolError('a session frame arrived with no id');
  }
  const id = Number(raw);
  if (!Number.isInteger(id) || id <= previous) {
    throw new CrucibleProtocolError(`event id ${raw} does not follow ${previous}`);
  }
  return id;
}

function pcm16(bytes: Uint8Array): Int16Array {
  if (bytes.length % 2 !== 0) {
    throw new CrucibleProtocolError(
      `an audio frame carried ${bytes.length} bytes, which is not a whole number of ` +
        '16-bit samples',
    );
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const samples = new Int16Array(bytes.length / 2);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = view.getInt16(index * 2, true);
  }
  return samples;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}
