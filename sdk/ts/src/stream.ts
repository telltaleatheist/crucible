/**
 * `stream()` — the live TTS session. PHASE3-TTS.md section 7.
 *
 * The Listen path, the in-app Play button, and the browser extension. It is
 * four HTTP calls and one SSE stream, and it is the reason the door is not a
 * WebSocket: **Node 20 has no global `WebSocket`** — it is behind
 * `--experimental-websocket` there and only ordinary in 22 — and Electron 33,
 * which is what BookForge ships, bundles Node 20.18 while this SDK runs in the
 * **main** process, where the renderer's browser `WebSocket` is not in scope.
 * Raising the floor to Node 22 does not help, because Electron's Node is
 * Electron's; adding `ws` breaks the zero-dependency rule this package has had
 * since phase 1; and an RFC 6455 client by hand is two hundred lines of
 * masking, fragmentation and close codes inside a client whose whole job is to
 * be boring.
 *
 * So this file is `fetch`, the same `readSseFrames` {@link CrucibleClient.events}
 * already uses, and nothing else.
 *
 * What the caller gets
 * --------------------
 * A session object that is also an `AsyncIterable`. Iterate it for the audio;
 * call `say`, `cancel`, `cancelAll` and `close` on it from anywhere. The
 * frames arrive interleaved across rows — a `done` for one row while another is
 * still emitting — because that is what the server sends and reordering them
 * here would defeat the whole point of a sub-sentence stream.
 *
 * It reattaches on its own, and that is a deliberate difference
 * ------------------------------------------------------------
 * `events()` never reconnects: a job goes on running whether anybody is
 * watching, its event log is kept for the life of the job, and a client can
 * resume whenever it likes. A session is the opposite. Its grace window is
 * fifteen seconds, and missing it costs the session, the rows in flight and the
 * listener's place in the paragraph. So a dropped stream is reattached here,
 * with `Last-Event-ID`, for as long as the window can still be open — which is
 * the behaviour the server's SSE door was chosen for, and a behaviour every
 * caller would otherwise have to write again and get wrong.
 *
 * A **refusal** is never retried. `unknown_session` and `replay_unavailable`
 * are the server saying the window has closed or the audio would have a hole in
 * it, and the correct answer to both is to tell the caller rather than to try
 * again quietly.
 */

import { decodeBase64 } from './base64.js';
import {
  CrucibleError,
  CrucibleProtocolError,
  CrucibleRefused,
  CrucibleUnreachable,
} from './errors.js';
import { readSseFrames } from './sse.js';
import { asObject, bool, nullableBool, nullableNum, num, str, type Json } from './shape.js';

/** Everything `stream(...)` needs. Neither field has a default. */
export interface StreamOptions {
  /**
   * The voice to speak in. It must ALREADY be resident: the streaming door
   * never loads, exactly as chat never loads, and refuses `voice_not_resident`
   * naming what is resident instead. Only a render job loads (section 6).
   */
  voice: string;
  /** The language every row of this session is spoken in. */
  language: string;
}

/** One sub-sentence chunk of audio, decoded. */
export interface StreamAudio {
  readonly kind: 'audio';
  /** The row's id — the caller's own, from {@link TtsStreamSession.say}. */
  readonly id: string;
  /**
   * Which chunk of this row it is. Strictly increasing within a row and never
   * restarting, so it stays a total order across a {@link StreamRestart}.
   */
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
  /**
   * Whether generation stopped at the frame cap rather than because the model
   * finished — the difference between a long sentence and a runaway.
   *
   * **`null` means "narrator did not say" and is never to be read as `false`.**
   * Against the pinned narrator it is always null: the cap it computes never
   * leaves the engine. A runaway reported as "not capped" is exactly the
   * failure this field exists to prevent.
   */
  readonly capped: boolean | null;
  /** Whether this row was stopped rather than finished. */
  readonly cancelled: boolean;
}

/**
 * A row starting again, and everything it already sent being void.
 *
 * narrator has **no per-row cancel**: its `cancel` aborts everything in flight.
 * So cancelling one row in a batch kills its neighbours too, and the server
 * resubmits the ones nobody cancelled rather than reporting them as cancelled
 * or failed. This frame is what says so: every {@link StreamAudio} for this id
 * with `seq < fromSeq` must be discarded, and the row's real audio begins at
 * `fromSeq`.
 *
 * It cannot happen on a `higgs-v3` voice, whose measured batch width is 1 — the
 * in-flight row IS the batch, so there is never a survivor. It can on Orpheus,
 * where the ramp dispatches eight.
 */
export interface StreamRestart {
  readonly kind: 'restart';
  readonly id: string;
  readonly restart: true;
  /** Discard every chunk of this row below this seq. */
  readonly fromSeq: number;
  readonly reason: string;
}

/** One row failing on its own. Its neighbours are unaffected. */
export interface StreamRowError {
  readonly kind: 'error';
  readonly id: string;
  readonly code: string;
  readonly message: string;
}

/** What iterating a session yields. */
export type StreamEvent = StreamAudio | StreamRowDone | StreamRestart | StreamRowError;

/** What `cancel` did. See {@link TtsStreamSession.cancel}. */
export type CancelOutcome = 'dropped' | 'aborting_batch' | 'already_finished';

/** A live TTS session. Iterate it for the audio; call the ops from anywhere. */
export interface TtsStreamSession extends AsyncIterable<StreamEvent> {
  readonly sessionId: string;
  readonly voice: string;
  /** `<voice>@<revision>` — the merge that is speaking, not just its name. */
  readonly fingerprint: string;
  readonly sampleRate: number;
  readonly backend: string;

  /**
   * Speak one row. Returns its id, **not its audio**: the audio comes out of
   * the iterator. A session that has never had its stream iterated refuses
   * this by name rather than generating into nothing.
   *
   * `take` defaults to 0 here and has no default on the wire. 0 is the engine's
   * own sampling, which is what asking for nothing gets; anything above it is
   * refused as `sampling_not_wired` until narrator grows a sampling channel.
   */
  say(id: string, text: string, take?: number): Promise<string>;

  /**
   * Stop one row.
   *
   * **What it costs depends on where the row is**, and the answer says which:
   * `dropped` means it had not been handed to the engine and cost nothing;
   * `aborting_batch` means the engine is generating it and its whole batch is
   * being thrown away to stop it, so its neighbours will arrive again behind a
   * {@link StreamRestart}; `already_finished` means it retired first, which is
   * the ordinary race on a live connection and not an error.
   */
  cancel(id: string): Promise<CancelOutcome>;

  /** Stop every row. Returns how many were still live. Nothing is restarted. */
  cancelAll(): Promise<number>;

  /** Close the session and free the voice. The iterator ends. */
  close(): Promise<void>;
}

/** What the session needs from the client, and nothing more. */
export interface StreamTransport {
  readonly url: string;
  fetch(path: string, init: RequestInit, authenticated: boolean): Promise<Response>;
  failure(response: Response): Promise<CrucibleError>;
  json(path: string, init: RequestInit, where: string): Promise<Json>;
}

/**
 * How long to keep reattaching a dropped stream before giving up.
 *
 * The server's grace window is 15 s (`crucible/ttsstream.py`'s
 * `GRACE_SECONDS`); after it the session is closed and every row in flight is
 * cancelled, so a reattach later than this cannot succeed and would only turn a
 * dead session into a slow error. A second of margin, because the window starts
 * when the server notices the drop and not when the client does.
 */
const REATTACH_BUDGET_MS = 16_000;

/** How long to wait between reattach attempts. Short: the budget is short. */
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
  const body = await transport.json(
    '/v1/tts/stream',
    { method: 'POST', headers: JSON_HEADERS, body: JSON.stringify({ voice, language }) },
    'stream',
  );
  return new Session(transport, {
    sessionId: str(body, 'session_id', 'stream'),
    voice: str(body, 'voice', 'stream'),
    fingerprint: str(body, 'fingerprint', 'stream'),
    sampleRate: num(body, 'sample_rate', 'stream'),
    backend: str(body, 'backend', 'stream'),
  });
}

interface Identity {
  sessionId: string;
  voice: string;
  fingerprint: string;
  sampleRate: number;
  backend: string;
}

class Session implements TtsStreamSession {
  readonly sessionId: string;
  readonly voice: string;
  readonly fingerprint: string;
  readonly sampleRate: number;
  readonly backend: string;

  readonly #transport: StreamTransport;
  #closed = false;

  constructor(transport: StreamTransport, identity: Identity) {
    this.#transport = transport;
    this.sessionId = identity.sessionId;
    this.voice = identity.voice;
    this.fingerprint = identity.fingerprint;
    this.sampleRate = identity.sampleRate;
    this.backend = identity.backend;
  }

  // ------------------------------------------------------------------ ops

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
      // Narrowed rather than passed through: each of the three names a
      // different cost, and a fourth is a contract change this client has not
      // been told about.
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
      // A session the server has already closed — the grace window ran out
      // while this client was deciding — is not a failure to close it.
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

  // ------------------------------------------------------------- the audio

  async *[Symbol.asyncIterator](): AsyncGenerator<StreamEvent, void, undefined> {
    let delivered = 0;
    let droppedAt: number | null = null;

    /** Give up, or wait and try again. Called for every kind of disconnection. */
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
        // A REFUSAL is the server's answer and travels straight back:
        // `unknown_session` means the window closed, `replay_unavailable` means
        // the audio would have a hole in it, and retrying either quietly is how
        // a client ends up playing a sentence that is missing its middle. Only
        // an unreachable server is a drop worth waiting out.
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
            return;
          }
          const yielded = this.#read(event, frame.data);
          if (yielded !== null) yield yielded;
        }
      } catch (cause) {
        // Sorting one kind of failure from the other, because they want
        // opposite answers. Anything this client raised — a frame it cannot
        // read, an id that does not follow, a session-wide `error` — is a fact
        // about the conversation and travels straight back. A socket that died
        // under the reader raises something else entirely (a TypeError from the
        // stream, a DOM error), and that is the drop this whole door was built
        // to survive, so it is caught and reattached rather than thrown.
        //
        // A consumer breaking out of its `for await` does NOT land here: a
        // return completion at a `yield` runs `finally` and skips `catch`.
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

      // The stream ended without a `closed` frame, which is a dropped
      // connection and not an ending. The server holds the session open for its
      // grace window precisely so this can be picked up again, and every frame
      // emitted while nobody was listening is replayed from `delivered`.
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

  /** One frame, as the caller's event — or null for one they need not see. */
  #read(event: string, data: string): StreamEvent | null {
    const body = asObject(parse(data, event), event);
    if (event === 'ready') {
      // The identity is already on the session object, and it is checked rather
      // than ignored: a `ready` naming another voice would mean this stream is
      // not the session that was opened.
      const voice = str(body, 'voice', 'ready');
      if (voice !== this.voice) {
        throw new CrucibleProtocolError(
          `session ${this.sessionId} was opened on ${this.voice} and its stream ` +
            `says ${voice}`,
        );
      }
      return null;
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
      // No id means the SESSION failed, not a row: narrator died, or said
      // something that is not a protocol message. Nothing more is coming, so
      // this throws rather than being yielded past.
      this.#closed = true;
      throw new CrucibleProtocolError(
        `session ${this.sessionId} failed: ${code}: ${message}`,
      );
    }
    // An event kind this client does not know. Thrown rather than skipped, for
    // the reason `events()` gives about the same case: a frame silently dropped
    // is a behaviour change nobody sees until a book is missing a sentence.
    throw new CrucibleProtocolError(
      `session ${this.sessionId} sent an event this client does not know: ${event}`,
    );
  }
}

// ------------------------------------------------------------------ helpers

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

/**
 * Bytes to samples, without assuming the caller's machine is little-endian.
 *
 * PCM16 on this wire is signed 16-bit **little-endian** — narrator's own
 * format, and the one ffmpeg is told on the render door (`-f s16le`). A plain
 * `new Int16Array(bytes.buffer)` would read it in the host's byte order, which
 * is right on x86 and arm64 and silently wrong anywhere else; and it would
 * throw outright on an odd offset, which a decoded base64 buffer is free to
 * have. A DataView says what it means.
 */
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
