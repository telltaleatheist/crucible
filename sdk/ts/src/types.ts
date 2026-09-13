/**
 * The shapes API v1 returns, hand-written from `docs/DESIGN.md` section 4 and
 * `crucible/api.py`. The e2e test is what keeps them honest.
 *
 * Convention: everything the client *models* is camelCase, because it is a
 * TypeScript API. The one exception is {@link Provenance}, which is the sidecar
 * document verbatim — clients must persist it beside the artifact (DESIGN.md
 * section 7), and rewriting its keys would corrupt the thing being persisted.
 */

/** The API contract version this client speaks. Sent as `X-Crucible-Api`. */
export const API_VERSION = 1;

/** `GET /v1/ping` — the only unauthenticated route. */
export interface Ping {
  /** Always `true`; anything else is a {@link CrucibleNotACrucible}. */
  readonly crucible: true;
  readonly name: string;
  readonly apiVersion: number;
}

/** One model a job type can serve, as advertised by `GET /v1/info`. */
export interface ModelDescriptor {
  readonly id: string;
  readonly revision: string;
  readonly source: string;
  readonly resident: boolean;
  readonly vramBytes: number;
}

/**
 * One job type this server offers, with the models it can serve.
 *
 * Most capabilities describe their models with DESIGN.md section 4's row. Two
 * are exceptions the contract makes on purpose, both for the same reason: their
 * rows are the rows of the route that lists them — `GET /v1/models` for `llm`
 * (PHASE2-LLM.md section 5), `GET /v1/voices` for `tts` (PHASE3-TTS.md section
 * 8) — the same shape from the same producer, so a model or a voice has one
 * description wherever a client finds it. Narrow with {@link isLlmCapability} or
 * {@link isTtsCapability} before reading a row.
 */
export type Capability = LlmCapability | TtsCapability | JobCapability;

/** Any capability other than `llm` and `tts`. */
export interface JobCapability {
  readonly jobType: string;
  readonly models: readonly ModelDescriptor[];
}

/** The `llm` capability: `GET /v1/models`' rows, carried inside `info()`. */
export interface LlmCapability {
  readonly jobType: 'llm';
  readonly models: readonly ModelInfo[];
}

/**
 * The `tts` capability: `GET /v1/voices`' rows, carried inside `info()`.
 *
 * The member is still called `models` — that is the key on the wire, and a
 * capability's models are whatever that job type serves. For `tts` the thing
 * served is a voice.
 */
export interface TtsCapability {
  readonly jobType: 'tts';
  readonly models: readonly VoiceInfo[];
}

/** Narrow a capability to the `llm` one, whose rows are {@link ModelInfo}. */
export function isLlmCapability(capability: Capability): capability is LlmCapability {
  return capability.jobType === 'llm';
}

/** Narrow a capability to the `tts` one, whose rows are {@link VoiceInfo}. */
export function isTtsCapability(capability: Capability): capability is TtsCapability {
  return capability.jobType === 'tts';
}

/** The accelerator the server owns. */
export interface GpuInfo {
  readonly vendor: string;
  readonly name: string;
  readonly vramBytes: number;
}

/** `GET /v1/info`. */
export interface ServerInfo {
  readonly server: {
    readonly name: string;
    readonly version: string;
    readonly apiVersion: number;
  };
  readonly host: {
    readonly platform: string;
    readonly arch: string;
    /** `cuda-linux` or `mlx-darwin`. Windows is never a backend. */
    readonly backend: string;
    readonly gpu: GpuInfo;
  };
  /**
   * What this server will accept as a `type` in `POST /v1/jobs`.
   *
   * Not the same list as {@link ServerInfo.capabilities}, and deliberately so: a
   * capability says what the server can *serve*, this says what to *ask it
   * with*, and one capability can be operated by several job types. `llm` is a
   * capability; `load-model` and `unload-model` are what you post, and neither
   * is a capability of its own.
   */
  readonly jobTypes: readonly string[];
  /**
   * One entry per capability, not one per postable job type. A model or a voice
   * appears in exactly one of them, in exactly one shape — `load-model`,
   * `unload-model` and `llm` listed `qwen3.5-9b` three times in two shapes until
   * 2026-09-13, which is the thing PHASE2-LLM.md section 5 exists to forbid.
   */
  readonly capabilities: readonly Capability[];
}

/** `GET /v1/health`. */
export interface Health {
  readonly status: 'ok' | 'warming' | 'busy';
  readonly queueDepth: number;
  /**
   * The ids of whatever is on the card. Keeps its phase-2 name and its phase-2
   * shape — a list of ids — because every phase-2 client reads it and one id is
   * one id whatever kind of thing it names. What kind it is, is
   * {@link Health.residentKind}.
   */
  readonly residentModels: readonly string[];
  /**
   * What *sort* of thing holds the card: `"llm"`, `"tts"`, or `null` when
   * nothing does (PHASE3-TTS.md section 8). Since one card holds one thing and
   * that thing may now be a voice, this is which door to knock on — `chat()` on
   * a server with a voice resident is `model_not_resident`, and the id in
   * `residentModels` does not say so on its own.
   *
   * It is a plain string and not a union on purpose. The set grows: PHASE4's
   * aligner lands here beside the other two, and a client that throws a protocol
   * error on a kind it has not heard of would break on the server that added it.
   */
  readonly residentKind: string | null;
}

/** `POST /v1/uploads`. */
export interface UploadResult {
  readonly blobId: string;
  readonly bytes: number;
  readonly sha256: string;
}

/** One named input on a job: either an uploaded blob or bytes carried inline. */
export type JobInput = { readonly blobId: string } | { readonly inline: Uint8Array };

/** The body of `POST /v1/jobs`. */
export interface JobRequest {
  readonly type: string;
  /** Omit for a job type that serves no models; the server refuses a mismatch. */
  readonly model?: string;
  readonly params: Readonly<Record<string, unknown>>;
  readonly inputs: Readonly<Record<string, JobInput>>;
}

/** A job's lifecycle state. `done`, `failed` and `cancelled` are terminal. */
export type JobState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

/** The server's named refusal or failure, as carried on a job and in events. */
export interface JobFailure {
  readonly code: string;
  readonly message: string;
}

/** `GET /v1/jobs/{id}`. */
export interface JobStatus {
  readonly jobId: string;
  readonly type: string;
  readonly model: string | null;
  readonly status: JobState;
  readonly progress: number;
  /** 0 while running, 1-based place in line while queued, `null` once terminal. */
  readonly position: number | null;
  readonly error: JobFailure | null;
  readonly artifacts: readonly string[];
  readonly created: string;
  readonly started: string | null;
  readonly finished: string | null;
}

/** `DELETE /v1/jobs/{id}`. A running job ends `cancelled` at its next checkpoint. */
export interface CancelResult {
  readonly jobId: string;
  readonly status: 'cancelled' | 'cancelling';
}

/**
 * One SSE event from `GET /v1/jobs/{id}/events`.
 *
 * `id` is the server's monotonic event counter, starting at 1; hand the last one
 * you saw back as `lastEventId` to resume a stream without losing an event.
 */
export type JobEvent =
  | { readonly id: number; readonly event: 'queued'; readonly data: QueuedData }
  | { readonly id: number; readonly event: 'warming'; readonly data: WarmingData }
  | { readonly id: number; readonly event: 'progress'; readonly data: ProgressData }
  | { readonly id: number; readonly event: 'chunk'; readonly data: ChunkData }
  | { readonly id: number; readonly event: 'artifact'; readonly data: ArtifactData }
  | { readonly id: number; readonly event: 'done'; readonly data: DoneData }
  | { readonly id: number; readonly event: 'failed'; readonly data: FailedData }
  | { readonly id: number; readonly event: 'cancelled'; readonly data: CancelledData }
  | UnknownEvent;

/**
 * An event kind this build of the client does not know.
 *
 * The server's event vocabulary GROWS — `chunk` was added for `tts` without
 * moving `api_version`, on the stated ground that a client which does not know
 * the kind still sees every `progress`, `artifact` and `done` it saw before
 * (PHASE3-TTS.md section 6). That argument is only true if the client actually
 * survives the unknown frame, and until 2026-09-13 it did not: `readEvent`
 * narrowed against a closed list and threw, so an older client watching ANY job
 * on a newer server lost the whole stream at the first `chunk`.
 *
 * So an unknown kind arrives here instead, with its name and its parsed data
 * intact, and is **never terminal** — the stream runs on to its real ending.
 *
 * This is not a softening of the no-fallbacks rule, and the line is worth
 * stating: the client stays strict about every kind it CLAIMS to understand — a
 * `chunk` missing `capped` is still a protocol error — and tolerant only of
 * kinds it makes no claim about at all. Refusing to parse is honest; refusing to
 * continue is not.
 */
export interface UnknownEvent {
  readonly id: number;
  readonly event: 'unknown';
  /** The name the server actually sent. */
  readonly kind: string;
  readonly data: Readonly<Record<string, unknown>>;
}

export interface QueuedData {
  readonly position: number | null;
}

/**
 * A model is being loaded for this job: one line of the engine's own readiness,
 * streamed as it happens. PHASE2-LLM.md section 5 pins the payload to
 * `{message}`; a `warming` frame without one is a protocol error, not an empty
 * message.
 */
export interface WarmingData {
  readonly message: string;
}

export interface ProgressData {
  readonly fraction: number;
  readonly message: string;
  /**
   * Every other key the job type put on this frame, verbatim — server spelling,
   * server types, nothing invented and nothing dropped.
   *
   * `JobContext.progress(fraction, message, **extra)` lets a job type send its
   * own measurements beside the fraction, because a fraction is not always the
   * useful number: `asr` sends `{stage, processed_s, total_s, cues}` so that a
   * client shows a moving position six minutes into an eighteen-hour book while
   * the percentage is still rounding to zero (PHASE4-AUDIO.md section 3). Those
   * keys are one job type's vocabulary, not API v1's, so they are carried rather
   * than modelled — the same decision, for the same reason, as {@link Provenance}
   * keeping the server's key names.
   *
   * `{}` when the frame carried only `fraction` and `message`, which is every
   * frame from a job type that sends no measurements of its own. That is not a
   * substituted default: it is the true answer to "what else was on the frame".
   */
  readonly extra: Readonly<Record<string, unknown>>;
}

/**
 * One rendered chunk of a `tts` job, measured (PHASE3-TTS.md section 6).
 *
 * This is the **whole guard interface**. Crucible measures and reports; it
 * decides nothing, never retakes, never re-splits and never substitutes.
 * BookForge's PaceTracker is the thing that judges, and this event carries
 * everything it has to judge with. One arrives per chunk that produced audio,
 * and none at all for a row that failed — that row is named in a `progress` line
 * when it happens and again in `done`'s `failed` list.
 *
 * Three of the seven fields are the request's or the server's own arithmetic and
 * two come off narrator's wire, and the split is why two of them are nullable:
 *
 * - `index` and `take` are the request's, unchanged.
 * - `chars` is **Crucible's own count of the text it sent**, not a number read
 *   off the reply: a number a subprocess echoes back is a number a subprocess
 *   can get wrong, and this one is already known exactly.
 * - `seconds` is measured from the PCM that actually arrived and then compared
 *   with the duration narrator reported; a disagreement over 50 ms fails the row
 *   rather than being reported. `charsPerSec` is those two divided.
 */
export interface ChunkData {
  /** The client's own chunk index, and the name of its artifact (`<index>.flac`). */
  readonly index: number;
  /** The duration of the audio that arrived, measured from its bytes. */
  readonly seconds: number;
  /** How many characters were sent, counted by the server. */
  readonly chars: number;
  /** `chars / seconds`. The pace this chunk was actually read at. */
  readonly charsPerSec: number;
  /**
   * How many tokens the engine spent, or **`null` meaning "narrator did not
   * say"** — see {@link ChunkData.capped}, which is null for the same reason and
   * must be read with the same care.
   */
  readonly tokens: number | null;
  /**
   * Whether generation stopped because it hit the frame cap rather than because
   * the model finished — the difference between **a long sentence and a
   * runaway**, which a duration cannot tell you and which is the reason this
   * event exists at all.
   *
   * **`null` means "narrator did not say", and is never to be read as `false`.**
   * narrator does not put the frame cap on its wire at the pinned sha:
   * `serve/worker.py` sends `{i, format, data, duration, sampleRate}` for a
   * retiring row and the cap it computed never leaves the engine, so Crucible
   * publishes `null` — explicitly, as a key that is present — rather than
   * guessing. A client that treated that null as `false` would read **every
   * runaway as a long sentence**, which is precisely the failure this field
   * exists to prevent, and it would do it silently.
   *
   * So the null is in the type, and the reader refuses a `chunk` frame that
   * omits the key altogether: "narrator did not say" has to be something the
   * server said, not something the client inferred from an absence. Narrow it
   * explicitly — `capped === true` is a runaway, `capped === false` is a
   * finished sentence, `capped === null` is no measurement and must be handled
   * as one — never `if (chunk.capped)`.
   *
   * PHASE3-TTS.md section 6 records the owed change on narrator's side.
   */
  readonly capped: boolean | null;
  /** Which rung of the voice's take ladder this render asked for. */
  readonly take: number;
}

export interface ArtifactData {
  readonly name: string;
}

/**
 * The terminal `done` event's payload.
 *
 * Different job types finish with different news: a producing job (`echo`,
 * later `tts`, `vlm-pages`) reports the artifacts it wrote, and `load-model`
 * reports the model that is now resident (PHASE2-LLM.md section 5). There is no
 * discriminant inside the frame — the caller already knows which job it
 * submitted — so this is one record with both fields optional rather than a
 * union the caller would have to narrow before reading `artifacts`.
 *
 * It is still checked, not loose: `artifacts` must be an array of strings and
 * `resident` a string or `null` wherever either appears, and a `done` frame
 * carrying *neither* is a {@link CrucibleProtocolError}.
 */
export interface DoneData {
  /** What a producing job wrote. Fetch each with `artifact(jobId, name)`. */
  readonly artifacts?: readonly string[];
  /**
   * What is resident **now**, on a `load-model` or `unload-model` job: the
   * model that was loaded, or `null` after an unload, when nothing is
   * (PHASE2-LLM.md section 5). Phase 2 keeps one model resident at a time, so
   * this is the whole story, not one entry of it.
   */
  readonly resident?: string | null;
  /**
   * Every other key the job type put on its `done` frame, verbatim — server
   * spelling, server types, nothing invented and nothing dropped. Exactly
   * {@link ProgressData.extra}, for exactly the same reason.
   *
   * The server builds this frame as `{"artifacts": [...], **job.done_extra}`
   * (`crucible/jobs/queue.py`), and a job type puts its own terminal news in
   * `done_extra`. Until 2026-09-13 this client read the two keys it modelled and
   * **silently dropped the rest**, which lost:
   *
   * - `tts`'s `failed: [{index, message}]`, `rendered`, `take` and
   *   `sample_rate`. PHASE3-TTS.md section 6 says in as many words that a client
   *   reading only the terminal event still learns exactly which indices it has
   *   to ask for again — and it could not, because the list never arrived.
   *   {@link readRenderResult} is the reader for it.
   * - `load-voice`'s `fingerprint` beside its `resident`, which is the string
   *   that says *which merge* of a fine-tune is on the card.
   *
   * `{}` when the frame carried only the modelled keys. That is not a
   * substituted default: it is the true answer to "what else was on the frame".
   */
  readonly extra: Readonly<Record<string, unknown>>;
}

export interface FailedData {
  readonly error: JobFailure;
}

export interface CancelledData {
  readonly status: 'cancelled';
}

/** The event names that end a stream. */
export const TERMINAL_EVENTS = ['done', 'failed', 'cancelled'] as const;

export type TerminalEventName = (typeof TERMINAL_EVENTS)[number];

/**
 * `<artifact>.provenance.json`, verbatim as the server wrote it (DESIGN.md
 * section 7). Persist this file beside the artifact: a finished audiobook says
 * which server rendered it. Keys are the server's, not camelCased, so that
 * `JSON.stringify(provenance)` round-trips the document.
 */
export interface Provenance {
  readonly server: { readonly name: string; readonly version: string };
  readonly backend: string;
  readonly job_type: string;
  readonly model: {
    readonly id: string;
    readonly revision: string | null;
    /** `<id>@<revision>`: which weights, not merely which model. */
    readonly fingerprint: string | null;
  } | null;
  readonly params: Readonly<Record<string, unknown>>;
  readonly started: string | null;
  readonly finished: string;
}

// --------------------------------------------------------------------- llm

/**
 * One model this server knows about, as `GET /v1/models` describes it
 * (PHASE2-LLM.md section 5).
 *
 * The four booleans are four different facts and none of them implies another:
 * `backendSupported` is "the manifest has a block for this host's backend",
 * `installed` is "the weights are on disk", `resident` is "an engine is serving
 * it right now", and `loadable` is "asking for it now would succeed" — which
 * also depends on the accelerator guard (section 4), so a model can be
 * installed and supported and still not loadable because someone else's process
 * holds the card.
 */
export interface ModelInfo {
  /** Crucible's id, stable across backends, e.g. `qwen3.5-9b`. */
  readonly id: string;
  readonly family: string;
  readonly paramsB: number;
  /**
   * The commit the manifest pins for **this host's** backend — the same sha the
   * weights were pulled at, so a client can record what it talked to. `null`
   * when `backendSupported` is false: a model this host cannot serve has no
   * revision here to name.
   */
  readonly revision: string | null;
  /**
   * `<id>@<revision>` — what to write down when you record what you talked to.
   *
   * The bare id does not identify bytes: one Crucible id serves different
   * weights on different backends, and a manifest can be re-pinned. This is
   * {@link ModelInfo.id} and {@link ModelInfo.revision} joined by the server, so
   * it never disagrees with them, and it is `null` wherever `revision` is — an
   * unpinned fingerprint would be worse than none, because it would read as a
   * pin.
   */
  readonly fingerprint: string | null;
  /**
   * What a client may put in a chat request's content parts — `"text"`,
   * `"image"` (PHASE3-VLM.md section 2).
   *
   * Unlike {@link ModelInfo.revision} and its nullable siblings this is **never
   * null**, on any host: it says what the model is offered *for*, which is the
   * same answer on a host whose backend cannot serve it at all. A page reader
   * picks an image-capable model off this rather than knowing one by name.
   */
  readonly modalities: readonly string[];
  readonly backendSupported: boolean;
  readonly installed: boolean;
  readonly resident: boolean;
  readonly loadable: boolean;
  /**
   * Why it is not loadable, in the server's words. Always present when
   * `loadable` is false — a refusal with no reason is a protocol error — and
   * absent when it is true.
   */
  readonly reason?: string;
  /**
   * Weights plus KV at `contextDefault`, measured on the host, not guessed.
   * `null` when `backendSupported` is false, for the same reason
   * {@link ModelInfo.revision} is: the figure lives in this backend's block,
   * and there is no block. Never `0` — a zero would read as "needs nothing".
   */
  readonly memoryBytesEstimate: number | null;
  /**
   * The manifest's intent: the context this host would serve this model at.
   * Compare {@link ModelInfo.maxModelLen}, which is what is being served.
   */
  readonly contextDefault: number;
  /**
   * The context in force **right now** — for the resident model, the one its
   * engine was actually started with. Size a request against this one: it is
   * the number the engine will measure `max_tokens` plus the prompt against.
   *
   * `null` when `backendSupported` is false, like {@link ModelInfo.revision} and
   * {@link ModelInfo.memoryBytesEstimate}: this host would not serve it at any
   * context.
   */
  readonly maxModelLen: number | null;
}

/**
 * OpenAI's structured-output request, passed through to the engine verbatim.
 *
 * `schema` is a JSON Schema document and is deliberately untyped here: it is the
 * grammar the engine's guided-decoding backend compiles, and this client is not
 * in the business of deciding which of JSON Schema an engine supports. A schema
 * the engine will not compile comes back as its own 400, which is the answer
 * that says what to fix.
 */
export type ResponseFormat =
  | { readonly type: 'text' }
  | { readonly type: 'json_object' }
  | {
      readonly type: 'json_schema';
      readonly json_schema: {
        readonly name: string;
        readonly schema: Readonly<Record<string, unknown>>;
        /** Whether the engine must follow the schema exactly. */
        readonly strict?: boolean;
        readonly description?: string;
      };
    };

/** One turn of a chat. `content` is text; this client sends no other part types. */
export interface ChatMessage {
  readonly role: 'system' | 'user' | 'assistant';
  readonly content: string;
}

/** What {@link CrucibleClient.chat} and {@link CrucibleClient.chatStream} take. */
export interface ChatOptions {
  /**
   * The model to talk to. Must be the resident one: the server never loads
   * implicitly, and answers 409 `model_not_resident` naming what is resident
   * instead (PHASE2-LLM.md section 5).
   */
  readonly model: string;
  readonly messages: readonly ChatMessage[];
  readonly temperature?: number;
  readonly topP?: number;
  readonly maxTokens?: number;
  readonly stop?: readonly string[];
  /**
   * The engine's sampling seed. Two identical requests at the same seed and the
   * same sampling give the same answer on the same engine; it says nothing
   * across engines or across backends.
   */
  readonly seed?: number;
  /**
   * Make the engine answer in a shape rather than in prose — OpenAI's
   * `response_format`, forwarded to the engine exactly as given.
   *
   * `{type: 'json_schema', json_schema: {name, schema, strict: true}}` is the
   * one structured-output mechanism the engines share, and the grammar inside it
   * is yours: Crucible does not read it, rewrite it or validate it. The reply
   * still arrives as {@link ChatResponse.content} — a string that happens to
   * hold JSON — because that is what the engine returns; parse it yourself and
   * check {@link ChatResponse.finishReason} first, since a `length` stop is a
   * truncated document and not a malformed one.
   *
   * On a reasoning model, consider `thinking: false` alongside it: a grammar
   * applied to a model that is still thinking routes the object into the
   * reasoning channel.
   */
  readonly responseFormat?: ResponseFormat;
  /**
   * Whether a reasoning model thinks before it answers.
   *
   * Qwen3.5 and its kind emit `reasoning` first and `content` after, so a short
   * `maxTokens` spends the whole budget thinking and returns a message with no
   * `content` at all. `false` sends `chat_template_kwargs: {enable_thinking:
   * false}` — read per request by mlx-lm's server and honoured by vLLM under
   * the same name — and `true` sends the same field set to `true`. Omit it and
   * nothing is sent: the model's own default stands.
   *
   * A model whose chat template does not know `enable_thinking` ignores it.
   */
  readonly thinking?: boolean;
  /** Aborts the request. See the README: the abort surfaces as a DOM `AbortError`. */
  readonly signal?: AbortSignal;
}

/** The tokens a completion cost, as the engine counted them. */
export interface ChatUsage {
  readonly promptTokens: number;
  readonly completionTokens: number;
  readonly totalTokens: number;
}

/**
 * One non-streamed completion: OpenAI's `chat.completion` read down to the part
 * a caller actually uses. The first choice is the only one — this client never
 * asks for `n > 1`.
 */
export interface ChatResponse {
  readonly id: string;
  /**
   * Crucible's model id — the one you asked for. The engine underneath may
   * answer to a different name (mlx-lm answers to the weights directory); the
   * server puts its own id back, so this is the same string on every backend.
   */
  readonly model: string;
  readonly content: string;
  /**
   * The engine's own word for why it stopped, surfaced rather than swallowed:
   * `stop`, `length`, `tool_calls`, or whatever else that engine says.
   *
   * It is a plain string and not a union on purpose — a value this client did
   * not anticipate must reach the caller, not become a protocol error — and
   * nothing normalises it in either direction. `length` means the answer is
   * truncated: both apps treat that as a degradation to record rather than an
   * answer to use, which only works if it arrives.
   */
  readonly finishReason: string;
  readonly usage: ChatUsage;
}

// --------------------------------------------------------------------- tts

/**
 * The band a client packs its chunks to, as the voice's manifest declares it
 * (PHASE3-TTS.md section 2).
 *
 * The server states the shape and the client does the packing — Crucible does no
 * chunking and no text normalisation, and a voice handed text outside its band
 * is a voice reading at the wrong speed. The three rates are always there and
 * always satisfy `min < pace < max`; the packing shape is one of three
 * arrangements, told apart by which of the other three are null:
 *
 * - a **band**: `safeMinChars` and `safeMaxChars` set, `targetChars` null — what
 *   the five fine-tunes declare, their training corpus's interquartile range;
 * - a **target**: `targetChars` set, the other two null — what the zero-shot
 *   voices declare;
 * - **neither**, all three null, which means pack to this backend's
 *   {@link VoiceInfo.maxChars}.
 *
 * The two shapes are never both set: the loader refuses a manifest declaring
 * both.
 */
export interface VoicePace {
  /** The measured pace this voice reads at. */
  readonly paceCharsPerSec: number;
  readonly maxCharsPerSec: number;
  readonly minCharsPerSec: number;
  readonly targetChars: number | null;
  readonly safeMinChars: number | null;
  readonly safeMaxChars: number | null;
}

/** How a voice is conditioned. The loader refuses any other word. */
export type VoiceKind = 'checkpoint' | 'zeroshot' | 'token';

/**
 * Where a voice's `memoryBytesEstimate` came from (PHASE3-TTS.md section 2,
 * difference 4).
 *
 * `measured` means somebody watched the card. `declared` means the number came
 * off the engine's own configured reservation, or off a sibling voice's
 * certificate — true enough to load against, not a measurement, and it rides on
 * the row precisely so that nothing downstream can mistake one for the other.
 * Every voice this build ships says `declared`.
 */
export type EstimateBasis = 'measured' | 'declared';

/**
 * One voice this server knows about, as `GET /v1/voices` describes it
 * (PHASE3-TTS.md section 2). These same rows are the `tts` capability's rows in
 * {@link CrucibleClient.info}.
 *
 * A voice is to `tts` what a {@link ModelInfo} is to `llm`, and the four
 * booleans mean exactly what they mean there: `backendSupported` is "the
 * manifest has a block for this host's backend", `installed` is "the weights are
 * on disk", `resident` is "narrator is serving it right now", `loadable` is
 * "everything this host needs is in place". `loadable` is a fact about the disk
 * and deliberately does not run nvidia-smi, so a row saying `loadable: true` can
 * still be refused at load time with `accelerator_busy`.
 *
 * **What is not here is not an omission.** `sampling`, the EOS levers, the
 * token-budget formula and the engine flags are engine tuning, they are the
 * server's, and publishing them would invite a client to send them back. What a
 * client gets is the shape it must pack to ({@link VoiceInfo.pace},
 * {@link VoiceInfo.maxChars}) and the identity it must record
 * ({@link VoiceInfo.fingerprint}).
 */
export interface VoiceInfo {
  /** Crucible's voice id, stable across backends, e.g. `deathstalker`. */
  readonly id: string;
  /** The name to put in front of a person. */
  readonly display: string;
  readonly kind: VoiceKind;
  /** The manifest's language tag, e.g. `en`. */
  readonly language: string;
  /**
   * Which of narrator's engines serves this voice, e.g. `higgs-v3`. On the row
   * because it decides which env a load needs, and therefore what a
   * {@link VoiceInfo.reason} about a missing env is talking about.
   */
  readonly narratorEngine: string;
  readonly backendSupported: boolean;
  readonly installed: boolean;
  readonly resident: boolean;
  readonly loadable: boolean;
  /**
   * Why it is not loadable, in the server's words; `null` when it is loadable.
   *
   * Unlike {@link ModelInfo.reason} this key is always present on the row — the
   * voice row carries `"reason": null` where the model row omits the key — and
   * the client reads it that way rather than tidying the difference away. What
   * does not differ is the rule: a row that is not loadable and does not say why
   * is a protocol error, because the operator cannot tell whether to pull
   * weights, install an env, free the card, or go to the other host.
   */
  readonly reason: string | null;
  /**
   * The commit this host's backend block pins. `null` when `backendSupported` is
   * false, together with {@link VoiceInfo.fingerprint},
   * {@link VoiceInfo.memoryBytesEstimate}, {@link VoiceInfo.estimateBasis} and
   * {@link VoiceInfo.maxChars}: all five live in the backend block this host
   * does not have, and `0` would read as "needs nothing" where an empty string
   * would read as a pin.
   */
  readonly revision: string | null;
  /** `<id>@<revision>`, joined by the server. What a render records. */
  readonly fingerprint: string | null;
  /** Null when `backendSupported` is false. Never `0`. */
  readonly memoryBytesEstimate: number | null;
  /** Null when `backendSupported` is false. */
  readonly estimateBasis: EstimateBasis | null;
  /**
   * **The** cap certificate for this (voice, backend): the most characters this
   * voice may be handed in one chunk. Per backend and staying per backend — that
   * every voice's two blocks carry the same number today is a coincidence of the
   * current catalog, not a property of the world.
   *
   * These are CHARACTERS, not tokens. Nothing in `tts` carries a token cap on
   * the wire: narrator derives the frame budget per chunk from the text it is
   * actually given. Null when `backendSupported` is false.
   */
  readonly maxChars: number | null;
  /**
   * The sample rate of the audio this voice produces. Never null — a client
   * writing FLACs cannot be handed one — and per voice rather than a constant,
   * because 24000 everywhere in today's catalog is exactly the kind of
   * coincidence that becomes a hard-coded number if it is not written down.
   */
  readonly sampleRate: number;
  /** How many rungs this voice's take ladder has. Never null. */
  readonly takes: number;
  /** Never null: the whole block, because a client that packs needs all of it. */
  readonly pace: VoicePace;
}

/**
 * One unit of work for {@link CrucibleClient.render}: the client's own index,
 * and the text to speak.
 *
 * **Chunking is the client's and stays the client's** (PHASE3-TTS.md section 1).
 * Crucible does no packing and no text normalisation; pack to the voice's own
 * {@link VoicePace} and {@link VoiceInfo.maxChars} before you get here, because
 * a chunk over the cap is refused (`chunk_too_long`) and never re-split — a
 * server that quietly cut a chunk in half would return two files where one was
 * asked for.
 */
export interface RenderChunk {
  /**
   * The client's number for this chunk, and the name of the artifact it
   * produces (`<index>.flac`). Crucible neither assigns it nor renumbers it: it
   * travels out as narrator's batch `i`, back on the retiring row, and into the
   * file name BookForge's assembly and resume already look for.
   */
  readonly index: number;
  readonly text: string;
}

/**
 * What {@link CrucibleClient.render} takes (PHASE3-TTS.md section 6).
 *
 * Nothing has a default. `language` and `take` are both decisions — a book
 * rendered in the wrong language, or at a rung the client did not choose, is a
 * silent substitution — and the server refuses a missing one rather than
 * picking.
 */
export interface RenderOptions {
  /**
   * The voice id, which is what `model` means for `tts`: a Higgs v3 voice *is*
   * the merged checkpoint the engine was started on, so the wire's word for "the
   * thing that produces the bytes" needs no second vocabulary here.
   *
   * Unlike {@link ChatOptions.model} this need **not** already be resident. A
   * render job is an operator's explicit order and owns the exclusive lane for
   * its whole duration, so if the wrong voice (or none) is on the card when the
   * job reaches the front of the lane, the job loads it and emits `warming`
   * events exactly as `loadVoice` does. That is the one asymmetry with `llm`,
   * and it is deliberate.
   */
  readonly voice: string;
  /** The manifest's language tag for this text, e.g. `en`. */
  readonly language: string;
  /**
   * Which rung of the voice's take ladder to render at. `0` is the engine's own
   * sampling, which is what asking for nothing gets. A rung past the end of the
   * ladder is `unknown_take` and is **never clamped**: a silent clamp is a
   * ladder that stops climbing without telling anyone.
   */
  readonly take: number;
  /** At least one. Two chunks may not share an index — an index is a file name. */
  readonly chunks: readonly RenderChunk[];
  /**
   * Aborts the submit itself. A 1,400-chunk book is a large POST, and this is
   * the caller's handle on it. It does **not** cancel a job that was already
   * queued — {@link CrucibleClient.cancel} does that, because by then the job
   * exists on the server and abandoning the socket would leave it running.
   */
  readonly signal?: AbortSignal;
}

/** One chunk of a render that produced no audio, as `done` names it. */
export interface RenderFailure {
  readonly index: number;
  /** narrator's own words: `No audio generated`, `cancelled`, an exception. */
  readonly message: string;
}

/**
 * A finished `tts` job's terminal news, read out of {@link DoneData.extra} by
 * {@link readRenderResult}.
 *
 * **A failed chunk is reported and the run continues** — one bad sentence never
 * sinks the other 1,399 — so a successful job can still have failures, and this
 * is the authoritative list of them. A missing `<index>.flac` is a file that is
 * not there, and BookForge's resume already knows how to ask for it again.
 */
export interface RenderResult {
  /** How many chunks produced audio. */
  readonly rendered: number;
  /** Every chunk that did not, and why. Empty on a clean run. */
  readonly failed: readonly RenderFailure[];
  /** The rung this render actually ran at. */
  readonly take: number;
  /** The rate the voice was loaded at, and the rate every FLAC was written at. */
  readonly sampleRate: number;
  /** The artifacts the job published — one `<index>.flac` per rendered chunk. */
  readonly artifacts: readonly string[];
}

/**
 * One artifact {@link CrucibleClient.writeArtifactsTo} has finished writing,
 * with its provenance sidecar already on disk beside it.
 */
export interface WrittenArtifact {
  /** The artifact's name on the server, e.g. `41.flac`. */
  readonly name: string;
  /** Where it was written, e.g. `Z:\books\the-mutineer\41.flac`. */
  readonly path: string;
  readonly bytes: number;
  /** Where its sidecar was written: `<path>.provenance.json`. */
  readonly provenancePath: string;
  /** The sidecar, parsed. The bytes on disk are the server's own, verbatim. */
  readonly provenance: Provenance;
}

/**
 * What {@link CrucibleClient.writeArtifactsTo} yields: the job's own events,
 * unchanged, interleaved with the files it has written.
 *
 * A union rather than a callback, because the writer must report its progress
 * **without swallowing the job's own events** — a caller still needs every
 * `chunk`, every `progress` and the terminal frame, and getting them through a
 * second channel while the events came through a first would be two clocks.
 * Narrow on `kind`, the same way you narrow a {@link JobEvent} on `event`.
 */
export type ArtifactWrite =
  | { readonly kind: 'event'; readonly event: JobEvent }
  | { readonly kind: 'written'; readonly written: WrittenArtifact };

// ------------------------------------------------------------- accelerator

/** One process the driver says is holding accelerator memory. */
export interface AcceleratorHolder {
  readonly pid: number;
  readonly name: string;
  /**
   * What this process holds — **`null` where the driver will not say**, which is
   * what happens under WDDM and wherever permissions withhold per-process
   * figures.
   *
   * That null is a refusal to answer and it is **not zero**. Rendering it as 0
   * tells a queue that a process holding several gigabytes is holding none,
   * which reads as "the card is free" — the one conclusion this whole route
   * exists to stop a caller drawing. Sum these only over the holders that
   * answered and treat the rest as unknown; {@link AcceleratorState.usedBytes}
   * and {@link AcceleratorState.unattributedBytes} are the figures that do not
   * depend on every process being willing to talk.
   */
  readonly bytes: number | null;
  /** Whether this pid is one of Crucible's own engine processes. */
  readonly ownedByCrucible: boolean;
}

/** What Crucible itself has on the card, from {@link AcceleratorState}. */
export interface AcceleratorResident {
  /**
   * The family of the resident thing, not the job type that put it there:
   * `"llm"` for an engine, `"tts"` for a voice. A plain string, like
   * {@link Health.residentKind} and for the same reason — PHASE4's aligner lands
   * here beside the other two.
   */
  readonly kind: string;
  readonly id: string;
  /** When it was loaded, ISO-8601. */
  readonly since: string;
  readonly memoryBytesEstimate: number;
}

/** The accelerator, at the moment it was probed. */
export interface AcceleratorGpu {
  readonly vendor: string;
  readonly name: string;
  /**
   * The live total from the probe, not the figure detection recorded at
   * start-up. On a real host they agree; where they would not, this is the one a
   * caller is about to make a decision on.
   */
  readonly totalBytes: number;
}

/**
 * `GET /v1/accelerator` — what is on the card right now, and which of it is
 * Crucible's (PHASE4-AUDIO.md section 5).
 *
 * **It reports; it never evicts.** Nothing here asks anybody to leave, and that
 * rule does not soften because more job types depend on the answer.
 */
export interface AcceleratorState {
  readonly backend: string;
  readonly gpu: AcceleratorGpu;
  readonly freeBytes: number;
  readonly usedBytes: number;
  /** What this server holds back for the desktop, from its own config. */
  readonly desktopAllowanceBytes: number;
  /**
   * VRAM in use that no listed holder accounts for, past the declared desktop
   * allowance — and the number that matters most on the host BookForge runs on.
   *
   * Under WSL2 the driver shim answers the compute-app query with an **empty
   * list** while a process inside that same VM holds 17 GB (measured on Owen's
   * PC, 2026-09-12). On that host {@link AcceleratorState.holders} is
   * misleadingly empty and this is the only honest report that the card is busy.
   *
   * `null` on `mlx-darwin`, where "used unified memory" is the OS doing its job
   * and attributing it to compute processes is not a question `vm_stat` can
   * answer.
   *
   * Never negative: the server clamps it at zero, because "VRAM that nothing
   * accounts for, past the allowance" cannot be less than none, and the negative
   * it used to publish on an idle card (−1.5 GiB on Owen's 3090 Ti) is headroom
   * a client would size a load against and not find. A negative here is an
   * older server's bug; it is surfaced as it arrived rather than corrected, so
   * that the bug is visible where it is rather than hidden in this client.
   */
  readonly unattributedBytes: number | null;
  /** What Crucible has loaded, or `null` when it has nothing loaded. */
  readonly resident: AcceleratorResident | null;
  /**
   * Every compute process the driver listed. Empty means the driver listed none
   * — which is not the same as the card being idle; see
   * {@link AcceleratorState.unattributedBytes}.
   */
  readonly holders: readonly AcceleratorHolder[];
  /** The probe's own one-line summary, for a log. */
  readonly detail: string;
}

// --------------------------------------------------------------------- asr

/**
 * What {@link CrucibleClient.asr} takes. Every field is required: the server
 * refuses a missing one, and a client that filled it in would be producing a
 * transcript under rules the caller never chose.
 */
export interface AsrOptions {
  /**
   * Which whisper. **There is no default and there will not be one** — an ASR
   * pass at the wrong size is a transcript that looks fine, is worse, and has
   * nothing in it to say so. `faster-whisper-tiny` through
   * `faster-whisper-large-v3`; {@link CrucibleClient.info}'s `asr` capability
   * lists what this build ships.
   */
  readonly model: string;
  /** The audio, as an uploaded blob or bytes carried inline. Exactly one file. */
  readonly audio: JobInput;
  /**
   * What to call that file on the server. **The extension is load-bearing**: the
   * input name becomes the file's name on disk and ffmpeg reads the container
   * from it, so send `book.m4b`, not `book`.
   */
  readonly filename: string;
  /**
   * A faster-whisper language code (`en`, `de`, …), or the literal `"auto"` to
   * have it detected — which is a *value* meaning "detect it", not an absence.
   *
   * Checked by the server against the tokenizer's own list, which is where that
   * list lives; this client does not keep a second copy of it to drift out of
   * date. An unknown code is a 400 naming the code, before the job is queued.
   */
  readonly language: string;
  /**
   * Whether whisper's voice-activity filter runs. Required, not defaulted: it is
   * `true` in BookForge, and a default here would mean a transcript quietly
   * produced under different rules than the caller assumed.
   */
  readonly vadFilter: boolean;
  /** Whether whisper emits per-word timestamps. Required, for the same reason. */
  readonly wordTimestamps: boolean;
}
