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

/**
 * One model a job type can serve, as advertised by `GET /v1/info` — DESIGN.md
 * section 4's row, for every capability but `llm` and `tts`.
 *
 * `installed` and `resident` are two facts and neither implies the other:
 * `installed` is "the weights are on disk at the revision the manifest pins"
 * (the puller's own stamp), `resident` is "an engine is serving it right now".
 * A puller reading this row to decide whether to pull reads `installed`; a
 * client deciding whether a job will start without a load reads `resident`.
 */
export interface ModelDescriptor {
  readonly id: string;
  readonly revision: string;
  readonly source: string;
  readonly installed: boolean;
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
export type Capability = LlmCapability | TtsCapability | JobCapability | RawCapability;

/** Any capability other than `llm` and `tts`, whose rows are descriptors. */
export interface JobCapability {
  readonly jobType: string;
  readonly models: readonly ModelDescriptor[];
}

/**
 * A capability whose rows this build cannot read, carried rather than refused.
 *
 * The same rule as {@link UnknownEvent}, one level out, and it is here because
 * the bug has already happened once: `tts`'s rows are voices rather than
 * descriptors, so a v0.2.0 client calling `info()` against a v0.3.0 server with
 * tts enabled threw `info.capabilities[2].models[0] has no field "source"` and
 * lost the whole call — measured on 2026-09-13 against a real server, not
 * reasoned about.
 *
 * `llm` and `tts` are the shapes this client CLAIMS, and it stays strict about
 * them: a malformed row in either is still a protocol error. Every other
 * capability is tried as a descriptor and, when it does not fit, carried here
 * with its rows exactly as they arrived — because a capability this build has
 * never heard of is not a broken server, it is a newer one, and `info()` is the
 * call a client makes to find out what it is talking to.
 */
export interface RawCapability {
  readonly jobType: string;
  /** The rows verbatim. Nothing has been checked beyond "it is an object". */
  readonly models: readonly Readonly<Record<string, unknown>>[];
  /** Why the descriptor shape did not fit, for a log rather than a branch. */
  readonly unreadable: string;
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

/**
 * Which half of the orchestrator/engine relation a process is.
 * PHASE17-ORCHESTRATOR.md section 1.
 *
 * A property of a PROCESS, never of an install: on a Windows machine with no
 * WSL, ONE install runs an orchestrator and an engine as two processes, and
 * only one of them answers any given `/v1/info`.
 */
export type CrucibleRole = 'engine' | 'orchestrator';

/**
 * How an orchestrator HOLDS its engine, and therefore what it may do to it.
 * PHASE15-HOST.md 4.1a, as PHASE17 3.2 spells it on the wire.
 *
 * A plain union rather than a closed check at the edge of the client: the set
 * can grow, and a client that threw a protocol error on an owner word it had
 * not heard of would break on the server that added one. `found` is the one
 * that matters to a reader — an engine the orchestrator did not start, which
 * it watches and never acts on.
 */
export type EngineOwner = 'wsl-unit' | 'child' | 'found';

/**
 * Who manages an engine, from {@link ServerInfo.managedBy}.
 *
 * The name and the url and no version: the orchestrator's own version is a
 * fact about the orchestrator, which is what `info()` against ITS address
 * answers. Two copies of a version string in two documents is two things to
 * keep in step across an upgrade that changes exactly one of them.
 */
export interface ManagedBy {
  readonly name: string;
  readonly url: string;
}

/**
 * The engine an orchestrator manages, from {@link ServerInfo.engine}.
 *
 * `name` and `backend` are `null` when the orchestrator could not read the
 * engine just now. `url` is never null while there is an engine at all: it is
 * a fact about the MACHINE rather than about the engine's health, and it is
 * the address {@link engineOf} hands back.
 */
export interface EngineRef {
  readonly name: string | null;
  readonly url: string;
  readonly backend: string | null;
  readonly owner: EngineOwner;
}

/**
 * What a page request IS, from {@link PagesEngine.request}.
 * PHASE15-HOST.md 3.10 fact 7; `crucible/pages.py` is the owner.
 *
 * Page reading has no job type of its own — a page is a chat completion with
 * one `image_url` part — so the CLIENT builds the request, and it was building
 * it out of constants pinned in its own source. A prompt and a pixel budget
 * are facts about the WEIGHTS, so the server publishes them and every backend
 * answers from the same function: vLLM, llama.cpp and mlx-vlm hand back the
 * same block, which is the whole of "an app cannot tell which one read a page".
 *
 * NOTHING HERE IS OPTIONAL AND NOTHING HERE HAS A DEFAULT. A field the server
 * did not send is refused by name rather than filled in, because a nearly-right
 * prompt or a nearly-right budget does not error — it answers worse, and costs
 * a whole book before anybody notices.
 */
export interface PageRequest {
  /** The Crucible model id to send as `model`. One id, three engines. */
  readonly model: string;
  /** What the CLIENT rasterises at. The bboxes come back in this frame's scale. */
  readonly dpi: number;
  /** The processor's own pixel limit — the frame the model's boxes are in. */
  readonly maxPixels: number;
  /**
   * The CEILING, never a budget. A client may send less — a per-page cap
   * derived from the book — and must re-read at the full ceiling any page that
   * came back with {@link PageRequest.truncatedFinishReason}.
   */
  readonly maxTokens: number;
  /** A layout is not a thing to be creative about. */
  readonly temperature: number;
  /** The model card's prompt, byte for byte. Never templated, never shortened. */
  readonly prompt: string;
  /** What the answer is shaped like, e.g. `dots-json`. The client owns the parser. */
  readonly dialect: string;
  /**
   * How many page requests a client may have open at once. The server's own
   * arithmetic for whether the weights FIT is sized for this number, so a
   * client that picked its own was reasoning about different work than the
   * machine it was talking to.
   */
  readonly concurrency: number;
  /** The `finish_reason` that means the model was still writing. */
  readonly truncatedFinishReason: string;
}

/**
 * `GET /v1/info`'s `pages_engine` — which engine reads a page HERE, and what a
 * page request is anywhere.
 *
 * The two halves are not equals. `engine` is for an operator looking at a
 * machine and a client must never branch on it; `request` is the load-bearing
 * one and says the same thing on every backend.
 */
export interface PagesEngine {
  /** `vllm`, `llama-server`, `mlx-vlm` — or `null`, meaning this host serves none. */
  readonly engine: string | null;
  readonly installed: boolean;
  /** Why, in words, for an operator. Never parsed. */
  readonly detail: string;
  /** Published whether or not this host can answer one. */
  readonly request: PageRequest;
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
  /**
   * Which half of the relation answered. PHASE17 3.1 and 3.2.
   *
   * **A server that predates Phase 17 reads as `'engine'`**, and that is a
   * fact its document states by its vintage rather than a default this client
   * fills — PHASE15 3.3's all-or-nothing reading rule, the same one `route`
   * gets. API version stays 1; every field of this phase is additive.
   */
  readonly role: CrucibleRole;
  /**
   * On an ENGINE: which orchestrator claimed it, or `null`.
   *
   * `null` is a complete, correct answer and never a fault. An engine nobody
   * claims is a whole Crucible — the Mac, a droplet, any `crucible serve` run
   * by hand. Always `null` on an orchestrator, which claims and is not
   * claimed.
   */
  readonly managedBy: ManagedBy | null;
  /**
   * On an ORCHESTRATOR: the one engine it manages, or `null` when the machine
   * has none. Always `null` on an engine, which IS the engine.
   *
   * Read it with {@link engineOf}, which is the whole of the rule.
   */
  readonly engine: EngineRef | null;
  /**
   * What a page request is on this server, or `null` where the document does
   * not carry the block at all.
   *
   * `null` IS THE VINTAGE, not an empty contract — PHASE15 3.3's all-or-nothing
   * reading rule, the same one {@link ServerInfo.role} gets. A server that
   * predates 3.10 fact 7 published no `pages_engine`, and no prompt or budget
   * is invented for it here: a client that needs the contract refuses that
   * server by name, because building a page request out of its own constants
   * is precisely what the block exists to stop.
   *
   * A host that serves no pages is NOT null — it answers with
   * {@link PagesEngine.engine} `null` and the request block beside it.
   */
  readonly pagesEngine: PagesEngine | null;
}

/**
 * What this server asked to stop and has not been told is gone.
 *
 * **Read it before you believe an idle server is free.** Crucible never
 * SIGKILLs a process holding CUDA — that wedges WSL2 until Windows reboots —
 * so when a `stop` goes unanswered the engine stays on the card and every
 * load, every claim and the streaming door answer `engine_still_stopping`
 * while `resident` reads `null` and the lane reads `ok`. Nothing in Crucible
 * clears this: a human stops those pids, which is why {@link Stopping.pids}
 * is here and not summarised.
 *
 * It is never both this and {@link Activity.resident}: `resident` is what may
 * be USED, this is what may only be waited for.
 */
export interface Stopping {
  /** `llm`, `tts`, … — what sort of thing was on the card. */
  readonly kind: string;
  /** The model or voice id that was resident. */
  readonly id: string;
  /** When the stop was asked for, in {@link ActivityJob.started}'s format. */
  readonly since: string;
  /** The pids still holding the card, ascending. Stop these by hand. */
  readonly pids: readonly number[];
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
  /**
   * What was told to go and has not, or `null`.
   *
   * On the smallest read this server has because it is the reason every load
   * is being refused, and {@link Health.status} cannot say it: `status`
   * reports the LANE, and `ok` there has always meant "no job is running",
   * never "the card is free". See {@link Stopping}.
   */
  readonly stopping: Stopping | null;
}

/**
 * One job as a bench reads it. Never its params — a chat prompt or a chapter of
 * a book is not something a whole-server read should spray at anyone holding the
 * token.
 */
export interface ActivityJob {
  readonly jobId: string;
  readonly type: string;
  readonly model: string | null;
  readonly status: string;
  readonly position: number | null;
  /** 0..1. A job HAS a denominator: the client posted every chunk up front. */
  readonly progress: number;
  readonly message: string | null;
  readonly created: string;
  readonly started: string | null;
  /** The submitting User-Agent. Null = it did not say; never a guessed name. */
  readonly client: string | null;
}

/**
 * An open TTS streaming session, as a bench reads it.
 *
 * **`progress` is `null` and always will be.** A render job knows its own
 * denominator; a session's rows arrive one `say` at a time, indefinitely, on a
 * reader's whim — so a percentage would be a percentage of the work that happens
 * to have arrived, a number that goes DOWN when more arrives. Owen named the
 * three BookForge surfaces that work this way on 2026-09-13 — the streaming
 * page, the correct-sentences/re-roll page and the browser extension: *"those
 * places are independent of a queue but claim a server while they run... that
 * means crucible wont always have a percent complete to hand back."*
 *
 * The field is present rather than omitted because on this wire a present null
 * is a statement and an absent key is not. Draw a spinner and the counts.
 */
export interface ActivityStreaming {
  readonly sessionId: string;
  readonly voice: string;
  readonly language: string;
  readonly narratorEngine: string;
  /** When the session opened, in {@link ActivityJob.started}'s format. */
  readonly since: string;
  /** Who opened it. Null = it did not say. */
  readonly client: string | null;
  /** Always null. See the interface docstring — this is not an omission. */
  readonly progress: null;
  /** Rows this session has been asked to say, ever. */
  readonly said: number;
  readonly finished: number;
  readonly inFlight: number;
  /** Seconds of audio delivered, measured from the bytes. */
  readonly seconds: number;
  readonly chars: number;
}

/**
 * One chat completion, while it is happening.
 *
 * A chat takes no lane and makes no job, so until 2026-09-13 a server grinding
 * through a 27B translation reported `running: []` and read as idle. These rows
 * are what it says instead — and they still gate nothing: `acceptsWork` stays
 * true while chats are in flight, because a vLLM engine batches and the server
 * really will take more.
 */
export interface ActivityChat {
  readonly id: number;
  /**
   * What this completion IS — a capability class name, from the client's
   * `X-Crucible-Act` header.
   *
   * **Null means the client did not say, and is never a guess.** Crucible cannot
   * tell a simplify from a translate: both are a chat against the same 27B and
   * the only difference is a prompt it does not own. Owen ruled on 2026-09-13
   * that a job must be named as what it is — *"before crucible, everything ran
   * under translate"* — so a wrong name here would be the defect, and an unknown
   * act is refused at the door rather than recorded.
   */
  readonly act: string | null;
  readonly model: string;
  /** The calling User-Agent. Null = it did not say. */
  readonly client: string | null;
  readonly since: string;
}

/**
 * A client's declared intention to keep using the resident model.
 *
 * **The hole it fills.** A chat completion holds nothing — no lane, no job, no
 * claim — deliberately, because the engine batches. That is right for one chat
 * and wrong for two thousand: a book translated block by block leaves this
 * server looking idle between any two blocks, and a `load-voice` submitted in
 * one of those gaps used to take the translator off the card mid-run.
 *
 * A timer ("a chat was seen within N seconds") would be a fact standing in for
 * a guess. The fact that exists is the client's intention, and only the client
 * has it — so the client says so, heartbeats while the run is alive, and
 * releases when it is done.
 *
 * **It is a refusal, not a reservation.** Holding one admits nothing and
 * reserves no lane; {@link ActivitySlot.acceptsWork} is untouched. It says one
 * thing: while it is open, nothing may move the model off the card.
 */
export interface Lease {
  readonly leaseId: string;
  /**
   * Which resident kind it holds: `llm`, `tts` or `align`.
   *
   * **The server decides this, never the caller.** The card holds one thing, so
   * the id passed to {@link CrucibleClient.lease} identifies it without a kind
   * and the server reads the kind off its own residency. A lease on a `tts` is
   * what makes a book rendered chapter by chapter one narrator load instead of
   * twenty.
   */
  readonly kind: string;
  /**
   * The id it is held on — a model, a voice or an aligner — which is always the
   * resident one.
   */
  readonly subject: string;
  /** The holder's User-Agent, as `/v1/activity` reports it. Null = it did not say. */
  readonly client: string | null;
  /** What the run IS: a capability class name. Never null — a lease must say. */
  readonly act: string;
  /** When it was taken, in {@link ActivityJob.started}'s format. */
  readonly since: string;
  /** When it stops being open unless something heartbeats it. */
  readonly expiresAt: string;
}

/**
 * The open lease, as `/v1/activity` reports it.
 *
 * {@link Lease} without the subject, and that is not an omission: a lease is
 * only ever on the resident thing, which the same read already reports as
 * `resident.id`. Repeating it would be one fact with two owners in one document
 * (ARCHITECTURE.md R1). `kind` survives the trim because the same fields are a
 * `leased` refusal's details, which arrive with no `resident` beside them and
 * whose whole subject is which jobs the lease refuses.
 */
export type ActivityLease = Omit<Lease, 'subject'>;

/** `GET /v1/activity` — what is on this server and how far along. */
export interface Activity {
  readonly server: {
    readonly name: string;
    readonly version: string;
    readonly apiVersion: number;
    readonly backend: string;
    readonly uptimeS: number;
  };
  readonly resident: {
    readonly kind: string;
    readonly id: string;
    readonly since: string;
    readonly memoryBytesEstimate: number | null;
    /**
     * What holds this resident thing, or `null` — **which is the stranded
     * card**, not an idle one.
     *
     * The server's own answer, from the one function that owns "what holds the
     * card" (`crucible/settle.py`'s four facts). `null` here with `resident`
     * set means nothing is coming back for it: a load that succeeded and was
     * never claimed, or a lease that lapsed with nothing asking again. A
     * reconciler acts on exactly this.
     *
     * `details` is the holding fact's OWN shape — a job's is the `server_busy`
     * body, a lease's is the lease receipt — so a client that can read a
     * refusal can read this without a second vocabulary.
     */
    readonly heldBy: {
      readonly fact: string;
      readonly who: string;
      readonly details: Record<string, unknown>;
    } | null;
    /**
     * Since when nothing has held it, or `null` because something does.
     *
     * **Never a poll artefact.** Both ways a card becomes unheld fire no event
     * — a successful load is exempt from settling, and a lapsed lease is read
     * rather than swept — so this is the later of two real timestamps: the
     * moment the exempt load returned, and the lapsed lease's own
     * `expires_at`. It means what it says even if nobody polled for ten
     * minutes.
     *
     * Non-null already implies "and nothing holds it right now": the server
     * checks the live holder before reporting the stamp, so a reconciler need
     * not check both.
     */
    readonly unclaimedSince: string | null;
  } | null;
  /**
   * What was told to go and has not, or `null`. See {@link Stopping}.
   *
   * The state {@link Activity.resident} cannot describe: `resident` is `null`
   * the moment the stop is asked for, and a bench reading only that drew an
   * idle machine that refuses everything.
   */
  readonly stopping: Stopping | null;
  /** The id of a model being loaded right now, or null. */
  readonly warming: string | null;
  /**
   * Who holds narrator's wire, or null.
   *
   * **Not the same question as the lane.** A streaming session holds the
   * resident engine without occupying the lane, so {@link ActivitySlot.busy} can
   * be 0 while this is set — which is exactly the state a bench used to read as
   * an idle machine while the browser extension was streaming from it.
   */
  readonly claim: { readonly heldBy: string } | null;
  /** The open streaming session, or null. */
  readonly streaming: ActivityStreaming | null;
  /**
   * Chat completions open right now, and what this engine will admit at once.
   *
   * **Counted, and — since 1.0.10 — bounded for a SERIAL engine.** mlx-lm
   * accepts every connection on a threading HTTP server and then generates on
   * one thread, so twelve accepted requests are one running and eleven waiting
   * with nothing on the wire saying so. A client sizes its pool from
   * `maxInFlight` rather than discovering the ceiling as a starved socket.
   */
  readonly chat: {
    readonly inFlight: number;
    /**
     * What this engine's chat door admits at once, or `null`.
     *
     * `null` means the resident engine states no concurrency — vLLM batches and
     * nothing has ever measured starvation against it — **or that nothing is
     * resident**, because the limit belongs to the engine. It never means
     * "unlimited", and a client must not read it as a licence to fan out.
     */
    readonly maxInFlight: number | null;
    /** Where `maxInFlight` came from, in a sentence. `null` when it is null. */
    readonly maxInFlightBasis: string | null;
    readonly rows: readonly ActivityChat[];
  };
  /**
   * The open lease on whatever is resident, or null.
   *
   * The intention behind the run, which nothing about this server could infer.
   * While it is non-null, a job that would move the leased thing off the card is
   * refused `leased` at the door — and nothing else changes, including the work
   * the lease was taken for: a render of the leased voice is admitted, because
   * it runs against what is already resident.
   */
  readonly lease: ActivityLease | null;
  readonly slots: { readonly accelerated: ActivitySlot };
  readonly running: readonly ActivityJob[];
  readonly queued: readonly ActivityJob[];
}

export interface ActivitySlot {
  /** THE LANE, and nothing else. A stream does not take it. */
  readonly busy: number;
  readonly of: number;
  readonly queueDepth: number;
  /**
   * The composition a caller actually wants before submitting: the lane is free
   * AND nobody holds the card. Derived by the server, once, so that three
   * benches do not each invent it and disagree.
   *
   * **Still not a reservation.** A client that reads `true` and submits is
   * racing every other client, and that race is settled at the door — `POST
   * /v1/jobs` admits one and refuses the other by name. Reading this is never
   * permission; only the door can say yes.
   */
  readonly acceptsWork: boolean;
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
  /**
   * Your own name for this work. Echoed back on the job record as
   * {@link JobStatus.clientRef}; this server never reads or parses it.
   *
   * Worth setting on anything long. If the server restarts mid-job it comes
   * back `interrupted` rather than vanishing, and this is how you recognise it
   * as yours after YOUR process has restarted too.
   */
  readonly clientRef?: string;
}

/**
 * A job's lifecycle state. `done`, `failed`, `cancelled` and `interrupted` are
 * all terminal.
 *
 * `interrupted` is the one a client must not treat as a failure: the server
 * stopped while the job was running — a deploy, a crash, a machine going away —
 * and the work it had already published is still there to collect. `failed` is
 * this server judging the work, which is a person's problem; an interruption is
 * weather, and the answer is to fetch what landed and re-ask for the rest.
 */
export type JobState = 'queued' | 'running' | 'done' | 'failed' | 'cancelled' | 'interrupted';

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
  /**
   * The lease this job opened, `null` if it opened none.
   *
   * **Readable after the stream is gone**, which is the whole reason it is on
   * the record and not only in the `done` frame: a lease id you cannot recover
   * is a hold nobody can release, and that is the stranding lease-on-load
   * exists to end. A client that lost its events stream reads it here.
   *
   * `null` on a loader means the load was not asked to hold anything. `null` on
   * every other job type means that type cannot hold a lease — a `tts` render
   * takes the lane instead, and the record simply does not carry the key.
   */
  readonly leaseId: string | null;
  /**
   * The client's own name for this work, echoed back. `null` when none was
   * given.
   *
   * Set it on submit when you will need to recognise this job after YOUR
   * process has restarted too — a job id you may have lost alongside
   * everything else is a poor key for that.
   */
  readonly clientRef: string | null;
  /**
   * When the server was found to have stopped while this job was running.
   * `null` for every other outcome.
   *
   * Set only by the restart that recovered it. Its presence is what tells an
   * `interrupted` job from one that ended some other way.
   */
  readonly interruptedAt: string | null;
  /**
   * The chunk index of every artifact this job published, ascending.
   *
   * **This is what a resume differences against.** For a render the artifacts
   * are `<index>.flac`, and parsing that filename in every client is a
   * documented contract re-implemented N times; this is the server saying it
   * once. Empty for a job whose artifacts are not indexed chunks.
   */
  readonly chunksDone: readonly number[];
  /**
   * How many indexed chunks the job was asked for, stated by the job type;
   * `null` for a job whose artifacts are not chunks, and until a render has
   * stated it. `chunksDone.length / chunksTotal` is done/total from one read.
   */
  readonly chunksTotal: number | null;
  /**
   * When the last chunk artifact landed (ISO-8601 UTC); `null` before any
   * did. Two reads a minute apart are a pace, with no engine log to tail.
   */
  readonly chunkAt: string | null;
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
 * One rendered chunk of a `tts` job (PHASE3-TTS.md section 6, amended by
 * PHASE6-REMOTE-RENDER.md sections 3 and 4).
 *
 * **The model judges, the server forwards, the client orders.** Crucible
 * measures the seven numbered fields and still decides nothing about a chunk —
 * it never retakes, never re-splits and never substitutes. What it no longer
 * claims is that those seven are the whole guard interface: {@link
 * ChunkData.guard} is the verdict the ENGINE'S own retake ladder already
 * reached, carried across unopened. One `ChunkData` arrives per chunk that
 * produced audio, and none at all for a row that failed — that row is named in a
 * `progress` line when it happens and again in `done`'s `failed` list.
 *
 * *What this replaced.* Until Owen's ruling of 2026-09-13 this comment said
 * "this is the whole guard interface… BookForge's PaceTracker is the thing that
 * judges, and this event carries everything it has to judge with". It did not,
 * and the PaceTracker was never in this path: narrator's serve world — the door
 * a Crucible render drives — rendered each Higgs chunk with one bare
 * `render_audio()`, no pace tracking, no re-roll and no split ladder, while its
 * audiobook world ran the same model through all three. The numbers therefore
 * described an unguarded single take. The guard now runs where the model runs,
 * and a client that acts on `capped` or `charsPerSec` instead of reading
 * `guard` is re-litigating a decision the engine has already made.
 *
 * Three of the seven measured fields are the request's or the server's own
 * arithmetic and two come off narrator's wire, and the split is why two of them
 * are nullable:
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
  /**
   * **The verdict the engine's own retake ladder reached about this chunk**, or
   * `null` meaning narrator sent none (Owen's ruling of 2026-09-13,
   * PHASE6-REMOTE-RENDER.md section 3).
   *
   * Deliberately typed as an opaque object and **not modelled**. Its contents
   * are narrator's: today `{verdict, clean, parts, band, takes}`, where
   * `verdict` is the ladder's own last action (`clean`, `short`, `long`,
   * `hole`, `rerolled`, `resplit`, `accepted-off-length`), `parts` is how many
   * text units the chunk was finally rendered as, `band` is the pace tracker's
   * edges and `takes` is the guard's event records verbatim. Crucible forwards
   * the object without reading inside it, and this client does the same, for one
   * reason: **a schema here would break the first time the ladder grows a rung**
   * — and it would break at the first guard fire on a real book, not at compile
   * time. Read what you need with your own narrowing, and treat a word you do
   * not recognise as news rather than as an error.
   *
   * `null` obeys {@link ChunkData.capped}'s rule one level up: it means narrator
   * sent no verdict — an engine with no guarded batch driver to offer, or a row
   * that failed before the ladder reached a decision — and it is **never to be
   * read as "the take was clean"**. `clean` is a key inside a verdict that
   * exists; the absence of a verdict says nothing about the take. The key itself
   * is always present, so an absent one is a protocol error and not a null.
   *
   * It is the **conclusion, not the evidence**. `verdict` is what the engine
   * decided; `takes` is why, for analytics and for a human eye. A client that
   * acts on `takes` is re-deciding something already decided by the only thing
   * that had the frame cap, the seed and the book's running pace in front of it.
   */
  readonly guard: Readonly<Record<string, unknown>> | null;
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
  /**
   * The model whose download this one's weights ARE, or null for a model that
   * owns its own (PHASE22-DECIDE.md section 2.9, `[model] weights_of`). A
   * fact of the manifest, the same on every host. `qwen3.5-9b-vl` says
   * `qwen3.5-9b`: one copy on disk, two rows — and switching between the two
   * is a full engine reload, so a picker chooses one per server session.
   */
  readonly weightsOf: string | null;
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
  /**
   * What this chat IS, for `GET /v1/activity` — a capability class name, sent
   * as the `X-Crucible-Act` header.
   *
   * A chat takes no lane and makes no job, so it is the one piece of work a
   * server can only report if the client says what it is: Crucible cannot tell
   * a `simplify` from a `translate`, since both are a chat against the same
   * model and the only difference is a prompt it does not own. The act is
   * recorded on the in-flight chat row and surfaces in
   * {@link Activity.chat}`.rows[].act`.
   *
   * **Omit it and no header is sent**, which the server records as `null` —
   * "it did not say". There is deliberately no default: a name nobody chose on
   * a bench is the thing this header exists to prevent.
   *
   * The vocabulary is the server's — exactly the capability classes
   * `GET /v1/capability` answers with — and this client does not keep a second
   * copy of it. A name that is not one comes back as the server's own 400
   * `unknown_act`, listing what it knows, BEFORE the completion runs.
   */
  readonly act?: string;
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

// ------------------------------------------------------------------ decide
//
// `POST /v1/decide`, PHASE22-DECIDE.md (2026-09-23): snap's decision model
// moved into Crucible as a door, sibling of the chat door. A state and a set of
// questions with FIXED answer sets go in; a probability distribution over each
// answer set comes out, read off ONE forward pass of the resident model. The
// prompt that makes an instruct model report a distribution is Crucible's (it
// is a fact about the weights, section 2.3) — the caller sends the ORDER: the
// state, the questions and their options, and nothing else.

/**
 * Pick one of 2–26 named options.
 *
 * **Order is meaning**: the server tags the options `A`..`Z` in the order the
 * object lists them. A JavaScript object lists integer-like keys (`"1"`,
 * `"42"`) first and in ascending order whatever order they were written in, so
 * give options names that are not integers when their order matters.
 */
export interface DecideChoiceQuestion {
  readonly type: 'choice';
  readonly instructions: string;
  /** Option name → the description the model reads. */
  readonly options: Readonly<Record<string, string>>;
}

/** Place the state on 2–10 unique, ORDERED levels; the answer's `score` is the expected 1-based level. */
export interface DecideScoreQuestion {
  readonly type: 'score';
  readonly instructions: string;
  readonly levels: readonly string[];
}

/** A statement the state does or does not make true; the answer is P(Yes). */
export interface DecideYesNoQuestion {
  readonly type: 'yesno';
  readonly instructions: string;
}

export type DecideQuestion = DecideChoiceQuestion | DecideScoreQuestion | DecideYesNoQuestion;

export interface DecideRequest {
  /**
   * The model to read the decision from. Must be the resident one, exactly as
   * for {@link ChatOptions.model}: the server answers 409 `model_not_resident`
   * and never loads a model to answer a decision. An upstream id
   * (`anthropic/…`) is refused 400 `decide_needs_logprobs` — no upstream
   * returns a distribution.
   */
  readonly model: string;
  /**
   * What the questions are about: a string, or any JSON value (the server
   * serialises a non-string as compact JSON). May be `""` only when `images`
   * are given.
   */
  readonly state: unknown;
  /**
   * Image FILES, base64-encoded (at most 8; more is 400 `too_many_images`).
   * A model whose manifest does not declare `image` refuses them 400
   * `model_text_only` naming the model (section 2.7). Omit it for none.
   */
  readonly images?: readonly string[];
  /** Question name → question. A name is a single path member. */
  readonly questions: Readonly<Record<string, DecideQuestion>>;
}

export interface DecideOptions {
  /**
   * What this decision IS, for `GET /v1/activity` — sent as `X-Crucible-Act`,
   * exactly as {@link ChatOptions.act}: omitted, no header is sent and the
   * server records `null`; an unknown name is the server's 400 `unknown_act`.
   */
  readonly act?: string;
  /** Aborts the request; the abort surfaces as a DOM `AbortError`. */
  readonly signal?: AbortSignal;
}

/**
 * The answer to a `choice`. `probabilities` is renormalised over the option
 * letters; `confidence` is its largest entry; `labelMass` is how much of the
 * engine's raw next-token mass the letters held before renormalising — low
 * means the model wanted to say something that was not an option.
 */
export interface DecideChoiceAnswer {
  readonly type: 'choice';
  readonly choice: string;
  readonly probabilities: Readonly<Record<string, number>>;
  readonly confidence: number;
  readonly labelMass: number;
}

/** The answer to a `score`: `score` = Σ(1-based level index × p), `level` the likeliest. */
export interface DecideScoreAnswer {
  readonly type: 'score';
  readonly score: number;
  readonly level: string;
  readonly probabilities: Readonly<Record<string, number>>;
  readonly confidence: number;
  readonly labelMass: number;
}

/** The answer to a `yesno`: `p` is the renormalised P(Yes). */
export interface DecideYesNoAnswer {
  readonly type: 'yesno';
  readonly p: number;
  readonly labelMass: number;
}

export type DecideAnswer = DecideChoiceAnswer | DecideScoreAnswer | DecideYesNoAnswer;

/** One completion the door sent the engine, timed by Crucible's wall clock. */
export interface DecideCallTiming {
  readonly wallMs: number;
  /** `usage.prompt_tokens`, as the engine counted it. */
  readonly promptTokens: number;
  /**
   * `usage.prompt_tokens_details.cached_tokens`, or **`null` when the engine
   * did not say — never 0**: a number nobody measured is not a measurement.
   */
  readonly cachedTokens: number | null;
}

export interface DecideTiming {
  readonly total: number;
  readonly perQuestion: Readonly<Record<string, DecideCallTiming>>;
  /** The shared-prefix prime, sent only when there is more than one question; `null` when none was. */
  readonly prime: DecideCallTiming | null;
}

export interface DecideResponse {
  /** The provenance triple every artifact sidecar carries: which weights made this decision. */
  readonly model: {
    readonly id: string;
    /** The revision the resident engine was started on; always stated, since a decision is read only from a resident engine. */
    readonly revision: string;
    /** `<id>@<revision>`. */
    readonly fingerprint: string;
  };
  /** The engine kind that answered (`vllm`, `llama-server`, …). */
  readonly engine: string;
  /** One answer per question asked, keyed by the question's name. */
  readonly answers: Readonly<Record<string, DecideAnswer>>;
  readonly timingMs: DecideTiming;
  readonly tokens: {
    readonly perQuestion: Readonly<Record<string, number>>;
    readonly images: number;
  };
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
  /**
   * The measured pace this voice reads at, with the two edges derived from it
   * — ALL THREE OR NONE, and `null` means nobody measured this voice.
   *
   * Optional as a group since 2026-09-18, and the reason is worth carrying
   * here rather than only in the server. They used to be required, so the two
   * voices that are the BASE WEIGHTS rather than a fine-tune met the
   * requirement by copying narrator's own Higgs v3 constants back to it: a
   * pace of 15.0, which is the divisor the frame cap is sized against and was
   * never measured as a speaking rate, inside a band written around a real
   * book pace nearer 17.2. narrator keeps a band's RATIOS and re-centres them
   * on the book's running median, so that pairing judged healthy chunks
   * run-ons and drove them to the bottom of the retake ladder.
   *
   * SO NOTHING DERIVES A CENTRE FROM A NULL — not the server and not this
   * client. A voice with no band is narrator reaching for its engine's own,
   * centred on the geometric mean of the edges, which is one owner for that
   * derivation. A client packing to these must ask whether they are there.
   */
  readonly paceCharsPerSec: number | null;
  readonly maxCharsPerSec: number | null;
  readonly minCharsPerSec: number | null;
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
  /**
   * A local (`path`) voice nothing holds: not resident, no lease naming it, no
   * queued or running job naming it. After a restart that is every screening
   * voice whose ladder ended without its DELETE. The server SAYS it and never
   * acts on it; `DELETE /v1/voices/{id}` is idempotent, so the ladder can. `false`
   * for every pinned voice (the pin owns it); `null` on a read that was not
   * asked to decide.
   */
  readonly orphan: boolean | null;
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
  /**
   * How many rungs this voice's take ladder has. Never null, never below 1 —
   * take 0 exists whether or not the manifest says so.
   *
   * **Read it before you spread candidates.** A `take` past the end is not
   * refused (it was `unknown_take` until 2026-09-19): it renders at the
   * voice's OWN sampling in take N's own seed lane, which is what a screening
   * sweep wants and what a retake ladder does NOT — a retake must not reuse the
   * settings that produced the problem, so a client climbing rungs reads this
   * to know where the DIFFERENT settings run out. Still never clamped. What
   * each rung MEANS is deliberately not published: the numbers are engine
   * tuning, they are the server's, and publishing them invites a client to send
   * them back.
   */
  readonly takes: number;
  /**
   * What the SERVER under narrator is sized by, or `null` for a voice that
   * declares no serving table (a shape the next narrator engine will have and
   * no manifest has today).
   *
   * **`maxNumSeqs` is the SERVED ARM's width** — `HIGGS_MAX_NUM_SEQS`, the
   * number Crucible starts narrator's serving stack at on a `cuda-linux`
   * server — and it is the ceiling a render's {@link RenderOptions.width} is
   * refused against THERE. That is why it is published at all: it was
   * deliberately NOT on the row until 2026-09-19, on the true-at-the-time
   * ground that a client had no decision to make with it, and giving a job its
   * own width made that false.
   *
   * **On `mlx-darwin` it is not the width anything runs at.** narrator starts
   * no server there and batches at `NARRATOR_HIGGS3_MLX_BATCH` out of a
   * measured tier table chosen by the machine's own memory — 64 on the 64 GB
   * Mac Studio against a manifest that says 16. Reading this number as "the
   * width my render will run at" is what cost the Mac a measured 2.3x between
   * Crucible 1.0.7 and 2026-09-20 (12.9x realtime / 189 sentences per minute
   * down to 5.5x / 78). On that arm, state a width only if you mean it, and
   * narrator answers for one it cannot run.
   *
   * `memFraction` and `contextLength` are `null` when the voice states none,
   * meaning narrator's own launcher defaults (0.60, and the Higgs builder's
   * 4096) — never a number this row invented. Each note is the measurement
   * that chose its number. Nothing here is ever SENT by a client: the row
   * publishes what the server chose, and `width` is the only thing a request
   * may say about any of it.
   */
  readonly serving: VoiceServing | null;
  /**
   * Whether loading this voice requires a reference clip
   * ({@link LoadVoiceOptions.reference}) — true for a `zeroshot` voice, false
   * for every other kind. Read it to decide whether to show a clip picker;
   * loading without one is refused `reference_required`, and loading a
   * checkpoint WITH one is refused `reference_not_allowed`.
   *
   * On a server built before the field existed no row carries it, and every
   * voice in such a document reads `false` — PHASE15-HOST.md section 3.3's
   * client reading rule, asked once of the whole document, all or nothing. A
   * document that states it on some rows and not others is refused
   * `voices_needs_reference_missing` rather than patched.
   */
  readonly needsReference: boolean;
  /** Never null: the whole block, because a client that packs needs all of it. */
  readonly pace: VoicePace;
}

/**
 * The recording a zero-shot voice is cloned from, sent with
 * {@link CrucibleClient.loadVoice} (PHASE3-TTS.md section 5).
 *
 * A zero-shot voice is the base weights plus somebody's voice: the weights are
 * the server's, pulled at the manifest's pin, and the CLIP is yours — a
 * per-client choice like the voice pick itself. It travels with the load,
 * which is the one moment it is needed.
 */
export interface VoiceReference {
  /**
   * The wav's bytes, base64, with **no `data:` prefix and no whitespace**. A
   * RIFF/WAVE container: the server reads its header, both of narrator's arms
   * want a wav, and anything else is `reference_malformed`.
   *
   * At most 30 seconds of audio — narrator's own cap, above which vllm-omni
   * answers "Reference audio too long". Two ~14 s clips joined into one wav is
   * the practical maximum, and a same-BOOK clip is worth far more than a
   * second one.
   */
  readonly data: string;
  /**
   * The BOOK-EXACT text spoken in the clip, and **never an ASR guess**.
   * Required: narrator refuses a clip without one at construction, because a
   * clone conditioned on a wrong or absent transcript is a whole book in a
   * subtly wrong voice, reported as success.
   */
  readonly transcript: string;
  /**
   * A short label for whoever reads the server's `/v1/activity` and wants to
   * know which of their clips is resident. Optional — the server always
   * reports a sha256 of the audio beside it, which is what tells two clients
   * apart when neither sent a name.
   */
  readonly name?: string;
}

/** What {@link CrucibleClient.loadVoice} takes beyond the voice id. */
export interface LoadVoiceOptions {
  /**
   * Required when the voice's row says {@link VoiceInfo.needsReference};
   * refused on any other kind.
   */
  readonly reference?: VoiceReference;
  /** Hold the voice from the instant it is resident. See {@link LeaseOnLoad}. */
  readonly lease?: LeaseOnLoad;
}

/**
 * Hold what a load makes resident, from the instant it exists.
 *
 * **Why you want this on every programmatic load.** A load that succeeds cannot
 * clear the card — its whole content is "be resident" — so without a lease the
 * window between `done` and your own `POST /v1/models/{id}/lease` is held by
 * NOTHING, and a client that dies in that window strands the card for ever,
 * because a quiet hold has no end. Ask for a lease and the same death is
 * bounded: the ttl runs out and the server clears the card itself.
 *
 * The job's `done` frame carries `lease_id`, and so does
 * {@link CrucibleClient.job} — a lease id you cannot recover is a hold nobody
 * can release.
 *
 * **Omit it for an operator-style load**, where a human will decide when the
 * card is free. Absent means exactly today's behaviour.
 */
export interface LeaseOnLoad {
  /**
   * What the run is for: one of the capability classes, the same vocabulary
   * `POST /v1/models/{id}/lease` takes. A value that is not one is refused by
   * name (`unknown_act`) rather than recorded.
   */
  readonly act: string;
  /**
   * Seconds. Bounded by the server (30-3600) and extended by each heartbeat —
   * this is how long the card survives your process disappearing, so short
   * enough to matter and long enough to outlive a slow act.
   */
  readonly ttlSeconds: number;
}

/** Options for {@link CrucibleClient.loadModel}. */
export interface LoadModelOptions {
  /** Hold the model from the instant it is resident. See {@link LeaseOnLoad}. */
  readonly lease?: LeaseOnLoad;
}

/**
 * One unit of work for {@link CrucibleClient.render}: the client's own index,
 * and the text to speak.
 *
 * **Chunking is the client's and stays the client's** (PHASE3-TTS.md section 1).
 * Crucible does no packing and no text normalisation; pack to the voice's own
 * {@link VoicePace} and {@link VoiceInfo.maxChars} before you get here.
 *
 * **The server no longer refuses an oversize chunk** (2026-09-19). It used to,
 * as `chunk_too_long`, and that refusal is retired rather than relaxed: the cap
 * is a measurement a screening checkpoint may not have, and a second TTS engine
 * would have its own frame arithmetic that this number describes nothing about.
 * A chunk over the cap now goes to the engine as sent and comes back measured —
 * which is how a sweep finds out what the cap actually is. Crucible still never
 * re-splits: a server that quietly cut a chunk in half would return two files
 * where one was asked for.
 */
/** `[voice.serving]` — what the server under narrator is sized by. */
export interface VoiceServing {
  readonly maxNumSeqs: number;
  readonly maxNumSeqsNote: string;
  readonly memFraction: number | null;
  readonly memFractionNote: string | null;
  readonly contextLength: number | null;
  readonly contextLengthNote: string | null;
}

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
   * Which rung of the voice's take ladder to render at, and which SEED LANE to
   * draw in. `0` is the engine's own sampling, which is what asking for nothing
   * gets.
   *
   * A take past the end of the declared ladder is legal since 2026-09-19 (it
   * was `unknown_take`): it renders at the voice's OWN sampling in take N's own
   * seed lane, which is what a screening sweep asks for when it names takes
   * 0..N on a voice with no ladder at all. Still **never clamped** — take 4 is
   * never take 2's numbers under take 4's name.
   */
  readonly take: number;
  /** At least one. Two chunks may not share an index — an index is a file name. */
  readonly chunks: readonly RenderChunk[];
  /**
   * Which ARM renders this batch (2026-09-19, PHASE18-UNCERTIFIED.md sections 4
   * and 6). `true` is narrator's guarded driver — the PaceTracker, the re-roll
   * on truncation/runaway/loop and the split ladder — measured against
   * {@link RenderOptions.band}. Absent, or `false`, is the bare arm: every
   * chunk rendered once as sent, nothing judged and nothing retaken, and
   * `guard` on every chunk row is then `null` at its most exact — nobody judged
   * it, because nobody was asked to.
   *
   * Bare is the default because it is the primitive. A guarded screen
   * under-counts the failures it exists to measure: a re-roll that succeeds is
   * indistinguishable from a good first draw.
   *
   * `retake: true` with no `band` is refused as `retake_without_band`.
   */
  readonly retake?: boolean;
  /**
   * The pace band the guarded arm measures against, in the voice manifest's own
   * spelling. All three positive, with `min < pace < max`, or the whole request
   * is refused as `band_malformed`.
   *
   * **The caller states it; the server never looks it up.** BookForge echoes
   * back the row it read from {@link CrucibleClient.voices}; a screening client
   * sends nothing and cannot be guarded. Looking it up is the shape that
   * produced two measured defects: a base voice satisfying a mandatory triple
   * with narrator's own frame-cap divisor (15.0, not a narration rate), and
   * deathstalker inheriting pace 16.64 onto weights that measured 15.91.
   *
   * Sent with `retake` false or absent, it is accepted, checked and not acted
   * on — the server was not asked to judge anything.
   */
  readonly band?: {
    readonly pace_chars_per_sec: number;
    readonly max_chars_per_sec: number;
    readonly min_chars_per_sec: number;
  };
  /**
   * How many of this job's chunks may be IN FLIGHT at once (2026-09-19,
   * amended 2026-09-20).
   *
   * **Absent sends nothing**, and the engine then renders at the width it was
   * STARTED at — `HIGGS_MAX_NUM_SEQS` from the voice's serving table on a
   * `cuda-linux` server, `NARRATOR_HIGGS3_MLX_BATCH` off a measured tier table
   * on a Mac. Until 2026-09-20 the server substituted the manifest's number
   * here, which narrowed every Mac batch from 64 to 16 and cost a measured
   * 12.9x realtime / 189 sentences per minute down to 5.5x / 78.
   *
   * A width ABOVE the engine's is refused as `width_over_serving` and never
   * clamped: a job that thought it was running 16 wide and was not would
   * report a throughput nobody can reproduce. On a `cuda-linux` server that
   * refusal comes from Crucible, against
   * {@link VoiceServing.maxNumSeqs}; on `mlx-darwin` it comes from narrator,
   * against the width it actually has.
   *
   * Narrowing restarts nothing. The server keeps the `--max-running-requests`
   * and CUDA-graph budget it was loaded with; this only caps what narrator
   * keeps in flight. Measured 2026-09-19: 0.60 mem fraction at 16 wide summed
   * to 24.2 GB on a 24 GB card, and WDDM then pages to host RAM 4-10x slower
   * with no error at all.
   */
  readonly width?: number;
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
  /**
   * THE FULL SAMPLING TRIPLE the engine actually applied — the voice's take-0
   * numbers with this take's rung laid over them, never the rung's override
   * alone (2026-09-19).
   *
   * It is here because sampling lives on the MANIFEST and not on the request,
   * which is the right shape and leaves exactly one hole: a manifest edited
   * between two runs makes two incomparable records that both say "take 0".
   * That is not hypothetical — every Higgs measurement before 2026-09-06 was
   * rendered at temperature 1.0 and the whole prior ladder record had to be
   * marked "at the wrong temperature" once already. Pinned here, a mismatch is
   * visible instead of silent.
   *
   * Manifest spelling (`top_p`, `top_k`), because it is a fact about the voice.
   */
  readonly sampling: Readonly<Record<string, number>>;
  /**
   * WHICH WEIGHTS RAN, in the `/v1/voices` row's own three words, so a ladder's
   * record is self-describing. `identity` is the pin's 40-character sha or a
   * local block's asserted string; `identityBasis` is `verified` or `asserted`
   * and says which, so a directory somebody pointed at cannot be mistaken for a
   * commit somebody fetched.
   */
  readonly voice: {
    readonly id: string;
    readonly identity: string;
    readonly identityBasis: string;
  };
  /**
   * The width the REQUEST stated — its {@link RenderOptions.width} — and
   * `null` when it stated none (2026-09-20). A throughput figure is comparable
   * against this and nothing else.
   *
   * `null` is a fact, not a gap: it means the engine rendered at the width it
   * was started at, which narrator knows and this server does not on every
   * arm. Until 2026-09-20 it reported the voice's `maxNumSeqs` instead, which
   * on a Mac named a width nothing ran at.
   */
  readonly width: number | null;
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

// -------------------------------------------------------------- capability

/**
 * One capability class's verdict, as `crucible capability` decided it on this
 * host (PHASE9-CAPABILITY.md; `crucible/config.py`'s `CapabilityRow`).
 *
 * A class is what a client asks for — "a translate-class model", "a voice" —
 * and this is the server's answer: the concrete thing it picked, or the number
 * that stopped it. `enabled: false` **is an answer, not an error**: a server
 * that cannot translate says so with the shortfall that decided it, and a
 * client renders "this machine cannot do that" without it looking like a fault.
 *
 * `selected` is `''` and `shortfallBytes` is `0` where they do not apply,
 * exactly as the server records them: its config is TOML, which has no null,
 * and a key that came and went would make "nothing fit" and "this record
 * predates the field" the same reading. Branch on `enabled`, never on the
 * emptiness of `selected`.
 */
export interface CapabilityRow {
  /** The class name a client asks by: `clean`, `translate`, `tts`, `asr`, … */
  readonly capability: string;
  readonly enabled: boolean;
  /** The candidate that won, or `''` when none did. */
  readonly selected: string;
  /** Why, in the server's words, whichever way it went. Never empty. */
  readonly reason: string;
  /**
   * How much more memory the SMALLEST candidate would have needed, or 0. The
   * number that turned the class off, as a number and not only inside
   * `reason` — a sentence is never load-bearing (ARCHITECTURE.md R4).
   */
  readonly shortfallBytes: number;
  /**
   * Where this class's work actually runs (PHASE15-HOST.md section 3.3).
   *
   * `upstream` means the operator routed it, `selected` is the
   * `<upstream>/<model>` id to send as a chat's `model`, and `reason` keeps
   * the local sentence after `the local answer would be: ` so nothing is lost
   * when they route back.
   */
  readonly route: 'local' | 'upstream';
}

/**
 * `GET /v1/capability` — what this server can hold, per class, and why not.
 *
 * The read a client makes before it decides what to ask for. Phase 9 made the
 * act-to-model mapping a per-host fact: `crucible install` probes the card and
 * picks the largest candidate that fits, so a 24 GB box serves `translate`
 * with a 4-bit 27B and a 12 GB box does not serve it at all. A client handed a
 * model id by configuration would be carrying one this server may have refused.
 *
 * **A record, not an authority.** `[jobs] enable_*` stays the one owner of
 * what the server offers; this says what the numbers were when somebody
 * decided. `totalBytes` is the card the decision was made on, so a reader can
 * tell a stale record from a current one — which is how a swapped GPU is
 * noticed without anybody writing down a date.
 */
export interface CapabilityRecord {
  /** `cuda-linux` or `mlx-darwin`: the backend the decision was made for. */
  readonly backendKind: string;
  /** The pool the decision was made on — the card, or unified memory on a Mac. */
  readonly totalBytes: number;
  /** The host's own reserve, subtracted before any candidate was measured. */
  readonly desktopAllowanceBytes: number;
  /** One verdict per class, in the server's report order. */
  readonly classes: readonly CapabilityRow[];
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

// ---------------------------------------------------------------------------
// The operator door — PHASE13-OPERATOR.md section 3
// ---------------------------------------------------------------------------
//
// Owen, 2026-09-14: *"crucible has its own ui. and it provides the token or
// whatever else we need to set it up on foundry or bookforge."* Everything
// below is that door's half of the wire, typed verbatim from section 3, and
// every field name is the doc's.

/**
 * `GET /v1/setup` — everything an app needs to be pointed at this server.
 *
 * **It carries the token, and that reveals nothing**: the route is behind the
 * bearer token, so the only caller who can read this already has it. What it
 * buys is that nobody types a secret twice — the operator page draws a
 * copyable {@link ServerSetup.pairing} line, and the person pasting it has not
 * seen a token at all.
 */
export interface ServerSetup {
  /** `crucible@mac-studio`. Contains an `@`, which is why pairing encodes it. */
  readonly name: string;
  readonly version: string;
  /** `cuda-linux` or `mlx-darwin`. Windows is never a backend. */
  readonly backend: string;
  /** What this process bound, e.g. `http://0.0.0.0:7100`. Not dialable as-is. */
  readonly bind: string;
  /**
   * The bind address made reachable: a wildcard bind becomes one entry per
   * non-loopback IPv4 interface, a concrete bind becomes exactly one.
   *
   * Never a hostname lookup — an interface the host has is a fact, a name
   * somebody else's DNS may resolve is not.
   */
  readonly urls: readonly string[];
  readonly token: string;
  /** One `crucible://` line per {@link ServerSetup.urls} entry, in order. */
  readonly pairing: readonly string[];
  /** `/v1/info`'s list, repeated so a page draws from one read. */
  readonly jobTypes: readonly string[];
  readonly configPath: string;
}

/**
 * The SIX things a subject can be. PHASE13-OPERATOR.md section 2 named five;
 * `engine` was added for PHASE15-HOST.md 3.10 and this sentence did not move
 * with it.
 *
 * It said "five" until 2026-09-16, when Foundry read the sentence instead of
 * the union, wrote a five-word mirror of it, and was saved by its compiler. A
 * mirror built by hand from the prose would have shipped a Windows-native
 * engine that could not fetch its own llama.cpp. COUNT THE UNION, and when a
 * member is added, the count above it is part of the change.
 */
export type SubjectKind =
  | 'model'
  | 'voice'
  | 'rvc'
  | 'rvc-base'
  | 'denoise'
  /**
   * The llama.cpp binaries a `llama-windows` server runs its GGUF models
   * with — `{kind: "engine", id: "llama-cpp"}` (PHASE15-HOST.md 3.10, fact 1).
   * A subject like any other, so the page's Tasks panel shows the download
   * with bytes exactly as it shows a weights pull; what makes it different is
   * only that its bytes are a pinned release's zips rather than a repo's.
   */
  | 'engine';

/**
 * One row of `GET /v1/catalog`: a pullable thing and where it stands here.
 *
 * Every field is DERIVED on the server from something that already owns it —
 * the weights stamp, the manifest, `crucible/lineup.py`, the residency — so a
 * client reading this is reading those, not a second table.
 */
export interface CatalogRow {
  readonly kind: SubjectKind;
  readonly id: string;
  /** The manifest's display name, or null where a manifest carries none. */
  readonly name: string | null;
  /** Which job type this subject belongs to: `llm`, `asr`, `align`, `tts`, … */
  readonly jobType: string;
  readonly installed: boolean;
  /** Bytes on disk, or null when it is not installed. */
  readonly installedBytes: number | null;
  /**
   * What a pull will fetch, where the manifest declares it — which is `rvc`,
   * `rvc-base` and `denoise`, whose weights are named files with pinned
   * digests. **Null for models and voices**, whose weights are a whole-repo
   * snapshot no manifest sizes. Never an estimate.
   */
  readonly expectedBytes: number | null;
  /**
   * The model whose download this row's weights are, or null (PHASE22 section
   * 2.9). On such a row the download's bytes are the BASE's row's, and
   * `installedBytes` / `expectedBytes` count only this row's own files — 0
   * where it has none, `expectedBytes` null where it adds a projector.
   */
  readonly sharesWeightsOf: string | null;
  /**
   * An alias's own files that are not on disk, empty when all are there; null
   * on every row that is not an alias. Why an alias is `installed: false` when
   * its base is installed.
   */
  readonly missingFiles: readonly string[] | null;
  /**
   * The capability classes this model is the FLOOR for — the smallest model
   * the class may run on at all. Only ever non-empty on a `model`.
   */
  readonly floors: readonly string[];
  /**
   * Null on every row this build of the server can produce, and that is the
   * honest value: no manifest schema carries a licence key, and reading one
   * off a repo name would be a claim about somebody else's weights.
   */
  readonly license: string | null;
  /** `hf:<repo>` — where the bytes come from. */
  readonly source: string;
  /** Is this the thing on the card right now? */
  readonly resident: boolean;
}

/** One `pull`: fetch a subject's weights. Refused if it is already installed. */
export interface PullTaskRequest {
  readonly type: 'pull';
  readonly kind: SubjectKind;
  readonly id: string;
}

/** One `install`: build a job type's env, then make it live (section 3.4). */
export interface InstallTaskRequest {
  readonly type: 'install';
  readonly jobType: string;
  /**
   * Required for `tts` and refused for anything else: on `cuda-linux` there is
   * one venv per narrator engine, because each pins its own serving stack
   * against its own torch and two of them cannot share one. The engines a
   * server will accept are its `/v1/capability` row's `narratorEngines`, and
   * there is no default even when that list holds one.
   */
  readonly narratorEngine?: string;
}

/**
 * An app's statement of what it needs from a server, as the JSON file it
 * vendors.
 *
 * **Keys are the server's, not camelCased, and that is deliberate.** The file
 * is written by `scripts/gen-modules.py` in the crucible repo and vendored by
 * each app byte for byte (PHASE13-OPERATOR.md section 5.4); posting it means
 * posting exactly those bytes. A camelCased mirror here would make every app
 * transform a file whose whole point is that it is not edited.
 */
export interface CrucibleModule {
  readonly name: string;
  /** Derived by the generator from the content, never typed by hand. */
  readonly version: string;
  readonly job_types: readonly {
    readonly type: string;
    readonly narrator_engine?: string;
  }[];
  /**
   * CAPABILITY CLASSES, unresolved, and the SERVER resolves them
   * (PHASE15-HOST.md 5.3a).
   *
   * The generator used to turn a class into one model id — the id its own
   * machine's backend would use — and the module then said `dots-ocr` to a
   * Mac that has no block for it, and `qwen3.8-27b-4bit` to a Mac whose
   * capability had selected `qwen3.8-27b`. PHASE9 puts the resolution in the
   * capability record, and the record is per machine, so the class travels
   * and the server decides.
   *
   * A class this engine has DISABLED is not a refusal: the task finishes and
   * reports it in {@link TaskStatus.unmet}.
   */
  readonly needs: readonly { readonly class: string }[];
  /**
   * Ids an app CHOSE: a voice, a whisper size, the rvc base. An explicit id
   * a backend cannot hold is still `unknown_subject` and still refuses the
   * whole module — the asymmetry is the point. A class is "give me whatever
   * serves this", which a machine can answer with "nothing here does". An id
   * is "give me this one", which it cannot.
   */
  readonly subjects: readonly {
    readonly kind: SubjectKind;
    readonly id: string;
  }[];
}

/** One class a `module` named that this engine does not serve. 5.3a. */
export interface UnmetNeed {
  readonly class: string;
  /**
   * The capability row's OWN sentence, verbatim. Never one the task wrote:
   * the row said why the class is off, and an app shows that.
   */
  readonly reason: string;
}

/** One `module`: an ordered list of installs and pulls, validated whole. */
export interface ModuleTaskRequest {
  readonly type: 'module';
  readonly module: CrucibleModule;
}

/**
 * One `engine`: move this Windows machine to the WSL2 engine (PHASE15-HOST.md
 * 4.7).
 *
 * **The server does not run it.** Only the host — `crucible host`, the tray
 * process — can run `wsl.exe`, prompt for administrator and survive the
 * reboot the move may need, so the Windows server hands this to the host's
 * loopback door and relays the host's events under the task id it returns.
 * A server that was not started by a host refuses `engine_move_needs_host`;
 * a server that is not a Windows one refuses `engine_move_not_here`.
 *
 * **Those two are the only refusals you get from the SUBMIT.** When a host
 * DID start the server, there is nothing to refuse at submit time — the
 * server hands the move over and relays — so the POST answers with a task id
 * and every failure of the door is named in the task's own `failed` event:
 * `host_unreachable` (the door did not answer: start the host again),
 * `host_install_failed` (the stream ended without saying whether the move
 * finished: read the host's log), or the code the host's own `failed` event
 * carried, verbatim. Read the task, not the submit.
 *
 * `target` is `'wsl'` and nothing else in this phase: moving BACK to Windows
 * is an explicit operator act (section 6) and is refused
 * `engine_target_unknown` rather than half-done.
 */
export interface EngineTaskRequest {
  readonly type: 'engine';
  readonly target: 'wsl';
}

/**
 * Restart this machine's engine, through its orchestrator.
 * PHASE17-ORCHESTRATOR.md 4.2.
 *
 * NO fields: there is exactly one engine on a machine and the orchestrator
 * knows which, so a `target` here would be a client naming a thing it cannot
 * see. Refused `engine_restart_needs_orchestrator` on a server no
 * orchestrator started, and `engine_not_ours` when the orchestrator merely
 * FOUND its engine and so has no unit it may name and no child it may kill.
 *
 * **The task's last event may never arrive.** The relay runs in the process
 * being restarted. Read `info()` for the answer, exactly as a page does
 * across the engine move's switch-over; a stream that ends with no terminal
 * event is the expected shape here, not a fault to report.
 */
export interface EngineRestartTaskRequest {
  readonly type: 'engine-restart';
}

export type TaskRequest =
  | PullTaskRequest
  | InstallTaskRequest
  | ModuleTaskRequest
  | EngineTaskRequest
  | EngineRestartTaskRequest;

/**
 * A task has no `queued`: it is admitted and running in the same act, because
 * a second submission is refused rather than parked (`task_busy`).
 */
export type TaskState = 'running' | 'done' | 'failed' | 'cancelled';

export const TASK_TERMINAL_STATES = ['done', 'failed', 'cancelled'] as const;

/** `GET /v1/tasks/{id}`. */
export interface TaskStatus {
  readonly taskId: string;
  /** `pull`, `install`, `module` or `engine`. */
  readonly type: string;
  /** The request body, echoed, in the server's own spelling. */
  readonly request: Readonly<Record<string, unknown>>;
  readonly state: TaskState;
  readonly error: JobFailure | null;
  readonly created: string;
  readonly started: string;
  readonly finished: string | null;
  /**
   * Classes a `module` named that this engine does not serve (5.3a).
   *
   * EMPTY and never absent, on every task type, so "nothing was unmet" and
   * "this server predates the field" are not one reading. A `done` task with
   * entries here did everything it could; the app shows "not on this engine"
   * beside the pulls it made.
   */
  readonly unmet: readonly UnmetNeed[];
}

/** One step of a task. For a `module`, one per entry plus the reload. */
export interface TaskStepData {
  readonly name: string;
  /** 1-based. */
  readonly index: number;
  readonly total: number;
  /**
   * What this server now offers, on the `reload` step only (section 3.4). A
   * client is told rather than having to diff two `/v1/info` reads.
   */
  readonly jobTypes?: readonly string[];
}

/** A pull's progress: bytes of one file. `bytesTotal` is null if unstated. */
export interface TaskBytesProgress {
  readonly bytesDone: number;
  readonly bytesTotal: number | null;
  readonly file: string;
}

/**
 * An install's progress: one line of the installer's own output.
 *
 * **It is not load-bearing** (ARCHITECTURE.md R4). It is pip's text, for a
 * person to read in a scrolling pane; every fact a client acts on is a `step`,
 * a `skipped`, a `done` or a `failed`.
 */
export interface TaskLineProgress {
  readonly line: string;
}

export type TaskProgressData = TaskBytesProgress | TaskLineProgress;

/** Narrow a `progress` frame to the installer's lines. */
export function isTaskLineProgress(data: TaskProgressData): data is TaskLineProgress {
  return 'line' in data;
}

/** Narrow a `progress` frame to a pull's byte counts. */
export function isTaskBytesProgress(data: TaskProgressData): data is TaskBytesProgress {
  return 'bytesDone' in data;
}

/** A module entry that was already true, so nothing was done for it. */
export interface TaskSkippedData {
  readonly reason: string;
}

/**
 * One frame of `GET /v1/tasks/{id}/events`.
 *
 * `unknown` is here for the reason it is on {@link JobEvent}: the server's
 * event vocabulary grows without moving `api_version`, and a client that threw
 * on a kind it had never heard of would lose the whole stream rather than one
 * frame. Strict about what it claims to understand, tolerant of what it makes
 * no claim about.
 */
export type TaskEvent =
  | { readonly id: number; readonly event: 'started'; readonly data: { readonly type: string } }
  | { readonly id: number; readonly event: 'step'; readonly data: TaskStepData }
  | { readonly id: number; readonly event: 'progress'; readonly data: TaskProgressData }
  | { readonly id: number; readonly event: 'skipped'; readonly data: TaskSkippedData }
  | { readonly id: number; readonly event: 'done'; readonly data: Readonly<Record<string, unknown>> }
  | { readonly id: number; readonly event: 'failed'; readonly data: JobFailure }
  | { readonly id: number; readonly event: 'cancelled'; readonly data: Readonly<Record<string, unknown>> }
  | UnknownEvent;

/** What `DELETE /v1/tasks/{id}` answers. `cancelling`, never `cancelled`. */
export interface TaskCancelResult {
  readonly taskId: string;
  /**
   * `cancelling`: the flag is set and the runner ends when it sees it — the
   * next chunk for a pull, the SIGTERM landing for an install. Watch the
   * stream for the `cancelled` event. Saying "cancelled" before a download
   * thread had stopped would be the ambiguous answer R3 forbids.
   */
  readonly status: 'cancelling';
}

// ---------------------------------------------------------------- settings
//
// PHASE15-HOST.md sections 3.1, 3.2 and 3.8. Owen, 2026-09-14: *"Settings live
// in the engine and nowhere else."* An app draws these and writes through
// `putSettings`; it holds no key, no route and no cloud model list of its own.

/** The three upstreams a Crucible speaks to. Exactly these names. */
export type UpstreamName = 'anthropic' | 'openai' | 'ollama';

/** Where one capability class's work runs on a server. */
export interface RouteSetting {
  /** `local` (this server's card) or `upstream` (the operator's account). */
  readonly route: 'local' | 'upstream';
  /**
   * For `local`, the selected local model, or `null` when nothing fits — and
   * also `null` when this server has decided nothing yet (`capability()`
   * refuses `capability_undecided`, which is how the two are told apart).
   * For `upstream`, the `<upstream>/<model>` id to send as a chat's `model`.
   */
  readonly model: string | null;
}

/**
 * One upstream card. **The key is never here**: `keyHint` is `…` followed by
 * its last four characters, and is rendered verbatim.
 */
export interface UpstreamSetting {
  readonly configured: boolean;
  /** `anthropic` and `openai`. Absent for `ollama`, which has no secret. */
  readonly keyHint?: string | null;
  /** `ollama`. Absent for the two hosted upstreams, whose address is fixed. */
  readonly url?: string | null;
}

/** An eligible local model, computed by the engine for a capability class. */
export interface LocalModelChoice {
  readonly id: string;
  readonly memoryBytesEstimate: number;
  readonly fits: boolean;
  readonly installed: boolean;
}

/** `GET /v1/settings` — the whole of what an app's settings window draws. */
export interface SettingsDocument {
  /**
   * Which local model serves each capability class. Null requests the engine's
   * automatic selection, and is a DECISION rather than an absence.
   *
   * REQUIRED since 2026-09-16. Both keys were optional so this SDK could read
   * an engine older than them; Owen ruled that population out of existence —
   * nothing is released, so nothing is legacy — and an optional field kept for
   * readers who do not exist is a branch every caller pays for.
   */
  readonly localModels: Readonly<Record<string, string | null>>;
  /** Empty when the engine has not measured its card yet; never absent. */
  readonly localModelChoices: Readonly<Record<string, readonly LocalModelChoice[]>>;
  /** One entry per routable class: `clean`, `translate`, `simplify`, `analysis`. */
  readonly routes: Readonly<Record<string, RouteSetting>>;
  readonly upstreams: Readonly<Record<UpstreamName, UpstreamSetting>>;
  readonly desktopAllowanceBytes: number;
  readonly backendKind: string;
}

/**
 * A `PUT /v1/settings` patch. Every field is optional and a patch is PARTIAL:
 * what it does not mention it does not change.
 *
 * Inside one request the server applies upstreams, then routes, then validates
 * the whole — which is what lets one call both configure an upstream and route
 * a class to it, and lets one call re-route away from an upstream and remove
 * it. A refusal applies nothing.
 */
export interface SettingsPatch {
  /** Explicit local selection; null restores the engine's automatic decision. */
  readonly localModels?: Readonly<Record<string, string | null>>;
  /** Class → `'local'` or an upstream model id. */
  readonly routes?: Readonly<Record<string, string>>;
  /** Name → its one field, or `null` to remove the upstream. */
  readonly upstreams?: Readonly<
    Partial<Record<UpstreamName, { key?: string; url?: string } | null>>
  >;
  readonly desktopAllowanceBytes?: number;
}

/**
 * What `testUpstream` answers. **It does not throw for the three test
 * refusals**, because all three are ordinary answers to "does this key work" —
 * a person pasting one expects to be told, not to have an exception raised at
 * their settings page. Auth, version and transport failures still throw, like
 * every other call.
 */
export type UpstreamTestResult =
  | { readonly ok: true; readonly models: string[] }
  | {
      readonly ok: false;
      readonly code:
        | 'upstream_unreachable'
        | 'upstream_rejected'
        | 'upstream_unconfigured';
      readonly message: string;
    };
