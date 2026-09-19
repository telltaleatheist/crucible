/**
 * The host's loopback door — `POST http://127.0.0.1:7101/install` — and this
 * package's client for it (PHASE15-HOST.md section 4.3).
 *
 * **Why there is a door at all.** The install sequence (PHASE14 section 4) used
 * to be walked twice on Windows: once by `install.ps1` and once by this package
 * through `wsl.exe`. PHASE15 gives the sequence ONE owner — `crucible host`,
 * the Windows-native tray process — and makes `@crucible/bootstrap` its client.
 * On win32 `install()` no longer walks `installSteps()`; it asks the host and
 * relays what the host says. Linux and macOS are untouched: there is no host
 * there, the machine IS the server, and the walk is still this package's.
 *
 * **Two callers, no third** (PHASE15 4.3, "Who calls the door"). The PRIMARY
 * one is the Windows server itself: the operator page posts
 * `POST /v1/tasks {"type":"engine","target":"wsl"}` (4.7), the server hands it
 * to this door and relays the events under its own task id. This package is
 * the SECOND caller, and it calls the door directly on a machine that has no
 * server at all yet — the very first install, before there is a page to open.
 * That is the question the next reader will have ("why does a library post to
 * a loopback port when there is a whole tasks API?") and that is the answer:
 * there is no server to post `/v1/tasks` to yet.
 *
 * **The other end of this wire is `crucible/host/door.py`.** Every type in the
 * "the wire" section below is that file's JSON, field for field. A change to
 * one is a change to the other; the file is named here so whoever edits either
 * finds the other. The envelope is `crucible/tasks.py`'s `append_event` —
 * `{"id", "event", "data"}` — BECAUSE of the relay above: a server that
 * reshaped these into task events would be a second owner of the shape.
 * The 4c state codes cross the wire VERBATIM — a host that cannot carry the
 * machine further sends `virtualization_disabled`, and this client refuses
 * with that name rather than wrapping it in a generic one, because the app's
 * next move is chosen from the code.
 *
 * **Bootstrap never elevates.** `wsl-states.ts` states the rule for
 * `run-elevated`: "a UAC prompt raised by a library, from a background probe,
 * is a dialog nobody asked for". It used to also never INSTALL the host, and
 * PHASE19 changed that half: `install.ps1` is per-user and raises no consent
 * dialog, and the UAC prompt a WSL install needs is raised later by the tray,
 * which is a process a person can see. What stays refused is elevation.
 *
 * **PHASE19 2.6 added two GETs.** The tray starts the move itself now, so the
 * door has to be watchable and not only drivable: {@link installStatus} reads
 * `GET /install` and {@link watchInstall} attaches to `GET /install/events`
 * from the move's current step. Neither ever posts.
 */
import { crucibleAppData } from './distro.js';
import { backendFor, releaseAssetUrl } from './release.js';
import { BootstrapRefusal } from './errors.js';
import type { InstallResult, InstallStep, JobTypeRequest } from './install.js';
import { processRunner, type OutputStream, type Runner } from './runner.js';
import { parseToml } from './toml.js';

// ------------------------------------------------------------- the address

/** The host's loopback port. One number, both ends (PHASE15 4.3). */
export const HOST_DOOR_PORT = 7101;

/**
 * The door's base URL. `127.0.0.1` and not `localhost`: the host binds the
 * loopback address, and `localhost` can resolve to `::1` first on a machine
 * whose hosts file says so — a connection refused that is not a door being shut.
 */
export const HOST_DOOR_URL = `http://127.0.0.1:${HOST_DOOR_PORT}`;

/** `POST` here to run the sequence; `GET` it for the status (PHASE19 2.6). */
export const HOST_INSTALL_PATH = '/install';

/** `GET` here to ATTACH to a move in flight, from its current step (PHASE19 2.6). */
export const HOST_INSTALL_EVENTS_PATH = '/install/events';

/**
 * The Windows host's relocatable entry point (PHASE15 4.4). A `.cmd` and
 * not an `.exe`: pip writes `Scripts\<name>.exe` launchers with the BUILDING
 * interpreter's absolute path baked into the binary, which a move breaks and
 * no shebang rewrite can reach, so the install writes a `%~dp0`-relative shim
 * instead. Its presence is what "the host is installed" means.
 */
export const HOST_ENTRY_POINT = 'crucible.cmd';

/**
 * `%LOCALAPPDATA%\Crucible\host` — where `install.ps1` puts the host runtime.
 *
 * The directory comes from {@link crucibleAppData}, which reads `LOCALAPPDATA`
 * from the environment and NEVER assembles it from a username, and which
 * refuses `host_unresponsive` when it is unset. That refusal's sentence names
 * the distro because `crucibleAppData` has one caller-facing spelling; the
 * CODE is the contract, and importing the function is what keeps
 * `%LOCALAPPDATA%\Crucible\` spelled in exactly one place — `wsl\`,
 * `downloads\`, `host\` and `config.toml` are all members of it (PHASE15 3.6).
 */
export function hostRuntimeDir(runner: Runner): string {
  return crucibleAppData(runner, 'host');
}

/** `%LOCALAPPDATA%\Crucible\config.toml` — the host-mode server's own config (PHASE15 3.6). */
export function hostConfigPath(runner: Runner): string {
  return crucibleAppData(runner, 'config.toml');
}

/**
 * Is the host runtime on this machine? A file test, not a ping: a host that is
 * installed and not answering is `host_unreachable`, which is a different
 * problem with a different answer (start it) from `host_not_installed`
 * (install it).
 */
export function hostInstalled(runner: Runner): boolean {
  return runner.fileExists(`${hostRuntimeDir(runner)}\\${HOST_ENTRY_POINT}`);
}

/** The line a person runs to put a host on this machine (PHASE15 4.4). */
export function hostInstallCommand(release: string): string {
  return `irm ${releaseAssetUrl(release, 'install.ps1')} | iex`;
}

/**
 * The door's bearer — the ENGINE token, read from the host's own config.
 *
 * There is no second token and none is minted here: PHASE15 3.5 says the
 * host-mode config's token and the guest's token after the migrate are THE
 * SAME token, so the door is authorised by the thing every client already
 * holds. A machine whose host has not run its first `init` has no token at
 * all, which is `host_no_token` by name — not an empty string, not a guess.
 */
export function hostToken(runner: Runner): string {
  const path = hostConfigPath(runner);
  if (!runner.fileExists(path)) {
    throw new BootstrapRefusal(
      'host_no_token',
      `${path} does not exist, so the host has no engine token yet and its door cannot be authorised. `
        + 'The host writes this file on its first `crucible init`; until then there is nothing to pair with.',
    );
  }
  let text: string;
  try {
    text = runner.readFile(path);
  } catch (err) {
    // Present and unreadable is a different fact from absent, and `config.toml`
    // already has a name for it.
    throw new BootstrapRefusal(
      'config_unreadable',
      `${path} exists and could not be read: ${(err as Error).message}`,
      { cause: err },
    );
  }
  let auth: unknown;
  try {
    auth = parseToml(text)['auth'];
  } catch (err) {
    throw new BootstrapRefusal(
      'config_unreadable',
      `${path} is not TOML this reader accepts (${(err as Error).message}). The server would refuse it too.`,
      { cause: err },
    );
  }
  const token = typeof auth === 'object' && auth !== null ? (auth as Record<string, unknown>)['token'] : undefined;
  if (typeof token !== 'string' || token.trim() === '') {
    throw new BootstrapRefusal(
      'host_no_token',
      `${path} has no [auth] token, so there is nothing to authorise the host's door with. `
        + 'A Crucible has no anonymous mode.',
    );
  }
  return token;
}

// ----------------------------------------------------------------- the wire

/**
 * THE DOOR'S EVENT SHAPE, pinned 2026-09-14 against `crucible/tasks.py`.
 *
 * `POST /install` answers `200 application/x-ndjson`, one JSON object per
 * line, and the envelope is byte-for-byte what `tasks.py`'s `append_event`
 * produces:
 *
 * ```text
 * {"id": 1, "event": "step",     "data": {"name": "server", "index": 2, "total": 7}}
 * {"id": 2, "event": "progress", "data": {"bytes_done": 4194304, "bytes_total": 120000000, "file": "…part00"}}
 * {"id": 3, "event": "state",    "data": {"code": "no_crucible_distro", "sentence": "…", "action": "run-elevated"}}
 * {"id": 4, "event": "line",     "data": {"text": "server: cpython-3.11.16…", "stream": "stdout"}}
 * {"id": 5, "event": "failed",   "data": {"code": "virtualization_disabled", "message": "…"}}
 * {"id": 6, "event": "done",     "data": {"server": {…}, "release": "…", "backend": "…", "crucible": "…", "steps": […]}}
 * ```
 *
 * Why it is that and not something prettier: 4.7 has the Windows server relay
 * this stream under a task id, and a relay that RESHAPES is a second owner of
 * the shape. So the envelope is tasks.py's, `failed` is called `failed`
 * because that is what tasks.py calls it, and the fields are snake_case
 * because every Crucible wire is, and this package reads every one of them
 * into a camelCase type in exactly the same way.
 *
 * **Two kinds tasks.py does not have, and why they are here.** `state` is the
 * 4c table's answer for this machine — the only stream that walks that table
 * is this one, and an `action` of `run-elevated` is how a caller knows a UAC
 * prompt is about to appear. `line` is a step's own output, which tasks.py
 * deliberately sends to stderr instead of the event stream; it is an event
 * here because `InstallOptions.onLine` is the whole reason a bootstrapping app
 * can show anything at all before there is a page. Neither is a thing to
 * "fix" back into tasks.py's set.
 *
 * **One divergence, named.** tasks.py's `done` carries `{}`. This one carries
 * the install's RESULT, because this door has a caller that is a library
 * function returning an {@link InstallResult} — `server`, `release`,
 * `backend`, `crucible` and the steps — and a `done` with nothing in it would
 * leave that caller with no way to learn any of it. On Windows it cannot go
 * and read the guest's config instead: that `wsl.exe` door is the one PHASE15
 * deletes. The relaying server (4.7) is free to drop `data` when it re-emits
 * `done` under its task id; the page does not need it.
 */
export type HostEvent =
  | { id: number; event: 'step'; data: HostStepData }
  | { id: number; event: 'progress'; data: HostProgressData }
  | { id: number; event: 'state'; data: HostStateData }
  | { id: number; event: 'line'; data: HostLineData }
  | { id: number; event: 'done'; data: HostDoneData }
  | { id: number; event: 'failed'; data: HostFailedData };

/** Every kind the door defines. A line naming anything else is refused, never skipped. */
export const HOST_EVENT_KINDS = ['step', 'progress', 'state', 'line', 'done', 'failed'] as const;
export type HostEventKind = (typeof HOST_EVENT_KINDS)[number];

/** `step`: one step of the PHASE14 sequence began. `index` is 1-based. */
export interface HostStepData {
  name: string;
  index: number;
  total: number;
}

/** `progress`: bytes, while a step downloads. `bytes_total` is null when the size is not known yet. */
export interface HostProgressData {
  bytes_done: number;
  bytes_total: number | null;
  file: string;
}

/** `state`: the 4c table's answer for this machine. `action` is the KIND, as `WslAction` spells it. */
export interface HostStateData {
  code: string;
  sentence: string;
  action: 'run' | 'run-elevated' | 'instruct' | 'link';
}

/** `line`: one line a step printed. The step it belongs to is the last `step` event's. */
export interface HostLineData {
  text: string;
  stream: OutputStream;
}

/** One finished step, as `done` reports it. The camelCase {@link InstallStep} is built from this. */
export interface HostDoneStep {
  name: string;
  argv: string[];
  status: 'running' | 'ok' | 'skipped';
  detail: string;
}

/** `done`: what the machine now has. See the divergence note on {@link HostEvent}. */
export interface HostDoneData {
  /** `[server] name`, the connect URL, and the config's path as the GUEST spells it. */
  server: { name: string; url: string; config_path: string };
  release: string;
  /** The engine the install produced. This door installs the WSL one, so: `cuda-linux`. */
  backend: string;
  /** `<CRUCIBLE_HOME>/server/bin/crucible` inside the guest. */
  crucible: string;
  steps: HostDoneStep[];
}

/** `failed`: the machine cannot be carried further. `code` is a 4c state code, verbatim. */
export interface HostFailedData {
  code: string;
  message: string;
  /** The exact command a person runs, when there is one. */
  command?: string | null;
  detail?: string | null;
}

/** The engine this door installs. `wsl` is the only target PHASE15 4.7 defines. */
export const HOST_INSTALL_TARGET = 'wsl';

/** The request body. `target` names the engine being moved to; anything else is `engine_target_unknown`. */
export interface HostInstallRequestBody {
  target: typeof HOST_INSTALL_TARGET;
  release: string;
  /** `"llm"` or `{"type": "tts", "narrator_engine": "higgs-v3"}` — snake_case, like the rest of the wire. */
  job_types: readonly (string | { type: string; narrator_engine: string })[];
  home?: string;
  bind?: { host?: string; port?: number };
}

/**
 * `globalThis.fetch`'s shape, injectable so the tests script a host without
 * opening a socket.
 */
export type HostFetch = typeof globalThis.fetch;

/**
 * The four callbacks a move's event stream feeds.
 *
 * Its own interface because BOTH readers take it — `requestHostInstall`, which
 * started the move, and {@link watchInstall}, which did not. A watcher that had
 * to name a release and a job list it is not asking for would be inventing two
 * required values to satisfy a type.
 */
export interface HostEventSinks {
  /** Every event, verbatim, as it arrives — including `progress`, which is where the bytes are. */
  onEvent?: (event: HostEvent) => void;
  /** Every line a step printed. The same callback `install()` was given. */
  onLine?: (line: string, stream: OutputStream, step: string) => void;
  /** Every step, as it begins. The same callback `install()` was given. */
  onStep?: (step: InstallStep) => void;
  /** Each 4c state the host reported on its way through the table. */
  onState?: (state: HostStateData) => void;
}

export interface HostInstallOptions extends HostEventSinks {
  /** Which release the host installs. Required here: `install()` has already resolved it. */
  release: string;
  jobTypes: readonly JobTypeRequest[];
  /** `CRUCIBLE_HOME` inside the guest. Omit for the server's own default. */
  home?: string;
  /** What `crucible init` binds. Omit for the server's own defaults (127.0.0.1:7100). */
  bind?: { host?: string; port?: number };
  /**
   * The `fetch` to use. Defaults to `globalThis.fetch`, which is the one
   * legitimate default in this file: it is the PLATFORM's own function on the
   * node this package declares (`engines: node >=20`), not a guess at a value
   * the caller forgot to supply.
   */
  fetchImpl?: HostFetch;
}

// ------------------------------------------------------------- the request

/** The wire's spelling of one job-type request. `narratorEngine` → `narrator_engine`. */
function wireJobType(request: JobTypeRequest): string | { type: string; narrator_engine: string } {
  return typeof request === 'string' ? request : { type: request.type, narrator_engine: request.narratorEngine };
}

/**
 * Ask the host to install, and relay its events as they arrive.
 *
 * Resolves with the {@link InstallResult} the host's `done` event describes.
 * Every other ending is a named refusal:
 *
 *   401                          `host_unauthorized`
 *   409                          `host_install_running`
 *   connection refused/timeout   `host_unreachable`
 *   a `failed` event             the code IT carries, verbatim
 *   a stream with no `done`      `host_install_failed`
 */
export async function requestHostInstall(
  options: HostInstallOptions,
  runner: Runner = processRunner(),
): Promise<InstallResult> {
  const token = hostToken(runner);
  const url = `${HOST_DOOR_URL}${HOST_INSTALL_PATH}`;
  const body: HostInstallRequestBody = {
    target: HOST_INSTALL_TARGET,
    release: options.release,
    job_types: options.jobTypes.map(wireJobType),
    // Absent means "the server's own default", which is a different
    // instruction from an explicit null — so an unset option is not sent.
    ...(options.home === undefined ? {} : { home: options.home }),
    ...(options.bind === undefined ? {} : { bind: options.bind }),
  };

  const doFetch = options.fetchImpl ?? globalThis.fetch;
  let response: Awaited<ReturnType<HostFetch>>;
  try {
    response = await doFetch(url, {
      method: 'POST',
      headers: { 'authorization': `Bearer ${token}`, 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    // fetch rejects for exactly one class of reason: the request never
    // completed — ECONNREFUSED, ENOTFOUND, a dropped socket, a timeout. The
    // host runtime is installed (`install()` checked) and the door did not answer.
    throw new BootstrapRefusal(
      'host_unreachable',
      `the Crucible host is installed on this machine and ${url} did not answer (${(err as Error).message}). `
        + 'Start it from the Startup item, or run `crucible host` from the host runtime.',
      { command: `${hostRuntimeDir(runner)}\\${HOST_ENTRY_POINT} host`, cause: err },
    );
  }

  if (response.status === 401) {
    throw new BootstrapRefusal(
      'host_unauthorized',
      `the host refused the engine token read from ${hostConfigPath(runner)}. `
        + 'The token was rotated and this config was not rewritten, or the host is running against another CRUCIBLE_HOME.',
      { detail: await readBodyText(response) },
    );
  }
  if (response.status === 409) {
    throw new BootstrapRefusal(
      'host_install_running',
      'an install is already running on this machine. There is one install per machine and the second caller waits '
        + 'rather than starting a second walk over the same distro.',
      { detail: await readBodyText(response) },
    );
  }
  if (!response.ok) {
    throw new BootstrapRefusal(
      'host_install_failed',
      `${url} answered ${response.status}, which is not a status the door defines (200, 401 or 409).`,
      { detail: await readBodyText(response) },
    );
  }

  const stream = response.body;
  if (stream === null) {
    throw new BootstrapRefusal(
      'host_install_failed',
      `${url} answered ${response.status} with no body, so no event was ever sent. `
        + 'The door streams newline-delimited JSON and a body-less 200 is not an install that happened.',
    );
  }

  return await readInstallStream(stream, url, options);
}

/** A failed response's body, for the `detail` of the refusal that names it. Never a throw. */
async function readBodyText(response: Awaited<ReturnType<HostFetch>>): Promise<string> {
  try {
    return (await response.text()).trim();
  } catch {
    return '';
  }
}

/**
 * The ndjson stream, one event per line, handed to the callbacks as it arrives.
 *
 * Parsed INCREMENTALLY off the body: an event is acted on the moment its
 * newline arrives, so a step that runs for twenty minutes prints as it goes
 * rather than at the end. A line split across two chunks is one event — the
 * remainder of a chunk is held for the next, which is the same defect
 * `runner.ts` holds partial UTF-8 back for, one layer up.
 */
async function readInstallStream(
  stream: ReadableStream<Uint8Array>,
  url: string,
  options: HostInstallOptions,
): Promise<InstallResult> {
  const reader = stream.getReader();
  const decoder = new TextDecoder('utf-8');
  let pending = '';
  let finished: InstallResult | null = null;
  // A `line` event carries no step name (tasks.py's does not), so the step it
  // belongs to is the last one announced. Derived, not defaulted: the sequence
  // is a sequence, and a line printed before any step has begun is the host
  // breaking its own ordering rather than a line with no owner.
  let currentStep: string | null = null;

  const take = (line: string, final: boolean): void => {
    // A bare newline carries no event. Skipping it is not defaulting a value:
    // there is no value there to default.
    if (line.trim() === '') return;
    const event = parseEventLine(line, url, final);
    if (event.event === 'step') currentStep = event.data.name;
    const result = handleEvent(event, currentStep, url, options);
    if (result !== null) finished = result;
  };

  try {
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      pending += decoder.decode(chunk.value, { stream: true });
      for (;;) {
        const newline = pending.indexOf('\n');
        if (newline < 0) break;
        const line = pending.slice(0, newline);
        pending = pending.slice(newline + 1);
        take(line, false);
      }
    }
  } finally {
    reader.releaseLock();
  }
  pending += decoder.decode();
  take(pending, true);

  if (finished === null) {
    throw new BootstrapRefusal(
      'host_install_failed',
      `${url} closed without a "done" event, so the install did not finish — it stopped. `
        + 'A truncated stream is not a success: the host died, was killed, or lost the connection mid-sequence, '
        + 'and what it had already done is on the machine.',
    );
  }
  return finished;
}

/** One line → one event, or the refusal that says the host broke its own contract. */
function parseEventLine(line: string, url: string, final: boolean): HostEvent {
  let raw: unknown;
  try {
    raw = JSON.parse(line);
  } catch (err) {
    throw new BootstrapRefusal(
      'host_install_failed',
      final
        ? `${url} ended mid-line: ${JSON.stringify(clip(line))} is not JSON, so the last event was cut in half `
          + 'and the stream ended without a "done".'
        : `${url} sent a line that is not JSON: ${JSON.stringify(clip(line))}. The door's body is newline-delimited JSON.`,
      { detail: clip(line), cause: err },
    );
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new BootstrapRefusal('host_install_failed', `${url} sent ${JSON.stringify(clip(line))}, which is not an event object.`);
  }
  const envelope = raw as Record<string, unknown>;
  const kind = envelope['event'];
  if (typeof kind !== 'string' || !(HOST_EVENT_KINDS as readonly string[]).includes(kind)) {
    throw new BootstrapRefusal(
      'host_install_failed',
      `${url} sent an event named ${JSON.stringify(kind)}; the door defines ${HOST_EVENT_KINDS.join(', ')}.`,
      { detail: clip(line) },
    );
  }
  const data = envelope['data'];
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    throw new BootstrapRefusal(
      'host_install_failed',
      `${url} sent a "${kind}" event with no data object. The envelope is tasks.py's: {"id", "event", "data"}.`,
      { detail: clip(line) },
    );
  }
  return raw as HostEvent;
}

/** Hand one event to the callbacks. Returns the result when it was the `done`. */
function handleEvent(event: HostEvent, currentStep: string | null, url: string, options: HostEventSinks): InstallResult | null {
  options.onEvent?.(event);
  switch (event.event) {
    case 'state':
      options.onState?.(event.data);
      // Nothing is dropped for want of a callback: a caller that only gave
      // `onLine` still reads the sentence the table wrote.
      options.onLine?.(event.data.sentence, 'stdout', 'state');
      return null;
    case 'step':
      // `argv` is empty because the door does not carry one: the commands run
      // inside the guest and the host is the thing that knows them. The final
      // `done` reports each step's argv, and that is where an app reads it.
      options.onStep?.({
        name: event.data.name,
        argv: [],
        status: 'running',
        detail: `step ${event.data.index} of ${event.data.total}`,
      });
      return null;
    case 'progress':
      // Bytes are not a line and are not a step transition. A caller that
      // wants them reads `onEvent`; inventing a progress LINE here would put a
      // sentence on the stream that no step ever printed.
      return null;
    case 'line':
      if (currentStep === null) {
        throw new BootstrapRefusal(
          'host_install_failed',
          `${url} sent a "line" event before any "step" event, so there is no step for `
            + `${JSON.stringify(clip(event.data.text))} to belong to. Every line the door sends is a step's output.`,
        );
      }
      options.onLine?.(event.data.text, event.data.stream, currentStep);
      return null;
    case 'failed':
      throw hostRefusal(event.data, url);
    case 'done':
      return doneResult(event.data, url);
  }
}

/**
 * A `failed` event, refused with ITS code. The host sends the 4c state codes
 * verbatim and the app's next move is chosen from the code, so wrapping
 * `virtualization_disabled` in a generic name would delete the only thing the
 * message was carrying.
 */
function hostRefusal(data: HostFailedData, url: string): BootstrapRefusal {
  if (typeof data.code !== 'string' || data.code.trim() === '') {
    return new BootstrapRefusal(
      'host_install_failed',
      `${url} sent a "failed" event with no code. Every refusal the door makes has a name (PHASE15 4.3).`,
      { detail: typeof data.message === 'string' ? data.message : '' },
    );
  }
  const message = typeof data.message === 'string' && data.message.trim() !== ''
    ? data.message
    : `the host refused the install: ${data.code}`;
  // The code is not checked against `BootstrapRefusalCode`: that union is a
  // compile-time type with no runtime list, and the 4c table's codes are
  // GENERATED into the host from `wsl-states.ts` (PHASE15 4.3), so the two
  // sides are already tied by `gen:install --check` rather than by a copy here.
  return new BootstrapRefusal(data.code as BootstrapRefusal['code'], message, {
    ...(typeof data.command === 'string' ? { command: data.command } : {}),
    ...(typeof data.detail === 'string' ? { detail: data.detail } : {}),
  });
}

/** The `done` event → an {@link InstallResult}. Every field is required; a missing one is named. */
export function doneResult(data: HostDoneData, url: string): InstallResult {
  const want = (value: unknown, field: string): string => {
    if (typeof value !== 'string' || value.trim() === '') {
      throw new BootstrapRefusal(
        'host_install_failed',
        `${url} sent a "done" event whose ${field} is ${JSON.stringify(value)}. `
          + 'The result an app gets back is built from this event, so a field that is not there is refused rather than filled in.',
      );
    }
    return value;
  };
  const server = data.server;
  if (typeof server !== 'object' || server === null) {
    throw new BootstrapRefusal('host_install_failed', `${url} sent a "done" event with no server object.`);
  }
  // This door installs the WSL engine (4.7: target `wsl`) and the guest is
  // Linux, so there is exactly one backend a successful install can report.
  // Asked for by name rather than checked against a list this file would then
  // have to keep in step with `release.ts`.
  const expected = backendFor('linux');
  const backend = want(data.backend, 'backend');
  if (backend !== expected) {
    throw new BootstrapRefusal(
      'host_install_failed',
      `${url} says it installed the ${JSON.stringify(backend)} backend. This door installs the WSL engine, `
        + `whose backend is ${expected}; a "done" claiming anything else describes an install this call did not ask for.`,
    );
  }
  if (!Array.isArray(data.steps)) {
    throw new BootstrapRefusal('host_install_failed', `${url} sent a "done" event whose steps is not an array.`);
  }
  return {
    steps: data.steps.map((step, index) => doneStep(step, index, url)),
    server: {
      name: want(server.name, 'server.name'),
      url: want(server.url, 'server.url'),
      configPath: want(server.config_path, 'server.config_path'),
    },
    release: want(data.release, 'release'),
    backend,
    crucible: want(data.crucible, 'crucible'),
  };
}

function doneStep(raw: unknown, index: number, url: string): InstallStep {
  const bad = (why: string): BootstrapRefusal =>
    new BootstrapRefusal('host_install_failed', `${url}: steps[${index}] ${why}.`);
  if (typeof raw !== 'object' || raw === null) throw bad('is not a step object');
  const step = raw as Record<string, unknown>;
  const name = step['name'];
  const argv = step['argv'];
  const status = step['status'];
  const detail = step['detail'];
  if (typeof name !== 'string' || name.trim() === '') throw bad(`has no name (${JSON.stringify(name)})`);
  if (!Array.isArray(argv) || argv.some((word) => typeof word !== 'string')) throw bad('argv is not an array of strings');
  if (status !== 'running' && status !== 'ok' && status !== 'skipped') throw bad(`status is ${JSON.stringify(status)}`);
  if (typeof detail !== 'string') throw bad(`detail is ${JSON.stringify(detail)}`);
  return { name, argv: argv as string[], status, detail };
}

/** A line, short enough to sit inside a message. */
function clip(line: string): string {
  return line.length > 200 ? `${line.slice(0, 200)}…` : line;
}

// ------------------------------------------- PHASE19 2.6: watching the move

/**
 * The five endings of `wsl-outcome.json` (PHASE19 2.2), verbatim.
 *
 * `crucible/host/outcome.py` is the owner; this is the same five words on the
 * wire, and an answer naming a sixth is refused rather than passed through as
 * a string an app would then `switch` on and fall off the end of.
 */
export const WSL_OUTCOME_STATES = ['done', 'reboot-pending', 'cannot', 'failed', 'declined'] as const;
export type WslOutcomeState = (typeof WSL_OUTCOME_STATES)[number];

/**
 * The endings after which the app stops waiting and coordinates (PHASE19 2.8).
 *
 * `done` and `cannot` are where the machine has landed — on the guest engine or
 * on the native one — and `reboot-pending` is where it will stay until somebody
 * presses Restart now. `failed` is NOT one: the tray retries it once, so an app
 * that gave up on the first failure would give up before the retry.
 * `declined` is terminal too: nothing further will happen on that machine.
 */
export const TERMINAL_OUTCOME_STATES: readonly WslOutcomeState[] = ['done', 'cannot', 'reboot-pending', 'declined'];

/** `wsl-outcome.json`, as `GET /install` reports it. Snake_case-free: the file has none. */
export interface WslOutcome {
  state: WslOutcomeState;
  /** A 4c state code or a task failure code. `null` for `done` and `declined`. */
  code: string | null;
  /** What a person reads. `null` for `done` and `declined`. */
  sentence: string | null;
  /** ISO-8601, UTC. */
  at: string;
  /** Which release the move was for. */
  release: string;
  /** How many consecutive tries this was. The tray retries a `failed` once. */
  attempts: number;
}

/** The tray's presence, as `presence.Presence` holds it. */
export interface HostPresenceData {
  distro: string;
  engine: string;
  owner: string;
  detail: string;
}

/** `GET /install`'s document. Three facts and no fourth. */
export interface InstallStatus {
  /** Is a move in flight right now? */
  running: boolean;
  /** How the last one ENDED, or null when none ever has. */
  outcome: WslOutcome | null;
  presence: HostPresenceData;
}

export interface InstallStatusOptions {
  /** See {@link HostInstallOptions.fetchImpl}. */
  fetchImpl?: HostFetch;
}

export interface WatchInstallOptions extends InstallStatusOptions, HostEventSinks {
  /**
   * How long to wait for the TRAY to make its own decision before deciding
   * there is nothing to watch.
   *
   * PHASE19 2.3 moved the move's start into the tray, and the tray makes that
   * decision only after its watch loop has settled a presence — so a caller
   * that asked the instant the tray came up would see `running: false` and no
   * outcome and be right for about a second. Absent, {@link DECISION_WAIT_MS}.
   */
  decisionWaitMs?: number;
  /** Between two polls while waiting for that decision. Absent, {@link DECISION_POLL_MS}. */
  pollMs?: number;
}

/**
 * How long {@link watchInstall} waits for the tray to decide, in ms.
 *
 * CITED, not chosen: `crucible/host/app.py`'s
 * `PRESENCE_SETTLE_CEILING_SECONDS` is the tray's OWN ceiling for that
 * decision, composed there from `presence.WATCH_SECONDS`,
 * `RECIPE_TIMEOUT_SECONDS` and `BOOT_WAIT_SECONDS` — and past it the tray
 * writes "presence never settled" and carries nothing. Waiting longer than the
 * thing being waited for is waiting for something that has already given up,
 * so this is that ceiling as of 1.0.5, with that file's own numbers:
 * `WATCH_SECONDS` 15 + 2 recipes x (`RECIPE_TIMEOUT_SECONDS` 60 +
 * `BOOT_WAIT_SECONDS` 30) = 195 s.
 */
export const DECISION_WAIT_MS = 195_000;

/** Between two `GET /install` polls while waiting. Four a second is a tray, not a load. */
export const DECISION_POLL_MS = 250;

/**
 * `GET /install` — is a move running, how did the last one end, what is the
 * tray's presence (PHASE19 2.6).
 *
 * The same named refusals as {@link requestHostInstall}: `host_unauthorized`,
 * `host_unreachable`, and the door's own code for anything else.
 */
export async function installStatus(
  options: InstallStatusOptions = {},
  runner: Runner = processRunner(),
): Promise<InstallStatus> {
  const url = `${HOST_DOOR_URL}${HOST_INSTALL_PATH}`;
  const response = await doorGet(url, options.fetchImpl, runner);
  let raw: unknown;
  try {
    raw = await response.json();
  } catch (err) {
    throw new BootstrapRefusal(
      'host_install_failed',
      `${url} answered 200 with a body that is not JSON: ${(err as Error).message}`,
      { cause: err },
    );
  }
  return readStatus(raw, url);
}

/**
 * Attach to this machine's move and follow it to its end (PHASE19 2.6).
 *
 * **IT NEVER STARTS ONE.** 2.3 put that in the tray, which is the process that
 * is already there at login and the only one that can resume across the reboot
 * `wsl --install` demands. This waits for the tray's decision, replays the ring
 * so the caller sees the step it joined at, follows the stream, and returns the
 * status once the move is over.
 *
 * Returns the status as it stands when there is nothing left to follow — which
 * on a machine that declined, or that has already finished, is immediate.
 */
export async function watchInstall(
  options: WatchInstallOptions = {},
  runner: Runner = processRunner(),
): Promise<InstallStatus> {
  const waitMs = options.decisionWaitMs ?? DECISION_WAIT_MS;
  const pollMs = options.pollMs ?? DECISION_POLL_MS;
  const deadline = Date.now() + waitMs;
  let status = await installStatus(options, runner);
  // THE TRAY MAY NOT HAVE DECIDED YET. Its decision waits on a presence
  // measurement, so "nothing running and nothing recorded" is a state a caller
  // can legitimately see for a second after the tray starts.
  while (!status.running && status.outcome === null && Date.now() < deadline) {
    await sleep(pollMs);
    status = await installStatus(options, runner);
  }
  for (;;) {
    const attached = await attachInstall(options, runner);
    if (!attached) break;
    status = await installStatus(options, runner);
    // A `failed` is retried by the tray ONCE (2.2), and that retry is a second
    // move on the same machine. Following it is what makes "watch until the
    // outcome is terminal" true rather than "watch the first attempt".
    if (!status.running) break;
  }
  return await installStatus(options, runner);
}

/** Follow the event stream to its end. False when the door had nothing to show. */
async function attachInstall(options: WatchInstallOptions, runner: Runner): Promise<boolean> {
  const url = `${HOST_DOOR_URL}${HOST_INSTALL_EVENTS_PATH}`;
  const response = await doorGet(url, options.fetchImpl, runner, { allow404: true });
  if (response.status === 404) return false;
  const stream = response.body;
  if (stream === null) {
    throw new BootstrapRefusal(
      'host_install_failed',
      `${url} answered ${response.status} with no body. The door streams newline-delimited JSON.`,
    );
  }
  await readEventStream(stream, url, options);
  return true;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** One authenticated GET on the door, with this file's named refusals. */
async function doorGet(
  url: string,
  fetchImpl: HostFetch | undefined,
  runner: Runner,
  options: { allow404?: boolean } = {},
): Promise<Awaited<ReturnType<HostFetch>>> {
  const token = hostToken(runner);
  const doFetch = fetchImpl ?? globalThis.fetch;
  let response: Awaited<ReturnType<HostFetch>>;
  try {
    response = await doFetch(url, { method: 'GET', headers: { 'authorization': `Bearer ${token}` } });
  } catch (err) {
    throw new BootstrapRefusal(
      'host_unreachable',
      `the Crucible host is installed on this machine and ${url} did not answer (${(err as Error).message}). `
        + 'Start it from the Startup item, or run `crucible host` from the host runtime.',
      { command: `${hostRuntimeDir(runner)}\\${HOST_ENTRY_POINT} host`, cause: err },
    );
  }
  if (response.status === 401) {
    throw new BootstrapRefusal(
      'host_unauthorized',
      `the host refused the engine token read from ${hostConfigPath(runner)}.`,
      { detail: await readBodyText(response) },
    );
  }
  if (options.allow404 === true && response.status === 404) return response;
  if (!response.ok) {
    const body = await readBodyText(response);
    throw new BootstrapRefusal(
      'host_install_failed',
      `${url} answered ${response.status}: ${body === '' ? '(no body)' : body}`,
      { detail: body },
    );
  }
  return response;
}

/** `GET /install`'s body, with every field required and a missing one named. */
function readStatus(raw: unknown, url: string): InstallStatus {
  const bad = (why: string): BootstrapRefusal =>
    new BootstrapRefusal('host_install_failed', `${url} ${why}.`);
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw bad('did not answer an object');
  const body = raw as Record<string, unknown>;
  if (typeof body['running'] !== 'boolean') throw bad(`says running=${JSON.stringify(body['running'])}`);
  const presence = body['presence'];
  if (typeof presence !== 'object' || presence === null) throw bad('answered no presence');
  const seen = presence as Record<string, unknown>;
  for (const field of ['distro', 'engine', 'owner', 'detail']) {
    if (typeof seen[field] !== 'string') throw bad(`presence.${field} is ${JSON.stringify(seen[field])}`);
  }
  const recorded = body['outcome'];
  return {
    running: body['running'],
    outcome: recorded === null || recorded === undefined ? null : readOutcome(recorded, url),
    presence: {
      distro: seen['distro'] as string,
      engine: seen['engine'] as string,
      owner: seen['owner'] as string,
      detail: seen['detail'] as string,
    },
  };
}

function readOutcome(raw: unknown, url: string): WslOutcome {
  const bad = (why: string): BootstrapRefusal =>
    new BootstrapRefusal('host_install_failed', `${url}: the outcome ${why}.`);
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw bad('is not an object');
  const body = raw as Record<string, unknown>;
  const state = body['state'];
  if (typeof state !== 'string' || !(WSL_OUTCOME_STATES as readonly string[]).includes(state)) {
    throw bad(`says state=${JSON.stringify(state)}; the states are ${WSL_OUTCOME_STATES.join(', ')}`);
  }
  const code = body['code'];
  const sentence = body['sentence'];
  const at = body['at'];
  const release = body['release'];
  const attempts = body['attempts'];
  if (code !== null && typeof code !== 'string') throw bad(`says code=${JSON.stringify(code)}`);
  if (sentence !== null && typeof sentence !== 'string') throw bad(`says sentence=${JSON.stringify(sentence)}`);
  if (typeof at !== 'string' || at === '') throw bad('carries no `at`');
  if (typeof release !== 'string' || release === '') throw bad('names no release');
  if (typeof attempts !== 'number' || !Number.isInteger(attempts)) throw bad(`says attempts=${JSON.stringify(attempts)}`);
  return { state: state as WslOutcomeState, code, sentence, at, release, attempts };
}

/**
 * The ndjson, dispatched to the callbacks. The SAME parser the POST's stream
 * uses — `parseEventLine` and `handleEvent` — so an attacher and the poster
 * cannot read one stream two ways.
 */
async function readEventStream(
  stream: ReadableStream<Uint8Array>,
  url: string,
  options: WatchInstallOptions,
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder('utf-8');
  let pending = '';
  let currentStep: string | null = null;
  const relay: HostEventSinks = {
    ...(options.onEvent === undefined ? {} : { onEvent: options.onEvent }),
    ...(options.onLine === undefined ? {} : { onLine: options.onLine }),
    ...(options.onStep === undefined ? {} : { onStep: options.onStep }),
    ...(options.onState === undefined ? {} : { onState: options.onState }),
  };
  const take = (line: string, final: boolean): void => {
    if (line.trim() === '') return;
    const event = parseEventLine(line, url, final);
    if (event.event === 'step') currentStep = event.data.name;
    // A WATCHER IS NOT THE CALLER OF THE MOVE. `failed` throws for the poster,
    // because its `install()` must refuse by that code; here it is an EVENT
    // about somebody else's move, and the outcome file is what says how it
    // ended. So the throw is caught and the stream is read to its end.
    try {
      handleEvent(event, currentStep, url, relay);
    } catch (err) {
      if (!(err instanceof BootstrapRefusal)) throw err;
    }
  };
  try {
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      pending += decoder.decode(chunk.value, { stream: true });
      for (;;) {
        const newline = pending.indexOf('\n');
        if (newline < 0) break;
        take(pending.slice(0, newline), false);
        pending = pending.slice(newline + 1);
      }
    }
  } finally {
    reader.releaseLock();
  }
  pending += decoder.decode();
  take(pending, true);
}
