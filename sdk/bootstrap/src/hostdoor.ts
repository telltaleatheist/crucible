import { crucibleAppData } from './distro.js';
import { backendFor, releaseAssetUrl } from './release.js';
import { BootstrapRefusal } from './errors.js';
import type { InstallResult, InstallStep, JobTypeRequest } from './install.js';
import { processRunner, type OutputStream, type Runner } from './runner.js';
import { parseToml } from './toml.js';

/** The host door's loopback port. */
export const HOST_DOOR_PORT = 7101;

/** The door's base URL. */
export const HOST_DOOR_URL = `http://127.0.0.1:${HOST_DOOR_PORT}`;

/** `POST` here to run the install sequence; `GET` it for the status. */
export const HOST_INSTALL_PATH = '/install';

/** `GET` here to attach to a move in flight from its current step. */
export const HOST_INSTALL_EVENTS_PATH = '/install/events';

/** The Windows host's relocatable entry point; its presence means the host is installed. */
export const HOST_ENTRY_POINT = 'crucible.cmd';

/** `%LOCALAPPDATA%\Crucible\host` — where `install.ps1` puts the host runtime. */
export function hostRuntimeDir(runner: Runner): string {
  return crucibleAppData(runner, 'host');
}

/** `%LOCALAPPDATA%\Crucible\config.toml`, the host-mode server's own config. */
export function hostConfigPath(runner: Runner): string {
  return crucibleAppData(runner, 'config.toml');
}

/** Is the host runtime on this machine? */
export function hostInstalled(runner: Runner): boolean {
  return runner.fileExists(`${hostRuntimeDir(runner)}\\${HOST_ENTRY_POINT}`);
}

/** The line a person runs to put a host on this machine. */
export function hostInstallCommand(release: string): string {
  return `irm ${releaseAssetUrl(release, 'install.ps1')} | iex`;
}

/** The door's bearer: the engine token, read from the host's own config. */
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

/** One event on the host door's ndjson stream. */
export type HostEvent =
  | { id: number; event: 'step'; data: HostStepData }
  | { id: number; event: 'progress'; data: HostProgressData }
  | { id: number; event: 'state'; data: HostStateData }
  | { id: number; event: 'line'; data: HostLineData }
  | { id: number; event: 'done'; data: HostDoneData }
  | { id: number; event: 'failed'; data: HostFailedData }
  | HostUnknownEvent;

/** An event kind this build does not know, carried to `onEvent` rather than refused. */
export interface HostUnknownEvent {
  id: number;
  event: 'unknown';
  /** The event name the host actually sent. */
  kind: string;
  data: Record<string, unknown>;
}

/** Every kind the door defines. */
export const HOST_EVENT_KINDS = ['step', 'progress', 'state', 'line', 'done', 'failed'] as const;
export type HostEventKind = (typeof HOST_EVENT_KINDS)[number];

/** `step`: one install step began; `index` is 1-based. */
export interface HostStepData {
  name: string;
  index: number;
  total: number;
}

/** `progress`: bytes while a step downloads; `bytes_total` is null when unknown. */
export interface HostProgressData {
  bytes_done: number;
  bytes_total: number | null;
  file: string;
}

/** `state`: the WSL state table's answer for this machine. */
export interface HostStateData {
  code: string;
  sentence: string;
  action: 'run' | 'run-elevated' | 'instruct' | 'link';
}

/** `line`: one line a step printed. */
export interface HostLineData {
  text: string;
  stream: OutputStream;
}

/** One finished step, as `done` reports it. */
export interface HostDoneStep {
  name: string;
  argv: string[];
  status: 'running' | 'ok' | 'skipped';
  detail: string;
}

/** `done`: what the machine now has. */
export interface HostDoneData {
  /** `[server] name`, the connect URL, and the config's path as the GUEST spells it. */
  server: { name: string; url: string; config_path: string };
  release: string;
  /** The engine the install produced. */
  backend: string;
  /** `<CRUCIBLE_HOME>/server/bin/crucible` inside the guest. */
  crucible: string;
  steps: HostDoneStep[];
}

/** `failed`: the machine cannot be carried further. */
export interface HostFailedData {
  code: string;
  message: string;
  /** The exact command a person runs, when there is one. */
  command?: string | null;
  detail?: string | null;
}

/** The engine this door installs. */
export const HOST_INSTALL_TARGET = 'wsl';

/** The `POST /install` request body. */
export interface HostInstallRequestBody {
  target: typeof HOST_INSTALL_TARGET;
  release: string;
  /**
   * `"llm"` or `{"type": "tts", "narrator_engine": "higgs-v3"}` — snake_case, like the rest of the
   * wire.
   */
  job_types: readonly (string | { type: string; narrator_engine: string })[];
  home?: string;
  bind?: { host?: string; port?: number };
}

/** `globalThis.fetch`'s shape, injectable so the tests script a host without opening a socket. */
export type HostFetch = typeof globalThis.fetch;

/** The four callbacks a move's event stream feeds. */
export interface HostEventSinks {
  /** Every event, verbatim, as it arrives — including `progress`, which is where the bytes are. */
  onEvent?: (event: HostEvent) => void;
  /** Every line a step printed. */
  onLine?: (line: string, stream: OutputStream, step: string) => void;
  /** Every step, as it begins. */
  onStep?: (step: InstallStep) => void;
  /** Each 4c state the host reported on its way through the table. */
  onState?: (state: HostStateData) => void;
}

export interface HostInstallOptions extends HostEventSinks {
  /** Which release the host installs. */
  release: string;
  jobTypes: readonly JobTypeRequest[];
  /** `CRUCIBLE_HOME` inside the guest. */
  home?: string;
  /** What `crucible init` binds. */
  bind?: { host?: string; port?: number };
  /** The `fetch` to use; defaults to `globalThis.fetch`. */
  fetchImpl?: HostFetch;
}

function wireJobType(request: JobTypeRequest): string | { type: string; narrator_engine: string } {
  return typeof request === 'string' ? request : { type: request.type, narrator_engine: request.narratorEngine };
}

/** Ask the host to install, and relay its events as they arrive. */
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
    throw new BootstrapRefusal(
      'host_unreachable',
      `the Crucible host is installed on this machine and ${url} did not answer (${(err as Error).message}). `
        + 'Start it from the Startup item, or run `crucible orchestrator` from the host runtime.',
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

async function readBodyText(response: Awaited<ReturnType<HostFetch>>): Promise<string> {
  try {
    return (await response.text()).trim();
  } catch {
    return '';
  }
}

async function readInstallStream(
  stream: ReadableStream<Uint8Array>,
  url: string,
  options: HostInstallOptions,
): Promise<InstallResult> {
  const reader = stream.getReader();
  const decoder = new TextDecoder('utf-8');
  let pending = '';
  let finished: InstallResult | null = null;
  let currentStep: string | null = null;

  const take = (line: string, final: boolean): void => {
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
  if (typeof kind !== 'string') {
    throw new BootstrapRefusal(
      'host_install_failed',
      `${url} sent an event named ${JSON.stringify(kind)}, which is not a name at all.`,
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
  if (!(HOST_EVENT_KINDS as readonly string[]).includes(kind)) {
    return {
      id: envelope['id'] as number,
      event: 'unknown',
      kind,
      data: data as Record<string, unknown>,
    };
  }
  return raw as HostEvent;
}

function handleEvent(event: HostEvent, currentStep: string | null, url: string, options: HostEventSinks): InstallResult | null {
  options.onEvent?.(event);
  switch (event.event) {
    case 'state':
      options.onState?.(event.data);
      options.onLine?.(event.data.sentence, 'stdout', 'state');
      return null;
    case 'step':
      options.onStep?.({
        name: event.data.name,
        argv: [],
        status: 'running',
        detail: `step ${event.data.index} of ${event.data.total}`,
      });
      return null;
    case 'progress':
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
    case 'unknown':
      return null;
  }
}

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
  return new BootstrapRefusal(data.code as BootstrapRefusal['code'], message, {
    ...(typeof data.command === 'string' ? { command: data.command } : {}),
    ...(typeof data.detail === 'string' ? { detail: data.detail } : {}),
  });
}

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

function clip(line: string): string {
  return line.length > 200 ? `${line.slice(0, 200)}…` : line;
}

/** The five endings of `wsl-outcome.json`, verbatim. */
export const WSL_OUTCOME_STATES = ['done', 'reboot-pending', 'cannot', 'failed', 'declined'] as const;
export type WslOutcomeState = (typeof WSL_OUTCOME_STATES)[number];

/** The outcome states after which an app stops waiting. */
export const TERMINAL_OUTCOME_STATES: readonly WslOutcomeState[] = ['done', 'cannot', 'reboot-pending', 'declined'];

/** `wsl-outcome.json`, as `GET /install` reports it. */
export interface WslOutcome {
  state: WslOutcomeState;
  /** A 4c state code or a task failure code. */
  code: string | null;
  /** What a person reads. */
  sentence: string | null;
  /** ISO-8601, UTC. */
  at: string;
  /** Which release the move was for. */
  release: string;
  /** How many consecutive tries this was. */
  attempts: number;
}

/** The tray's presence, as `presence.Presence` holds it. */
export interface HostPresenceData {
  distro: string;
  engine: string;
  owner: string;
  detail: string;
}

/** `GET /install`'s document. */
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
  /** How long to wait for the tray to decide before concluding there is nothing to watch. */
  decisionWaitMs?: number;
  /** Between two polls while waiting for that decision. */
  pollMs?: number;
}

/** How long {@link watchInstall} waits for the tray to decide, in ms. */
export const DECISION_WAIT_MS = 195_000;

/** Between two `GET /install` polls while waiting. */
export const DECISION_POLL_MS = 250;

/** `GET /install`: whether a move is running, how the last one ended, and the tray's presence. */
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

/** Attach to this machine's engine move and follow it to its end; never starts one. */
export async function watchInstall(
  options: WatchInstallOptions = {},
  runner: Runner = processRunner(),
): Promise<InstallStatus> {
  const waitMs = options.decisionWaitMs ?? DECISION_WAIT_MS;
  const pollMs = options.pollMs ?? DECISION_POLL_MS;
  const deadline = Date.now() + waitMs;
  let status = await installStatus(options, runner);
  while (!status.running && status.outcome === null && Date.now() < deadline) {
    await sleep(pollMs);
    status = await installStatus(options, runner);
  }
  for (;;) {
    const attached = await attachInstall(options, runner);
    if (!attached) break;
    status = await installStatus(options, runner);
    if (!status.running) break;
  }
  return await installStatus(options, runner);
}

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
        + 'Start it from the Startup item, or run `crucible orchestrator` from the host runtime.',
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
