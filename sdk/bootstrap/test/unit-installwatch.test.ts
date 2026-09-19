/**
 * `installStatus()` and `watchInstall()` — PHASE19-AUTOMATIC-WSL.md 2.6.
 *
 * The door is a fake `fetch`, so nothing opens a socket and nothing needs a
 * tray: what is under test is the CLIENT's reading of `GET /install` and
 * `GET /install/events`, and its rule that it never starts a move.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  DECISION_POLL_MS,
  installStatus,
  TERMINAL_OUTCOME_STATES,
  watchInstall,
  WSL_OUTCOME_STATES,
  type HostEvent,
  type InstallStep,
} from '../src/index.js';
import {
  fakeWatchDoor,
  FAKE_PRESENCE,
  FakeRunner,
  HOST_CMD,
  HOST_CONFIG,
  HOST_CONFIG_PATH,
  HOST_DONE,
  refusal,
  WIN_ENV,
} from './fake.js';

const winRunner = (): FakeRunner =>
  new FakeRunner({ platform: 'win32', env: WIN_ENV, files: { [HOST_CMD]: '@echo off', [HOST_CONFIG_PATH]: HOST_CONFIG } }, []);

const CANNOT = {
  state: 'cannot',
  code: 'virtualization_disabled',
  sentence: "Windows cannot start a virtual machine: HCS_E_HYPERV_NOT_INSTALLED",
  at: '2026-09-19T01:02:03+00:00',
  release: '1.0.5',
  attempts: 1,
};

test('the five outcome states are the five `crucible/host/outcome.py` writes', () => {
  assert.deepEqual([...WSL_OUTCOME_STATES], ['done', 'reboot-pending', 'cannot', 'failed', 'declined']);
  // `failed` is NOT terminal: the tray retries it once (2.2), so an app that
  // stopped waiting on the first failure would stop before the retry.
  assert.deepEqual([...TERMINAL_OUTCOME_STATES], ['done', 'cannot', 'reboot-pending', 'declined']);
  assert.equal(TERMINAL_OUTCOME_STATES.includes('failed' as never), false);
});

test('installStatus reads the three facts, with the bearer from the host config', async () => {
  const door = fakeWatchDoor({ statuses: [{ running: true, outcome: CANNOT }] });
  const status = await installStatus({ fetchImpl: door.fetchImpl }, winRunner());
  assert.equal(status.running, true);
  assert.equal(status.outcome?.state, 'cannot');
  assert.equal(status.outcome?.code, 'virtualization_disabled');
  assert.equal(status.outcome?.attempts, 1);
  assert.deepEqual(status.presence, FAKE_PRESENCE);
  assert.equal(door.requests[0]?.method, 'GET');
  assert.equal(door.requests[0]?.url, 'http://127.0.0.1:7101/install');
  assert.equal(door.requests[0]?.authorization, 'Bearer host-token-not-a-secret');
});

test('a machine that has never run one answers outcome: null, and that is a fact not a blank', async () => {
  const door = fakeWatchDoor({ statuses: [{ running: false }] });
  const status = await installStatus({ fetchImpl: door.fetchImpl }, winRunner());
  assert.equal(status.outcome, null);
  assert.equal(status.running, false);
});

test('an outcome naming a state nobody defined is refused, never passed through', async () => {
  const door = fakeWatchDoor({ statuses: [{ outcome: { ...CANNOT, state: 'sideways' } }] });
  const r = await refusal(installStatus({ fetchImpl: door.fetchImpl }, winRunner()));
  assert.equal(r.code, 'host_install_failed');
  assert.match(r.message, /the states are done, reboot-pending, cannot, failed, declined/);
});

test('an outcome with no release is refused: an ending is about a release', async () => {
  const door = fakeWatchDoor({ statuses: [{ outcome: { ...CANNOT, release: '' } }] });
  const r = await refusal(installStatus({ fetchImpl: door.fetchImpl }, winRunner()));
  assert.equal(r.code, 'host_install_failed');
  assert.match(r.message, /names no release/);
});

test('watchInstall replays the ring and follows, then answers with the outcome', async () => {
  const steps: InstallStep[] = [];
  const lines: string[] = [];
  const events: HostEvent['event'][] = [];
  const door = fakeWatchDoor({
    statuses: [
      { running: true },
      { running: false, outcome: { ...CANNOT, state: 'done', code: null, sentence: null } },
    ],
    events: [
      ['step', { name: 'import-distro', index: 2, total: 11 }],
      ['progress', { bytes_done: 4194304, bytes_total: 356515840, file: 'ubuntu-noble-wsl-amd64-wsl.rootfs.tar.gz' }],
      ['line', { text: "Downloading Ubuntu's own WSL image", stream: 'stdout' }],
      HOST_DONE,
    ],
  });
  const status = await watchInstall({
    fetchImpl: door.fetchImpl,
    onStep: (step) => steps.push(step),
    onLine: (line, _stream, step) => lines.push(`${step}: ${line}`),
    onEvent: (event) => events.push(event.event),
  }, winRunner());

  assert.deepEqual(steps.map((s) => s.name), ['import-distro']);
  assert.deepEqual(lines, ['import-distro: Downloading Ubuntu\'s own WSL image']);
  assert.deepEqual(events, ['step', 'progress', 'line', 'done']);
  assert.equal(status.outcome?.state, 'done');
  // It never posts: the tray started the move (2.3).
  assert.equal(door.requests.some((r) => r.method === 'POST'), false);
});

test('a `failed` event on a watched stream is an event, not this caller\'s refusal', async () => {
  // The poster's `install()` must refuse with that code; a WATCHER is reading
  // somebody else's move, and the outcome file is what says how it ended.
  const door = fakeWatchDoor({
    statuses: [{ running: true }, { running: false, outcome: { ...CANNOT, state: 'failed', code: 'rootfs_download_failed' } }],
    events: [
      ['step', { name: 'import-distro', index: 2, total: 11 }],
      ['failed', { code: 'rootfs_download_failed', message: 'the image would not download' }],
    ],
  });
  const status = await watchInstall({ fetchImpl: door.fetchImpl }, winRunner());
  assert.equal(status.outcome?.state, 'failed');
  assert.equal(status.outcome?.code, 'rootfs_download_failed');
});

test('nothing to watch is not an error: a declined machine answers at once', async () => {
  const declined = { state: 'declined', code: null, sentence: null, at: '2026-09-19T00:00:00+00:00', release: '1.0.5', attempts: 0 };
  const door = fakeWatchDoor({ statuses: [{ running: false, outcome: declined }] });
  const status = await watchInstall({ fetchImpl: door.fetchImpl }, winRunner());
  assert.equal(status.outcome?.state, 'declined');
  // One status read, one 404 from the events door, one status read back.
  assert.deepEqual(door.requests.map((r) => new URL(r.url).pathname), ['/install', '/install/events', '/install']);
});

test('watchInstall waits for the TRAY to decide before giving up on it', async () => {
  // 2.3 starts the move only after the tray's watch loop settles a presence,
  // so "nothing running and nothing recorded" is a state a caller legitimately
  // sees for a moment after the tray comes up.
  const door = fakeWatchDoor({
    statuses: [{ running: false }, { running: false }, { running: true }, { running: false, outcome: { ...CANNOT, state: 'done', code: null, sentence: null } }],
    events: [HOST_DONE],
  });
  const status = await watchInstall({ fetchImpl: door.fetchImpl, pollMs: 1, decisionWaitMs: 5_000 }, winRunner());
  assert.equal(status.outcome?.state, 'done');
  const polls = door.requests.filter((r) => new URL(r.url).pathname === '/install').length;
  assert.ok(polls >= 4, `it polled ${polls} times while the tray was deciding`);
});

test('a tray that never decides is given up on rather than waited for forever', async () => {
  const door = fakeWatchDoor({ statuses: [{ running: false }] });
  const status = await watchInstall({ fetchImpl: door.fetchImpl, pollMs: 1, decisionWaitMs: 20 }, winRunner());
  assert.equal(status.outcome, null);
  assert.equal(status.running, false);
});

test('the poll interval and the decision wait are named constants, not numbers in a loop', () => {
  assert.equal(DECISION_POLL_MS, 250);
});

test('a door that will not answer is host_unreachable, naming what to start', async () => {
  const door = fakeWatchDoor({ statuses: [] });
  const fetchImpl = (async () => { throw new Error('connect ECONNREFUSED 127.0.0.1:7101'); }) as typeof door.fetchImpl;
  const r = await refusal(installStatus({ fetchImpl }, winRunner()));
  assert.equal(r.code, 'host_unreachable');
  assert.match(r.command ?? '', /crucible\.cmd host$/);
});
