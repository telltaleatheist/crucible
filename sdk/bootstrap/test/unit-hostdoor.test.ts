/**
 * The host's loopback door, from this side (PHASE15-HOST.md 4.3): where the
 * pack is, which token authorises it, what the request looks like on the wire,
 * and every way the stream can end — each refused by its own name.
 *
 * The event shape under test is `crucible/tasks.py`'s envelope,
 * `{"id", "event", "data"}`, because 4.7 has the Windows server relay this
 * stream under a task id and a relay that reshapes would be a second owner of
 * the shape.
 *
 * Nothing here opens a socket. {@link fakeHostDoor} is a `fetchImpl` that
 * replays scripted chunks, which is also how the "a line split across two
 * chunks is still one event" case is provable at all.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  HOST_DOOR_PORT,
  HOST_DOOR_URL,
  HOST_ENTRY_POINT,
  hostConfigPath,
  hostInstallCommand,
  hostInstalled,
  hostPackDir,
  hostToken,
  requestHostInstall,
  type HostEvent,
  type InstallStep,
  type OutputStream,
} from '../src/index.js';
import {
  doorLine,
  fakeHostDoor,
  FakeRunner,
  HOST_CMD,
  HOST_CONFIG,
  HOST_CONFIG_PATH,
  HOST_DIR,
  HOST_DONE,
  HOST_DONE_DATA,
  refusal,
  WIN_ENV,
} from './fake.js';

/** A Windows machine with the host pack unpacked and its config written. */
function hostMachine(files: Record<string, string> = {}): FakeRunner {
  return new FakeRunner({ platform: 'win32', env: WIN_ENV, files: { [HOST_CMD]: '@echo off', [HOST_CONFIG_PATH]: HOST_CONFIG, ...files } }, []);
}

/** A `step` event, which is what every `line` after it belongs to. */
const STEP = (name: string, index = 1, total = 7): readonly [string, unknown] => ['step', { name, index, total }];

test('the door is 127.0.0.1:7101, and the URL is built from the port rather than spelled twice', () => {
  assert.equal(HOST_DOOR_PORT, 7101);
  assert.equal(HOST_DOOR_URL, 'http://127.0.0.1:7101');
});

test('the host pack is %LOCALAPPDATA%\\Crucible\\host, read from the environment and never assembled', () => {
  const runner = hostMachine();
  assert.equal(hostPackDir(runner), HOST_DIR);
  assert.equal(hostConfigPath(runner), HOST_CONFIG_PATH);
  // A username appears nowhere in the code that built it: the env said so.
  assert.equal(hostPackDir(new FakeRunner({ platform: 'win32', env: { LOCALAPPDATA: 'D:\\appdata' } }, [])), 'D:\\appdata\\Crucible\\host');
});

test('LOCALAPPDATA unset refuses host_unresponsive rather than guessing a path', async () => {
  const runner = new FakeRunner({ platform: 'win32', env: {} }, []);
  const r = await refusal(Promise.resolve().then(() => hostPackDir(runner)));
  assert.equal(r.code, 'host_unresponsive');
  assert.match(r.message, /LOCALAPPDATA is not set/);
});

test('"the host is installed" is crucible.cmd — the .cmd, because pip\'s .exe launchers do not survive a move', () => {
  assert.equal(HOST_ENTRY_POINT, 'crucible.cmd');
  assert.equal(hostInstalled(hostMachine()), true);
  assert.equal(hostInstalled(new FakeRunner({ platform: 'win32', env: WIN_ENV, files: {} }, [])), false);
  // A pack directory with the .exe launchers and no .cmd is NOT an installed host.
  const exeOnly = new FakeRunner({ platform: 'win32', env: WIN_ENV, files: { [`${HOST_DIR}\\Scripts\\crucible.exe`]: 'MZ' } }, []);
  assert.equal(hostInstalled(exeOnly), false);
});

test('hostToken reads [auth] token out of the host\'s own config.toml', () => {
  assert.equal(hostToken(hostMachine()), 'host-token-not-a-secret');
});

test('no config.toml at all refuses host_no_token by name', async () => {
  const runner = new FakeRunner({ platform: 'win32', env: WIN_ENV, files: { [HOST_CMD]: '@echo off' } }, []);
  const r = await refusal(Promise.resolve().then(() => hostToken(runner)));
  assert.equal(r.code, 'host_no_token');
  assert.match(r.message, /does not exist/);
});

for (const [label, text] of [
  ['no [auth] table', '[server]\nname = "x"\nhost = "127.0.0.1"\nport = 7100\n'],
  ['an [auth] table with no token', '[server]\nname = "x"\n\n[auth]\nrotated = 1\n'],
  ['an empty token', '[auth]\ntoken = ""\n'],
] as const) {
  test(`a config with ${label} refuses host_no_token, never an empty bearer`, async () => {
    const runner = new FakeRunner({ platform: 'win32', env: WIN_ENV, files: { [HOST_CMD]: '@echo off', [HOST_CONFIG_PATH]: text } }, []);
    const r = await refusal(Promise.resolve().then(() => hostToken(runner)));
    assert.equal(r.code, 'host_no_token');
    assert.match(r.message, /no \[auth\] token/);
  });
}

test('a config.toml that is not TOML is config_unreadable — present and wrong is not the same fact as absent', async () => {
  const runner = new FakeRunner({ platform: 'win32', env: WIN_ENV, files: { [HOST_CMD]: '@echo off', [HOST_CONFIG_PATH]: '{ "auth": {} }' } }, []);
  const r = await refusal(Promise.resolve().then(() => hostToken(runner)));
  assert.equal(r.code, 'config_unreadable');
});

test('the request: POST /install, the engine token as the bearer, target wsl and the job list in the body', async () => {
  const door = fakeHostDoor({ events: [HOST_DONE] });
  const result = await requestHostInstall(
    {
      release: '0.6.0',
      jobTypes: ['llm', { type: 'tts', narratorEngine: 'higgs-v3' }],
      home: '/srv/crucible',
      bind: { host: '0.0.0.0', port: 7200 },
      fetchImpl: door.fetchImpl,
    },
    hostMachine(),
  );
  assert.equal(door.requests.length, 1);
  const request = door.requests[0];
  assert.equal(request?.url, 'http://127.0.0.1:7101/install');
  assert.equal(request?.method, 'POST');
  assert.equal(request?.authorization, 'Bearer host-token-not-a-secret');
  assert.equal(request?.contentType, 'application/json');
  assert.deepEqual(request?.body, {
    target: 'wsl',
    release: '0.6.0',
    // snake_case on the wire, like every other Crucible body.
    job_types: ['llm', { type: 'tts', narrator_engine: 'higgs-v3' }],
    home: '/srv/crucible',
    bind: { host: '0.0.0.0', port: 7200 },
  });
  assert.equal(result.release, '0.6.0');
  assert.equal(result.crucible, '/home/crucible/.crucible/server/bin/crucible');
  assert.equal(result.server.configPath, HOST_DONE_DATA.server.config_path);
});

test('an unset home or bind is ABSENT from the body, not null: "the server\'s default" is a different instruction', async () => {
  const door = fakeHostDoor({ events: [HOST_DONE] });
  await requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine());
  assert.deepEqual(door.requests[0]?.body, { target: 'wsl', release: '0.6.0', job_types: ['echo'] });
});

test('every event reaches the callbacks as it arrives: state, step, progress, line, done', async () => {
  const events: HostEvent[] = [];
  const lines: string[] = [];
  const steps: string[] = [];
  const door = fakeHostDoor({
    events: [
      ['state', { code: 'no_crucible_distro', sentence: 'There is no Crucible distro on this machine yet.', action: 'run-elevated' }],
      STEP('server-pack', 2, 7),
      ['progress', { bytes_done: 4194304, bytes_total: 120000000, file: 'crucible-env-server-cuda-linux-0.6.0.tar.zst.part00' }],
      ['line', { text: 'server-pack: part00', stream: 'stdout' }],
      STEP('install-llm', 4, 7),
      ['line', { text: 'WARNING: slow mirror', stream: 'stderr' }],
      HOST_DONE,
    ],
  });
  const result = await requestHostInstall(
    {
      release: '0.6.0',
      jobTypes: ['llm'],
      fetchImpl: door.fetchImpl,
      onEvent: (event) => events.push(event),
      onLine: (line, stream: OutputStream, step) => lines.push(`${step}/${stream}: ${line}`),
      onStep: (step: InstallStep) => steps.push(`${step.name}:${step.status}:${step.detail}`),
    },
    hostMachine(),
  );
  assert.deepEqual(events.map((e) => e.event), ['state', 'step', 'progress', 'line', 'step', 'line', 'done']);
  // The ids are the envelope's, monotonic and 1-based, as tasks.py writes them.
  assert.deepEqual(events.map((e) => e.id), [1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual(steps, ['server-pack:running:step 2 of 7', 'install-llm:running:step 4 of 7']);
  // A line carries no step of its own: it belongs to the last step announced.
  assert.deepEqual(lines, [
    'state/stdout: There is no Crucible distro on this machine yet.',
    'server-pack/stdout: server-pack: part00',
    'install-llm/stderr: WARNING: slow mirror',
  ]);
  assert.deepEqual(result.steps.map((s) => s.name), ['host-facts', 'server-pack', 'init', 'service-install', 'linger', 'capability-write']);
  assert.deepEqual(result.server, {
    name: HOST_DONE_DATA.server.name,
    url: HOST_DONE_DATA.server.url,
    configPath: HOST_DONE_DATA.server.config_path,
  });
});

test('the bytes are on the progress event and nowhere else: no invented progress LINE', async () => {
  const lines: string[] = [];
  const bytes: number[] = [];
  const door = fakeHostDoor({
    events: [
      STEP('server-pack', 2, 7),
      ['progress', { bytes_done: 1024, bytes_total: null, file: 'part00' }],
      HOST_DONE,
    ],
  });
  await requestHostInstall({
    release: '0.6.0',
    jobTypes: ['echo'],
    fetchImpl: door.fetchImpl,
    onLine: (line) => lines.push(line),
    onEvent: (event) => {
      if (event.event === 'progress') bytes.push(event.data.bytes_done);
    },
  }, hostMachine());
  assert.deepEqual(lines, [], 'a byte count is not a line a step printed');
  assert.deepEqual(bytes, [1024]);
});

test('a line before any step is the host breaking its own ordering, and is refused rather than given a made-up owner', async () => {
  const door = fakeHostDoor({ events: [['line', { text: 'hello', stream: 'stdout' }], HOST_DONE] });
  const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
  assert.equal(r.code, 'host_install_failed');
  assert.match(r.message, /before any "step" event/);
});

test('an ndjson line split across two chunks is ONE event, not two halves and a parse error', async () => {
  const whole = doorLine(2, 'line', { text: 'backend:  cuda-linux', stream: 'stdout' });
  const cut = Math.floor(whole.length / 2);
  const lines: string[] = [];
  const door = fakeHostDoor({
    chunks: [
      doorLine(1, 'step', { name: 'init', index: 3, total: 7 }),
      whole.slice(0, cut),
      whole.slice(cut),
      doorLine(3, 'done', HOST_DONE_DATA),
    ],
  });
  await requestHostInstall(
    { release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl, onLine: (line) => lines.push(line) },
    hostMachine(),
  );
  assert.deepEqual(lines, ['backend:  cuda-linux']);
});

test('two events arriving in ONE chunk are two events', async () => {
  const lines: string[] = [];
  const door = fakeHostDoor({
    chunks: [
      doorLine(1, 'step', { name: 'init', index: 3, total: 7 })
        + doorLine(2, 'line', { text: 'one', stream: 'stdout' })
        + doorLine(3, 'line', { text: 'two', stream: 'stdout' })
        + doorLine(4, 'done', HOST_DONE_DATA),
    ],
  });
  await requestHostInstall(
    { release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl, onLine: (line) => lines.push(line) },
    hostMachine(),
  );
  assert.deepEqual(lines, ['one', 'two']);
});

test('a done event on the last line WITHOUT a trailing newline still finishes the install', async () => {
  const door = fakeHostDoor({ chunks: [doorLine(1, 'done', HOST_DONE_DATA).trimEnd()] });
  const result = await requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine());
  assert.equal(result.backend, 'cuda-linux');
});

test('a stream that ends without a done event is host_install_failed — a truncated stream is not a success', async () => {
  const door = fakeHostDoor({
    events: [STEP('server-pack', 2, 7), ['line', { text: 'server-pack: part00', stream: 'stdout' }]],
  });
  const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
  assert.equal(r.code, 'host_install_failed');
  assert.match(r.message, /closed without a "done" event/);
});

test('a trailing PARTIAL line at end of stream is host_install_failed, named, not silently dropped', async () => {
  const door = fakeHostDoor({ chunks: [doorLine(1, 'step', { name: 'init', index: 1, total: 7 }), '{"id":2,"event":"do'] });
  const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
  assert.equal(r.code, 'host_install_failed');
  assert.match(r.message, /ended mid-line/);
  assert.equal(r.detail, '{"id":2,"event":"do');
});

test('a mid-stream line that is not JSON is host_install_failed naming the line', async () => {
  const door = fakeHostDoor({ chunks: ['Traceback (most recent call last):\n', doorLine(1, 'done', HOST_DONE_DATA)] });
  const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
  assert.equal(r.code, 'host_install_failed');
  assert.match(r.message, /sent a line that is not JSON/);
});

test('an event the door does not define is refused rather than ignored', async () => {
  const door = fakeHostDoor({ events: [['started', { type: 'engine' }], HOST_DONE] });
  const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
  assert.equal(r.code, 'host_install_failed');
  assert.match(r.message, /an event named "started"/);
});

test('an envelope with no data object is refused: the shape is tasks.py\'s, not a flat blob', async () => {
  const door = fakeHostDoor({ chunks: [`${JSON.stringify({ id: 1, event: 'done', server: {} })}\n`] });
  const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
  assert.equal(r.code, 'host_install_failed');
  assert.match(r.message, /with no data object/);
});

for (const [code, message, command] of [
  ['virtualization_disabled', 'Virtualization is turned off in this machine\'s firmware.', null],
  ['wsl1_only', 'WSL is there and WSL2 is not.', 'wsl --set-default-version 2'],
  ['distro_import_failed', 'wsl --import crucible failed.', null],
] as const) {
  test(`a failed event carrying ${code} refuses with THAT code, never a generic one`, async () => {
    const door = fakeHostDoor({
      events: [
        ['state', { code, sentence: message, action: 'instruct' }],
        ['failed', { code, message, ...(command === null ? {} : { command }), detail: 'wsl.exe said 0x80370102' }],
      ],
    });
    const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
    assert.equal(r.code, code);
    assert.equal(r.message, message);
    assert.equal(r.command, command);
    assert.equal(r.detail, 'wsl.exe said 0x80370102');
  });
}

test('a failed event with no code is the host breaking its own contract: host_install_failed', async () => {
  const door = fakeHostDoor({ events: [['failed', { message: 'something went wrong' }]] });
  const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
  assert.equal(r.code, 'host_install_failed');
  assert.match(r.message, /"failed" event with no code/);
});

test('401 is host_unauthorized: the token this side read is not the token the host holds', async () => {
  const door = fakeHostDoor({ status: 401, text: 'unauthorized' });
  const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
  assert.equal(r.code, 'host_unauthorized');
  assert.match(r.message, /refused the engine token read from/);
  assert.equal(r.detail, 'unauthorized');
});

test('409 is host_install_running: one install per machine, and the second caller waits', async () => {
  const door = fakeHostDoor({ status: 409, text: 'an install is in flight' });
  const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
  assert.equal(r.code, 'host_install_running');
  assert.match(r.message, /already running on this machine/);
});

test('a refused connection is host_unreachable — the pack is there and the door is not answering', async () => {
  const door = fakeHostDoor({ rejectWith: Object.assign(new Error('connect ECONNREFUSED 127.0.0.1:7101'), { code: 'ECONNREFUSED' }) });
  const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
  assert.equal(r.code, 'host_unreachable');
  assert.match(r.message, /ECONNREFUSED/);
  assert.equal(r.command, `${HOST_DIR}\\crucible.cmd host`);
});

for (const [label, error] of [
  ['a name that does not resolve', new Error('getaddrinfo ENOTFOUND localhost')],
  ['a timeout', Object.assign(new Error('The operation was aborted due to timeout'), { name: 'TimeoutError' })],
] as const) {
  test(`${label} is host_unreachable too: fetch rejects only when the request never completed`, async () => {
    const door = fakeHostDoor({ rejectWith: error });
    const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
    assert.equal(r.code, 'host_unreachable');
  });
}

test('a status the door does not define is host_install_failed, with what it said', async () => {
  const door = fakeHostDoor({ status: 500, text: 'internal server error' });
  const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
  assert.equal(r.code, 'host_install_failed');
  assert.match(r.message, /answered 500/);
  assert.equal(r.detail, 'internal server error');
});

for (const [label, done, pattern] of [
  ['no crucible path', { ...HOST_DONE_DATA, crucible: '' }, /whose crucible is ""/],
  ['no release', { ...HOST_DONE_DATA, release: undefined }, /whose release is undefined/],
  ['a server object with no url', { ...HOST_DONE_DATA, server: { name: 'n', config_path: 'p' } }, /whose server\.url is undefined/],
  ['a camelCase configPath', { ...HOST_DONE_DATA, server: { name: 'n', url: 'u', configPath: 'p' } }, /whose server\.config_path is undefined/],
  ['the Windows backend', { ...HOST_DONE_DATA, backend: 'llama-windows' }, /This door installs the WSL engine, whose backend is cuda-linux/],
  ['steps that are not an array', { ...HOST_DONE_DATA, steps: 'six' }, /whose steps is not an array/],
  ['a step with no name', { ...HOST_DONE_DATA, steps: [{ argv: [], status: 'ok', detail: '' }] }, /steps\[0\] has no name/],
  ['a step with a status nobody defines', { ...HOST_DONE_DATA, steps: [{ name: 'init', argv: [], status: 'went-fine', detail: '' }] }, /steps\[0\] status is "went-fine"/],
] as const) {
  test(`a done event with ${label} is refused rather than filled in`, async () => {
    const door = fakeHostDoor({ events: [['done', done]] });
    const r = await refusal(requestHostInstall({ release: '0.6.0', jobTypes: ['echo'], fetchImpl: door.fetchImpl }, hostMachine()));
    assert.equal(r.code, 'host_install_failed');
    assert.match(r.message, pattern);
  });
}

test('the install.ps1 line is built from RELEASE_REPO and the release, never typed out', () => {
  assert.equal(
    hostInstallCommand('0.6.0'),
    'irm https://github.com/telltaleatheist/crucible/releases/download/v0.6.0/install.ps1 | iex',
  );
  assert.equal(
    hostInstallCommand('1.2.3'),
    'irm https://github.com/telltaleatheist/crucible/releases/download/v1.2.3/install.ps1 | iex',
  );
});
