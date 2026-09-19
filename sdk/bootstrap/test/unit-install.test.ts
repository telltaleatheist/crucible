/**
 * `install()` — two shapes, one per kind of machine (PHASE15-HOST.md 4.3).
 *
 * **linux and darwin:** the sequence by name, the interpreter and the wheel
 * fetched with the machine's own curl, the token never shown. The machine IS
 * the server, so `install()` walks `installSteps()` exactly as it always did.
 *
 * **win32:** two branches and nothing else — ask the host, or refuse
 * `host_not_installed` carrying the `irm … | iex` line. The sequence has ONE
 * implementation and it is `crucible host`'s, so there is no wsl.exe in any
 * argv this file asserts on a Windows machine any more; the walk's tests below
 * therefore run on `linux`, which is the platform that still performs it.
 * (Before PHASE15 they ran on `win32` through `wsl.exe -d Ubuntu --exec`; the
 * guest-side commands they pin are unchanged — only the transport went.)
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { BOOTSTRAP_VERSION, BootstrapStepFailed, install as installCurrent, planJobTypes, type InstallStep, type OutputStream } from '../src/index.js';
import { interpreterFor, interpreterUrl } from '../src/interpreter.js';
import { wheelShaUrl, wheelUrl } from '../src/release.js';
import { guestProbeScript } from '../src/runtime.js';
import {
  ARCHIVE,
  CRUCIBLE_BIN,
  DEST,
  DOWNLOADS,
  fakeHostDoor,
  FakeRunner,
  GUEST_BARE,
  GUEST_INSTALLED,
  GUEST_CONFIG,
  HOST_CMD,
  HOST_CONFIG,
  HOST_CONFIG_PATH,
  HOST_DIR,
  HOST_DONE,
  HOST_DONE_DATA,
  WHEEL,
  PIN,
  PY_SHA,
  refusal,
  WIN_ENV,
  type Expectation,
} from './fake.js';

// These fixtures model a specific published release, including archive/stamp
// names. Keep that release explicit when the SDK's default version advances.
const install: typeof installCurrent = (options, runner) =>
  installCurrent({ release: '0.6.0', ...options }, runner);

test('the default install command requests this SDK release, not the fixture release', async () => {
  const result = await refusal(installCurrent({ jobTypes: ['llm'], onLine: () => {} }, winRunner({})));
  assert.equal(result.command, `irm https://github.com/telltaleatheist/crucible/releases/download/v${BOOTSTRAP_VERSION}/install.ps1 | iex`);
});

/** On linux the command runs as it is: no transport, no wrapping. */
const N = (...argv: string[]): string[] => [...argv];
/** The script a `bash -c` call carries, wherever it is in the argv. */
const script = (argv: readonly string[]): string => argv[2] ?? '';
const PROBE = (home?: string): string[] => N('bash', '-c', guestProbeScript(home));
const CONFIG_PATH = '/home/owen/.crucible/config.toml';
/** The config read on a native host is a FILE read, not a call: it lives in `files`. */
const HAS_CONFIG = { [CONFIG_PATH]: GUEST_CONFIG };

const WHEEL_SHA = 'c'.repeat(64);

/**
 * The four commands the WHEEL half is, in order. It runs on every install.
 *
 * A FUNCTION OF THE HOME AND THE SHA TOOL, because two of these tests are
 * about exactly those: `{home}` moves every path, and a Mac has `shasum -a
 * 256` where Linux has `sha256sum`. A fixture that hard-coded either would be
 * a fixture that could not be used to test it.
 */
const wheelFetch = (home = '/home/owen/.crucible', sha = ['sha256sum']): Expectation[] => {
  const wheel = `${home}/downloads/crucible-0.6.0-py3-none-any.whl`;
  return [
    { argv: (argv) => script(argv).includes(`-o '${wheel}' '${wheelUrl('0.6.0')}'`) },
    { argv: N('curl', '-fsSL', '--retry', '3', wheelShaUrl('0.6.0')), stdout: `${WHEEL_SHA}  crucible-0.6.0-py3-none-any.whl\n` },
    { argv: N(...sha, wheel), stdout: `${WHEEL_SHA}  ${wheel}\n` },
    { argv: (argv) => script(argv).includes('pip install --upgrade --no-input') && script(argv).includes('python_sha256=%s') },
  ];
};

/** The three the INTERPRETER half is. Skipped when the stamp already names it. */
const pythonFetch = (pin = PIN, home = '/home/owen/.crucible', sha = ['sha256sum']): Expectation[] => {
  const archive = `${home}/downloads/${pin.asset}`;
  return [
    { argv: (argv) => script(argv).includes(`-o '${archive}' '${interpreterUrl(pin)}'`) },
    { argv: N(...sha, archive), stdout: `${pin.sha256}  ${archive}\n` },
    { argv: (argv) => script(argv).includes('tar -xzf') && script(argv).includes('activate_crucible_runtime') },
  ];
};

const WHEEL_FETCH = wheelFetch();
const PYTHON_FETCH = pythonFetch();

/** A clean first install: the interpreter, then the wheel. */
const SERVER: Expectation[] = [...PYTHON_FETCH, ...WHEEL_FETCH];

function collector(): { lines: string[]; steps: string[]; onLine: (line: string, stream: OutputStream, step: string) => void; onStep: (step: InstallStep) => void } {
  const lines: string[] = [];
  const steps: string[] = [];
  return {
    lines,
    steps,
    onLine: (line, stream, step) => lines.push(`${step}/${stream}: ${line}`),
    onStep: (step) => steps.push(`${step.name}:${step.status}`),
  };
}

/** A linux machine. `files` is what exists on it before the install starts. */
function linuxRunner(expectations: readonly Expectation[], files: Record<string, string> = {}): FakeRunner {
  return new FakeRunner({ platform: 'linux', homedir: '/home/owen', files }, expectations);
}

test('planJobTypes: flags in request order, installers only for installable types, duplicates folded', () => {
  const plan = planJobTypes(['echo', 'llm', { type: 'tts', narratorEngine: 'higgs-v3' }, 'rvc', 'denoise', 'llm']);
  assert.deepEqual(plan.enableFlags, ['--enable-echo', '--enable-llm', '--enable-tts', '--enable-rvc', '--enable-denoise']);
  assert.deepEqual(plan.installs, [
    { type: 'llm', argv: ['install', 'llm', '--verbose'] },
    { type: 'tts', argv: ['install', 'tts', '--narrator-engine', 'higgs-v3', '--verbose'] },
    { type: 'rvc', argv: ['install', 'rvc', '--verbose'] },
  ]);
});

for (const [label, requests, pattern] of [
  ['an empty list', [], /jobTypes is empty/],
  ['an unknown type', ['vlm'], /unknown job type "vlm"/],
  ['a bare tts', ['tts'], /tts must name its narrator engine/],
  ['tts with no engine', [{ type: 'tts', narratorEngine: '' }], /tts must name its narrator engine/],
  ['tts twice', [{ type: 'tts', narratorEngine: 'a' }, { type: 'tts', narratorEngine: 'b' }], /listed twice/],
  ['denoise without rvc', ['denoise'], /denoise shares the rvc env/],
] as const) {
  test(`planJobTypes refuses ${label} as bad_job_type`, async () => {
    const r = await refusal(Promise.resolve().then(() => planJobTypes(requests as never)));
    assert.equal(r.code, 'bad_job_type');
    assert.match(r.message, pattern);
  });
}

// ---------------------------------------------------------- the native walk

test('linux: the whole PHASE14 sequence, each step by name, the token redacted everywhere it could show', async () => {
  const c = collector();
  let tokenSeen: string | null = null;
  const expectations: Expectation[] = [
    { argv: PROBE(), stdout: GUEST_BARE },
    ...SERVER,
    {
      argv: (argv) => {
        const ok = argv.slice(0, 2).join(' ') === `${CRUCIBLE_BIN} init` && argv[2] === '--token' && argv.slice(4).join(' ') === '--enable-llm --enable-tts';
        tokenSeen = argv[3] ?? null;
        return ok;
      },
      lines: [['backend:  cuda-linux', 'stdout'], ['token:    minted; print it with `crucible token --show`', 'stdout']],
    },
    { argv: N(CRUCIBLE_BIN, 'install', 'llm', '--verbose'), lines: [['  Collecting vllm==0.29.0', 'stdout'], ['installed in 400s', 'stdout']] },
    { argv: N(CRUCIBLE_BIN, 'install', 'tts', '--narrator-engine', 'higgs-v3', '--verbose'), lines: [['installed in 300s', 'stdout']] },
    { argv: N(CRUCIBLE_BIN, 'service', 'install'), lines: [['enabled and started crucible.service', 'stdout']] },
    { argv: N(CRUCIBLE_BIN, 'local', 'register') },
    { argv: N(CRUCIBLE_BIN, 'local', 'install-cli') },
    { argv: N(CRUCIBLE_BIN, 'local', 'install-desktop') },
    { argv: N(CRUCIBLE_BIN, 'capability', '--write'), lines: [['recorded in /home/owen/.crucible/config.toml', 'stdout']] },
    { argv: N(CRUCIBLE_BIN, 'local', 'start', '--json') },
  ];
  const runner = linuxRunner(expectations);
  // The config is absent before init and present after: `init` is the step that writes it.
  let written = false;
  runner.fileExists = (path: string): boolean => (path === CONFIG_PATH ? written : false);
  runner.readFile = (): string => GUEST_CONFIG;
  const result = await install({
    jobTypes: ['llm', { type: 'tts', narratorEngine: 'higgs-v3' }],
    onLine: c.onLine,
    onStep: (step) => {
      if (step.name === 'init' && step.status === 'ok') written = true;
      c.onStep(step);
    },
  }, runner);
  runner.assertDrained();

  assert.deepEqual(result.steps.map((s) => `${s.name}:${s.status}`), [
    'host-facts:ok', 'server:ok', 'init:ok', 'install-llm:ok', 'install-tts:ok', 'service-install:ok', 'local-register:ok', 'local-install-cli:ok', 'local-install-desktop:ok', 'capability-write:ok', 'local-start:ok',
  ]);
  assert.deepEqual(result.server, { name: 'crucible@owens-pc-wsl', url: 'http://127.0.0.1:7100', configPath: CONFIG_PATH });
  assert.equal(result.release, '0.6.0');
  assert.equal(result.backend, 'cuda-linux');
  assert.equal(result.crucible, CRUCIBLE_BIN);
  assert.equal('token' in result.server, false);

  assert.ok(tokenSeen !== null && (tokenSeen as string).length >= 40, 'a token was minted and passed');
  const init = result.steps.find((s) => s.name === 'init');
  assert.deepEqual(init?.argv, [CRUCIBLE_BIN, 'init', '--token', '<redacted>', '--enable-llm', '--enable-tts']);
  const everything = JSON.stringify(result) + c.lines.join('\n') + c.steps.join('\n');
  assert.equal(everything.includes(tokenSeen as string), false, 'the token appears nowhere the host can log');

  const server = result.steps.find((s) => s.name === 'server');
  assert.match(server?.detail ?? '', /python 3\.11\.16 from python-build-standalone into \/home\/owen\/\.crucible\/server, then the 0\.6\.0 wheel/);
  assert.ok(c.lines.includes('install-llm/stdout:   Collecting vllm==0.29.0'));
  assert.equal(c.steps[0], 'host-facts:ok');
  assert.equal(c.steps[1], 'server:running');
  assert.equal(c.steps[2], 'server:ok');
  // NO CONDA, anywhere in the argv this install would run. `pip install` IS
  // here now and that is the change: PHASE20 made the wheel the deploy, so the
  // thing the old assertion forbade is the thing the new sequence is.
  const spelled = runner.calls.map((call) => call.argv.join(' ')).join('\n');
  assert.equal(/conda/.test(spelled), false, 'the conda walk is gone, not hidden');
});

test('the download happens with the machine\'s own tools: curl, sha256sum, tar, atomic rename', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_BARE },
    ...SERVER,
    { argv: N(CRUCIBLE_BIN, 'service', 'install') },
    { argv: N(CRUCIBLE_BIN, 'local', 'register') },
    { argv: N(CRUCIBLE_BIN, 'local', 'install-cli') },
    { argv: N(CRUCIBLE_BIN, 'local', 'install-desktop') },
    { argv: N(CRUCIBLE_BIN, 'capability', '--write') },
    { argv: N(CRUCIBLE_BIN, 'local', 'start', '--json') },
  ], HAS_CONFIG);
  await install({ jobTypes: ['echo'], onLine: c.onLine }, runner);
  runner.assertDrained();
  const commands = runner.calls.map((call) => (call.argv[0] === 'bash' ? script(call.argv) : call.argv.join(' ')));
  const curls = commands.filter((line) => line.includes('curl -fL'));
  assert.equal(curls.length, 2, 'one for the interpreter, one for the wheel');
  for (const line of curls) {
    assert.equal(line.includes('--continue-at'), false, 'nothing here is big enough to resume: it is fetched whole and hashed');
    assert.equal(line.includes('/mnt/'), false, 'nothing crosses /mnt/c');
  }
  assert.ok(commands.includes(`sha256sum ${ARCHIVE}`), 'the interpreter is hashed beside the archive');
  assert.ok(commands.includes(`sha256sum ${WHEEL}`), 'so is the wheel');
  assert.ok(commands.some((line) => line.includes(`tar -xzf '${ARCHIVE}' -C '${DEST}.partial'`)));
  assert.ok(commands.some((line) => line.includes(`_crucible_dest='${DEST}'`) && line.includes('mv "$_crucible_partial" "$_crucible_dest"')), 'the unpack is renamed into place, never unpacked over');
});

test('an interpreter whose digest is already stamped is not re-fetched; the wheel still installs', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_INSTALLED },
    ...WHEEL_FETCH,
    { argv: N(CRUCIBLE_BIN, 'service', 'install') },
    { argv: N(CRUCIBLE_BIN, 'local', 'register') },
    { argv: N(CRUCIBLE_BIN, 'local', 'install-cli') },
    { argv: N(CRUCIBLE_BIN, 'local', 'install-desktop') },
    { argv: N(CRUCIBLE_BIN, 'capability', '--write') },
    { argv: N(CRUCIBLE_BIN, 'local', 'start', '--json') },
  ], HAS_CONFIG);
  const result = await install({ jobTypes: ['echo'], onLine: c.onLine }, runner);
  runner.assertDrained();
  const server = result.steps.find((s) => s.name === 'server');
  assert.equal(server?.status, 'ok');
  assert.match(server?.detail ?? '', /python 3\.11\.16 was already at \/home\/owen\/\.crucible\/server; installed the 0\.6\.0 wheel into it/);
  assert.equal(runner.calls.some((call) => call.argv.join(' ').includes('python-build-standalone')), false);
});

test('an interpreter digest that is not the pin deletes the archive and refuses by name', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_BARE },
    ...PYTHON_FETCH.slice(0, 1),
    { argv: N('sha256sum', ARCHIVE), stdout: `${'9'.repeat(64)}  ${ARCHIVE}\n` },
    { argv: N('bash', '-c', `rm -f '${ARCHIVE}'`) },
  ]);
  const r = await refusal(install({ jobTypes: ['echo'], onLine: c.onLine }, runner));
  runner.assertDrained();
  assert.equal(r.code, 'runtime_sha_mismatch');
  assert.match(r.message, /hashes 9{64}/);
  assert.match(r.message, /The archive was deleted/);
});

test('a host with no curl or tar refuses before anything is fetched', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: `${GUEST_BARE}missing=curl\n` },
  ]);
  const r = await refusal(install({ jobTypes: ['echo'], onLine: c.onLine }, runner));
  runner.assertDrained();
  assert.equal(r.code, 'guest_missing_tool');
  assert.equal(r.command, 'sudo apt-get update && sudo apt-get install -y curl');
});

test('init is skipped when a config already exists, and its token is kept', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_INSTALLED },
    ...WHEEL_FETCH,
    { argv: N(CRUCIBLE_BIN, 'install', 'asr', '--verbose') },
    { argv: N(CRUCIBLE_BIN, 'service', 'install') },
    { argv: N(CRUCIBLE_BIN, 'local', 'register') },
    { argv: N(CRUCIBLE_BIN, 'local', 'install-cli') },
    { argv: N(CRUCIBLE_BIN, 'local', 'install-desktop') },
    { argv: N(CRUCIBLE_BIN, 'capability', '--write') },
    { argv: N(CRUCIBLE_BIN, 'local', 'start', '--json') },
  ], HAS_CONFIG);
  const result = await install({ jobTypes: ['asr', 'echo'], onLine: c.onLine }, runner);
  runner.assertDrained();
  const init = result.steps.find((s) => s.name === 'init');
  assert.equal(init?.status, 'skipped');
  assert.match(init?.detail ?? '', /config exists at \/home\/owen\/\.crucible\/config\.toml; its token is kept/);
  assert.equal(result.steps.some((s) => s.name === 'install-echo'), false, 'echo has no installer');
});

test('a failing step stops the sequence with its name, exit code, tail and the steps done', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_INSTALLED },
    ...WHEEL_FETCH,
    { argv: N(CRUCIBLE_BIN, 'install', 'llm', '--verbose'), code: 1, lines: [['recipe: llm/cuda-linux', 'stdout'], ['ERROR: env_smoke_failed', 'stderr']] },
  ], HAS_CONFIG);
  const err = await refusal(install({ jobTypes: ['llm'], onLine: c.onLine }, runner));
  runner.assertDrained();
  assert.equal(err.code, 'step_failed');
  const failed = err.error as BootstrapStepFailed;
  assert.ok(failed instanceof BootstrapStepFailed);
  assert.equal(failed.step, 'install-llm');
  assert.equal(failed.exitCode, 1);
  // `server` is done: the wheel installed. It is a step that ALWAYS runs now,
  // where the pack step used to be skipped whenever its stamp matched.
  assert.deepEqual(failed.stepsDone, ['host-facts', 'server']);
  assert.deepEqual(failed.tail, ['recipe: llm/cuda-linux', '! ERROR: env_smoke_failed']);
  assert.match(failed.message, /install step "install-llm" exited 1 inside this machine/);
});

for (const detail of ['local_start_failed: engine did not answer', 'unauthorized: engine info returned HTTP 401']) {
  test(`install refuses readiness failure after service installation: ${detail}`, async () => {
    const runner = linuxRunner([
      { argv: PROBE(), stdout: GUEST_INSTALLED }, ...WHEEL_FETCH,
      { argv: N(CRUCIBLE_BIN, 'service', 'install') },
      { argv: N(CRUCIBLE_BIN, 'local', 'register') },
      { argv: N(CRUCIBLE_BIN, 'local', 'install-cli') },
      { argv: N(CRUCIBLE_BIN, 'local', 'install-desktop') },
      { argv: N(CRUCIBLE_BIN, 'capability', '--write') },
      { argv: N(CRUCIBLE_BIN, 'local', 'start', '--json'), code: 1, lines: [[detail, 'stderr']] },
    ], HAS_CONFIG);
    const result = await refusal(install({ jobTypes: ['echo'], onLine: () => {} }, runner));
    runner.assertDrained();
    const error = result.error as BootstrapStepFailed;
    assert.equal(error.step, 'local-start');
    assert.ok(error.tail.some(line => line.includes(detail)));
    assert.equal(error.stepsDone.at(-1), 'capability-write');
  });
}

test('install does not return while local authenticated readiness is pending', async () => {
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_INSTALLED }, ...WHEEL_FETCH,
    { argv: N(CRUCIBLE_BIN, 'service', 'install') },
    { argv: N(CRUCIBLE_BIN, 'local', 'register') },
    { argv: N(CRUCIBLE_BIN, 'local', 'install-cli') },
    { argv: N(CRUCIBLE_BIN, 'local', 'install-desktop') },
    { argv: N(CRUCIBLE_BIN, 'capability', '--write') },
    { argv: N(CRUCIBLE_BIN, 'local', 'start', '--json') },
  ], HAS_CONFIG);
  let enter!: () => void;
  let ready!: () => void;
  const entered = new Promise<void>(resolve => { enter = resolve; });
  const pending = new Promise<void>(resolve => { ready = resolve; });
  const stream = runner.stream.bind(runner);
  runner.stream = async (argv, options) => {
    if (argv.slice(1).join(' ') === 'local start --json') {
      enter();
      await pending;
    }
    return stream(argv, options);
  };
  let completed = false;
  const installing = install({ jobTypes: ['echo'], onLine: () => {} }, runner).then(result => {
    completed = true;
    return result;
  });
  await entered;
  assert.equal(completed, false);
  ready();
  const result = await installing;
  assert.equal(result.steps.at(-1)?.name, 'local-start');
  assert.equal(result.steps.at(-1)?.status, 'ok');
  runner.assertDrained();
});

test('a step that never returns is a failure naming the timeout, not a hang', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_INSTALLED },
    ...WHEEL_FETCH,
    { argv: () => true, failure: 'crucible did not answer within 1800s' },
  ], HAS_CONFIG);
  const err = await refusal(install({ jobTypes: ['llm'], onLine: c.onLine }, runner));
  const failed = err.error as BootstrapStepFailed;
  assert.equal(failed.step, 'install-llm');
  assert.equal(failed.exitCode, null);
  assert.match(failed.message, /did not answer within 1800s/);
});

test('a broken existing config is refused by name rather than re-initialised over', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_INSTALLED },
    ...WHEEL_FETCH,
  ], { [CONFIG_PATH]: '[server]\nname = "n"\n' });
  const r = await refusal(install({ jobTypes: ['llm'], onLine: c.onLine }, runner));
  assert.equal(r.code, 'config_missing_key');
  assert.equal(r.command, 'crucible init --force');
});

test('{home} travels as CRUCIBLE_HOME into every crucible verb, and {bind} into init', async () => {
  const c = collector();
  const CRUCIBLE = '/srv/crucible/server/bin/crucible';
  const ENV = { CRUCIBLE_HOME: '/srv/crucible' };
  const runner = linuxRunner([
    { argv: PROBE('/srv/crucible'), stdout: `home=/srv/crucible\nuser=owen\nfree_kib=400000000\ncrucible=${CRUCIBLE}\nversion=crucible 0.6.0\npython_sha256=${PY_SHA}\nrelease=0.6.0\n` },
    ...wheelFetch('/srv/crucible'),
    { argv: (argv) => argv.slice(0, 2).join(' ') === `${CRUCIBLE} init` && argv[2] === '--token' && argv.slice(4).join(' ') === '--host 0.0.0.0 --port 7200 --enable-echo', env: ENV },
    { argv: N(CRUCIBLE, 'service', 'install'), env: ENV },
    { argv: N(CRUCIBLE, 'local', 'register'), env: ENV },
    { argv: N(CRUCIBLE, 'local', 'install-cli'), env: ENV },
    { argv: N(CRUCIBLE, 'local', 'install-desktop'), env: ENV },
    { argv: N(CRUCIBLE, 'capability', '--write'), env: ENV },
    { argv: N(CRUCIBLE, 'local', 'start', '--json'), env: ENV },
  ]);
  let written = false;
  runner.fileExists = (path: string): boolean => (path === '/srv/crucible/config.toml' ? written : false);
  runner.readFile = (): string => GUEST_CONFIG;
  const result = await install({
    home: '/srv/crucible',
    bind: { host: '0.0.0.0', port: 7200 },
    jobTypes: ['echo'],
    onLine: c.onLine,
    onStep: (step) => {
      if (step.name === 'init' && step.status === 'ok') written = true;
    },
  }, runner);
  runner.assertDrained();
  assert.equal(result.server.configPath, '/srv/crucible/config.toml');
});

test('darwin: the same steps run natively, shasum instead of sha256sum, no linger step at all', async () => {
  const c = collector();
  const N = (...argv: string[]): string[] => [...argv];
  const runner = new FakeRunner({ platform: 'darwin', homedir: '/Users/owen' }, [
    { argv: N('bash', '-c', guestProbeScript(undefined)), stdout: 'home=/Users/owen/.crucible\nuser=owen\nfree_kib=900000000\n' },
    // THE MAC'S OWN PIN, and its own sha tool. Both come off the same tables
    // the Linux expectations above use, so a pin edited for one backend cannot
    // leave this test passing about the other.
    ...pythonFetch(interpreterFor('mlx-darwin'), '/Users/owen/.crucible', ['shasum', '-a', '256']),
    ...wheelFetch('/Users/owen/.crucible', ['shasum', '-a', '256']),
    { argv: (argv) => argv[1] === 'init' },
    { argv: ['/Users/owen/.crucible/server/bin/crucible', 'install', 'llm', '--verbose'] },
    { argv: ['/Users/owen/.crucible/server/bin/crucible', 'service', 'install'] },
    { argv: ['/Users/owen/.crucible/server/bin/crucible', 'local', 'register'] },
    { argv: ['/Users/owen/.crucible/server/bin/crucible', 'local', 'install-cli'] },
    { argv: ['/Users/owen/.crucible/server/bin/crucible', 'local', 'install-desktop'] },
    { argv: ['/Users/owen/.crucible/server/bin/crucible', 'capability', '--write'] },
    { argv: ['/Users/owen/.crucible/server/bin/crucible', 'local', 'start', '--json'] },
  ]);
  // The config read on darwin is a file read, not a call: absent before init, present after.
  let written = false;
  runner.fileExists = (path: string): boolean => (path === '/Users/owen/.crucible/config.toml' ? written : false);
  runner.readFile = (): string => GUEST_CONFIG;
  const onStep = (step: InstallStep): void => {
    if (step.name === 'init' && step.status === 'ok') written = true;
  };
  const result = await install({ jobTypes: ['llm'], onLine: c.onLine, onStep }, runner);
  runner.assertDrained();
  assert.equal(result.backend, 'mlx-darwin');
  assert.equal(result.server.configPath, '/Users/owen/.crucible/config.toml');
  assert.equal(result.steps.some((s) => s.name === 'linger'), false, 'launchd has no linger question');
});

// ------------------------------------------------------- win32: the two branches

/** A Windows machine with the host runtime installed and its host-mode config written. */
function winRunner(files: Record<string, string> = { [HOST_CMD]: '@echo off', [HOST_CONFIG_PATH]: HOST_CONFIG }): FakeRunner {
  return new FakeRunner({ platform: 'win32', env: WIN_ENV, files }, []);
}

test('win32 + a host: install() asks the door and returns the HOST\'s result, spawning nothing itself', async () => {
  const c = collector();
  const door = fakeHostDoor({
    events: [
      ['state', { code: 'no_crucible_distro', sentence: 'There is no Crucible distro on this machine yet.', action: 'run-elevated' }],
      ['step', { name: 'server', index: 2, total: 7 }],
      ['line', { text: 'server: cpython-3.11.16', stream: 'stdout' }],
      HOST_DONE,
    ],
  });
  const runner = winRunner();
  const result = await install({
    jobTypes: ['llm', { type: 'tts', narratorEngine: 'higgs-v3' }],
    release: '0.6.0',
    onLine: c.onLine,
    onStep: c.onStep,
    fetchImpl: door.fetchImpl,
  }, runner);

  // The request, field for field.
  assert.equal(door.requests.length, 1);
  assert.equal(door.requests[0]?.url, 'http://127.0.0.1:7101/install');
  assert.equal(door.requests[0]?.method, 'POST');
  assert.equal(door.requests[0]?.authorization, 'Bearer host-token-not-a-secret');
  assert.deepEqual(door.requests[0]?.body, {
    target: 'wsl',
    release: '0.6.0',
    job_types: ['llm', { type: 'tts', narrator_engine: 'higgs-v3' }],
  });

  // The result is the host's, verbatim — including the linger step, which only
  // the host can perform now because only the host has the guest.
  assert.deepEqual(result.steps.map((s) => s.name), ['host-facts', 'server', 'init', 'service-install', 'linger', 'capability-write']);
  assert.deepEqual(result.server, {
    name: HOST_DONE_DATA.server.name,
    url: HOST_DONE_DATA.server.url,
    configPath: HOST_DONE_DATA.server.config_path,
  });
  assert.equal(result.release, '0.6.0');
  assert.equal(result.backend, 'cuda-linux');
  assert.equal(result.crucible, '/home/crucible/.crucible/server/bin/crucible');

  // Relayed through the callbacks install() already had.
  assert.deepEqual(c.steps, ['server:running']);
  assert.deepEqual(c.lines, [
    'state/stdout: There is no Crucible distro on this machine yet.',
    'server/stdout: server: cpython-3.11.16',
  ]);

  // The whole point of 4.3: bootstrap does not ALSO walk the sequence. Nothing
  // was spawned on this machine at all — no wsl.exe, no probe, no curl.
  assert.deepEqual(runner.calls, []);
});

test('win32 + a host: {home} and {bind} travel over the wire, not through a wsl.exe argv', async () => {
  const c = collector();
  const door = fakeHostDoor({ events: [HOST_DONE] });
  await install({
    jobTypes: ['echo'],
    release: '0.6.0',
    home: '/srv/crucible',
    bind: { host: '0.0.0.0', port: 7200 },
    onLine: c.onLine,
    fetchImpl: door.fetchImpl,
  }, winRunner());
  assert.deepEqual(door.requests[0]?.body, {
    target: 'wsl',
    release: '0.6.0',
    job_types: ['echo'],
    home: '/srv/crucible',
    bind: { host: '0.0.0.0', port: 7200 },
  });
});

test('win32 + a host: onHostEvent sees every event verbatim, including the 4c states and the bytes', async () => {
  const c = collector();
  const seen: string[] = [];
  const door = fakeHostDoor({
    events: [
      ['state', { code: 'wsl_ready', sentence: 'WSL2 is ready.', action: 'run' }],
      ['step', { name: 'server', index: 2, total: 7 }],
      ['progress', { bytes_done: 4194304, bytes_total: 120000000, file: 'part00' }],
      ['line', { text: 'backend:  cuda-linux', stream: 'stdout' }],
      HOST_DONE,
    ],
  });
  await install({
    jobTypes: ['echo'],
    release: '0.6.0',
    onLine: c.onLine,
    onHostEvent: (event) => {
      if (event.event === 'state') seen.push(`state:${event.data.code}`);
      else if (event.event === 'progress') seen.push(`progress:${event.data.bytes_done}/${event.data.bytes_total}`);
      else seen.push(event.event);
    },
    fetchImpl: door.fetchImpl,
  }, winRunner());
  assert.deepEqual(seen, ['state:wsl_ready', 'step', 'progress:4194304/120000000', 'line', 'done']);
});

test('win32 with NO host: host_not_installed, carrying the exact irm line for the release asked for', async () => {
  const c = collector();
  const runner = winRunner({});
  const r = await refusal(install({ jobTypes: ['llm'], release: '0.6.0', onLine: c.onLine }, runner));
  assert.equal(r.code, 'host_not_installed');
  assert.equal(r.command, 'irm https://github.com/telltaleatheist/crucible/releases/download/v0.6.0/install.ps1 | iex');
  assert.match(r.message, new RegExp(HOST_DIR.replace(/\\/g, '\\\\')));
  // Named, not performed: a library does not elevate.
  assert.deepEqual(runner.calls, []);
});

test('win32 with NO host: the irm line names the release the caller asked for, not this package\'s', async () => {
  const c = collector();
  const r = await refusal(install({ jobTypes: ['llm'], release: '9.9.9', onLine: c.onLine }, winRunner({})));
  assert.equal(r.command, 'irm https://github.com/telltaleatheist/crucible/releases/download/v9.9.9/install.ps1 | iex');
});

test('win32: a malformed job list is bad_job_type here, without needing a host to say so', async () => {
  const c = collector();
  const r = await refusal(install({ jobTypes: ['tts'] as never, onLine: c.onLine }, winRunner({})));
  assert.equal(r.code, 'bad_job_type');
  assert.match(r.message, /tts must name its narrator engine/);
});

test('win32: a failed event from the host is install()\'s refusal, with the 4c code the host sent', async () => {
  const c = collector();
  const door = fakeHostDoor({
    events: [['failed', { code: 'virtualization_disabled', message: 'Virtualization is turned off in this machine\'s firmware.' }]],
  });
  const r = await refusal(install({ jobTypes: ['echo'], release: '0.6.0', onLine: c.onLine, fetchImpl: door.fetchImpl }, winRunner()));
  assert.equal(r.code, 'virtualization_disabled');
});

test('win32: a host whose config has no token refuses host_no_token before anything is sent', async () => {
  const c = collector();
  const door = fakeHostDoor({ events: [HOST_DONE] });
  const r = await refusal(install(
    { jobTypes: ['echo'], release: '0.6.0', onLine: c.onLine, fetchImpl: door.fetchImpl },
    winRunner({ [HOST_CMD]: '@echo off' }),
  ));
  assert.equal(r.code, 'host_no_token');
  assert.equal(door.requests.length, 0, 'an unauthorised request is not sent and then refused');
});
