/**
 * `install()` — two shapes, one per kind of machine (PHASE15-HOST.md 4.3).
 *
 * **linux and darwin:** the PHASE14 sequence by name, the pack fetched with the
 * machine's own curl, the token never shown. The machine IS the server, so
 * `install()` walks `installSteps()` exactly as it always did.
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

import { BootstrapStepFailed, install, planJobTypes, type InstallStep, type OutputStream } from '../src/index.js';
import { envpacksUrl } from '../src/envpacks.js';
import { guestProbeScript } from '../src/pack.js';
import {
  ARCHIVE,
  CRUCIBLE_BIN,
  DEST,
  DOWNLOADS,
  ENVPACKS_JSON,
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
  PACK_SHA,
  refusal,
  WIN_ENV,
  type Expectation,
} from './fake.js';

/** On linux the command runs as it is: no transport, no wrapping. */
const N = (...argv: string[]): string[] => [...argv];
/** The script a `bash -c` call carries, wherever it is in the argv. */
const script = (argv: readonly string[]): string => argv[2] ?? '';
const PROBE = (home?: string): string[] => N('bash', '-c', guestProbeScript(home));
const MANIFEST = { argv: N('curl', '-fsSL', '--retry', '3', envpacksUrl('0.6.0')), stdout: ENVPACKS_JSON };
const CONFIG_PATH = '/home/owen/.crucible/config.toml';
/** The config read on a native host is a FILE read, not a call: it lives in `files`. */
const HAS_CONFIG = { [CONFIG_PATH]: GUEST_CONFIG };

const PART0 = 'crucible-env-server-cuda-linux-0.6.0.tar.zst.part00';
const PART1 = 'crucible-env-server-cuda-linux-0.6.0.tar.zst.part01';
const BASE = 'https://github.com/telltaleatheist/crucible/releases/download/v0.6.0';

/** The seven commands a server-pack fetch is, in order. */
const PACK_FETCH: Expectation[] = [
  { argv: N('bash', '-c', `rm -f '${ARCHIVE}' && mkdir -p '${DOWNLOADS}'`) },
  { argv: (argv) => script(argv).startsWith(`curl -fL --retry 3 --retry-delay 2 --continue-at - --create-dirs -o '${DOWNLOADS}/${PART0}' '${BASE}/${PART0}'`) },
  { argv: (argv) => script(argv).includes(`${BASE}/${PART1}`) },
  { argv: N('sha256sum', ARCHIVE), stdout: `${PACK_SHA}  ${ARCHIVE}\n` },
  { argv: (argv) => script(argv).startsWith(`rm -rf '${DEST}.partial'`) && script(argv).includes('tar --zstd -xf') },
  { argv: (argv) => script(argv).includes(`'${DEST}.partial/bin/crucible' --version`) },
  { argv: (argv) => script(argv).startsWith(`rm -rf '${DEST}' && mv '${DEST}.partial' '${DEST}'`) && script(argv).includes('sha256=%s') },
];

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
    MANIFEST,
    ...PACK_FETCH,
    {
      argv: (argv) => {
        const ok = argv.slice(0, 2).join(' ') === `${CRUCIBLE_BIN} init` && argv[2] === '--token' && argv.slice(4).join(' ') === '--enable-llm --enable-tts';
        tokenSeen = argv[3] ?? null;
        return ok;
      },
      lines: [['backend:  cuda-linux', 'stdout'], ['token:    minted; print it with `crucible token --show`', 'stdout']],
    },
    { argv: N(CRUCIBLE_BIN, 'install', 'llm', '--verbose'), lines: [['  downloading llm pack', 'stdout'], ['installed in 400s', 'stdout']] },
    { argv: N(CRUCIBLE_BIN, 'install', 'tts', '--narrator-engine', 'higgs-v3', '--verbose'), lines: [['installed in 300s', 'stdout']] },
    { argv: N(CRUCIBLE_BIN, 'service', 'install'), lines: [['enabled and started crucible.service', 'stdout']] },
    { argv: N(CRUCIBLE_BIN, 'capability', '--write'), lines: [['recorded in /home/owen/.crucible/config.toml', 'stdout']] },
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
    'host-facts:ok', 'server-pack:ok', 'init:ok', 'install-llm:ok', 'install-tts:ok', 'service-install:ok', 'capability-write:ok',
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

  const pack = result.steps.find((s) => s.name === 'server-pack');
  assert.match(pack?.detail ?? '', /unpacked 2 part\(s\) into \/home\/owen\/\.crucible\/server \(Python 3\.11\.13\)/);
  assert.ok(c.lines.includes('install-llm/stdout:   downloading llm pack'));
  assert.equal(c.steps[0], 'host-facts:ok');
  assert.equal(c.steps[1], 'server-pack:running');
  assert.equal(c.steps[2], 'server-pack:ok');
  // NOTHING pip, nothing conda, anywhere in the argv this install would run.
  const spelled = runner.calls.map((call) => call.argv.join(' ')).join('\n');
  assert.equal(/conda|pip install|\.whl/.test(spelled), false, 'the conda and wheel path is gone, not hidden');
});

test('the pack download happens with the machine\'s own tools: curl, sha256sum, tar --zstd, atomic rename', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_BARE },
    MANIFEST,
    ...PACK_FETCH,
    { argv: N(CRUCIBLE_BIN, 'service', 'install') },
    { argv: N(CRUCIBLE_BIN, 'capability', '--write') },
  ], HAS_CONFIG);
  await install({ jobTypes: ['echo'], onLine: c.onLine }, runner);
  runner.assertDrained();
  const commands = runner.calls.map((call) => (call.argv[0] === 'bash' ? script(call.argv) : call.argv.join(' ')));
  const curls = commands.filter((line) => line.startsWith('curl -fL'));
  assert.equal(curls.length, 2, 'one curl per part');
  for (const line of curls) {
    assert.match(line, /--continue-at -/, 'a killed part resumes');
    assert.match(line, /cat '\/home\/owen\/\.crucible\/downloads\/[^']+' >> '\/home\/owen\/\.crucible\/downloads\/crucible-env-server-cuda-linux-0\.6\.0\.tar\.zst'/);
    assert.match(line, /&& rm -f /, 'the part is deleted once appended: peak extra disk is one part');
    assert.equal(line.includes('/mnt/'), false, 'nothing crosses /mnt/c');
  }
  assert.ok(commands.includes(`sha256sum ${ARCHIVE}`), 'the digest is computed beside the archive');
  assert.ok(commands.some((line) => line.includes(`tar --zstd -xf '${ARCHIVE}' -C '${DEST}.partial'`)));
  assert.ok(commands.some((line) => line.includes(`mv '${DEST}.partial' '${DEST}'`)), 'the unpack is renamed into place, never unpacked over');
});

test('a pack whose stamp already matches the manifest is skipped, and nothing is downloaded', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_INSTALLED },
    MANIFEST,
    { argv: N(CRUCIBLE_BIN, 'service', 'install') },
    { argv: N(CRUCIBLE_BIN, 'capability', '--write') },
  ], HAS_CONFIG);
  const result = await install({ jobTypes: ['echo'], onLine: c.onLine }, runner);
  runner.assertDrained();
  const pack = result.steps.find((s) => s.name === 'server-pack');
  assert.equal(pack?.status, 'skipped');
  assert.match(pack?.detail ?? '', /already the 0\.6\.0 pack/);
});

test('a sha that is not the manifest\'s deletes the archive and refuses by name', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_BARE },
    MANIFEST,
    ...PACK_FETCH.slice(0, 3),
    { argv: N('sha256sum', ARCHIVE), stdout: `${'9'.repeat(64)}  ${ARCHIVE}\n` },
    { argv: N('bash', '-c', `rm -f '${ARCHIVE}'`) },
  ]);
  const r = await refusal(install({ jobTypes: ['echo'], onLine: c.onLine }, runner));
  runner.assertDrained();
  assert.equal(r.code, 'pack_sha_mismatch');
  assert.match(r.message, /hashes 9{64} and the 0\.6\.0 manifest says a{64}/);
  assert.match(r.message, /The archive was deleted/);
});

test('the disk pre-flight refuses with the numbers BEFORE anything is fetched', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: 'home=/home/owen/.crucible\nuser=owen\nfree_kib=100000\n' },
    MANIFEST,
  ]);
  const r = await refusal(install({ jobTypes: ['echo'], onLine: c.onLine }, runner));
  runner.assertDrained();
  assert.equal(r.code, 'pack_disk');
  assert.match(r.message, /needs 0\.5 GiB free .* and there is 0\.1 GiB/);
  assert.match(r.message, /Nothing was downloaded/);
});

test('a release with no server pack for this backend is pack_not_published, never a build', async () => {
  const c = collector();
  const manifest = JSON.stringify({ schema: 1, version: '0.6.0', packs: [JSON.parse(ENVPACKS_JSON).packs[2]] });
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_BARE },
    { argv: MANIFEST.argv, stdout: manifest },
  ]);
  const r = await refusal(install({ jobTypes: ['echo'], onLine: c.onLine }, runner));
  assert.equal(r.code, 'pack_not_published');
  assert.match(r.message, /publishes no "server" pack for cuda-linux/);
  assert.match(r.message, /lists llm for that backend/);
});

test('a host with no curl, tar or zstd refuses before the manifest is even asked for', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: `${GUEST_BARE}missing=zstd\n` },
  ]);
  const r = await refusal(install({ jobTypes: ['echo'], onLine: c.onLine }, runner));
  runner.assertDrained();
  assert.equal(r.code, 'guest_missing_tool');
  assert.equal(r.command, 'sudo apt-get update && sudo apt-get install -y zstd');
});

test('init is skipped when a config already exists, and its token is kept', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_INSTALLED },
    MANIFEST,
    { argv: N(CRUCIBLE_BIN, 'install', 'asr', '--verbose') },
    { argv: N(CRUCIBLE_BIN, 'service', 'install') },
    { argv: N(CRUCIBLE_BIN, 'capability', '--write') },
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
    MANIFEST,
    { argv: N(CRUCIBLE_BIN, 'install', 'llm', '--verbose'), code: 1, lines: [['recipe: llm/cuda-linux', 'stdout'], ['ERROR: pack_not_published', 'stderr']] },
  ], HAS_CONFIG);
  const err = await refusal(install({ jobTypes: ['llm'], onLine: c.onLine }, runner));
  runner.assertDrained();
  assert.equal(err.code, 'step_failed');
  const failed = err.error as BootstrapStepFailed;
  assert.ok(failed instanceof BootstrapStepFailed);
  assert.equal(failed.step, 'install-llm');
  assert.equal(failed.exitCode, 1);
  assert.deepEqual(failed.stepsDone, ['host-facts']);
  assert.deepEqual(failed.tail, ['recipe: llm/cuda-linux', '! ERROR: pack_not_published']);
  assert.match(failed.message, /install step "install-llm" exited 1 inside this machine/);
});

test('a step that never returns is a failure naming the timeout, not a hang', async () => {
  const c = collector();
  const runner = linuxRunner([
    { argv: PROBE(), stdout: GUEST_INSTALLED },
    MANIFEST,
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
    MANIFEST,
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
    { argv: PROBE('/srv/crucible'), stdout: `home=/srv/crucible\nuser=owen\nfree_kib=400000000\ncrucible=${CRUCIBLE}\nversion=crucible 0.6.0\nsha256=${PACK_SHA}\nrelease=0.6.0\n` },
    MANIFEST,
    { argv: (argv) => argv.slice(0, 2).join(' ') === `${CRUCIBLE} init` && argv[2] === '--token' && argv.slice(4).join(' ') === '--host 0.0.0.0 --port 7200 --enable-echo', env: ENV },
    { argv: N(CRUCIBLE, 'service', 'install'), env: ENV },
    { argv: N(CRUCIBLE, 'capability', '--write'), env: ENV },
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
    { argv: N('curl', '-fsSL', '--retry', '3', envpacksUrl('0.6.0')), stdout: ENVPACKS_JSON },
    { argv: N('bash', '-c', `rm -f '/Users/owen/.crucible/downloads/crucible-env-server-mlx-darwin-0.6.0.tar.zst' && mkdir -p '/Users/owen/.crucible/downloads'`) },
    { argv: (argv) => (argv[2] ?? '').includes('crucible-env-server-mlx-darwin-0.6.0.tar.zst.part00') },
    { argv: N('shasum', '-a', '256', '/Users/owen/.crucible/downloads/crucible-env-server-mlx-darwin-0.6.0.tar.zst'), stdout: `${'b'.repeat(64)}  x\n` },
    { argv: (argv) => (argv[2] ?? '').includes('tar --zstd -xf') },
    { argv: (argv) => (argv[2] ?? '').includes('--version') },
    { argv: (argv) => (argv[2] ?? '').includes('mv ') },
    { argv: (argv) => argv[1] === 'init' },
    { argv: ['/Users/owen/.crucible/server/bin/crucible', 'install', 'llm', '--verbose'] },
    { argv: ['/Users/owen/.crucible/server/bin/crucible', 'service', 'install'] },
    { argv: ['/Users/owen/.crucible/server/bin/crucible', 'capability', '--write'] },
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

/** A Windows machine with the host pack unpacked and its host-mode config written. */
function winRunner(files: Record<string, string> = { [HOST_CMD]: '@echo off', [HOST_CONFIG_PATH]: HOST_CONFIG }): FakeRunner {
  return new FakeRunner({ platform: 'win32', env: WIN_ENV, files }, []);
}

test('win32 + a host: install() asks the door and returns the HOST\'s result, spawning nothing itself', async () => {
  const c = collector();
  const door = fakeHostDoor({
    events: [
      ['state', { code: 'no_crucible_distro', sentence: 'There is no Crucible distro on this machine yet.', action: 'run-elevated' }],
      ['step', { name: 'server-pack', index: 2, total: 7 }],
      ['line', { text: 'server-pack: part00', stream: 'stdout' }],
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
  assert.deepEqual(result.steps.map((s) => s.name), ['host-facts', 'server-pack', 'init', 'service-install', 'linger', 'capability-write']);
  assert.deepEqual(result.server, {
    name: HOST_DONE_DATA.server.name,
    url: HOST_DONE_DATA.server.url,
    configPath: HOST_DONE_DATA.server.config_path,
  });
  assert.equal(result.release, '0.6.0');
  assert.equal(result.backend, 'cuda-linux');
  assert.equal(result.crucible, '/home/crucible/.crucible/server/bin/crucible');

  // Relayed through the callbacks install() already had.
  assert.deepEqual(c.steps, ['server-pack:running']);
  assert.deepEqual(c.lines, [
    'state/stdout: There is no Crucible distro on this machine yet.',
    'server-pack/stdout: server-pack: part00',
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
      ['step', { name: 'server-pack', index: 2, total: 7 }],
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
