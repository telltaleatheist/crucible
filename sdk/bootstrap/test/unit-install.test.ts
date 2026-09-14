/** `install()` — the PHASE14 sequence by name, the pack fetched in the guest, the token never shown. */
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
  FakeRunner,
  GUEST_BARE,
  GUEST_INSTALLED,
  GUEST_CONFIG,
  PACK_SHA,
  refusal,
  WSL_LIST,
  type Expectation,
} from './fake.js';

const W = (...argv: string[]): string[] => ['wsl.exe', '-d', 'Ubuntu', '--exec', ...argv];
/** The guest, entered as root. `wsl.exe -u root` is not an escalation (linger.ts). */
const R = (...argv: string[]): string[] => ['wsl.exe', '-d', 'Ubuntu', '-u', 'root', '--exec', ...argv];
const LIST = { argv: ['wsl.exe', '-l', '-v'], stdout: WSL_LIST };
const WHOAMI = { argv: W('id', '-un'), stdout: 'owen\n' };
const LINGER_ON = { argv: R('loginctl', 'show-user', 'owen', '-p', 'Linger'), stdout: 'Linger=yes\n' };
const LINGER_OFF = { argv: R('loginctl', 'show-user', 'owen', '-p', 'Linger'), stdout: 'Linger=no\n' };
const GRANT = { argv: R('loginctl', 'enable-linger', 'owen') };
const PROBE = (home?: string): string[] => W('bash', '-c', guestProbeScript(home));
const MANIFEST = { argv: W('curl', '-fsSL', '--retry', '3', envpacksUrl('0.6.0')), stdout: ENVPACKS_JSON };
const READ = (argv: readonly string[]): boolean => argv[4] === 'bash' && (argv[6] ?? '').includes('config.toml');
const CONFIG_OK = { argv: READ, stdout: `/home/owen/.crucible/config.toml\n${GUEST_CONFIG}` };
const NO_CONFIG = { argv: READ, code: 3, stderr: '/home/owen/.crucible/config.toml\n' };

const PART0 = 'crucible-env-server-cuda-linux-0.6.0.tar.zst.part00';
const PART1 = 'crucible-env-server-cuda-linux-0.6.0.tar.zst.part01';
const BASE = 'https://github.com/telltaleatheist/crucible/releases/download/v0.6.0';

/** The seven guest commands a server-pack fetch is, in order. */
const PACK_FETCH: Expectation[] = [
  { argv: W('bash', '-c', `rm -f '${ARCHIVE}' && mkdir -p '${DOWNLOADS}'`) },
  { argv: (argv) => (argv[6] ?? '').startsWith(`curl -fL --retry 3 --retry-delay 2 --continue-at - --create-dirs -o '${DOWNLOADS}/${PART0}' '${BASE}/${PART0}'`) },
  { argv: (argv) => (argv[6] ?? '').includes(`${BASE}/${PART1}`) },
  { argv: W('sha256sum', ARCHIVE), stdout: `${PACK_SHA}  ${ARCHIVE}\n` },
  { argv: (argv) => (argv[6] ?? '').startsWith(`rm -rf '${DEST}.partial'`) && (argv[6] ?? '').includes('tar --zstd -xf') },
  { argv: (argv) => (argv[6] ?? '').includes(`'${DEST}.partial/bin/crucible' --version`) },
  { argv: (argv) => (argv[6] ?? '').startsWith(`rm -rf '${DEST}' && mv '${DEST}.partial' '${DEST}'`) && (argv[6] ?? '').includes('sha256=%s') },
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

test('win32: the whole PHASE14 sequence, each step by name, the token redacted everywhere it could show', async () => {
  const c = collector();
  let tokenSeen: string | null = null;
  const expectations: Expectation[] = [
    LIST,
    { argv: PROBE(), stdout: GUEST_BARE },
    MANIFEST,
    ...PACK_FETCH,
    NO_CONFIG,
    {
      argv: (argv) => {
        const ok = argv.slice(0, 6).join(' ') === W(CRUCIBLE_BIN, 'init').join(' ') && argv[6] === '--token' && argv.slice(8).join(' ') === '--enable-llm --enable-tts';
        tokenSeen = argv[7] ?? null;
        return ok;
      },
      lines: [['backend:  cuda-linux', 'stdout'], ['token:    minted; print it with `crucible token --show`', 'stdout']],
    },
    { argv: W(CRUCIBLE_BIN, 'install', 'llm', '--verbose'), lines: [['  downloading llm pack', 'stdout'], ['installed in 400s', 'stdout']] },
    { argv: W(CRUCIBLE_BIN, 'install', 'tts', '--narrator-engine', 'higgs-v3', '--verbose'), lines: [['installed in 300s', 'stdout']] },
    { argv: W(CRUCIBLE_BIN, 'service', 'install'), lines: [['enabled and started crucible.service', 'stdout'], ['linger: OFF for owen', 'stdout']] },
    WHOAMI,
    LINGER_OFF,
    GRANT,
    { argv: W(CRUCIBLE_BIN, 'capability', '--write'), lines: [['recorded in /home/owen/.crucible/config.toml', 'stdout']] },
    CONFIG_OK,
  ];
  const runner = new FakeRunner({ platform: 'win32' }, expectations);
  const result = await install({
    distro: 'Ubuntu',
    jobTypes: ['llm', { type: 'tts', narratorEngine: 'higgs-v3' }],
    onLine: c.onLine,
    onStep: c.onStep,
  }, runner);
  runner.assertDrained();

  assert.deepEqual(result.steps.map((s) => `${s.name}:${s.status}`), [
    'host-facts:ok', 'server-pack:ok', 'init:ok', 'install-llm:ok', 'install-tts:ok', 'service-install:ok', 'linger:ok', 'capability-write:ok',
  ]);
  assert.deepEqual(result.server, { name: 'crucible@owens-pc-wsl', url: 'http://127.0.0.1:7100', configPath: 'Ubuntu:/home/owen/.crucible/config.toml' });
  assert.equal(result.release, '0.6.0');
  assert.equal(result.backend, 'cuda-linux');
  assert.equal(result.crucible, CRUCIBLE_BIN);
  assert.equal('token' in result.server, false);

  assert.ok(tokenSeen !== null && (tokenSeen as string).length >= 40, 'a token was minted and passed');
  const init = result.steps.find((s) => s.name === 'init');
  // A step records the argv as the TARGET sees it; the wsl.exe wrapping is the transport's.
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

test('the pack download happens INSIDE the guest: curl, sha256sum, tar --zstd, atomic rename', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    { argv: PROBE(), stdout: GUEST_BARE },
    MANIFEST,
    ...PACK_FETCH,
    CONFIG_OK,
    { argv: W(CRUCIBLE_BIN, 'service', 'install') },
    WHOAMI,
    LINGER_ON,
    { argv: W(CRUCIBLE_BIN, 'capability', '--write') },
    CONFIG_OK,
  ]);
  await install({ distro: 'Ubuntu', jobTypes: ['echo'], onLine: c.onLine }, runner);
  runner.assertDrained();
  const guest = runner.calls.map((call) => (call.argv[4] === 'bash' ? call.argv[6] ?? '' : call.argv.slice(4).join(' ')));
  const curls = guest.filter((line) => line.startsWith('curl -fL'));
  assert.equal(curls.length, 2, 'one curl per part');
  for (const line of curls) {
    assert.match(line, /--continue-at -/, 'a killed part resumes');
    assert.match(line, /cat '\/home\/owen\/\.crucible\/downloads\/[^']+' >> '\/home\/owen\/\.crucible\/downloads\/crucible-env-server-cuda-linux-0\.6\.0\.tar\.zst'/);
    assert.match(line, /&& rm -f /, 'the part is deleted once appended: peak extra disk is one part');
    assert.equal(line.includes('/mnt/'), false, 'nothing crosses /mnt/c');
  }
  assert.ok(guest.includes(`sha256sum ${ARCHIVE}`), 'the digest is computed in the guest');
  assert.ok(guest.some((line) => line.includes(`tar --zstd -xf '${ARCHIVE}' -C '${DEST}.partial'`)));
  assert.ok(guest.some((line) => line.includes(`mv '${DEST}.partial' '${DEST}'`)), 'the unpack is renamed into place, never unpacked over');
});

test('a pack whose stamp already matches the manifest is skipped, and nothing is downloaded', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    { argv: PROBE(), stdout: GUEST_INSTALLED },
    MANIFEST,
    CONFIG_OK,
    { argv: W(CRUCIBLE_BIN, 'service', 'install') },
    WHOAMI,
    LINGER_ON,
    { argv: W(CRUCIBLE_BIN, 'capability', '--write') },
    CONFIG_OK,
  ]);
  const result = await install({ distro: 'Ubuntu', jobTypes: ['echo'], onLine: c.onLine }, runner);
  runner.assertDrained();
  const pack = result.steps.find((s) => s.name === 'server-pack');
  assert.equal(pack?.status, 'skipped');
  assert.match(pack?.detail ?? '', /already the 0\.6\.0 pack/);
});

test('a sha that is not the manifest\'s deletes the archive and refuses by name', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    { argv: PROBE(), stdout: GUEST_BARE },
    MANIFEST,
    ...PACK_FETCH.slice(0, 3),
    { argv: W('sha256sum', ARCHIVE), stdout: `${'9'.repeat(64)}  ${ARCHIVE}\n` },
    { argv: W('bash', '-c', `rm -f '${ARCHIVE}'`) },
  ]);
  const r = await refusal(install({ distro: 'Ubuntu', jobTypes: ['echo'], onLine: c.onLine }, runner));
  runner.assertDrained();
  assert.equal(r.code, 'pack_sha_mismatch');
  assert.match(r.message, /hashes 9{64} and the 0\.6\.0 manifest says a{64}/);
  assert.match(r.message, /The archive was deleted/);
});

test('the disk pre-flight refuses with the numbers BEFORE anything is fetched', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    { argv: PROBE(), stdout: 'home=/home/owen/.crucible\nuser=owen\nfree_kib=100000\n' },
    MANIFEST,
  ]);
  const r = await refusal(install({ distro: 'Ubuntu', jobTypes: ['echo'], onLine: c.onLine }, runner));
  runner.assertDrained();
  assert.equal(r.code, 'pack_disk');
  assert.match(r.message, /needs 0\.5 GiB free .* and there is 0\.1 GiB/);
  assert.match(r.message, /Nothing was downloaded/);
});

test('a release with no server pack for this backend is pack_not_published, never a build', async () => {
  const c = collector();
  const manifest = JSON.stringify({ schema: 1, version: '0.6.0', packs: [JSON.parse(ENVPACKS_JSON).packs[2]] });
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    { argv: PROBE(), stdout: GUEST_BARE },
    { argv: MANIFEST.argv, stdout: manifest },
  ]);
  const r = await refusal(install({ distro: 'Ubuntu', jobTypes: ['echo'], onLine: c.onLine }, runner));
  assert.equal(r.code, 'pack_not_published');
  assert.match(r.message, /publishes no "server" pack for cuda-linux/);
  assert.match(r.message, /lists llm for that backend/);
});

test('a guest with no curl, tar or zstd refuses before the manifest is even asked for', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    { argv: PROBE(), stdout: `${GUEST_BARE}missing=zstd\n` },
  ]);
  const r = await refusal(install({ distro: 'Ubuntu', jobTypes: ['echo'], onLine: c.onLine }, runner));
  runner.assertDrained();
  assert.equal(r.code, 'guest_missing_tool');
  assert.equal(r.command, 'sudo apt-get update && sudo apt-get install -y zstd');
});

test('init is skipped when a config already exists, and its token is kept', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    { argv: PROBE(), stdout: GUEST_INSTALLED },
    MANIFEST,
    CONFIG_OK,
    { argv: W(CRUCIBLE_BIN, 'install', 'asr', '--verbose') },
    { argv: W(CRUCIBLE_BIN, 'service', 'install') },
    WHOAMI,
    LINGER_ON,
    { argv: W(CRUCIBLE_BIN, 'capability', '--write') },
    CONFIG_OK,
  ]);
  const result = await install({ distro: 'Ubuntu', jobTypes: ['asr', 'echo'], onLine: c.onLine }, runner);
  runner.assertDrained();
  const init = result.steps.find((s) => s.name === 'init');
  assert.equal(init?.status, 'skipped');
  assert.match(init?.detail ?? '', /config exists at Ubuntu:\/home\/owen\/\.crucible\/config\.toml; its token is kept/);
  assert.equal(result.steps.some((s) => s.name === 'install-echo'), false, 'echo has no installer');
});

test('a failing step stops the sequence with its name, exit code, tail and the steps done', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    { argv: PROBE(), stdout: GUEST_INSTALLED },
    MANIFEST,
    CONFIG_OK,
    { argv: W(CRUCIBLE_BIN, 'install', 'llm', '--verbose'), code: 1, lines: [['recipe: llm/cuda-linux', 'stdout'], ['ERROR: pack_not_published', 'stderr']] },
  ]);
  const err = await refusal(install({ distro: 'Ubuntu', jobTypes: ['llm'], onLine: c.onLine }, runner));
  runner.assertDrained();
  assert.equal(err.code, 'step_failed');
  const failed = err.error as BootstrapStepFailed;
  assert.ok(failed instanceof BootstrapStepFailed);
  assert.equal(failed.step, 'install-llm');
  assert.equal(failed.exitCode, 1);
  assert.deepEqual(failed.stepsDone, ['host-facts']);
  assert.deepEqual(failed.tail, ['recipe: llm/cuda-linux', '! ERROR: pack_not_published']);
  assert.match(failed.message, /install step "install-llm" exited 1 inside WSL distro "Ubuntu"/);
});

test('a step that never returns is a failure naming the timeout, not a hang', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    { argv: PROBE(), stdout: GUEST_INSTALLED },
    MANIFEST,
    CONFIG_OK,
    { argv: () => true, failure: 'wsl.exe did not answer within 1800s' },
  ]);
  const err = await refusal(install({ distro: 'Ubuntu', jobTypes: ['llm'], onLine: c.onLine }, runner));
  const failed = err.error as BootstrapStepFailed;
  assert.equal(failed.step, 'install-llm');
  assert.equal(failed.exitCode, null);
  assert.match(failed.message, /did not answer within 1800s/);
});

test('a broken existing config is refused by name rather than re-initialised over', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    { argv: PROBE(), stdout: GUEST_INSTALLED },
    MANIFEST,
    { argv: READ, stdout: '/home/owen/.crucible/config.toml\n[server]\nname = "n"\n' },
  ]);
  const r = await refusal(install({ distro: 'Ubuntu', jobTypes: ['llm'], onLine: c.onLine }, runner));
  assert.equal(r.code, 'config_missing_key');
  assert.equal(r.command, 'crucible init --force');
});

test('{home} travels as `env CRUCIBLE_HOME=…` into every crucible verb, and {bind} into init', async () => {
  const c = collector();
  const E = (...argv: string[]): string[] => W('env', 'CRUCIBLE_HOME=/srv/crucible', ...argv);
  const runner = new FakeRunner({ platform: 'win32' }, [
    LIST,
    { argv: PROBE('/srv/crucible'), stdout: 'home=/srv/crucible\nuser=owen\nfree_kib=400000000\ncrucible=/srv/crucible/server/bin/crucible\nversion=crucible 0.6.0\n' + `sha256=${PACK_SHA}\nrelease=0.6.0\n` },
    MANIFEST,
    { argv: (argv) => (argv[6] ?? '').startsWith("p='/srv/crucible'/config.toml"), code: 3, stderr: '/srv/crucible/config.toml' },
    { argv: (argv) => argv.slice(0, 8).join(' ') === E('/srv/crucible/server/bin/crucible', 'init').join(' ') && argv[8] === '--token' && argv.slice(10).join(' ') === '--host 0.0.0.0 --port 7200 --enable-echo' },
    { argv: E('/srv/crucible/server/bin/crucible', 'service', 'install') },
    { argv: W('env', 'CRUCIBLE_HOME=/srv/crucible', 'id', '-un'), stdout: 'owen\n' },
    LINGER_ON,
    { argv: E('/srv/crucible/server/bin/crucible', 'capability', '--write') },
    { argv: READ, stdout: `/srv/crucible/config.toml\n${GUEST_CONFIG}` },
  ]);
  await install({ distro: 'Ubuntu', home: '/srv/crucible', bind: { host: '0.0.0.0', port: 7200 }, jobTypes: ['echo'], onLine: c.onLine }, runner);
  runner.assertDrained();
  for (const call of runner.calls) assert.equal(call.env, undefined, 'on win32 nothing is passed as a Windows-side env');
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
