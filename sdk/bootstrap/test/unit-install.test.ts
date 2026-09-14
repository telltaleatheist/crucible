/** `install()` — the sequence by name, the token never shown, a failing step's tail. */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { BootstrapStepFailed, install, planJobTypes, DEFAULT_CONDA_ROOTS, type InstallStep, type OutputStream } from '../src/index.js';
import { interpreterScript } from '../src/host.js';
import { CRUCIBLE_BIN, FakeRunner, GUEST_CONFIG, INTERPRETER_OK, PYTHON_BIN, refusal, type Expectation } from './fake.js';

const W = (...argv: string[]): string[] => ['wsl.exe', '-d', 'Ubuntu', '--exec', ...argv];
const INTERP = W('bash', '-c', interpreterScript(DEFAULT_CONDA_ROOTS));
const READ = (argv: readonly string[]): boolean => argv[4] === 'bash' && (argv[6] ?? '').includes('config.toml');
const CONFIG_OK = { argv: READ, stdout: `/home/owen/.crucible/config.toml\n${GUEST_CONFIG}` };
const NO_CONFIG = { argv: READ, code: 3, stderr: '/home/owen/.crucible/config.toml\n' };

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

test('win32: the whole sequence, each step by name, the token redacted everywhere it could show', async () => {
  const c = collector();
  let tokenSeen: string | null = null;
  const expectations: Expectation[] = [
    { argv: INTERP, stdout: INTERPRETER_OK },
    { argv: W(PYTHON_BIN, '-m', 'pip', 'install', '/mnt/c/dl/crucible-0.5.0-py3-none-any.whl'), lines: [['Collecting crucible', 'stdout'], ['Successfully installed crucible-0.5.0', 'stdout']] },
    NO_CONFIG,
    {
      argv: (argv) => {
        const ok = argv.slice(0, 6).join(' ') === W(CRUCIBLE_BIN, 'init').join(' ') && argv[6] === '--token' && argv.slice(8).join(' ') === '--enable-llm --enable-tts';
        tokenSeen = argv[7] ?? null;
        return ok;
      },
      lines: [['backend:  cuda-linux', 'stdout'], ['token:    minted; print it with `crucible token --show`', 'stdout']],
    },
    { argv: W(CRUCIBLE_BIN, 'install', 'llm', '--verbose'), lines: [['  Collecting vllm', 'stdout'], ['installed in 400s', 'stdout']] },
    { argv: W(CRUCIBLE_BIN, 'install', 'tts', '--narrator-engine', 'higgs-v3', '--verbose'), lines: [['installed in 300s', 'stdout']] },
    { argv: W(CRUCIBLE_BIN, 'service', 'install'), lines: [['enabled and started crucible.service', 'stdout'], ['linger: OFF for owen', 'stdout']] },
    { argv: W(CRUCIBLE_BIN, 'capability', '--write'), lines: [['recorded in /home/owen/.crucible/config.toml', 'stdout']] },
    CONFIG_OK,
  ];
  const runner = new FakeRunner({ platform: 'win32', files: { 'C:\\dl\\crucible-0.5.0-py3-none-any.whl': '' } }, expectations);
  const result = await install({
    distro: 'Ubuntu',
    jobTypes: ['llm', { type: 'tts', narratorEngine: 'higgs-v3' }],
    wheel: 'C:\\dl\\crucible-0.5.0-py3-none-any.whl',
    onLine: c.onLine,
    onStep: c.onStep,
  }, runner);
  runner.assertDrained();

  assert.deepEqual(result.steps.map((s) => `${s.name}:${s.status}`), [
    'interpreter:ok', 'pip-install:ok', 'init:ok', 'install-llm:ok', 'install-tts:ok', 'service-install:ok', 'capability-write:ok',
  ]);
  assert.deepEqual(result.server, { name: 'crucible@owens-pc-wsl', url: 'http://127.0.0.1:7100', configPath: 'Ubuntu:/home/owen/.crucible/config.toml' });
  assert.equal('token' in result.server, false);

  assert.ok(tokenSeen !== null && (tokenSeen as string).length >= 40, 'a token was minted and passed');
  const init = result.steps.find((s) => s.name === 'init');
  // A step records the argv as the TARGET sees it; the wsl.exe wrapping is the transport's.
  assert.deepEqual(init?.argv, [CRUCIBLE_BIN, 'init', '--token', '<redacted>', '--enable-llm', '--enable-tts']);
  const everything = JSON.stringify(result) + c.lines.join('\n') + c.steps.join('\n');
  assert.equal(everything.includes(tokenSeen as string), false, 'the token appears nowhere the host can log');

  assert.deepEqual(c.lines.slice(0, 2), ['pip-install/stdout: Collecting crucible', 'pip-install/stdout: Successfully installed crucible-0.5.0']);
  assert.ok(c.lines.includes('install-llm/stdout:   Collecting vllm'));
  assert.ok(c.lines.includes('service-install/stdout: linger: OFF for owen'));
  assert.equal(c.steps[0], 'interpreter:ok');
  assert.equal(c.steps[1], 'pip-install:running');
  assert.equal(c.steps[2], 'pip-install:ok');
  assert.ok(runner.calls.filter((call) => call.streamed).length === 6, 'every crucible verb and pip streamed');
});

test('init is skipped when a config already exists, and its token is kept', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32', files: { 'C:\\dl\\w.whl': '' } }, [
    { argv: INTERP, stdout: INTERPRETER_OK },
    { argv: W(PYTHON_BIN, '-m', 'pip', 'install', '/mnt/c/dl/w.whl') },
    CONFIG_OK,
    { argv: W(CRUCIBLE_BIN, 'install', 'asr', '--verbose') },
    { argv: W(CRUCIBLE_BIN, 'service', 'install') },
    { argv: W(CRUCIBLE_BIN, 'capability', '--write') },
    CONFIG_OK,
  ]);
  const result = await install({ distro: 'Ubuntu', jobTypes: ['asr', 'echo'], wheel: 'C:\\dl\\w.whl', onLine: c.onLine }, runner);
  runner.assertDrained();
  const init = result.steps.find((s) => s.name === 'init');
  assert.equal(init?.status, 'skipped');
  assert.match(init?.detail ?? '', /config exists at Ubuntu:\/home\/owen\/\.crucible\/config\.toml; its token is kept/);
  assert.equal(result.steps.some((s) => s.name === 'install-echo'), false, 'echo has no installer');
});

test('a failing step stops the sequence with its name, exit code, tail and the steps done', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32', files: { 'C:\\dl\\w.whl': '' } }, [
    { argv: INTERP, stdout: INTERPRETER_OK },
    { argv: W(PYTHON_BIN, '-m', 'pip', 'install', '/mnt/c/dl/w.whl') },
    CONFIG_OK,
    { argv: W(CRUCIBLE_BIN, 'install', 'llm', '--verbose'), code: 1, lines: [['recipe: llm/cuda-linux', 'stdout'], ['ERROR: No matching distribution found for vllm==0.11.0', 'stderr'], ['crucible: the env did not come out installed', 'stderr']] },
  ]);
  const err = await refusal(install({ distro: 'Ubuntu', jobTypes: ['llm'], wheel: 'C:\\dl\\w.whl', onLine: c.onLine }, runner));
  runner.assertDrained();
  assert.equal(err.code, 'step_failed');
  const failed = err.error as BootstrapStepFailed;
  assert.ok(failed instanceof BootstrapStepFailed);
  assert.equal(failed.step, 'install-llm');
  assert.equal(failed.exitCode, 1);
  assert.deepEqual(failed.stepsDone, ['interpreter', 'pip-install']);
  assert.deepEqual(failed.tail, ['recipe: llm/cuda-linux', '! ERROR: No matching distribution found for vllm==0.11.0', '! crucible: the env did not come out installed']);
  assert.match(failed.message, /install step "install-llm" exited 1 inside WSL distro "Ubuntu"/);
  assert.equal(failed.detail, failed.tail.join('\n'));
});

test('a step that never returns is a failure naming the timeout, not a hang', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32', files: { 'C:\\dl\\w.whl': '' } }, [
    { argv: INTERP, stdout: INTERPRETER_OK },
    { argv: () => true, failure: 'wsl.exe did not answer within 1800s' },
  ]);
  const err = await refusal(install({ distro: 'Ubuntu', jobTypes: ['llm'], wheel: 'C:\\dl\\w.whl', onLine: c.onLine }, runner));
  const failed = err.error as BootstrapStepFailed;
  assert.equal(failed.step, 'pip-install');
  assert.equal(failed.exitCode, null);
  assert.match(failed.message, /did not answer within 1800s/);
});

test('the wheel: a URL passes verbatim; a missing file is wheel_missing; a mapped drive is network_path', async () => {
  const c = collector();
  const url = new FakeRunner({ platform: 'win32' }, [
    { argv: INTERP, stdout: INTERPRETER_OK },
    { argv: W(PYTHON_BIN, '-m', 'pip', 'install', 'https://github.com/telltaleatheist/crucible/releases/download/v0.5.0/crucible-0.5.0-py3-none-any.whl'), code: 2 },
  ]);
  const u = await refusal(install({ distro: 'Ubuntu', jobTypes: ['llm'], wheel: 'https://github.com/telltaleatheist/crucible/releases/download/v0.5.0/crucible-0.5.0-py3-none-any.whl', onLine: c.onLine }, url));
  assert.equal((u.error as BootstrapStepFailed).step, 'pip-install');

  const missing = new FakeRunner({ platform: 'win32' }, [{ argv: INTERP, stdout: INTERPRETER_OK }]);
  const m = await refusal(install({ distro: 'Ubuntu', jobTypes: ['llm'], wheel: 'C:\\dl\\gone.whl', onLine: c.onLine }, missing));
  assert.equal(m.code, 'wheel_missing');

  const mapped = new FakeRunner({ platform: 'win32', files: { 'Z:\\dl\\w.whl': '' }, realpaths: { 'Z:\\dl\\w.whl': '\\\\TITAN\\iO\\dl\\w.whl' } }, [{ argv: INTERP, stdout: INTERPRETER_OK }]);
  const n = await refusal(install({ distro: 'Ubuntu', jobTypes: ['llm'], wheel: 'Z:\\dl\\w.whl', onLine: c.onLine }, mapped));
  assert.equal(n.code, 'network_path');
});

test('no interpreter is the named refusal, before pip runs', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32', files: { 'C:\\dl\\w.whl': '' } }, [{ argv: INTERP, stdout: 'home=/home/owen\nconda=/home/owen/miniforge3\n' }]);
  const r = await refusal(install({ distro: 'Ubuntu', jobTypes: ['llm'], wheel: 'C:\\dl\\w.whl', onLine: c.onLine }, runner));
  assert.equal(r.code, 'no_python');
  assert.equal(r.command, '/home/owen/miniforge3/bin/conda create -n crucible python=3.11 -y');
  runner.assertDrained();
});

test('a broken existing config is refused by name rather than re-initialised over', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'win32', files: { 'C:\\dl\\w.whl': '' } }, [
    { argv: INTERP, stdout: INTERPRETER_OK },
    { argv: () => true },
    { argv: READ, stdout: '/home/owen/.crucible/config.toml\n[server]\nname = "n"\n' },
  ]);
  const r = await refusal(install({ distro: 'Ubuntu', jobTypes: ['llm'], wheel: 'C:\\dl\\w.whl', onLine: c.onLine }, runner));
  assert.equal(r.code, 'config_missing_key');
  assert.equal(r.command, 'crucible init --force');
});

test('{home} travels as `env CRUCIBLE_HOME=…` into every crucible verb, and {bind} into init', async () => {
  const c = collector();
  const E = (...argv: string[]): string[] => W('env', 'CRUCIBLE_HOME=/srv/crucible', ...argv);
  const runner = new FakeRunner({ platform: 'win32', files: { 'C:\\dl\\w.whl': '' } }, [
    { argv: INTERP, stdout: INTERPRETER_OK },
    { argv: E(PYTHON_BIN, '-m', 'pip', 'install', '/mnt/c/dl/w.whl') },
    { argv: (argv) => (argv[6] ?? '').startsWith("p='/srv/crucible'/config.toml"), code: 3, stderr: '/srv/crucible/config.toml' },
    { argv: (argv) => argv.slice(0, 8).join(' ') === E(CRUCIBLE_BIN, 'init').join(' ') && argv[8] === '--token' && argv.slice(10).join(' ') === '--host 0.0.0.0 --port 7200 --enable-echo' },
    { argv: E(CRUCIBLE_BIN, 'service', 'install') },
    { argv: E(CRUCIBLE_BIN, 'capability', '--write') },
    { argv: READ, stdout: `/srv/crucible/config.toml\n${GUEST_CONFIG}` },
  ]);
  await install({ distro: 'Ubuntu', home: '/srv/crucible', bind: { host: '0.0.0.0', port: 7200 }, jobTypes: ['echo'], wheel: 'C:\\dl\\w.whl', onLine: c.onLine }, runner);
  runner.assertDrained();
  for (const call of runner.calls) assert.equal(call.env, undefined, 'on win32 nothing is passed as a Windows-side env');
});

test('darwin: the same verbs run natively, {home} as a real env, the wheel path as it is', async () => {
  const c = collector();
  const runner = new FakeRunner({ platform: 'darwin', homedir: '/Users/owen', files: { '/Users/owen/Downloads/w.whl': '' } }, [
    { argv: ['bash', '-c', interpreterScript(DEFAULT_CONDA_ROOTS)], stdout: 'home=/Users/owen\nconda=/Users/owen/miniforge3\npython=/Users/owen/miniforge3/envs/crucible/bin/python\nversion=Python 3.11.13\n' },
    { argv: ['/Users/owen/miniforge3/envs/crucible/bin/python', '-m', 'pip', 'install', '/Users/owen/Downloads/w.whl'], env: { CRUCIBLE_HOME: '/srv/c' } },
    { argv: (argv) => argv[1] === 'init', env: { CRUCIBLE_HOME: '/srv/c' } },
    { argv: ['/Users/owen/miniforge3/envs/crucible/bin/crucible', 'install', 'llm', '--verbose'], env: { CRUCIBLE_HOME: '/srv/c' } },
    { argv: ['/Users/owen/miniforge3/envs/crucible/bin/crucible', 'service', 'install'], env: { CRUCIBLE_HOME: '/srv/c' } },
    { argv: ['/Users/owen/miniforge3/envs/crucible/bin/crucible', 'capability', '--write'], env: { CRUCIBLE_HOME: '/srv/c' } },
  ]);
  // The config read on darwin is a file read, not a call: absent before init, present after.
  let written = false;
  const originalExists = runner.fileExists.bind(runner);
  runner.fileExists = (path: string): boolean => (path === '/srv/c/config.toml' ? written : originalExists(path));
  runner.readFile = (): string => GUEST_CONFIG;
  const onStep = (step: InstallStep): void => {
    if (step.name === 'init' && step.status === 'ok') written = true;
  };
  const result = await install({ home: '/srv/c', jobTypes: ['llm'], wheel: '/Users/owen/Downloads/w.whl', onLine: c.onLine, onStep }, runner);
  runner.assertDrained();
  assert.equal(result.server.configPath, '/srv/c/config.toml');
});
