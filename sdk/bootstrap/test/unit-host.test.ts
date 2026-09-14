/** `detectHost()` on the three platforms, every null with its refusal. */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { detectHost, DEFAULT_CONDA_ROOTS } from '../src/index.js';
import { interpreterScript, NVIDIA_SMI_SCRIPT, parseNvidiaSmi } from '../src/host.js';
import { CRUCIBLE_BIN, FakeRunner, INTERPRETER_OK, PYTHON_BIN, refusal, WSL_LIST } from './fake.js';

const LIST = ['wsl.exe', '-l', '-v'];
const GPU = ['wsl.exe', '-d', 'Ubuntu', '--exec', 'bash', '-c', NVIDIA_SMI_SCRIPT];
const INTERP = ['wsl.exe', '-d', 'Ubuntu', '--exec', 'bash', '-c', interpreterScript(DEFAULT_CONDA_ROOTS)];
const SMI = 'NVIDIA GeForce RTX 3090 Ti, 24564\n';

test('the interpreter script: conda by test -x at the three roots in order, never which; the env\'s python beside it', () => {
  const script = interpreterScript(DEFAULT_CONDA_ROOTS);
  assert.match(script, /for c in "\$HOME\/anaconda3" "\$HOME\/miniconda3" "\$HOME\/miniforge3"; do if test -x "\$c\/bin\/conda"; then conda="\$c"; break; fi; done/);
  assert.match(script, /p="\$conda\/envs\/crucible\/bin\/python"/);
  assert.doesNotMatch(script, /which/);
  assert.doesNotMatch(script, /\\/);
  assert.equal(interpreterScript(['/opt/homebrew/Caskroom/miniconda/base']).includes("for c in '/opt/homebrew/Caskroom/miniconda/base'; do"), true);
  assert.throws(() => interpreterScript(['relative/x']), /must start with ~\/ or \//);
});

test('parseNvidiaSmi reads name and MiB exactly as crucible/backend.py does', () => {
  assert.deepEqual(parseNvidiaSmi(SMI), { vendor: 'nvidia', name: 'NVIDIA GeForce RTX 3090 Ti', vramBytes: 24564 * 1024 * 1024 });
  assert.equal(parseNvidiaSmi(''), null);
  assert.equal(parseNvidiaSmi('a, b, c'), null);
  assert.equal(parseNvidiaSmi('card, lots'), null);
});

test('win32: the guest\'s facts, read through the default distro, in three calls', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: GPU, stdout: SMI },
    { argv: INTERP, stdout: INTERPRETER_OK },
  ]);
  const facts = await detectHost({}, runner);
  assert.deepEqual(facts, {
    platform: 'win32',
    wsl: { present: true, distros: [{ name: 'Ubuntu', state: 'Running', version: 2, default: true }], default: 'Ubuntu', probed: 'Ubuntu' },
    gpu: { vendor: 'nvidia', name: 'NVIDIA GeForce RTX 3090 Ti', vramBytes: 25757220864 },
    python: { path: PYTHON_BIN, version: '3.11.9' },
    conda: { root: '/home/owen/anaconda3' },
    refusals: [],
  });
  assert.ok(CRUCIBLE_BIN.startsWith('/home/owen/anaconda3/envs/crucible/bin/'));
  runner.assertDrained();
});

test('win32: {distro} picks the guest to read through, and is recorded as probed', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: '  NAME    STATE    VERSION\r\n* Ubuntu  Running  2\r\n  Other   Stopped  2\r\n' },
    { argv: ['wsl.exe', '-d', 'Other', '--exec', 'bash', '-c', NVIDIA_SMI_SCRIPT], stdout: SMI },
    { argv: ['wsl.exe', '-d', 'Other', '--exec', 'bash', '-c', interpreterScript(DEFAULT_CONDA_ROOTS)], stdout: INTERPRETER_OK },
  ]);
  const facts = await detectHost({ distro: 'Other' }, runner);
  assert.equal(facts.wsl?.probed, 'Other');
  assert.equal(facts.wsl?.default, 'Ubuntu');
});

test('win32: no wsl.exe is wsl_missing with the install command and the reboot note', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [{ argv: LIST, failure: 'spawn wsl.exe ENOENT' }]);
  const r = await refusal(detectHost({}, runner));
  assert.equal(r.code, 'wsl_missing');
  assert.equal(r.command, 'wsl --install -d Ubuntu');
  assert.match(r.message, /REBOOT Windows/);
});

test('win32: wsl.exe failing, or listing nothing, is wsl_missing too; a timeout is wsl_read_failed', async () => {
  const failing = new FakeRunner({ platform: 'win32' }, [{ argv: LIST, code: 1, stderr: 'The Windows Subsystem for Linux is not installed.' }]);
  const f = await refusal(detectHost({}, failing));
  assert.equal(f.code, 'wsl_missing');
  assert.match(f.message, /not installed/);
  assert.equal(f.command, 'wsl --install -d Ubuntu');

  const empty = new FakeRunner({ platform: 'win32' }, [{ argv: LIST, stdout: '  NAME  STATE  VERSION\r\n' }]);
  const e = await refusal(detectHost({}, empty));
  assert.equal(e.code, 'wsl_missing');
  assert.match(e.message, /lists no distribution/);

  const slow = new FakeRunner({ platform: 'win32' }, [{ argv: LIST, failure: 'wsl.exe did not answer within 15s' }]);
  const s = await refusal(detectHost({}, slow));
  assert.equal(s.code, 'wsl_read_failed');
});

test('win32: distros but no default and no {distro} is no_wsl_distro', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [{ argv: LIST, stdout: '  NAME    STATE    VERSION\r\n  Ubuntu  Stopped  2\r\n' }]);
  const r = await refusal(detectHost({}, runner));
  assert.equal(r.code, 'no_wsl_distro');
  assert.match(r.message, /lists Ubuntu and marks none as default/);
});

test('win32: no nvidia-smi in the guest is gpu null with no_nvidia_driver naming the Windows driver', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: GPU, code: 3 },
    { argv: INTERP, stdout: INTERPRETER_OK },
  ]);
  const facts = await detectHost({}, runner);
  assert.equal(facts.gpu, null);
  assert.equal(facts.refusals.length, 1);
  assert.equal(facts.refusals[0]?.code, 'no_nvidia_driver');
  assert.match(facts.refusals[0]?.message ?? '', /looked on PATH and at \/usr\/lib\/wsl\/lib\/nvidia-smi/);
  assert.match(facts.refusals[0]?.command ?? '', /NVIDIA Windows driver/);
  assert.match(facts.refusals[0]?.command ?? '', /Do NOT install a driver inside the distro/);
});

test('win32: nvidia-smi failing, or answering nonsense, is no_nvidia_driver with its words', async () => {
  const failing = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: GPU, code: 9, stderr: 'NVIDIA-SMI has failed because it couldn\'t communicate with the NVIDIA driver.' },
    { argv: INTERP, stdout: INTERPRETER_OK },
  ]);
  const f = await detectHost({}, failing);
  assert.equal(f.refusals[0]?.code, 'no_nvidia_driver');
  assert.match(f.refusals[0]?.message ?? '', /exited 9.*couldn't communicate/);

  const nonsense = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: GPU, stdout: 'not,a,gpu,line' },
    { argv: INTERP, stdout: INTERPRETER_OK },
  ]);
  const n = await detectHost({}, nonsense);
  assert.equal(n.refusals[0]?.code, 'no_nvidia_driver');
  assert.match(n.refusals[0]?.message ?? '', /unparseable/);
});

test('win32: a probe that cannot run in the guest is wsl_read_failed, thrown', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: GPU, failure: 'wsl.exe did not answer within 60s' },
  ]);
  const r = await refusal(detectHost({}, runner));
  assert.equal(r.code, 'wsl_read_failed');
  assert.match(r.message, /run nvidia-smi inside WSL distro "Ubuntu"/);
});

test('no conda anywhere is no_conda, naming the roots and the miniforge command for the platform', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: GPU, stdout: SMI },
    { argv: INTERP, stdout: 'home=/home/owen\n' },
  ]);
  const facts = await detectHost({}, runner);
  assert.equal(facts.python, null);
  assert.equal(facts.conda, null);
  assert.equal(facts.refusals[0]?.code, 'no_conda');
  assert.match(facts.refusals[0]?.message ?? '', /none of ~\/anaconda3, ~\/miniconda3, ~\/miniforge3 has bin\/conda/);
  assert.match(facts.refusals[0]?.message ?? '', /conda create -n crucible python=3\.11 -y/);
  assert.match(facts.refusals[0]?.command ?? '', /Miniforge3-Linux-x86_64\.sh.*-p "\$HOME\/miniforge3"/);
});

test('conda without the crucible env is no_python with the exact conda create', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: GPU, stdout: SMI },
    { argv: INTERP, stdout: 'home=/home/owen\nconda=/home/owen/miniforge3\n' },
  ]);
  const facts = await detectHost({}, runner);
  assert.equal(facts.python, null);
  assert.deepEqual(facts.conda, { root: '/home/owen/miniforge3' });
  assert.equal(facts.refusals[0]?.code, 'no_python');
  assert.equal(facts.refusals[0]?.command, '/home/owen/miniforge3/bin/conda create -n crucible python=3.11 -y');
});

test('a crucible env that is not 3.11 is reported as a fact AND refused as no_python', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: GPU, stdout: SMI },
    { argv: INTERP, stdout: INTERPRETER_OK.replace('3.11.9', '3.12.3') },
  ]);
  const facts = await detectHost({}, runner);
  assert.deepEqual(facts.python, { path: PYTHON_BIN, version: '3.12.3' });
  assert.equal(facts.refusals[0]?.code, 'no_python');
  assert.match(facts.refusals[0]?.message ?? '', /is Python 3\.12\.3; the server needs 3\.11/);
  assert.match(facts.refusals[0]?.command ?? '', /conda remove -n crucible --all -y && .*conda create -n crucible python=3\.11 -y/);
});

test('an interpreter that will not say its version is no_python', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: GPU, stdout: SMI },
    { argv: INTERP, stdout: 'home=/home/owen\nconda=/home/owen/anaconda3\npython=/home/owen/anaconda3/envs/crucible/bin/python\nversion=bash: python: cannot execute\n' },
  ]);
  const facts = await detectHost({}, runner);
  assert.equal(facts.python, null);
  assert.equal(facts.refusals[0]?.code, 'no_python');
  assert.match(facts.refusals[0]?.message ?? '', /would not report a version/);
});

test('darwin: uname, sysctl and the interpreter script, natively — unified memory is the card', async () => {
  const runner = new FakeRunner({ platform: 'darwin' }, [
    { argv: ['uname', '-m'], stdout: 'arm64\n' },
    { argv: ['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize'], stdout: 'Apple M1 Ultra\n68719476736\n' },
    { argv: ['bash', '-c', interpreterScript(['/opt/homebrew/Caskroom/miniconda/base'])], stdout: 'home=/Users/owen\nconda=/opt/homebrew/Caskroom/miniconda/base\npython=/opt/homebrew/Caskroom/miniconda/base/envs/crucible/bin/python\nversion=Python 3.11.13\n' },
  ]);
  const facts = await detectHost({ condaRoots: ['/opt/homebrew/Caskroom/miniconda/base'] }, runner);
  assert.equal(facts.wsl, null);
  assert.deepEqual(facts.gpu, { vendor: 'apple', name: 'Apple M1 Ultra', vramBytes: 68719476736 });
  assert.equal(facts.python?.version, '3.11.13');
  assert.deepEqual(facts.refusals, []);
});

test('darwin: an Intel Mac is not_apple_silicon, and the miniforge command names the arm64 asset', async () => {
  const runner = new FakeRunner({ platform: 'darwin' }, [
    { argv: ['uname', '-m'], stdout: 'x86_64\n' },
    { argv: () => true, stdout: 'home=/Users/owen\n' },
  ]);
  const facts = await detectHost({}, runner);
  assert.equal(facts.gpu, null);
  assert.equal(facts.refusals[0]?.code, 'not_apple_silicon');
  assert.equal(facts.refusals[1]?.code, 'no_conda');
  assert.match(facts.refusals[1]?.command ?? '', /Miniforge3-MacOSX-arm64\.sh/);
});

test('darwin: sysctl failing is not_apple_silicon with its words', async () => {
  const runner = new FakeRunner({ platform: 'darwin' }, [
    { argv: ['uname', '-m'], stdout: 'arm64\n' },
    { argv: () => true, code: 1, stderr: 'sysctl: unknown oid' },
    { argv: () => true, stdout: 'home=/Users/owen\n' },
  ]);
  const facts = await detectHost({}, runner);
  assert.equal(facts.gpu, null);
  assert.match(facts.refusals[0]?.message ?? '', /unknown oid/);
});

test('linux: nvidia-smi and the interpreter, natively; a failed spawn is host_unresponsive', async () => {
  const runner = new FakeRunner({ platform: 'linux' }, [
    { argv: ['bash', '-c', NVIDIA_SMI_SCRIPT], stdout: SMI },
    { argv: ['bash', '-c', interpreterScript(DEFAULT_CONDA_ROOTS)], stdout: INTERPRETER_OK },
  ]);
  const facts = await detectHost({}, runner);
  assert.equal(facts.gpu?.name, 'NVIDIA GeForce RTX 3090 Ti');
  assert.equal(facts.python?.path, PYTHON_BIN);

  const dead = new FakeRunner({ platform: 'linux' }, [{ argv: () => true, failure: 'spawn bash ENOENT' }]);
  const r = await refusal(detectHost({}, dead));
  assert.equal(r.code, 'host_unresponsive');
});

test('an unsupported platform is refused by name', async () => {
  const r = await refusal(detectHost({}, new FakeRunner({ platform: 'freebsd' }, [])));
  assert.equal(r.code, 'unsupported_platform');
});
