/** `detectHost()` on the three platforms, every null with its refusal, no conda anywhere. */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { detectHost } from '../src/index.js';
import { NVIDIA_SMI_SCRIPT, parseNvidiaSmi, pickProbeDistro } from '../src/host.js';
import { guestProbeScript } from '../src/runtime.js';
import { probeArgv } from '../src/wsl-states.js';
import { CRUCIBLE_BIN, FakeRunner, GUEST_BARE, GUEST_INSTALLED, refusal, WSL_LIST, WSL_LIST_WITH_CRUCIBLE } from './fake.js';

const LIST = ['wsl.exe', '-l', '-v'];
const GPU = ['wsl.exe', '-d', 'Ubuntu', '--exec', 'bash', '-c', NVIDIA_SMI_SCRIPT];
const PROBE = ['wsl.exe', '-d', 'Ubuntu', '--exec', 'bash', '-c', guestProbeScript(undefined)];
const SMI = 'NVIDIA GeForce RTX 3090 Ti, 24564\n';
const INPUTS = { release: '0.6.0' };
/** After the facts, detectHost walks the 4c table: --status, then -l -v again. */
const STATE_OK: { argv: string[]; stdout: string }[] = [
  { argv: probeArgv('wsl-status', INPUTS), stdout: 'WSL version: 2.3.26.0\r\nDefault Version: 2\r\n' },
  { argv: probeArgv('wsl-list', INPUTS), stdout: WSL_LIST },
];

test('the guest probe script: home, user, free space, the two tools, the runtime and its stamp', () => {
  const script = guestProbeScript(undefined);
  assert.match(script, /h="\$\{CRUCIBLE_HOME:-\$HOME\/\.crucible\}"/);
  assert.match(script, /for t in curl tar; do command -v/);
  assert.match(script, /c="\$h\/server\/bin\/crucible"/);
  assert.doesNotMatch(script, /conda/);
  assert.doesNotMatch(script, /\\/);
  assert.match(guestProbeScript('/srv/crucible'), /^h='\/srv\/crucible'; /);
});

test('parseNvidiaSmi reads name and MiB exactly as crucible/backend.py does', () => {
  assert.deepEqual(parseNvidiaSmi(SMI), { vendor: 'nvidia', name: 'NVIDIA GeForce RTX 3090 Ti', vramBytes: 24564 * 1024 * 1024 });
  assert.equal(parseNvidiaSmi(''), null);
  assert.equal(parseNvidiaSmi('a, b, c'), null);
  assert.equal(parseNvidiaSmi('card, lots'), null);
});

test('win32: the guest\'s facts, read through the default distro, with the pack it already has', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: GPU, stdout: SMI },
    { argv: PROBE, stdout: GUEST_INSTALLED },
    ...STATE_OK,
  ]);
  const facts = await detectHost({}, runner);
  runner.assertDrained();
  assert.equal(facts.platform, 'win32');
  assert.equal(facts.backend, 'cuda-linux');
  assert.deepEqual(facts.wsl, { present: true, distros: [{ name: 'Ubuntu', state: 'Running', version: 2, default: true }], default: 'Ubuntu', probed: 'Ubuntu' });
  assert.deepEqual(facts.gpu, { vendor: 'nvidia', name: 'NVIDIA GeForce RTX 3090 Ti', vramBytes: 25757220864 });
  assert.equal(facts.guest?.home, '/home/owen/.crucible');
  assert.equal(facts.guest?.user, 'owen');
  assert.equal(facts.server?.crucible, CRUCIBLE_BIN);
  assert.equal(facts.server?.release, '0.6.0');
  assert.deepEqual(facts.refusals, []);
  // The 4c row that matched: WSL is fine, there is simply no crucible distro here.
  assert.equal(facts.wslState?.code, 'no_crucible_distro');
});

test('win32: a machine with no server runtime says so as a STATE, not a refusal', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: GPU, stdout: SMI },
    { argv: PROBE, stdout: GUEST_BARE },
    ...STATE_OK,
  ]);
  const facts = await detectHost({}, runner);
  assert.equal(facts.server, null);
  assert.deepEqual(facts.refusals, [], 'install() puts one there; a fresh machine is not an error');
});

test('win32: the crucible distro wins over {distro}, and is recorded as probed', async () => {
  const OURS = (...argv: string[]): string[] => ['wsl.exe', '-d', 'crucible', '--exec', ...argv];
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: OURS('bash', '-c', NVIDIA_SMI_SCRIPT), stdout: SMI },
    { argv: OURS('bash', '-c', guestProbeScript(undefined)), stdout: GUEST_INSTALLED },
    { argv: probeArgv('wsl-status', INPUTS), stdout: 'Default Version: 2\r\n' },
    { argv: probeArgv('wsl-list', INPUTS), stdout: WSL_LIST_WITH_CRUCIBLE },
    { argv: probeArgv('wsl-conf', INPUTS), stdout: '# crucible-rootfs\n[boot]\nsystemd=true\n' },
    { argv: probeArgv('app-distro-conf', { ...INPUTS, appDistro: 'Ubuntu' }), stdout: '[boot]\nsystemd=true\n' },
    { argv: probeArgv('guest-root', INPUTS), stdout: '0\n' },
  ]);
  const facts = await detectHost({ distro: 'Ubuntu' }, runner);
  runner.assertDrained();
  assert.equal(facts.wsl?.probed, 'crucible');
  assert.equal(facts.wslState?.code, 'wsl_ready');
});

test('pickProbeDistro: ours, then the named one, then the default, then a refusal', () => {
  const listed = (names: string[], byDefault: string | null): { distros: { name: string; version: number; default: boolean; state: string }[]; default: string | null } =>
    ({ distros: names.map((name) => ({ name, version: 2, default: name === byDefault, state: 'Stopped' })), default: byDefault });
  assert.equal(pickProbeDistro(listed(['Ubuntu', 'crucible'], 'Ubuntu'), 'Ubuntu'), 'crucible');
  assert.equal(pickProbeDistro(listed(['Ubuntu', 'Other'], 'Ubuntu'), 'Other'), 'Other');
  assert.equal(pickProbeDistro(listed(['Ubuntu'], 'Ubuntu'), undefined), 'Ubuntu');
  assert.throws(() => pickProbeDistro(listed(['Ubuntu'], null), undefined), /no_wsl_distro|marks none as default/);
});

test('win32: no wsl.exe is wsl_missing with the install command', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [{ argv: LIST, failure: 'spawn wsl.exe ENOENT' }]);
  const r = await refusal(detectHost({}, runner));
  assert.equal(r.code, 'wsl_missing');
  assert.equal(r.command, 'wsl --install --no-distribution');
});

test('win32: wsl.exe failing, or listing nothing, is wsl_missing too; a timeout is wsl_read_failed', async () => {
  const failing = new FakeRunner({ platform: 'win32' }, [{ argv: LIST, code: 1, stderr: 'The Windows Subsystem for Linux is not installed.' }]);
  const f = await refusal(detectHost({}, failing));
  assert.equal(f.code, 'wsl_missing');
  assert.match(f.message, /not installed/);

  const empty = new FakeRunner({ platform: 'win32' }, [{ argv: LIST, stdout: '  NAME  STATE  VERSION\r\n' }]);
  const e = await refusal(detectHost({}, empty));
  assert.equal(e.code, 'no_wsl_distro');

  const slow = new FakeRunner({ platform: 'win32' }, [{ argv: LIST, failure: 'wsl.exe did not answer within 15s' }]);
  const s = await refusal(detectHost({}, slow));
  assert.equal(s.code, 'wsl_read_failed');
});

test('win32: no nvidia-smi in the guest is gpu null with no_nvidia_driver naming the Windows driver', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: GPU, code: 3 },
    { argv: PROBE, stdout: GUEST_INSTALLED },
    ...STATE_OK,
  ]);
  const facts = await detectHost({}, runner);
  assert.equal(facts.gpu, null);
  assert.equal(facts.refusals.length, 1);
  assert.equal(facts.refusals[0]?.code, 'no_nvidia_driver');
  assert.match(facts.refusals[0]?.message ?? '', /looked on PATH and at \/usr\/lib\/wsl\/lib\/nvidia-smi/);
  assert.match(facts.refusals[0]?.command ?? '', /NVIDIA Windows driver/);
  assert.match(facts.refusals[0]?.command ?? '', /Do NOT install a driver inside the distro/);
});

test('win32: a guest missing zstd is a refusal beside the facts, with apt-get', async () => {
  const runner = new FakeRunner({ platform: 'win32' }, [
    { argv: LIST, stdout: WSL_LIST },
    { argv: GPU, stdout: SMI },
    { argv: PROBE, stdout: `${GUEST_BARE}missing=zstd\nmissing=curl\n` },
    ...STATE_OK,
  ]);
  const facts = await detectHost({}, runner);
  assert.equal(facts.refusals[0]?.code, 'guest_missing_tool');
  assert.equal(facts.refusals[0]?.command, 'sudo apt-get install -y zstd curl');
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

test('darwin: uname, sysctl and the guest probe, natively — unified memory is the card, no WSL table', async () => {
  const runner = new FakeRunner({ platform: 'darwin' }, [
    { argv: ['uname', '-m'], stdout: 'arm64\n' },
    { argv: ['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize'], stdout: 'Apple M1 Ultra\n68719476736\n' },
    { argv: ['bash', '-c', guestProbeScript(undefined)], stdout: 'home=/Users/owen/.crucible\nuser=owen\nfree_kib=900000000\n' },
  ]);
  const facts = await detectHost({}, runner);
  runner.assertDrained();
  assert.equal(facts.wsl, null);
  assert.equal(facts.wslState, null);
  assert.equal(facts.backend, 'mlx-darwin');
  assert.deepEqual(facts.gpu, { vendor: 'apple', name: 'Apple M1 Ultra', vramBytes: 68719476736 });
  assert.equal(facts.guest?.freeBytes, 900000000 * 1024);
  assert.deepEqual(facts.refusals, []);
});

test('darwin: an Intel Mac is not_apple_silicon', async () => {
  const runner = new FakeRunner({ platform: 'darwin' }, [
    { argv: ['uname', '-m'], stdout: 'x86_64\n' },
    { argv: () => true, stdout: 'home=/Users/owen/.crucible\nuser=owen\nfree_kib=1\n' },
  ]);
  const facts = await detectHost({}, runner);
  assert.equal(facts.gpu, null);
  assert.equal(facts.refusals[0]?.code, 'not_apple_silicon');
});

test('darwin: sysctl failing is not_apple_silicon with its words', async () => {
  const runner = new FakeRunner({ platform: 'darwin' }, [
    { argv: ['uname', '-m'], stdout: 'arm64\n' },
    { argv: () => true, code: 1, stderr: 'sysctl: unknown oid' },
    { argv: () => true, stdout: 'home=/Users/owen/.crucible\nuser=owen\nfree_kib=1\n' },
  ]);
  const facts = await detectHost({}, runner);
  assert.equal(facts.gpu, null);
  assert.match(facts.refusals[0]?.message ?? '', /unknown oid/);
});

test('linux: nvidia-smi and the guest probe, natively; a failed spawn is host_unresponsive', async () => {
  const runner = new FakeRunner({ platform: 'linux' }, [
    { argv: ['bash', '-c', NVIDIA_SMI_SCRIPT], stdout: SMI },
    { argv: ['bash', '-c', guestProbeScript(undefined)], stdout: GUEST_INSTALLED },
  ]);
  const facts = await detectHost({}, runner);
  assert.equal(facts.gpu?.name, 'NVIDIA GeForce RTX 3090 Ti');
  assert.equal(facts.server?.crucible, CRUCIBLE_BIN);
  assert.equal(facts.backend, 'cuda-linux');

  const dead = new FakeRunner({ platform: 'linux' }, [{ argv: () => true, failure: 'spawn bash ENOENT' }]);
  const r = await refusal(detectHost({}, dead));
  assert.equal(r.code, 'host_unresponsive');
});

test('an unsupported platform is refused by name', async () => {
  const r = await refusal(detectHost({}, new FakeRunner({ platform: 'freebsd' }, [])));
  assert.equal(r.code, 'unsupported_platform');
});
