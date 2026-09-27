import { BootstrapRefusal } from './errors.js';
import { CRUCIBLE_DISTRO, listDistros } from './distro.js';
import { backendFor, type ServerBackend } from './release.js';
import { probeGuest, type GuestFacts, type InstalledRuntime } from './runtime.js';
import { processRunner, type Runner } from './runner.js';
import { describeTarget, refuseIfUnrun, resolveTarget, runOn, type Target } from './target.js';
import { BOOTSTRAP_VERSION } from './version.js';
import { detectWslState, type WslState } from './wsl-states.js';
import type { WslDistro } from './wsl.js';

/** Where WSL2 puts the driver's nvidia-smi, after PATH (`crucible/backend.py`). */
export const WSL_NVIDIA_SMI = '/usr/lib/wsl/lib/nvidia-smi';

const PROBE_TIMEOUT_MS = 60_000;

export interface WslFacts {
  present: true;
  distros: WslDistro[];
  /** The distro wsl.exe marks `*`, or null when none is. */
  default: string | null;
  /** Which distro the gpu / pack facts were read through. */
  probed: string;
}

export interface GpuFacts {
  vendor: 'nvidia' | 'apple';
  name: string;
  /** Total memory; unified memory on Apple Silicon. */
  vramBytes: number;
}

export interface HostFacts {
  platform: NodeJS.Platform;
  /** The backend this host is: `cuda-linux` or `mlx-darwin`. */
  backend: ServerBackend;
  /** win32 only; null elsewhere. */
  wsl: WslFacts | null;
  /** win32 only: the first WSL state table row that matches this machine. */
  wslState: WslState | null;
  gpu: GpuFacts | null;
  /** `<CRUCIBLE_HOME>`, the guest user, free disk, missing tools — one probe. */
  guest: GuestFacts | null;
  /** The server runtime already installed, or null. */
  server: InstalledRuntime | null;
  /** One named refusal per null above, with the command that clears it. */
  refusals: BootstrapRefusal[];
}

export interface DetectOptions {
  /** win32: the app's own WSL distro setting. */
  distro?: string;
  /** `CRUCIBLE_HOME`, as the target spells it. */
  home?: string;
  /** The release the 4c network probe checks reachability of. */
  release?: string;
}

/** Which distro `detectHost` reads through. */
export function pickProbeDistro(listed: { distros: WslDistro[]; default: string | null }, named: string | undefined): string {
  if (listed.distros.some((entry) => entry.name === CRUCIBLE_DISTRO)) return CRUCIBLE_DISTRO;
  if (named !== undefined && named.trim() !== '') return named;
  if (listed.default !== null) return listed.default;
  throw new BootstrapRefusal(
    'no_wsl_distro',
    `WSL lists ${listed.distros.map((entry) => entry.name).join(', ')} and marks none as default, and there is no `
      + `"${CRUCIBLE_DISTRO}" distro; name one with {distro}.`,
  );
}

export async function detectHost(options: DetectOptions = {}, runner: Runner = processRunner()): Promise<HostFacts> {
  const platform = runner.platform;
  const backend = backendFor(platform);
  const refusals: BootstrapRefusal[] = [];

  let wsl: WslFacts | null = null;
  let wslState: WslState | null = null;
  let target: Target;
  if (platform === 'win32') {
    const listed = await listDistros(runner);
    const probed = pickProbeDistro(listed, options.distro);
    wsl = { present: true, distros: listed.distros, default: listed.default, probed };
    target = { kind: 'wsl', distro: probed };
  } else {
    target = resolveTarget(runner, undefined);
  }

  const gpu = platform === 'darwin' ? await appleGpu(runner, target, refusals) : await nvidiaGpu(runner, target, refusals);

  let guest: GuestFacts | null = null;
  try {
    guest = await probeGuest(runner, target, options.home);
  } catch (err) {
    if (!(err instanceof BootstrapRefusal)) throw err;
    refusals.push(err);
  }
  if (guest !== null && guest.missingTools.length > 0) {
    refusals.push(new BootstrapRefusal(
      'guest_missing_tool',
      `${describeTarget(target)} has no ${guest.missingTools.join(', ')}; a pack is fetched with curl and unpacked with tar --zstd.`,
      { command: platform === 'darwin' ? `brew install ${guest.missingTools.join(' ')}` : `sudo apt-get install -y ${guest.missingTools.join(' ')}` },
    ));
  }

  if (platform === 'win32') {
    wslState = await detectWslState(
      {
        release: options.release ?? BOOTSTRAP_VERSION,
        ...(options.distro === undefined ? {} : { appDistro: options.distro }),
      },
      runner,
      PROBE_TIMEOUT_MS,
    );
  }

  return { platform, backend, wsl, wslState, gpu, guest, server: guest?.server ?? null, refusals };
}

const NO_NVIDIA_SMI_EXIT = 3;

export const NVIDIA_SMI_SCRIPT =
  `for s in "$(command -v nvidia-smi)" ${WSL_NVIDIA_SMI}; do `
  + 'if [ -n "$s" ] && test -x "$s"; then exec "$s" --query-gpu=name,memory.total --format=csv,noheader,nounits; fi; '
  + `done; exit ${NO_NVIDIA_SMI_EXIT}`;

async function nvidiaGpu(runner: Runner, target: Target, refusals: BootstrapRefusal[]): Promise<GpuFacts | null> {
  const result = await runOn(runner, target, ['bash', '-c', NVIDIA_SMI_SCRIPT], { timeoutMs: PROBE_TIMEOUT_MS });
  refuseIfUnrun(target, result, 'run nvidia-smi');
  const where = describeTarget(target);
  const driverCommand = target.kind === 'wsl'
    ? 'Install the NVIDIA Windows driver (GeForce/Studio, R470 or newer) from https://www.nvidia.com/drivers — WSL2 receives CUDA through it. Do NOT install a driver inside the distro.'
    : 'Install the NVIDIA driver for this Linux host (Ubuntu: sudo ubuntu-drivers install), then reboot.';
  if (result.code === NO_NVIDIA_SMI_EXIT) {
    refusals.push(new BootstrapRefusal(
      'no_nvidia_driver',
      `no nvidia-smi inside ${where}: looked on PATH and at ${WSL_NVIDIA_SMI}.`,
      { command: driverCommand },
    ));
    return null;
  }
  if (result.code !== 0) {
    const saidIt = (result.stderr.trim() || result.stdout.trim()) || 'no output';
    refusals.push(new BootstrapRefusal(
      'no_nvidia_driver',
      `nvidia-smi inside ${where} exited ${result.code}: ${saidIt}`,
      { command: driverCommand, detail: saidIt },
    ));
    return null;
  }
  const parsed = parseNvidiaSmi(result.stdout);
  if (parsed === null) {
    refusals.push(new BootstrapRefusal(
      'no_nvidia_driver',
      `nvidia-smi inside ${where} answered something unparseable: ${JSON.stringify(result.stdout.trim().slice(0, 120))}`,
      { command: driverCommand, detail: result.stdout.trim() },
    ));
    return null;
  }
  return parsed;
}

/** `name, MiB` — the first GPU, exactly as `crucible/backend.py` reads it. */
export function parseNvidiaSmi(stdout: string): GpuFacts | null {
  const first = stdout.split(/\r?\n/).map((line) => line.trim()).find((line) => line.length > 0);
  if (first === undefined) return null;
  const parts = first.split(',').map((part) => part.trim());
  if (parts.length !== 2) return null;
  const [name, mibText] = parts as [string, string];
  if (!/^\d+$/.test(mibText)) return null;
  return { vendor: 'nvidia', name, vramBytes: Number(mibText) * 1024 * 1024 };
}

async function appleGpu(runner: Runner, target: Target, refusals: BootstrapRefusal[]): Promise<GpuFacts | null> {
  const arch = await runOn(runner, target, ['uname', '-m'], { timeoutMs: PROBE_TIMEOUT_MS });
  refuseIfUnrun(target, arch, 'run uname');
  if (arch.stdout.trim() !== 'arm64') {
    refusals.push(new BootstrapRefusal(
      'not_apple_silicon',
      `this Mac is ${arch.stdout.trim() || 'of unknown architecture'}; mlx-darwin needs Apple Silicon (arm64). There is no backend for an Intel Mac.`,
    ));
    return null;
  }
  const sysctl = await runOn(runner, target, ['sysctl', '-n', 'machdep.cpu.brand_string', 'hw.memsize'], { timeoutMs: PROBE_TIMEOUT_MS });
  refuseIfUnrun(target, sysctl, 'run sysctl');
  const lines = sysctl.stdout.split(/\r?\n/).map((line) => line.trim()).filter((line) => line.length > 0);
  const chip = lines[0];
  const memsize = lines[1];
  if (sysctl.code !== 0 || chip === undefined || memsize === undefined || !/^\d+$/.test(memsize)) {
    refusals.push(new BootstrapRefusal(
      'not_apple_silicon',
      `sysctl would not name the chip and memory (exit ${sysctl.code}): ${(sysctl.stderr || sysctl.stdout).trim()}`,
    ));
    return null;
  }
  return { vendor: 'apple', name: chip, vramBytes: Number(memsize) };
}

/** The `crucible` console script beside an interpreter. */
export function consoleScriptBeside(pythonPath: string): string {
  const slash = pythonPath.lastIndexOf('/');
  if (slash < 0) throw new Error(`not an absolute interpreter path: ${pythonPath}`);
  return `${pythonPath.slice(0, slash)}/crucible`;
}
