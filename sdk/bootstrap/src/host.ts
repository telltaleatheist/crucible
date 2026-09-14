/**
 * `detectHost()` — what this machine can run a Crucible with, measured.
 *
 * On win32 the facts are the GUEST's, read through `wsl.exe`: the card the
 * guest's `nvidia-smi` sees, the conda and interpreter inside the distro. On
 * darwin and linux they are the machine's own. Every null carries a named
 * refusal in `refusals`, with the command the host must run; nothing is
 * inferred from a null.
 *
 * The server interpreter is ONE thing: `<conda root>/envs/crucible/bin/python`,
 * a Python 3.11. `crucible install <type>` builds every job type's venv from
 * it, so this package's job is only to find that one interpreter — conda found
 * by `test -x` at `~/anaconda3 | ~/miniconda3 | ~/miniforge3`, in order, never
 * `which`, so the conda whose `envs/` the interpreter lands in is the one used
 * (docs/FROM-FOUNDRY-WSL-VLLM.md section 2).
 */
import { BootstrapRefusal } from './errors.js';
import { processRunner, type RunResult, type Runner } from './runner.js';
import { describeTarget, refuseIfUnrun, resolveTarget, runOn, type Target } from './target.js';
import { parseWslList, shellQuote, wslListArgv, type WslDistro } from './wsl.js';

/** Where conda is looked for, in order. Overridable per call, never guessed past. */
export const DEFAULT_CONDA_ROOTS: readonly string[] = ['~/anaconda3', '~/miniconda3', '~/miniforge3'];
/** The conda env the server interpreter lives in. `crucible/README.md`'s install recipe names it. */
export const SERVER_ENV_NAME = 'crucible';
/** The interpreter the server needs, major.minor. */
export const SERVER_PYTHON = '3.11';
/** Where WSL2 puts the driver's nvidia-smi, after PATH (`crucible/backend.py`). */
export const WSL_NVIDIA_SMI = '/usr/lib/wsl/lib/nvidia-smi';

const LIST_TIMEOUT_MS = 15_000;
const PROBE_TIMEOUT_MS = 60_000;

export interface WslFacts {
  present: true;
  distros: WslDistro[];
  /** The distro wsl.exe marks `*`, or null when none is. */
  default: string | null;
  /** Which distro the gpu / python facts were read through. */
  probed: string;
}

export interface GpuFacts {
  vendor: 'nvidia' | 'apple';
  name: string;
  /** Total memory. On Apple Silicon that is the machine's unified memory. */
  vramBytes: number;
}

export interface PythonFacts {
  /** The server interpreter, as the target spells it. */
  path: string;
  /** `3.11.9` — whatever `--version` said, without the word Python. */
  version: string;
}

export interface HostFacts {
  platform: NodeJS.Platform;
  /** win32 only; null elsewhere. */
  wsl: WslFacts | null;
  gpu: GpuFacts | null;
  python: PythonFacts | null;
  /** The conda root the interpreter was looked for under, or null when none was found. */
  conda: { root: string } | null;
  /** One named refusal per null above, with the command that clears it. Empty when nothing is missing. */
  refusals: BootstrapRefusal[];
}

export interface DetectOptions {
  /** win32: which distro to read the facts through. Defaults to the one wsl.exe marks default, and says so in `wsl.probed`. */
  distro?: string;
  /** Where to look for conda, in order. Defaults to {@link DEFAULT_CONDA_ROOTS}. */
  condaRoots?: readonly string[];
}

export async function detectHost(options: DetectOptions = {}, runner: Runner = processRunner()): Promise<HostFacts> {
  const platform = runner.platform;
  const roots = options.condaRoots ?? DEFAULT_CONDA_ROOTS;
  const refusals: BootstrapRefusal[] = [];

  let wsl: WslFacts | null = null;
  let target: Target;
  if (platform === 'win32') {
    const listed = await listDistros(runner);
    const probed = options.distro ?? listed.default;
    if (probed === null) {
      throw new BootstrapRefusal(
        'no_wsl_distro',
        `WSL lists ${listed.distros.map((d) => d.name).join(', ')} and marks none as default; name one with {distro}.`,
      );
    }
    wsl = { present: true, distros: listed.distros, default: listed.default, probed };
    target = { kind: 'wsl', distro: probed };
  } else {
    target = resolveTarget(runner, undefined);
  }

  const gpu = platform === 'darwin' ? await appleGpu(runner, target, refusals) : await nvidiaGpu(runner, target, refusals);
  const interpreter = await probeInterpreter(runner, target, roots);
  for (const refusal of interpreter.refusals) refusals.push(refusal);

  return { platform, wsl, gpu, python: interpreter.python, conda: interpreter.conda, refusals };
}

// ---------------------------------------------------------------------- WSL

const WSL_INSTALL = 'wsl --install -d Ubuntu';
const WSL_INSTALL_NOTE = 'Run it in an elevated PowerShell, then REBOOT Windows; the first launch of the distro asks for a username and password.';

async function listDistros(runner: Runner): Promise<{ distros: WslDistro[]; default: string | null }> {
  const result = await runner.run(wslListArgv(), { timeoutMs: LIST_TIMEOUT_MS });
  if (result.failure !== null) {
    if (/ENOENT/.test(result.failure)) {
      throw new BootstrapRefusal('wsl_missing', `wsl.exe is not on this machine: ${result.failure}. ${WSL_INSTALL_NOTE}`, { command: WSL_INSTALL });
    }
    throw new BootstrapRefusal('wsl_read_failed', `wsl.exe -l -v did not answer: ${result.failure}`, { detail: (result.stderr || result.stdout).trim() });
  }
  if (result.code !== 0) {
    const said = (result.stderr.trim() || result.stdout.trim()) || `exit ${result.code}`;
    throw new BootstrapRefusal('wsl_missing', `WSL is not usable here (wsl.exe -l -v said: ${said}). ${WSL_INSTALL_NOTE}`, { command: WSL_INSTALL, detail: said });
  }
  const parsed = parseWslList(result.stdout);
  if (parsed.distros.length === 0) {
    throw new BootstrapRefusal('wsl_missing', `WSL is installed but lists no distribution. ${WSL_INSTALL_NOTE}`, { command: WSL_INSTALL, detail: result.stdout.trim() });
  }
  return parsed;
}

// ---------------------------------------------------------------------- GPU

/** Exit 3 = no nvidia-smi anywhere it is looked for. Anything else is nvidia-smi's own exit. */
export const NVIDIA_SMI_SCRIPT =
  `for s in "$(command -v nvidia-smi)" ${WSL_NVIDIA_SMI}; do `
  + 'if [ -n "$s" ] && test -x "$s"; then exec "$s" --query-gpu=name,memory.total --format=csv,noheader,nounits; fi; '
  + 'done; exit 3';

async function nvidiaGpu(runner: Runner, target: Target, refusals: BootstrapRefusal[]): Promise<GpuFacts | null> {
  const result = await runOn(runner, target, ['bash', '-c', NVIDIA_SMI_SCRIPT], { timeoutMs: PROBE_TIMEOUT_MS });
  refuseIfUnrun(target, result, 'run nvidia-smi');
  const where = describeTarget(target);
  const driverCommand = target.kind === 'wsl'
    ? 'Install the NVIDIA Windows driver (GeForce/Studio, R470 or newer) from https://www.nvidia.com/drivers — WSL2 receives CUDA through it. Do NOT install a driver inside the distro.'
    : 'Install the NVIDIA driver for this Linux host (Ubuntu: sudo ubuntu-drivers install), then reboot.';
  if (result.code === 3) {
    refusals.push(new BootstrapRefusal(
      'no_nvidia_driver',
      `no nvidia-smi inside ${where}: looked on PATH and at ${WSL_NVIDIA_SMI}.`,
      { command: driverCommand },
    ));
    return null;
  }
  if (result.code !== 0) {
    const said = (result.stderr.trim() || result.stdout.trim()) || 'no output';
    refusals.push(new BootstrapRefusal(
      'no_nvidia_driver',
      `nvidia-smi inside ${where} exited ${result.code}: ${said}`,
      { command: driverCommand, detail: said },
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

// --------------------------------------------------------------- interpreter

/**
 * One bash script, exit 0 always, printing `key=value` lines: `home`, then
 * `conda` if a root passed `test -x`, then `python` and `version` if the server
 * env's interpreter is there. No backslash, no host-side `$` — with `--exec`
 * the guest's bash is the first thing that reads it.
 */
export function interpreterScript(roots: readonly string[]): string {
  const spelled = roots.map((root) => {
    if (root.startsWith('~/')) return `"$HOME/${root.slice(2)}"`;
    if (root.startsWith('/')) return shellQuote(root);
    throw new Error(`condaRoots entries must start with ~/ or /: ${JSON.stringify(root)}`);
  });
  return 'conda=""; '
    + `for c in ${spelled.join(' ')}; do if test -x "$c/bin/conda"; then conda="$c"; break; fi; done; `
    + 'echo "home=$HOME"; '
    + 'if [ -n "$conda" ]; then echo "conda=$conda"; '
    + `p="$conda/envs/${SERVER_ENV_NAME}/bin/python"; `
    + 'if test -x "$p"; then echo "python=$p"; echo "version=$("$p" --version 2>&1)"; fi; fi; exit 0';
}

export interface InterpreterFacts {
  python: PythonFacts | null;
  conda: { root: string } | null;
  refusals: BootstrapRefusal[];
}

function parseKeyValues(result: RunResult): Map<string, string> {
  const values = new Map<string, string>();
  for (const raw of result.stdout.split(/\r?\n/)) {
    const eq = raw.indexOf('=');
    if (eq > 0) values.set(raw.slice(0, eq).trim(), raw.slice(eq + 1).trim());
  }
  return values;
}

function miniforgeCommand(platform: NodeJS.Platform): string {
  const asset = platform === 'darwin' ? 'Miniforge3-MacOSX-arm64.sh' : 'Miniforge3-Linux-x86_64.sh';
  return `curl -L -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/${asset} && bash Miniforge3.sh -b -p "$HOME/miniforge3"`;
}

/** Find the server interpreter, or say by name what is missing and how to get it. */
export async function probeInterpreter(runner: Runner, target: Target, roots: readonly string[]): Promise<InterpreterFacts> {
  const result = await runOn(runner, target, ['bash', '-c', interpreterScript(roots)], { timeoutMs: PROBE_TIMEOUT_MS });
  refuseIfUnrun(target, result, 'look for the server interpreter');
  if (result.code !== 0) {
    refuseIfUnrun(target, { ...result, failure: `bash exited ${result.code}` }, 'look for the server interpreter');
  }
  const values = parseKeyValues(result);
  const where = describeTarget(target);
  const condaRoot = values.get('conda');
  if (condaRoot === undefined) {
    const inside = target.kind === 'wsl' ? ` (run it inside ${where})` : '';
    return {
      python: null,
      conda: null,
      refusals: [new BootstrapRefusal(
        'no_conda',
        `no conda inside ${where}: none of ${roots.join(', ')} has bin/conda. Install miniforge${inside}, then create the server env: conda create -n ${SERVER_ENV_NAME} python=${SERVER_PYTHON} -y`,
        { command: miniforgeCommand(runner.platform) },
      )],
    };
  }
  const conda = { root: condaRoot };
  const create = `${condaRoot}/bin/conda create -n ${SERVER_ENV_NAME} python=${SERVER_PYTHON} -y`;
  const pythonPath = values.get('python');
  if (pythonPath === undefined) {
    return {
      python: null,
      conda,
      refusals: [new BootstrapRefusal(
        'no_python',
        `conda at ${condaRoot} inside ${where} has no "${SERVER_ENV_NAME}" env (${condaRoot}/envs/${SERVER_ENV_NAME}/bin/python is not there). The server needs a Python ${SERVER_PYTHON} of its own.`,
        { command: create },
      )],
    };
  }
  const versionLine = values.get('version') ?? '';
  const match = /Python (\d+\.\d+\.\d+)/.exec(versionLine);
  if (match === null || match[1] === undefined) {
    return {
      python: null,
      conda,
      refusals: [new BootstrapRefusal(
        'no_python',
        `${pythonPath} inside ${where} would not report a version (said: ${JSON.stringify(versionLine)}). Recreate the env.`,
        { command: `${condaRoot}/bin/conda remove -n ${SERVER_ENV_NAME} --all -y && ${create}` },
      )],
    };
  }
  const version = match[1];
  const python = { path: pythonPath, version };
  if (!version.startsWith(`${SERVER_PYTHON}.`)) {
    return {
      python,
      conda,
      refusals: [new BootstrapRefusal(
        'no_python',
        `${pythonPath} inside ${where} is Python ${version}; the server needs ${SERVER_PYTHON}. Recreate the env.`,
        { command: `${condaRoot}/bin/conda remove -n ${SERVER_ENV_NAME} --all -y && ${create}` },
      )],
    };
  }
  return { python, conda, refusals: [] };
}

/** The `crucible` console script beside an interpreter. Guest-spelled: always `/`. */
export function consoleScriptBeside(pythonPath: string): string {
  const slash = pythonPath.lastIndexOf('/');
  if (slash < 0) throw new Error(`not an absolute interpreter path: ${pythonPath}`);
  return `${pythonPath.slice(0, slash)}/crucible`;
}
