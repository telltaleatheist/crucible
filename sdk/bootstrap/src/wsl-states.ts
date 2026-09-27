import type { BootstrapRefusalCode } from './errors.js';
import { CRUCIBLE_DISTRO, WSL_CONF_MARKER } from './distro.js';
import { wheelUrl } from './release.js';
import type { RunResult, Runner } from './runner.js';
import { parseWslList } from './wsl.js';

/** What a state needs done about it. */
export type WslAction =
  | { kind: 'run-elevated'; argv: readonly string[] }
  | { kind: 'run'; argv: readonly string[] }
  | { kind: 'instruct'; text: string }
  | { kind: 'link'; url: string };

/** The commands the table can run, each at most once per detection. */
export type ProbeKey = 'wsl-status' | 'wsl-list' | 'wsl-conf' | 'app-distro-conf' | 'guest-network' | 'guest-disk' | 'guest-root';

export interface WslStateDef {
  code: BootstrapRefusalCode | 'wsl_ready';
  /** Which command answers this row. */
  probe: ProbeKey;
  /** Whether the tray can carry a machine past this state without a person. */
  automatic: boolean;
  /** Whether this row applies to this caller; a disabled row's probe is not run. */
  enabled?: boolean;
  /** Whether this evidence means this state. */
  means: (result: RunResult, seen: Evidence) => boolean;
  /** What a person reads. */
  sentence: (result: RunResult, seen: Evidence) => string;
  action: (result: RunResult, seen: Evidence) => WslAction;
}

/** Everything the probes have answered so far, for rows that need two facts. */
export interface Evidence {
  results: Partial<Record<ProbeKey, RunResult>>;
  /** The distros `wsl -l -v` listed, once it has been run. */
  distros: { name: string; version: number }[];
}

export interface WslStateInputs {
  /** The release whose WHEEL the network probe fetches, to prove a route. */
  release: string;
  /** The app's own WSL distro setting, when it has one. */
  appDistro?: string;
  /** What the install needs free, in bytes; absent skips the disk row. */
  requiredBytes?: number;
  /** Probe the guest's route to the release; off by default. */
  checkNetwork?: boolean;
}

/** `wsl.exe --status` / `-l -v` / a command inside a distro — the table's four probes. */
export function probeArgv(key: ProbeKey, inputs: WslStateInputs): string[] {
  switch (key) {
    case 'wsl-status':
      return ['wsl.exe', '--status'];
    case 'wsl-list':
      return ['wsl.exe', '-l', '-v'];
    case 'wsl-conf':
      return ['wsl.exe', '-d', CRUCIBLE_DISTRO, '-u', 'root', '--exec', 'bash', '-c', 'test -f /etc/wsl.conf && cat /etc/wsl.conf || true'];
    case 'app-distro-conf':
      return ['wsl.exe', '-d', inputs.appDistro ?? CRUCIBLE_DISTRO, '-u', 'root', '--exec', 'bash', '-c', 'test -f /etc/wsl.conf && cat /etc/wsl.conf || true'];
    case 'guest-network':
      return ['wsl.exe', '-d', CRUCIBLE_DISTRO, '--exec', 'bash', '-c', networkProbeScript()];
    case 'guest-disk':
      return ['wsl.exe', '-d', CRUCIBLE_DISTRO, '--exec', 'bash', '-c', 'df -Pk "$HOME" | awk \'NR==2 {print $4}\''];
    case 'guest-root':
      return ['wsl.exe', '-d', CRUCIBLE_DISTRO, '-u', 'root', '--exec', 'id', '-u'];
  }
}

export const INDEXES_PLACEHOLDER = '{indexes}';

export function networkProbeScript(): string {
  return `set -e; for u in ${INDEXES_PLACEHOLDER}; do `
    + 'curl -fsSL -I -m 20 -o /dev/null "$u" || { echo "$u could not be reached" >&2; exit 1; }; '
    + 'done';
}

const NO_HYPERVISOR = /HCS_E_HYPERV_NOT_INSTALLED|0x80370102|hypervisor|virtual machine platform/i;

const FIRMWARE = 'Virtualization is turned off in this machine\'s firmware. Restart, open the BIOS/UEFI setup '
  + '(usually Del or F2 during boot), and enable Intel VT-x (Intel) or SVM Mode (AMD). Then run Enable WSL again.';

function said(result: RunResult): string {
  return (result.stderr.trim() || result.stdout.trim() || result.failure || `exit ${result.code}`).slice(0, 400);
}

function freeBytes(result: RunResult): number | null {
  const kib = result.stdout.trim().split(/\s+/)[0] ?? '';
  return /^\d+$/.test(kib) ? Number(kib) * 1024 : null;
}

export function gib(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

/** The WSL state table, in detection order. */
export function wslStates(inputs: WslStateInputs): WslStateDef[] {
  const required = inputs.requiredBytes ?? 0;
  return [
    {
      code: 'virtualization_disabled',
      automatic: false,
      probe: 'wsl-status',
      means: (result) => (result.failure !== null || result.code !== 0) && NO_HYPERVISOR.test(`${result.stdout}${result.stderr}${result.failure ?? ''}`),
      sentence: (result) => `Windows cannot start a virtual machine: ${said(result)}`,
      action: () => ({ kind: 'instruct', text: FIRMWARE }),
    },
    {
      code: 'wsl_missing',
      automatic: true,
      probe: 'wsl-status',
      means: (result) => result.failure !== null || result.code !== 0,
      sentence: (result) => result.failure !== null
        ? 'This machine has no wsl.exe: the Windows Subsystem for Linux has never been enabled.'
        : `wsl.exe is there and not usable: ${said(result)}`,
      action: () => ({ kind: 'run-elevated', argv: ['wsl.exe', '--install', '--no-distribution'] }),
    },
    {
      code: 'wsl1_only',
      automatic: true,
      probe: 'wsl-status',
      means: (result) => /Default Version:\s*1\b/i.test(result.stdout),
      sentence: () => 'WSL is set to version 1, which has no GPU. Crucible needs WSL2.',
      action: () => ({ kind: 'run', argv: ['wsl.exe', '--set-default-version', '2'] }),
    },
    {
      code: 'no_crucible_distro',
      automatic: true,
      probe: 'wsl-list',
      means: (_result, seen) => !seen.distros.some((entry) => entry.name === CRUCIBLE_DISTRO),
      sentence: () => `Crucible has no Linux of its own on this machine yet (the "${CRUCIBLE_DISTRO}" distribution). `
        + 'Installing one takes a download and touches nothing you already have.',
      action: () => ({ kind: 'run', argv: ['wsl.exe', '--import', CRUCIBLE_DISTRO, '<install dir>', '<rootfs>', '--version', '2'] }),
    },
    {
      code: 'distro_not_systemd',
      automatic: true,
      probe: 'wsl-conf',
      means: (result) => !/systemd\s*=\s*true/i.test(result.stdout),
      sentence: () => `The "${CRUCIBLE_DISTRO}" distribution is not running systemd, so the Crucible service cannot start in it. `
        + 'It is ours: this is repaired without asking.',
      action: () => ({ kind: 'run', argv: ['wsl.exe', '--terminate', CRUCIBLE_DISTRO] }),
    },
    {
      code: 'foreign_distro_not_systemd',
      automatic: false,
      probe: 'app-distro-conf',
      enabled: inputs.appDistro !== undefined && inputs.appDistro !== CRUCIBLE_DISTRO,
      means: (result, seen) => seen.distros.some((entry) => entry.name === inputs.appDistro)
        && !/systemd\s*=\s*true/i.test(result.stdout),
      sentence: () => `"${inputs.appDistro}" is the distribution this app was told to use, and it is not running systemd. `
        + 'Crucible will not write to a distribution you chose without being asked.',
      action: () => ({
        kind: 'instruct',
        text: `Add [boot] systemd=true to /etc/wsl.conf in "${inputs.appDistro}" and run `
          + `wsl --terminate ${inputs.appDistro}, or let Crucible import its own distribution instead.`,
      }),
    },
    {
      code: 'guest_no_network',
      automatic: false,
      probe: 'guest-network',
      enabled: inputs.checkNetwork === true,
      means: (result) => result.failure !== null || result.code !== 0,
      sentence: (result) => `The "${CRUCIBLE_DISTRO}" distribution cannot reach one of the places this install `
        + `downloads from: ${said(result)}. A VPN or a proxy on this machine usually explains it; `
        + 'there is nothing to install until it can.',
      action: () => ({ kind: 'link', url: wheelUrl(inputs.release) }),
    },
    {
      code: 'guest_no_disk',
      automatic: false,
      probe: 'guest-disk',
      enabled: required > 0,
      means: (result) => {
        const free = freeBytes(result);
        return free !== null && free < required;
      },
      sentence: (result) => {
        const free = freeBytes(result);
        return `This install needs ${gib(required)} free and the "${CRUCIBLE_DISTRO}" distribution has `
          + `${free === null ? 'an unreadable amount' : gib(free)}. Nothing has been downloaded.`;
      },
      action: () => ({ kind: 'instruct', text: 'Free some space on the drive WSL keeps its disk on, then try again.' }),
    },
    {
      code: 'guest_root_unreachable',
      automatic: false,
      probe: 'guest-root',
      means: (result) => result.failure !== null || result.code !== 0 || result.stdout.trim() !== '0',
      sentence: (result) => `The "${CRUCIBLE_DISTRO}" distribution will not let Crucible in as root (${said(result)}). `
        + 'The server is installed as a system service, which needs root to write, so there is nothing to install until it does.',
      action: () => ({
        kind: 'instruct',
        text: `Enable the root account in "${CRUCIBLE_DISTRO}", or let Crucible import its own distribution, which grants root through wsl.exe with no password.`,
      }),
    },
    {
      code: 'wsl_ready',
      automatic: true,
      probe: 'wsl-list',
      means: () => true,
      sentence: () => `WSL2 is ready and the "${CRUCIBLE_DISTRO}" distribution is there.`,
      action: () => ({ kind: 'instruct', text: 'Nothing to do.' }),
    },
  ];
}

/** A state, as detected: the row, the evidence that matched it, and the words. */
export interface WslState {
  code: WslStateDef['code'];
  sentence: string;
  action: WslAction;
  /** {@link WslStateDef.automatic}, carried through. */
  automatic: boolean;
  /** What the probe said, for a log. */
  evidence: string;
}

/** The first state that matches, with probes run lazily and cached. */
export async function detectWslState(
  inputs: WslStateInputs,
  runner: Runner,
  timeoutMs = 60_000,
): Promise<WslState> {
  const seen: Evidence = { results: {}, distros: [] };
  const ask = async (key: ProbeKey): Promise<RunResult> => {
    const cached = seen.results[key];
    if (cached !== undefined) return cached;
    const result = await runner.run(probeArgv(key, inputs), { timeoutMs });
    seen.results[key] = result;
    if (key === 'wsl-list' && result.failure === null && result.code === 0) {
      seen.distros = parseWslList(result.stdout).distros.map((entry) => ({ name: entry.name, version: entry.version }));
    }
    return result;
  };
  for (const row of wslStates(inputs)) {
    if (row.enabled === false) continue;
    const result = await ask(row.probe);
    if (!row.means(result, seen)) continue;
    return {
      code: row.code,
      sentence: row.sentence(result, seen),
      action: row.action(result, seen),
      automatic: row.automatic,
      evidence: said(result),
    };
  }
  throw new Error('wslStates: no row matched, and the table must be total');
}

/** The argv an app passes to PowerShell to run a `run-elevated` action under UAC. */
export function elevatedArgv(action: WslAction): string[] {
  if (action.kind !== 'run-elevated') throw new Error(`elevatedArgv: ${action.kind} is not an elevated action`);
  const [program, ...rest] = action.argv;
  if (program === undefined) throw new Error('elevatedArgv: nothing to run');
  const quoted = rest.map((argument) => `'${argument.replace(/'/g, "''")}'`).join(',');
  const list = quoted === '' ? '' : ` -ArgumentList ${quoted}`;
  return [
    'powershell.exe',
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-Command',
    `Start-Process -Verb RunAs -Wait -FilePath '${program.replace(/'/g, "''")}'${list}`,
  ];
}

export { WSL_CONF_MARKER };
