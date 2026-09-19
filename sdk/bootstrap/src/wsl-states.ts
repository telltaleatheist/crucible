/**
 * Every WSL state, detected and answered by name — PHASE14-ENVPACKS.md 4c,
 * as DATA so that `install.ps1` and this package cannot drift (one owner).
 *
 * The shape is deliberate. A state is four things:
 *
 *   probe     what to run, and what its output MEANS
 *   code      the refusal name, so an app can switch on it
 *   sentence  what a person reads, in their words
 *   action    what clears it: a command we run, a command the APP runs
 *             elevated, an instruction only a person can carry out, or a link
 *
 * **Bootstrap never elevates.** `run-elevated` is data: the argv the app passes
 * to `Start-Process -Verb RunAs` ({@link elevatedArgv} spells it), at a moment
 * the app chooses — a UAC prompt raised by a library, from a background probe,
 * is a dialog nobody asked for.
 *
 * The probes are keyed and cached, so the ten rows below cost four commands,
 * and a row only ever reads evidence it declared. {@link detectWslState}
 * returns the FIRST row that matches, which is why the order is the order of
 * the table in the doc: the deepest cause first, so "virtualization is off in
 * the firmware" is never reported as "WSL is not installed".
 */
import type { BootstrapRefusalCode } from './errors.js';
import { CRUCIBLE_DISTRO, WSL_CONF_MARKER } from './distro.js';
import { wheelUrl } from './release.js';
import type { RunResult, Runner } from './runner.js';
import { parseWslList } from './wsl.js';

/** What a state needs done about it. */
export type WslAction =
  /** The app runs this through `Start-Process -Verb RunAs`; see {@link elevatedArgv}. */
  | { kind: 'run-elevated'; argv: readonly string[] }
  /** Ours to run, no elevation involved. */
  | { kind: 'run'; argv: readonly string[] }
  /** Only a person can do this: a firmware setting, a reboot, a decision. */
  | { kind: 'instruct'; text: string }
  /** Something to download or read. */
  | { kind: 'link'; url: string };

/** The commands the table can run. Each is run at most once per detection. */
export type ProbeKey = 'wsl-status' | 'wsl-list' | 'wsl-conf' | 'app-distro-conf' | 'guest-network' | 'guest-disk' | 'guest-root';

export interface WslStateDef {
  code: BootstrapRefusalCode | 'wsl_ready';
  /** Which command answers this row. */
  probe: ProbeKey;
  /**
   * CAN THE TRAY CARRY A MACHINE PAST THIS STATE WITHOUT A PERSON?
   * (PHASE19-AUTOMATIC-WSL.md 1 and 2.1.)
   *
   * True for every row whose action is something we run — `run`,
   * `run-elevated`, and the reboot a `run-elevated` leads to. False for
   * `instruct` and `link`, which are a firmware setting, a VPN, a disk or a
   * distro somebody else owns: a person changes something first, and the tray
   * writes `cannot` and stops.
   *
   * IT IS A FIELD AND NOT A FUNCTION OF `action.kind`, because `wsl_ready` is
   * the one row where the two disagree: its action is `instruct` with the text
   * "Nothing to do.", and a machine that is already there is the most
   * automatic state of all. So the partition is DATA — every row states it —
   * and `gen-install-scripts.ts` ASSERTS the derivation for every other row,
   * which is what stops this from becoming a second opinion about a row's
   * action.
   */
  automatic: boolean;
  /**
   * Whether this row applies to this caller at all. A disabled row's probe is
   * NOT run — which is the difference between "the disk is fine" and "nobody
   * asked about the disk", and it is why reading a machine's facts does not
   * reach the internet.
   */
  enabled?: boolean;
  /** Does this evidence mean this state? Pure; the runner is not here. */
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
  /** The app's own WSL distro setting, when it has one. Only used by the "somebody else's distro" row. */
  appDistro?: string;
  /** What the install needs free, in bytes, when that is known. Absent skips the disk row. */
  requiredBytes?: number;
  /**
   * Probe the guest's route to the release. Off by default ON PURPOSE: reading
   * the facts about a machine must not reach the internet, and the only caller
   * that needs this row is one that is about to download gigabytes.
   */
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
      // EVERY PLACE THE INSTALL DOWNLOADS FROM, one cheap HEAD each, and the
      // FIRST one it cannot reach named on stderr (PHASE19 2.12).
      //
      // It used to be one `curl` at the release wheel, which proved one route
      // out of five: a VPN or a proxy that passes GitHub and blocks PyPI,
      // `download.pytorch.org`, the SGLang index or Hugging Face passed the
      // probe and failed minutes later inside pip, with pip's own message.
      //
      // `{indexes}` is filled in by `crucible/host/wslstate.py` from the
      // RECIPE FILES (`jobenv.recipe_index_urls`) plus the interpreter pin and
      // this release's wheel. The list is not spelled here, and it is not
      // spelled in the generated table either: a list written down anywhere
      // drifts the first time a recipe gains an `--extra-index-url`, and the
      // drift is invisible — the probe goes on passing and pip goes on
      // failing. What lives here is the SHAPE of the probe, which is this
      // package's to own.
      return ['wsl.exe', '-d', CRUCIBLE_DISTRO, '--exec', 'bash', '-c', networkProbeScript()];
    case 'guest-disk':
      return ['wsl.exe', '-d', CRUCIBLE_DISTRO, '--exec', 'bash', '-c', 'df -Pk "$HOME" | awk \'NR==2 {print $4}\''];
    case 'guest-root':
      return ['wsl.exe', '-d', CRUCIBLE_DISTRO, '-u', 'root', '--exec', 'id', '-u'];
  }
}

/**
 * The placeholder `crucible/host/wslstate.py` fills with the space-separated
 * list of URLs to probe. It is a `{name}` because that is the seam the
 * generator and `wslstate.render`/`substitute` already use for every other
 * value the CALLER measures (`{release}`, `{app_distro}`, `{required}`).
 */
export const INDEXES_PLACEHOLDER = '{indexes}';

/**
 * One HEAD per index, stopping at the first that will not answer and naming it.
 *
 * `-I` rather than a body: an index page is megabytes and the question is
 * whether the machine can reach it at all. `-L` because a release asset is a
 * redirect to a CDN and the CDN is the thing a proxy blocks. The URL goes to
 * STDERR because `said()` prefers stderr, so the sentence a person reads names
 * the host that failed rather than curl's exit code.
 */
export function networkProbeScript(): string {
  return `set -e; for u in ${INDEXES_PLACEHOLDER}; do `
    + 'curl -fsSL -I -m 20 -o /dev/null "$u" || { echo "$u could not be reached" >&2; exit 1; }; '
    + 'done';
}

/** The error text Windows prints when the hypervisor is not available. */
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

/**
 * The table, in order. `wslStates(inputs)` rather than a constant because three
 * rows are about numbers the caller measures (the release, the app's distro,
 * how much disk the pack needs) and a row that invented them would be guessing.
 */
export function wslStates(inputs: WslStateInputs): WslStateDef[] {
  const required = inputs.requiredBytes ?? 0;
  return [
    {
      // FIRST, because `wsl --status` also fails when the feature is off, and a
      // person told to "enable WSL" on a machine with VT-x disabled will press
      // that button forever.
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
      // A distro the PERSON chose. Same probe, opposite answer: we say what is
      // wrong and ask, because writing /etc/wsl.conf in somebody's Ubuntu and
      // terminating it is a change to their machine.
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
      // The one hand-over that remains: a distro whose root account is disabled
      // (or WSL1, which has no `-u root` at all). Everything else above we can
      // do; this one we cannot.
      //
      // IT USED TO BE CALLED `linger_unreadable` AND USED TO BE ABOUT LINGER.
      // Two things changed on 2026-09-16. The guest's server became a SYSTEM
      // unit, so root is no longer what makes it survive a logout — it is what
      // lets it be INSTALLED: /etc/systemd/system is root's and so is the
      // system manager. And `linger_unreadable` is already a real refusal from
      // linger.ts about actual linger, so the old name was one code meaning two
      // different machine states.
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
      // The one row whose `automatic` is not its action kind: the action is
      // `instruct` "Nothing to do." and a machine already there is carried
      // past it by doing nothing at all.
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
  /** {@link WslStateDef.automatic}, carried through, so a caller branches on the row's own answer. */
  automatic: boolean;
  /** What the probe said, for a log. */
  evidence: string;
}

/**
 * The FIRST state that matches. Probes are run lazily, in row order, and cached
 * — a healthy machine costs `--status`, `-l -v`, one `cat`, one `curl`, one
 * `df` and one `id`, and a machine with no WSL costs exactly one.
 */
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
  // Unreachable while the last row's `means` is `() => true`; a table whose
  // last row stops being total is a bug, not a state.
  throw new Error('wslStates: no row matched, and the table must be total');
}

/**
 * The argv an APP passes to PowerShell to run a `run-elevated` action under
 * UAC. Bootstrap builds it and never runs it: when the consent dialog appears
 * is the app's decision, and a library that raises one from a probe is a
 * library that pops a dialog nobody asked for.
 */
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

/** The marker the systemd rows are really asking about, re-exported so the generator has one import. */
export { WSL_CONF_MARKER };
