/**
 * Linger, on win32, is GRANTED — because there is nobody to hand it to.
 *
 * A systemd USER unit dies with the user's last session unless
 * `loginctl enable-linger <user>` has been granted, and a Crucible that dies
 * when a shell closes is the failure `ensureRunning()` exists to prevent. Until
 * 2026-09-14 this package REPORTED the fact and handed the operator a `sudo`
 * line — the rule being that elevation is the host app's to obtain, never this
 * package's to attempt.
 *
 * **That rule does not apply inside WSL, and the measurement is why.** Verified
 * on Owen's PC, 2026-09-14:
 *
 *     wsl.exe -d Ubuntu -u root --exec id -u                    → 0
 *     wsl.exe -d Ubuntu -u root --exec loginctl show-user telltale -p Linger
 *                                                               → Linger=yes
 *
 * `wsl.exe -u root` grants root **with no password prompt and no sudo** — it is
 * how the guest is entered, not an escalation performed inside it. So there is
 * no elevation to obtain: the thing that was being handed over is something
 * this package can simply do, idempotently, in one command, on the machine the
 * app was asked to install a server on. Handing a person a `sudo` line for a
 * command that needs no sudo is not caution, it is a step somebody skips.
 *
 * **macOS is untouched**, and not by omission: launchd agents have no linger
 * question at all. Native Linux is untouched too — there `sudo` really is
 * elevation, the host app really is the one that can obtain it, and PHASE11's
 * reporting is still the right shape.
 *
 * THE ONE HAND-OVER THAT REMAINS. A guest that will not give root through
 * `-u root` — WSL1, or a distro whose root account is disabled — is a **named
 * refusal carrying the command**, never a guess and never a silent skip. "I
 * could not ask" and "it is off" are different answers, and only one of them
 * means the server will survive a logout.
 */
import { BootstrapRefusal } from './errors.js';
import type { Runner } from './runner.js';
import { describeTarget, runOn, type Target } from './target.js';
import { wslRootArgv } from './wsl.js';

const LINGER_TIMEOUT_MS = 60_000;

/** What `ensureLinger` did, and to whom. Reported as a step, never assumed. */
export interface LingerOutcome {
  /**
   * Whether the guest user lingers now. `true` after a grant as well as after
   * finding it already on; this is the state, not the action.
   */
  linger: boolean;
  /** Did THIS call grant it, or was it already there? */
  granted: boolean;
  /** The guest user it is about. */
  user: string;
  /** What ran to grant it, or `[]` when nothing needed to. */
  argv: readonly string[];
  /** One sentence, for the step's detail. */
  detail: string;
}

/** `Linger=yes` / `Linger=no` from `loginctl show-user -p Linger`, or null. */
export function parseLinger(stdout: string): boolean | null {
  const match = /^\s*Linger\s*=\s*(yes|no)\s*$/im.exec(stdout);
  if (match === null) return null;
  return match[1]?.toLowerCase() === 'yes';
}

/** The `sudo` line a person runs when this package could not (the last resort). */
export function lingerCommand(user: string): string {
  return `wsl.exe -d <distro> -u root --exec loginctl enable-linger ${user}`;
}

/**
 * Make the guest user linger, or say by name why not. **win32/WSL only.**
 *
 * A non-WSL target returns `null` — there is nothing here for it, and a
 * function that invented an outcome for macOS would be reporting a fact about
 * a mechanism that has no such question.
 */
export async function ensureLinger(
  runner: Runner,
  target: Target,
  home: Readonly<Record<string, string>> | undefined,
): Promise<LingerOutcome | null> {
  if (target.kind !== 'wsl') return null;
  const where = describeTarget(target);

  // WHO, asked of the guest rather than assumed from the Windows username.
  // They differ on most installs — `telltale` inside Ubuntu, `tellt` on
  // Windows — and `loginctl enable-linger` with no argument would enable it
  // for whoever is calling, which here is root.
  // Through `runOn`, so `CRUCIBLE_HOME` travels the way every other verb's
  // does — as `env K=V` INSIDE the guest. Passed as a Windows-side env it
  // would reach wsl.exe and not the distro, which is the transport mistake
  // `target.ts` exists to make impossible to repeat.
  const whoami = await runOn(runner, target, ['id', '-un'], {
    timeoutMs: LINGER_TIMEOUT_MS,
    ...(home === undefined ? {} : { env: home }),
  });
  const user = whoami.stdout.trim();
  if (whoami.failure !== null || whoami.code !== 0 || user === '') {
    throw new BootstrapRefusal(
      'linger_unreadable',
      `could not ask ${where} which user it runs as (${whoami.failure ?? `exit ${whoami.code}`}), `
        + 'so this package cannot say whether the service survives a logout',
      { command: lingerCommand('<user>'), detail: (whoami.stderr || whoami.stdout).trim() },
    );
  }

  // The root calls go through `wslRootArgv` directly and carry no environment:
  // `runOn` cannot add `-u root`, and `loginctl` reads nothing from
  // `CRUCIBLE_HOME` — an env var passed here would be noise on the one command
  // whose argv a reader will want to compare against what they would type.
  const read = await runner.run(
    wslRootArgv(target.distro, ['loginctl', 'show-user', user, '-p', 'Linger']),
    { timeoutMs: LINGER_TIMEOUT_MS },
  );
  const current = read.failure === null && read.code === 0 ? parseLinger(read.stdout) : null;
  if (current === null) {
    // THE ONE HAND-OVER THAT REMAINS. A guest with no root through `-u root`
    // (WSL1, a distro with the root account disabled) lands here, and so does
    // a loginctl that answered something else. Refused by name with the
    // command, never guessed at in either direction: "off" would grant
    // something nobody asked for, "on" would promise a server that dies with
    // the next logout.
    const said = (read.stderr.trim() || read.stdout.trim()) || 'no output';
    throw new BootstrapRefusal(
      'linger_unreadable',
      `could not read linger for ${user} in ${where} (${read.failure ?? `exit ${read.code}`}): ${said}. `
        + 'Without it this package cannot say whether the service survives a logout, and it will '
        + 'not guess in either direction.',
      { command: lingerCommand(user), detail: said },
    );
  }
  if (current) {
    return {
      linger: true,
      granted: false,
      user,
      argv: [],
      detail: `${user} already lingers in ${where}; the service survives a logout and starts at boot`,
    };
  }

  const argv = wslRootArgv(target.distro, ['loginctl', 'enable-linger', user]);
  const grant = await runner.run(argv, { timeoutMs: LINGER_TIMEOUT_MS });
  if (grant.failure !== null || grant.code !== 0) {
    const said = (grant.stderr.trim() || grant.stdout.trim()) || 'no output';
    throw new BootstrapRefusal(
      'linger_failed',
      `loginctl enable-linger ${user} ${grant.failure ?? `exited ${grant.code}`} in ${where}: ${said}`,
      { command: lingerCommand(user), detail: said },
    );
  }
  return {
    linger: true,
    granted: true,
    user,
    argv,
    detail: `granted linger to ${user} in ${where}, so the service survives a logout and starts at boot`,
  };
}
