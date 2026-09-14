/**
 * Where a command runs: inside a WSL distro, or on this machine.
 *
 * Windows is never a backend (DESIGN.md section 2), so on win32 every command
 * goes through `wsl.exe -d <distro> --exec …` and the distro is REQUIRED — there
 * is no default distro here on purpose, for the reason BookForge's `local.ts`
 * gives: "the default distro" is whatever `wsl --set-default` last said, and a
 * server read from the wrong guest is a wrong server. On darwin and linux the
 * command runs as it is. Anything else is refused by name.
 */
import { BootstrapRefusal } from './errors.js';
import type { RunResult, Runner, StreamOptions } from './runner.js';
import { wslArgv } from './wsl.js';

export type Target = { kind: 'wsl'; distro: string } | { kind: 'native' };

export function resolveTarget(runner: Runner, distro: string | undefined): Target {
  const platform = runner.platform;
  if (platform === 'win32') {
    if (distro === undefined || distro.trim() === '') {
      throw new BootstrapRefusal(
        'no_wsl_distro',
        'the local Crucible on Windows runs inside WSL2, and no WSL distro was named. There is no '
          + 'default distro here on purpose: the server read from the wrong guest is the wrong server. '
          + 'detectHost() lists them.',
      );
    }
    return { kind: 'wsl', distro };
  }
  if (platform === 'darwin' || platform === 'linux') return { kind: 'native' };
  throw new BootstrapRefusal(
    'unsupported_platform',
    `there is no Crucible backend for ${platform}: cuda-linux (Linux, or WSL2 on Windows) and mlx-darwin are the two`,
  );
}

/** Describe a target for a message: `WSL distro "Ubuntu"` or `this machine`. */
export function describeTarget(target: Target): string {
  return target.kind === 'wsl' ? `WSL distro "${target.distro}"` : 'this machine';
}

export interface TargetRunOptions {
  timeoutMs: number;
  env?: Readonly<Record<string, string>>;
}

/**
 * The argv that runs `argv` on the target, with `env` set for it. Inside a
 * distro the environment travels as `env K=V … argv` — a program, not a shell,
 * so nothing is expanded on the way in.
 */
export function commandFor(
  target: Target,
  argv: readonly string[],
  env: Readonly<Record<string, string>> | undefined,
): { argv: string[]; env: Readonly<Record<string, string>> | undefined } {
  if (target.kind === 'wsl') {
    const guest = env === undefined
      ? [...argv]
      : ['env', ...Object.entries(env).map(([key, value]) => `${key}=${value}`), ...argv];
    return { argv: wslArgv(target.distro, guest), env: undefined };
  }
  return { argv: [...argv], env };
}

export function runOn(runner: Runner, target: Target, argv: readonly string[], options: TargetRunOptions): Promise<RunResult> {
  const command = commandFor(target, argv, options.env);
  return runner.run(command.argv, command.env === undefined
    ? { timeoutMs: options.timeoutMs }
    : { timeoutMs: options.timeoutMs, env: command.env });
}

export function streamOn(
  runner: Runner,
  target: Target,
  argv: readonly string[],
  options: TargetRunOptions & Pick<StreamOptions, 'onLine'>,
): Promise<RunResult> {
  const command = commandFor(target, argv, options.env);
  return runner.stream(command.argv, command.env === undefined
    ? { timeoutMs: options.timeoutMs, onLine: options.onLine }
    : { timeoutMs: options.timeoutMs, onLine: options.onLine, env: command.env });
}

/**
 * A probe that could not run at all — spawn error or timeout — is a refusal
 * about the transport, not an answer about the thing probed. Named per target.
 */
export function refuseIfUnrun(target: Target, result: RunResult, what: string): void {
  if (result.failure === null) return;
  if (target.kind === 'wsl') {
    throw new BootstrapRefusal(
      'wsl_read_failed',
      `could not ${what} inside ${describeTarget(target)}: ${result.failure}`,
      { detail: (result.stderr || result.stdout).trim() },
    );
  }
  throw new BootstrapRefusal(
    'host_unresponsive',
    `could not ${what} on this machine: ${result.failure}`,
    { detail: (result.stderr || result.stdout).trim() },
  );
}
