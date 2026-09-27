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

export function describeTarget(target: Target): string {
  return target.kind === 'wsl' ? `WSL distro "${target.distro}"` : 'this machine';
}

export interface TargetRunOptions {
  timeoutMs: number;
  env?: Readonly<Record<string, string>>;
}

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
