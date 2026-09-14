/**
 * Every refusal this package surfaces has a name, and most carry a command.
 *
 * The rule (PHASE5-APPS.md section 6, CLAUDE.md "no fallbacks"): a missing
 * prerequisite is a NAMED refusal carrying the exact command the HOST must run.
 * Elevation, a reboot, a sudo password — those are the host app's to obtain,
 * never this package's to attempt. So a refusal here is never "something went
 * wrong"; it is one of the codes below, with `command` set whenever there is a
 * thing a person can type to make it go away.
 */

export type BootstrapRefusalCode =
  /** Not win32, darwin or linux. There is no backend for it (DESIGN.md section 2). */
  | 'unsupported_platform'
  /** win32 with no usable WSL: no `wsl.exe`, or it lists no distribution. */
  | 'wsl_missing'
  /** win32, and the caller named no distro. There is no default here on purpose. */
  | 'no_wsl_distro'
  /** `wsl.exe` ran and failed, timed out, or answered something unparseable. */
  | 'wsl_read_failed'
  /** A probe on this machine (not through WSL) could not be spawned or timed out. */
  | 'host_unresponsive'
  /** cuda-linux host (or guest) with no `nvidia-smi` on PATH or at the WSL location. */
  | 'no_nvidia_driver'
  /** darwin on something other than arm64. `mlx-darwin` is Apple Silicon only. */
  | 'not_apple_silicon'
  /** No conda at any of the roots searched (`test -x`, in order). */
  | 'no_conda'
  /** Conda is there, but no `envs/crucible/bin/python` under it, or not a 3.11. */
  | 'no_python'
  /** No `config.toml` where the local server would keep one. A state, not a bug. */
  | 'no_local_config'
  /** The config exists and is not TOML, or could not be read. */
  | 'config_unreadable'
  /** The config parses but lacks a key the server itself requires. */
  | 'config_missing_key'
  /** A Windows path on a UNC share or a mapped drive: WSL2 has no mount for it. */
  | 'network_path'
  /** A path that is not an absolute `X:\...` Windows path, so it has no `/mnt` spelling. */
  | 'not_a_windows_path'
  /** The wheel the host named is not on disk. */
  | 'wheel_missing'
  /** A job type this build cannot install, or one spelled without what it needs. */
  | 'bad_job_type'
  /** One of `install()`'s steps exited non-zero. See {@link BootstrapStepFailed}. */
  | 'step_failed'
  /** `crucible service status` says no definition is installed on this host. */
  | 'service_not_installed'
  /** The service is installed and would not come up, or could not be asked. */
  | 'service_failed'
  /** The server's URL did not answer at all. */
  | 'unreachable'
  /** Something answered, but it is not a Crucible. */
  | 'not_a_crucible'
  /** The server refused the token its own config.toml holds. */
  | 'wrong_token'
  /** The server speaks a different major API version than the SDK this was built with. */
  | 'version_mismatch';

export interface BootstrapRefusalOptions {
  /** The exact command the host must run, when there is one. Never a guess. */
  command?: string;
  /** Verbatim output that explains the refusal — a status page, a stderr tail. */
  detail?: string;
  cause?: unknown;
}

/** Base class for everything `@crucible/bootstrap` refuses deliberately. */
export class BootstrapRefusal extends Error {
  readonly code: BootstrapRefusalCode;
  /** What to run to make this refusal go away, or null when nothing can be typed. */
  readonly command: string | null;
  /** Verbatim evidence, or null. */
  readonly detail: string | null;

  constructor(code: BootstrapRefusalCode, message: string, options: BootstrapRefusalOptions = {}) {
    super(message);
    this.name = new.target.name;
    this.code = code;
    this.command = options.command ?? null;
    this.detail = options.detail ?? null;
    if ('cause' in options) {
      // `cause` is ES2022; assign it explicitly so the CJS build carries it too.
      (this as Error & { cause?: unknown }).cause = options.cause;
    }
  }
}

/**
 * One of `install()`'s steps failed. Carries the step's name, the exit code (or
 * the reason there is none), the tail of what it printed, and every step that
 * finished before it — partial work survives failure (ARCHITECTURE.md R6), and
 * a caller that only sees "step 4 failed" cannot tell the operator what steps
 * 1 to 3 already did.
 */
export class BootstrapStepFailed extends BootstrapRefusal {
  readonly step: string;
  readonly exitCode: number | null;
  /** The last lines the step printed, stdout and stderr interleaved as they arrived. */
  readonly tail: readonly string[];
  readonly stepsDone: readonly string[];

  constructor(
    step: string,
    exitCode: number | null,
    tail: readonly string[],
    stepsDone: readonly string[],
    message: string,
    options: BootstrapRefusalOptions = {},
  ) {
    super('step_failed', message, options);
    this.step = step;
    this.exitCode = exitCode;
    this.tail = tail;
    this.stepsDone = stepsDone;
  }
}
