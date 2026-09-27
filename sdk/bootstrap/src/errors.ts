/** Every refusal code this package can raise. */

export type BootstrapRefusalCode =
  | 'unsupported_platform'
  | 'wsl_missing'
  | 'no_wsl_distro'
  | 'wsl_read_failed'
  | 'host_unresponsive'
  | 'no_nvidia_driver'
  | 'not_apple_silicon'
  | 'guest_missing_tool'
  | 'no_server_runtime'
  | 'release_channel_unreadable'
  | 'install_would_downgrade'
  | 'rollback_version_mismatch'
  | 'host_rollback_unsupported'
  | 'runtime_download_failed'
  | 'runtime_sha_mismatch'
  | 'runtime_unpack_failed'
  | 'runtime_install_failed'
  | 'guest_no_disk'
  | 'no_local_config'
  | 'config_unreadable'
  | 'config_missing_key'
  | 'network_path'
  | 'not_a_windows_path'
  | 'wsl1_only'
  | 'virtualization_disabled'
  | 'no_crucible_distro'
  | 'distro_not_systemd'
  | 'foreign_distro_not_systemd'
  | 'two_local_crucibles'
  | 'distro_unmarked'
  | 'distro_import_failed'
  | 'guest_no_network'
  | 'bad_job_type'
  | 'step_failed'
  | 'service_not_installed'
  | 'service_failed'
  | 'unreachable'
  | 'not_a_crucible'
  | 'wrong_token'
  | 'version_mismatch'
  | 'guest_root_unreachable'
  | 'host_not_installed'
  | 'host_unreachable'
  | 'host_unauthorized'
  | 'host_no_token'
  | 'host_install_running'
  | 'host_install_failed'
  | 'host_install_unwitnessed';

export interface BootstrapRefusalOptions {
  /** The exact command the host must run, when there is one. */
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
      (this as Error & { cause?: unknown }).cause = options.cause;
    }
  }
}

/** One of `install()`'s steps failed. */
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
