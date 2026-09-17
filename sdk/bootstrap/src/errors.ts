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
  /** The guest (or the Mac) lacks a tool the pack install needs: `curl`, `tar`, `zstd`, the sha tool. */
  | 'guest_missing_tool'
  /** No `<CRUCIBLE_HOME>/server/bin/crucible`: this machine has no server pack yet. `install()` puts one there. */
  | 'no_server_pack'
  /** `envpacks.json` for this release could not be fetched, or is not the manifest. */
  | 'pack_manifest_unreadable'
  /** The manifest has no pack for this (name, backend). Never a quiet build (PHASE14 section 2). */
  | 'pack_not_published'
  /** A part would not download. Carries the URL that failed. */
  | 'pack_download_failed'
  /** The reassembled archive's sha256 is not the manifest's. The archive is deleted. */
  | 'pack_sha_mismatch'
  /** Not enough free disk for `unpacked_bytes` + one part. Carries the numbers, before anything is fetched. */
  | 'pack_disk'
  /** `tar --zstd` would not unpack the archive. The `.partial` directory is left for reading. */
  | 'pack_unpack_failed'
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
  /** WSL is there, WSL2 is not: the default version is 1 and no distro can run a Crucible. */
  | 'wsl1_only'
  /** The hypervisor is not available: virtualization is off in the firmware. Software cannot fix it. */
  | 'virtualization_disabled'
  /** Windows, WSL present, and no `crucible` distro imported yet. A state; `ensureDistro()` clears it. */
  | 'no_crucible_distro'
  /** The `crucible` distro exists without `[boot] systemd=true`. Ours to repair. */
  | 'distro_not_systemd'
  /** A distro somebody chose by hand has no systemd. Theirs; we ask before writing to it. */
  | 'foreign_distro_not_systemd'
  /** A `crucible` distro exists AND another distro holds a config. Which one is `local` is not guessed. */
  | 'two_local_crucibles'
  /** A `crucible` distro exists without the rootfs marker and with a config in it: not ours to re-import. */
  | 'distro_unmarked'
  /** `wsl --import` failed. Carries what wsl.exe said. */
  | 'distro_import_failed'
  /** The guest has no route to the release (a VPN, a proxy). Carries the URL that failed. */
  | 'guest_no_network'
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
  | 'version_mismatch'
  /**
   * win32 only. The guest would not say whether its user lingers — no root
   * through `wsl.exe -u root` (WSL1, or a distro with root disabled), or a
   * `loginctl` that answered something else. The one hand-over that remains:
   * "off" would grant something nobody asked for and "on" would promise a
   * server that dies with the next logout, so neither is guessed.
   */
  | 'linger_unreadable'
  /**
   * win32 only, and a WSL STATE rather than a linger answer. The distribution
   * would not let Crucible in as root through `wsl.exe -u root` (WSL1, or a
   * distro whose root account is disabled). Since 2026-09-16 the guest's server
   * is a SYSTEM unit, so root is what lets it be installed at all — not merely
   * what makes it survive a logout, which is what `linger_unreadable` is about.
   */
  | 'guest_root_unreachable'
  /** win32 only. `loginctl enable-linger` ran as root and failed. */
  | 'linger_failed'
  /**
   * win32 only. There is no `%LOCALAPPDATA%\Crucible\host\crucible.cmd` on this
   * machine, so the process that owns the install sequence (PHASE15 4.3) is not
   * here. The answer is `install.ps1`, which this refusal carries as `command`:
   * a library does not download and run an elevated installer of its own accord.
   */
  | 'host_not_installed'
  /** win32 only. The host pack IS installed and its loopback door did not answer. */
  | 'host_unreachable'
  /** win32 only. The host refused the engine token this side read from its config (401). */
  | 'host_unauthorized'
  /**
   * win32 only. The host has no `[auth] token` to authorise its door with,
   * which is only true before its first host-mode `crucible init` (503).
   */
  | 'host_no_token'
  /**
   * win32 only. An install is already in flight on this machine (409). There is
   * one install per machine; the second caller waits rather than starting a
   * second walk over the same distro.
   */
  | 'host_install_running'
  /**
   * win32 only. The host's stream ended without a `done` event, or broke the
   * door's own contract (a line that is not JSON, an event with no name, a
   * `done` missing a field). A truncated stream is not a success.
   */
  | 'host_install_failed';

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
