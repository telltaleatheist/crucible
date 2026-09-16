# Installation and network audit — 2026-09-16

The Windows default remains native Crucible. WSL is a guided, optional upgrade;
Tailscale is optional sharing. Neither is a prerequisite for a working local install.
This follows PHASE15's native-Windows amendment and its single Windows control owner.

## Confirmed defects corrected

- Settings and capability writes discarded `[server] advertise`. Every rewriting
  call now preserves it. Configuration replacement is atomic, so a serialization or
  interrupted staging failure cannot truncate the existing token and provider keys.
- Native Windows wildcard binding called POSIX `getifaddrs` through `CDLL(None)` and
  crashed. Windows now enumerates `Get-NetIPAddress` structured data, retaining LAN
  and Tailscale interfaces and excluding loopback/link-local addresses.
- LAN detection considered any two matching port numbers a valid forward, regardless
  of its source/destination. It now checks all four fields. Mirrored networking in a
  config file no longer claims verified remote reachability.
- WSL upgrade automatically added a wildcard LAN portproxy. Engine installation now
  leaves forwarding/firewall configuration unchanged; sharing requires explicit choice.
- Windows `-Root` did not propagate to the runtime, and the host independently ignored
  `CRUCIBLE_HOME`. Both now use the same absolute root.
- Installs did not provision an ordinary CLI or the new desktop/lifecycle contract.
  The shared native step list and generated standalone scripts now register the local
  contract, provision a CLI, and install desktop presence. Linux explicitly skips the
  GUI; macOS server packs include its menu-bar dependencies.
- Windows and POSIX upgrades deleted the installed pack while processes could still use it.
  A replacement is staged and smoke-tested first. The staged controller requests
  orderly shutdown; failure keeps the installed pack. The previous pack is renamed
  before activation and retained if activation fails. An interrupted previous swap
  produces `upgrade_recovery_required`, never silent deletion of the backup.
  The explicit POSIX source-build route uses the same activation transaction, with
  console scripts made relocatable before activation.
- PID-only controller/tray checks had a simultaneous-launch race. Kernel-held locks
  now exclude competing processes, and failed tray startup releases its lock and PID.
  Upgrade shutdown distinguishes positively refused connections from timeouts and
  refuses an answering unmanaged engine. Console opening now passes the token using
  the parameter the console actually reads. macOS installs load the menu-bar agent
  immediately, and refuse overwriting a foreign Crucible.app.
  Upgrade negotiation accepts the authenticated controller's lifecycle contract 1,
  or the explicitly supported legacy 0.6.0 `/quit` contract. It verifies controller
  process and port termination, plus native child termination; the legacy Windows
  host upgrade leaves its separately installed WSL engine running. HTTP errors on
  the control port report an incompatible service without spawning another host.
  The tray's captured process must actually exit even after removing its PID file,
  preventing runtime replacement during interpreter cleanup. Installation records
  retain a source build's venv interpreter symlink rather than dereferencing it to
  the system Python.
- Uninstall continued removing environments/config after service shutdown failed.
  Shutdown and deregistration failures now stop destructive cleanup. Uninstall keeps
  `jobs/` and `uploads/`, including partial job output, even with `--purge-weights`.
  No distro is unregistered, and models remain unless explicitly purged.
- Guided WSL installation stopped at a hard-coded missing-rootfs refusal. It now
  downloads that release's image/checksum, verifies the digest, imports WSL2, and
  checks the Crucible ownership marker. Image naming/repository constants are
  generated from the existing SDK owner. An unrelated or unmarked distro is never
  overwritten or unregistered; partial imports are retained with a recovery error.
- The migration's stop-native/switch-pairing steps were log-only promises. They now
  require host callbacks and execute them before reporting completion. The guest is
  restarted after importing the token, then authenticated before weight migration.
  Both migration HTTP transports now send the required API-version header.
- The guest installer applied `CRUCIBLE_RELEASE` to curl, not the shell executing
  the installer, and could pipe a failed download into a successful empty shell.
  Download and execution are now separate, with an explicit `--release` argument.

## Managed sharing contract

`crucible sharing enable|status|reconcile|disable` owns a single Tailscale Serve raw
TCP forward. It verifies the exact target, checks authenticated local engine access,
and publishes through `PUT /v1/settings` using `tailscale_advertise`. This field is a
host-owned projection separate from operator-authored `advertise`; `/v1/setup`
combines both with the actual bind-derived addresses. It never changes the bind.

`<home>/sharing.json` records durable intent before mutation. Interrupted publication
can be repaired with `reconcile`. Existing matching forwards require explicit
`enable --adopt`; conflicting forwards are refused, not overwritten. Disable removes
only an unchanged owned forward and withdraws only the managed advertisement. External
drift retains the ownership record and reports failure. No `tailscale serve reset`
is used, and unrelated Tailscale listeners are untouched.

Status distinguishes configured/degraded/disabled and always reports
`remote_reachability: not_tested`: local inspection cannot establish a Mac's network
path. A disconnected node or changed DNS/forward is degraded. Persistent configured
addresses survive an outage rather than disappearing without explanation. Tailscale
background Serve persists itself across restarts; reconcile repairs recorded intent.
Successful local starts reconcile the saved opt-in; a sharing outage is returned as
degraded sharing while the working local engine remains running.

The desktop menu offers Enable/Stop Tailscale sharing. A matching pre-existing
forward produces an explicit Use existing Tailscale sharing action before adoption.
Action failures remain visible across health refreshes until the next action. The
CLI remains available for detailed status and advanced management. Tailscale itself
must already be installed and connected; failure is shown without affecting local use.

The implementation uses [Tailscale's documented Serve commands](https://tailscale.com/docs/reference/tailscale-cli/serve):
background raw TCP forwarding, JSON status, and per-port `off`.

## CLI ownership

Windows owns `<home>/bin/crucible.cmd` and only the user-PATH segment it inserted.
POSIX owns `~/.local/bin/crucible`; it does not rewrite shell profiles. A shell lacking
`~/.local/bin` still needs that standard directory added. Launchers capture the
registered interpreter, working directory and `CRUCIBLE_HOME`, so they do not depend
on the caller's current directory. Upgrades and uninstall compare the content digest;
externally modified or unrelated commands are preserved and reported.

## Verification and limits

- Final stable-source full Python suite in WSL: **1,940 passed, 14 skipped,
  zero failures**, in 8 minutes 44 seconds. Two existing test-library deprecation
  warnings remain. Native Windows focused lifecycle/sharing tests: **58 passed**.
- Final source candidate: **0.6.1**, with client/bootstrap versions, generated
  installers, module manifests, and consumer vendor archives coordinated. It is
  not a published binary release; see the release-readiness report.

- Focused Python tests cover sharing ownership/adoption/conflicts, interrupted
  publication, offline/drift status, address settings round trips, custom Windows
  roots, CLI upgrade/uninstall, preserved job data, and stop-failure safety.
- Bootstrap's installer-generation and installation suites pass, including the newly
  shared registration/CLI/desktop steps. The generated PowerShell script parses with
  PowerShell's parser.
- Actual Windows interface enumeration was exercised successfully on this host.
- No Tailscale forwarding or live service configuration was changed by this audit.
- Multi-gigabyte release pack downloads/builds, a clean Windows VM reinstall, Mac
  install/reboot, and failure injection during a real pack swap remain release-level
  validation. Tests using fake runners do not prove those OS operations.
- Follow-up verification: 58 Python lifecycle/desktop/sharing/release-smoke tests passed; all 269
  bootstrap tests passed, including four actual shell activation fixtures (success,
  failed shutdown, failed activation with rollback, interrupted prior swap refusal).
  These tests exercise temporary runtimes, not the user's live installation.
  Relocated core pack smoke tests now verify each installer-consumed lifecycle
  command's parser before an archive can be published, without running those actions.
- The source tree's new bootstrap commands require the updated core runtime. The
  previously published 0.6.0 packs do not include that lifecycle contract. A complete
  new release with matching installers/core packs is required before fresh-install
  readiness can be claimed. Existing release tooling builds the complete pack matrix;
  there is no supported core-only asset patch operation.
- A running published 0.6.0 native Windows controller may terminate only its old
  command wrapper when asked to quit, leaving the Python engine alive. The new
  staged updater detects the answering engine and safely refuses the swap. Fixing
  child ownership in the new release does not change the already-running old
  process; unattended upgrade success from that legacy state is not claimed. No
  process-kill workaround guesses at ownership. The old owned WSL-engine case uses
  a separate guest runtime and is unaffected by replacing the Windows host.
- Final targeted regressions for interpreter-symlink preservation and actual tray
  process termination passed with the lifecycle suite (39 tests, CPU fixtures only).
