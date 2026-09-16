# Local lifecycle and desktop installation

The normal Windows installation runs the native `llama-windows` engine immediately.
WSL is an optional guided upgrade owned by the Windows controller. Applications
never select a WSL distribution or spawn an engine to start their local service.

`installation.json` under the resolved Crucible home is versioned installation
metadata. Schema 1 publishes `platform`, `release`, `home`, and `control` with an
absolute `command`, argument array and absolute `cwd`. It contains no credentials
and makes no claim that a service is healthy. The installer and service-install
command publish it atomically. Re-running installation repairs a moved runtime's
record. Existing developer installations can explicitly register with
`crucible local register`; applications must not guess their Python paths.

`@crucible/bootstrap` owns reading/validating that record and invoking
`localStatus`, `startLocal`, and `stopLocal`. The command's environment names the
same Crucible home. Missing records, broken installations, failed authentication,
wrong services, unreachable services and intentionally stopped services are
different outcomes. Starting succeeds only after the paired engine answers, with
its identity and authenticated info checked. A timeout alone never means stopped.

The Windows controller runs independently with `orchestrator --headless`.
The desktop shortcut starts `local tray`, which ensures the controller exists.
Closing that icon leaves the controller, native child or WSL hold, and work intact.
Explicit Stop records the operator's intent and the watch does not recover it.
Start clears that intent. Authenticated controller `/local/start`, `/local/stop`,
and `/local/status` operations share its lifecycle implementation. The existing
restart and guided-upgrade operations remain controller-owned.

The native child is the pack's Python process, not a command-shell wrapper.
Its private inherited stdin pipe carries lifetime: closing it, or losing the
controller, requests graceful Uvicorn/lifespan shutdown. The controller waits
for the child to exit and retains ownership on timeout. Its own main loop waits
for cleanup to finish before exiting. The authenticated controller info publishes
`local_lifecycle_version: 1` so upgraders negotiate the shutdown contract without
guessing from patch versions.

On macOS the engine remains a launchd service; the menu bar is an independent
user application installed under `~/Applications`. Linux uses its service manager
without installing a desktop component. Tray close/uninstall is cooperative and
must complete before deleting its runtime. Installers provision owned CLI
launchers and refuse to overwrite unrelated commands.

The guided Windows-to-WSL move must execute the native-child stop and pairing
switch before emitting completion. The controller suppresses recovery during the
move, authenticates to the guest through Windows after the switch, takes the WSL
hold, and claims the guest. Logged intentions are not completed steps.

Validation: lifecycle regression tests exercise identity validation against a real
local HTTP server, bad credentials/transport distinctions, persistent stop intent,
and failure to stop. Bootstrap tests exercise record discovery, malformed records,
missing runtimes and failed control processes. Platform smoke results and release
limitations are recorded in the integration audit rather than inferred from these
unit tests.
