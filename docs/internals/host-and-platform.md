# Host and platform internals

How Crucible lives on a machine: the Windows orchestrator (`crucible/host/`), the WSL guest,
the service managers, LAN and Tailscale exposure, pairing, process lifetime, and uninstall.
These are the constraints and measured facts a maintainer needs that the code cannot say by
itself. Plans and rationale live in `docs/PHASE15-HOST.md`, `PHASE17-ORCHESTRATOR.md` and
`PHASE19-AUTOMATIC-WSL.md`; this file is the short list of things that must stay true.

## Standing owner rulings

- **A local Crucible is a service, not an app's child** (2026-09-13). Nobody owns it, it
  survives the app that started it, and nothing can be orphaned that was never a child.
- **Windows is the driver** (2026-09-18): *"windows is the driver; the thing moving wsl
  forward."* The Windows host carries the guest to its own release; one release per machine.
- **Pairing is open by default** (2026-09-17): *"ollama allows anybody to connect if they can
  reach it. make that the case with crucible servers as well."* `[auth] open_pairing = false`
  restores approval (short code, operator list, decision door all still work).
- **Restart sentences name "Update and restart"** (2026-09-26): *"make it clear that the user
  has to reboot and update, not just reboot ... the update logic is the route through which wsl
  installs."* Every restart sentence also says someone must sign in afterwards (the tray
  resumes from the Startup folder, which runs only at sign-in).
- **Nobody spells a guest path** (2026-09-26). A hand step is a bug; when a guest command is
  needed, `crucible guest <words>` forwards it.
- **Pinned ffmpeg from our own `tools` release** (2026-09-26): *"downloading from gh releases
  is probably the most trustworthy/correct way."* No system package, no runtime download.
- **Pinned Silero VAD for `asr` speech_only** (2026-09-27): *"We can use the speech detector,
  that's fine."*
- **Virtualization-off is re-checked at every start** (2026-09-26), so a person who fixes the
  BIOS and restarts never has to find Try again.

## Hard rules for WSL and GPU processes

- **Never SIGKILL a process that may be in a WSL2 GPU (dxg/CUDA) wait.** It wedges the distro
  until Windows reboots. POSIX stop is SIGTERM to the process group, then a timeout reported
  by name (`procgroup.ask_to_stop`). Only native win32 processes are tree-terminated
  (`procgroup.terminate_tree`, `taskkill /T /F`), because there the kill goes through the
  Windows driver the way Task Manager's does.
- **Never `wsl --shutdown`.** It stops every distro on the machine, including ones Crucible
  does not own, and takes GPU processes down uncleanly. The only VM-level act Crucible takes is
  the generated state table's per-distro `wsl --terminate <distro>` repair.
- **Never `wsl --unregister`.** `uninstall --wsl-too` runs the guest's own `crucible
  uninstall` and prints the unregister line for the operator; unregister destroys the whole
  ext4 irreversibly.
- **Every wsl.exe call uses `--exec`.** The implicit-shell form pre-expands `$var` on the
  Windows side, where it is empty. When the guest must expand something (`$HOME`,
  `$CRUCIBLE_HOME`), the spelling is `--exec bash -lc '<script>' <argv0> <words...>`, with
  the words as positional parameters so no quoting on either side can change them.
- **Guest commands name the user in Crucible's own distro** (`presence.guest_argv`: `-u
  crucible`). `[user] default=crucible` in `wsl.conf` only takes effect at the distro's next
  start; commands run straight after the import ran as root, the engine landed in
  `/root/.crucible`, and a later run minted a second home with a new token (every later
  upgrade then failed 401). A consented foreign distro keeps its own default user.
- **A distro terminates seconds after its last wsl.exe session ends**, even with systemd units
  running and linger on. Only a Windows-side process can keep the VM up, so the orchestrator
  holds `wsl -d <distro> --exec sleep infinity` for as long as a WSL engine is meant to run
  (`PresenceWatcher.hold`), re-takes it every watch tick (including the first hold, for an
  owner decided late by the watch), and on an upgrade quit leaves a bounded
  `sleep 600` hold for the next tray (`hand_over`) started before its own is released.
- **User systemd is unreachable inside WSL.** WSLg overmounts `/run/user/<uid>`, hiding the
  user manager's D-Bus socket; `systemctl --user` fails and the orchestrator's unit probe
  with it. So in WSL the server is a **system** unit with `User=` naming the installing
  account, `WantedBy=multi-user.target`, driven via `wsl.exe -u root` (no password).
  `/run/dbus` is not overmounted. Outside WSL the unit stays a user unit.
- **Detect WSL from `/proc/sys/kernel/osrelease`**, not `WSL_DISTRO_NAME`/`WSL_INTEROP`,
  which are absent from a unit's environment.
- **Root inside WSL** (`service.root_prefix`): passwordless `sudo -n` first (Crucible's distro
  grants it; interop can be unregistered by systemd's binfmt handling, giving `Exec format
  error: 'wsl.exe'`), then the `wsl.exe -u root` interop door.
- **Secrets never go through `/mnt/c`.** DrvFs synthesises permissions from the Windows ACL,
  so a 0600 file there is readable by every Windows process. The carried config is written
  inside the guest with `umask 077`, as base64 on the argv (the runner has no stdin), and as
  the user that reads it.
- **Windows drives on a guest PATH (`/mnt/<drive>/...`) are never searched** for tools; an
  `.exe` there only runs while interop works.
- **Children of the orchestrator start in `CRUCIBLE_HOME`.** A wsl.exe that inherits the
  orchestrator's cwd (inside `Crucible\host`) keeps a handle on that directory after exit and
  blocks the next upgrade's `Move-Item`. The systemd unit uses `CRUCIBLE_HOME` as
  `WorkingDirectory` for the same reason. Tray children get `CREATE_NO_WINDOW`.

## Reading wsl.exe

- wsl.exe is two programs: its own messages are **UTF-16LE**, anything it `--exec`s is relayed
  **UTF-8**, and one command can produce both. `runner._decode_pipe` decides per stream from
  the bytes (a UTF-16LE BOM, or a zero high byte under every ASCII char in the opening
  window), with `errors="replace"`, then normalises CRLF. Never ask `subprocess` for text.
- Key on the machine-readable **error code** (`Wsl/Service/WSL_E_...`, `HCS_E_...`), never
  the localised prose (`wslstate.read_wsl_answer`). The `wsl -l -v` header is recognised by
  its last column not being an integer.
- Before WSL is live, the inbox stub prints its **usage screen**; three option names in one
  reply identify it.
- `wsl -l -v` with WSL live and no distros **exits non-zero** ("no installed
  distributions"). The authority for "nothing registered" is the registry key
  `HKCU\...\Lxss` (`presence.registered_wsl_distros`); any other failure is unreadable, not
  empty (`presence.read_wsl_distros`, the single reader).
- `WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED` means "features not live" (off, or awaiting a
  restart), not "missing". `WSL_E_DEFAULT_DISTRO_NOT_FOUND` from `--status` is a live WSL.
- `Win32_OptionalFeature.InstallState` (1 enabled, 2 disabled, 3 absent) can say "enabled"
  while CBS still has the package pending and no `lxss.sys` exists. It is logged and gates
  the noisy pending signals, never decides alone. `Get-CimInstance` answers a standard user;
  `Get-WindowsOptionalFeature` needs elevation.
- **Restart owed** (`LiveWsl.restart_owed`): CBS `RebootPending` alone suffices once wsl.exe
  says the component is required. `PendingFileRenameOperations` and Windows Update's
  `RebootRequired` count only when both features report enabled (they are often set for
  other programs' reasons).
- **Boot time** comes from `GetTickCount64`, which runs through sleep and Fast Startup; a Fast
  Startup "shut down" never commits servicing, so it must not count as the restart asked for.
- Every runner call has a timeout; a failure is a `RunResult`, not an exception; argv arrays
  only. `RunResult.said` keeps the **tail** of output (a failing `install.sh` prints curl's
  progress meter first and its error last).

## The orchestrator (`crucible/host/`)

**Imports.** Importing `crucible.host` must not need pystray, Pillow or tkinter: pytest runs
in WSL without them. `tray.py` imports them inside functions and makes no decisions; the menu
is a pure function (`menu.menu_model`) tested cell by cell. Paths take the environment as a
parameter (`paths.py`) and `LOCALAPPDATA`/`APPDATA` are read, never assembled from a username.

**Nothing in the host runs a model.** It picks which server runs, starts it at login, boots
the guest, watches, and runs the install move.

**One per machine.** `processlock` holds a kernel lock; `host.pid` only names the holder so
"another one is running" has a pid in it. A pid whose liveness check is access-denied is
treated as alive.

**Start order** (`app.run`): Startup item if absent → consent read (before the watcher,
because it decides which distro is watched) → presence → pairing file → claim (needs the
owner and the pairing token) → door (before the carry thread, so a mid-move `POST /install`
gets 409 and attaches) → tray and watch → carry thread (waits on the watch's first settled
presence, bounded by `WATCH_SECONDS + RECIPE_TIMEOUT_SECONDS + BOOT_WAIT_SECONDS`).

### Ownership (`menu.Owner`)

- `WSL_UNIT`: the guest's unit in the managed distro; the host may restart and stop it.
- `HOST_CHILD`: the `llama-windows` server this process spawned; Quit takes it down.
- `FOUND`: already answering when the host looked. **Never restarted, stopped, replaced or
  claimed**, and the WSL install is not offered over it. Both the menu and the act refuse
  (`_refuse_acting_on_a_found_engine`).
- `NONE`: no engine.

`Host.start` **pings before starting anything**: on a machine whose engine lives in a
hand-installed distro, the distro probe answers `absent` and starting a child would put a
second server on 7100. `poll` resolves an owner when it is `NONE` and something answers
(otherwise `engine_token` stays empty and every authenticated door, `/quit` included,
answers 503). `probe_distro` returning `unknown` is never read as `absent` (a second import
cannot be undone). The engine hunt (`find_engine`) only asks **running** distros; asking a
stopped one boots it. A ping that gets any HTTP status, including 4xx, means "up".

**Consent** (`[orchestrator] distro = "<name>"` in the Windows `config.toml`) lets the
orchestrator manage a distro Crucible did not import. It is permission, not a fact: the unit
is probed (`systemctl is-enabled crucible.service` as root on the system manager, reading the
**printed** state because `disabled` exits non-zero), and only a unit that exists makes the
owner `WSL_UNIT`. A present but unusable value is refused, never ignored.
`PresenceWatcher.distro` is the single owner of "which distro"; every guest act asks it (the
guest carry once used `EngineInstall`'s default and ran `install.sh` in a distro that did not
exist). `[orchestrator] wsl = "never"` keeps a machine native on purpose; any other value is
refused.

**Tokens.** A guest engine's token is read from the pairing line the host copied out of the
guest; a host-mode child's from the host's own config. No fallback between them (the wrong one
gives a 401 against a working door). The Windows pairing file is a **copy** of the guest's
line, never a second composition (it would carry the wrong token).

### Presence and watch

Watch every 15 s. **One recovery per down-edge**, then a named state and a menu item; the
host never loops on restart because the systemd unit is `Restart=always`. Recovery starts the
guest's system unit (touches only Crucible's unit, so allowed in any distro); host mode
respawns the child. Every down-to-up edge re-asserts the claim.

### The door (127.0.0.1:7101, `door.py`)

- Loopback only; `serve` takes no wildcard. `http.server`, not FastAPI: the tray must start
  in well under a second and hold no event loop or server stack.
- Bearer = the **engine's** token, read through a callable because the token changes during
  the move (`migrate-config` gives the guest the Windows token). `host_no_token` is 503 and
  names the actual cause (`engine_token_detail`).
- `/v1/ping` is unauthenticated (it tells "wrong token" from "not a Crucible"); everything
  else takes the bearer.
- `POST /install` 409 `host_install_running` is normal: the tray usually started the move
  already. `GET /install` gives running/outcome/presence; `GET /install/events` replays a
  200-event ring then follows. Ring read and subscribe happen under one lock so no event is
  lost or duplicated. `_move_open` (more events can arrive) is distinct from the claim.
- `engine_not_ours` is refused before the stream opens (a status code, not a last ndjson
  line). Every stream ends with a terminal event, even if the sequence threw first.
- `POST /quit` writes and flushes its 200 **before** stopping, and is not refusable during an
  install: `taskkill` without `/F` is a no-op on a console-less `pythonw`, and `/F` runs none
  of `quit()`, so this route is the only orderly stop. `X-Crucible-Handover: 1` (sent by
  `local shutdown` on upgrade) leaves the bounded distro hold behind.
- Request bodies are capped; the handler logs to `host.log`, never stderr (nowhere under
  `pythonw`).

### Quit order (`Host.quit`)

Release the claim while the engine still answers → drop the hold → stop the child (only when
the owner is `HOST_CHILD`; a found engine is not our child even if the distro probe said
absent) → end. It logs before acting. Main thread waits for cleanup because request threads
are daemons.

### Orchestrator/engine relation (`peer.py`)

- A claim is a **statement of fact**, never a permission check; the engine never consults it.
- Not persisted: it dies with the engine process and is re-asserted within one watch tick.
- Same URL re-claiming succeeds; URLs are normalised only by trailing slash. `force` is never
  sent by an orchestrator. Releasing nothing is success; releasing another's claim is refused.
- `uptime_s` is monotonic, so a re-answering engine is recognisably a new process.
- Orchestrator `/v1/info` reads the engine's capabilities through on every request, uncached;
  an unreadable engine gives `capabilities: []` and null name/backend. An orchestrator serves
  zero job types. Peer calls are `urllib` only (tray start-up budget).

### Log (`host.log`)

`%LOCALAPPDATA%\Crucible\host.log`, the tray's only output. Rolled at 2 MiB, one previous file
kept. Opened per write (no handler opened at import time). **ASCII only**: Windows PowerShell
5.1 reads BOM-less files as ANSI; known punctuation is transliterated, anything else becomes
`?`, NULs from UTF-16 are dropped. Timestamps are local time.

### Startup item

`%APPDATA%\...\Startup\Crucible.lnk`, per-user; no Task Scheduler (needs elevation), no
service (cannot own a tray icon). Target is `pythonw.exe` with a small entry point that binds
`CRUCIBLE_HOME` (Explorer does not inherit the installer's environment); a `.cmd` would flash a
console. Written with PowerShell's `WScript.Shell` COM (no pywin32). `CreateShortcut` rewrites
in place, so install is idempotent. The entry is what makes "restart, then Crucible continues"
true.

### Tray and menu

`install-engine` is absent (not greyed) unless the distro is absent or unknown. Restart is
offered in every state except during an install, including FAILED. Open log is always
present. Try again appears only for outcome `cannot` or `failed`; a disabled restart line
shows while `reboot-pending`. `menu.outcome_items` is shared with `desktop.py` so both icons
offer the same items. The icon is drawn with Pillow, not shipped.

## The Windows to WSL move (`installer.py`, `outcome.py`)

- **One sequence, two callers**: the tray at start (`decide_engine`) and `POST /install`. Both
  go through `door.run_recorded` under the door's claim, so there is one event numbering.
  `install.sh` (generated from `sdk/bootstrap/src/steps.ts`) is the only definition of the
  guest half; the host runs it, never restates it.
- **Decision table** (`Host.decide_engine`): found → nothing; `wsl = "never"` → declined;
  transient `cannot` → re-probe what is live and resume if live; virtualization-off `cannot` →
  re-check hypervisor; other `cannot` → nothing until Try again; `failed` with attempts ≥ 2 →
  nothing; `reboot-pending` → resume; otherwise probe the table.
- **Two inits, one token**: `install.sh` mints a token; the host then runs `crucible init
  --force --config-from <file>` carrying `auth.token`, `[routes]`, `[upstreams]` and
  `[accelerator]` (extracted textually, never round-tripped through a TOML writer). Host,
  port, name, backend and job flags belong to the guest.
- **Model retirement follows verified activation.** Pull each subject into the guest and wait
  until the guest's catalog says installed; stop the Windows server; verify guest ownership and
  pairing; only then retire native files through the catalog owner functions
  (`StoppedWindowsCatalog`). After activation port 7100 is the guest's: never send a Windows
  deletion there. A persistent cleanup record lets an interrupted deletion resume even after
  its install stamp is gone. At any instant a subject exists on one side or both.
- `subject_in_use` is waited out (60 rounds × 5 s) and then fails by the holder's name, never
  skipped. `weights_shared` (a base an alias holds) is retried next round; a round that
  deferred everything and removed nothing is refused rather than spun.
- **Guest catalog** is reached with `curl` inside the distro (Windows holds 7100 during the
  move), `-w` status not `-f`, so the refusal code survives. Catalog parsing is strict: a
  half-parsed catalog would delete Windows weights prematurely. The key is `rows`.
- **The walk** (`_walk`): detect, repair, re-detect. A repair that did not change the answer
  ends the walk (otherwise it loops forever). **Check what is live before any UAC**: running
  `wsl --install` again while a restart is owed re-pends the servicing transaction.
- **Restart budget** 3 (`RESTART_BUDGET`): a restart is spent only if Windows actually booted
  since the ask. The first restart after an enable is often deferred behind a staged update;
  the second commits both. Codes: `wsl_reboot_required` → `wsl_reboot_still_owed` →
  `wsl_reboot_again` (budget spent; still re-probed at every start).
- `_guest_ready` re-runs the state table once the distro exists (guest rows: systemd, network,
  root reachable). `pack_disk` stays off (the guest payload is ~30 MB).
- **Import**: Canonical's WSL image with Canonical's own digest row, constants generated from
  `sdk/bootstrap/src/distro.ts`. A sums file that does not name our file is refused. Finishing
  the import creates the `crucible` user, passwordless sudo and the `/etc/wsl.conf` marker.
  The disk line reports the Windows drive, not `df` inside the vhdx (a virtual maximum).
- **First boot** of the image with cloud-init takes ~40 s; the post-move guest wait is sized
  for that, and the last real error (not a fixed "token" sentence) is reported.
- **Guest carry** (`upgrade_guest`): older or unreadable → run the same `install.sh --release`;
  equal → nothing; newer → `guest_ahead_of_host`, left alone (never backwards). Runs in the
  distro the host claimed, only when the owner is `WSL_UNIT`, after the watch settles, and
  never raises out of its thread.
- The LAN door step runs after the pairing switch (the guest must be answering) and before
  weights migration (a UAC prompt hours later has nobody in front of it); it follows the
  existing `landoor.json` preference when not told.

### Outcome file (`wsl-outcome.json`)

Five states: `done`, `declined` (no code/sentence), `reboot-pending`, `cannot`, `failed`
(always code and sentence). `failed` is retried once at the next start (attempts counts
consecutive failures). `restarts` counts consecutive restart asks. `classify` derives
`cannot` from the generated table's `automatic` partition. The record is written **before**
the terminal event is emitted. `outcome.read` refuses a present but unreadable file by name;
the orchestrator reads through `read_or_quarantine`, which moves such a file to
`wsl-outcome.json.bad-<utc stamp>` (`host/quarantine.py`), logs where it went, and decides
again from nothing. Nobody is asked to delete a file. Uninstall removes it.

### Records Crucible repairs by quarantine (`host/quarantine.py`)

A record Crucible owns that will not parse is never a hand step: it is moved aside to
`<name>.bad-<utc stamp>` next to where it was, the log names the new path, and the code
carries on as if the record were absent. Three records take this path:

- `wsl-outcome.json` (above).
- `migration-cleanup.json`: `installer.quarantine_bad_cleanup_record` runs before a resume
  (`_resume_model_cleanup`) and before an install on a WSL-owned machine. A bad record used to
  make the cleanup retry every 300 s forever and every later `POST /install` refuse. Now the
  cleanup it described is dropped: the Windows copies of the moved models stay on disk (cost:
  disk only; `crucible uninstall --purge-weights` removes them with the rest).
- `landoor.json`: an unreadable LAN sharing record no longer fails the `lan-door` step; it is
  quarantined, sharing stays off, and the line names `crucible lan enable`.

### Half-finished imports and distros Crucible did not make

`_import_distro` finds `<home>\wsl` non-empty with no `crucible` distro registered when an
earlier `wsl --import` was killed. If the directory holds only what an import writes
(`IMPORT_ARTEFACTS`, i.e. `ext4.vhdx`), it is removed and the import runs again in the same
walk. Anything else refuses `distro_import_incomplete` naming the directory, what is in it and
the exact `Remove-Item -Recurse -Force` command. A registered `crucible` distro whose
`/etc/wsl.conf` lacks `WSL_CONF_MARKER` is refused `distro_unmarked` naming the distro, that
Crucible did not make it, `wsl --unregister crucible`, and that this deletes the distro's files.

### The controller port (`host/portholder.py`)

When 7101 cannot be bound, or answers as something other than the orchestrator
(`wrong_controller` in `local.py`), the message names the occupant: `netstat -ano -p tcp`
gives the listening pid (rows are matched by a `:0` foreign address, never by the localised
state word) and `tasklist /FI "PID eq N" /FO CSV` its image name. The sentence is "port 7101
is held by <name> (pid N); stop it or run `crucible local shutdown`"; when the lookup cannot
name it, the `netstat -ano | findstr :7101` command is given instead.

### Console after `install.ps1` (`installwatch.py`)

Waits for a real ending (up to 2 min for the tray to decide), prints steps and the ending in
ASCII wrapped words. An outcome older than the install start (`--since`) is printed as
history, never as this run's result. With `--brief` (an app ran the script) it keeps the short
wait. When the door is not answering after the decision window, the console starts the
controller itself (`local._spawn_controller`, the same call `retry.py` and `desktop.py` use)
and waits `CONTROLLER_START_SECONDS`; only when that spawn fails or the door still does not
answer does it print the sign-out-and-back-in advice, and then with the host log path on its
own line. Every "did not start" message in `retry.py` and `local.py` names `<home>\host.log`;
every "reinstall" names `paths.INSTALL_ONE_LINER`, the same `irm ... install.ps1 | iex` line
the README documents (a test holds them equal).

### One `alive` (`processlock.alive`)

`OpenProcess` failing with `ERROR_ACCESS_DENIED` means the pid exists and belongs to somebody
else, so it is alive. `host/app.py` (`_alive`), `uninstall.py` (`_alive`), `desktop.py` and
`local.py` all read the one implementation in `processlock.py`; the copy in `uninstall.py`
that read a denied handle as dead is gone (it could have called a live tray stale and planned
around it).

## LAN door (`host/landoor.py`, `lan.py`)

- The WSL engine binds `127.0.0.1` and keeps doing so; `0.0.0.0` inside the NAT'd guest widens
  exposure without reachability. The crossing is a Windows-side fact.
- **Mirrored networking** (`networkingMode=mirrored` in `.wslconfig`) needs nothing. Otherwise
  a **portproxy** `v4tov4 listenport=7100 listenaddress=0.0.0.0 connectport=7100
  connectaddress=127.0.0.1`: aimed at loopback, so it survives the guest's DHCP changes.
- **A forward alone is a dead door**: the listener belongs to the Windows forwarding service,
  so an inbound allow (`RULE_NAME`, **Private profile only**) is also required. Both rows are
  reported separately.
- Private-ness is judged **per interface** (a Private Tailscale adapter plus a Public Ethernet
  shut the LAN while "any network Private" said yes). A Public network is only marked Private
  when a person says so, in the same prompt. Hyper-V internal switches (WSL vEthernet,
  Default Switch) are never offered as addresses.
- `admits` decides from the effective firewall profile settings (enabled, block-all-inbound,
  group policy ignoring local rules) and the rule; it cannot see routers that isolate clients
  or third-party firewalls. A self-dial proves the forward and engine, not the firewall; a dial
  from the guest is judged by the vEthernet profile. The final test is `crucible pair` on the
  other machine.
- `netsh` output is localised: rows are parsed by their numbers, rule absence by exit code (1
  absent, 0 present). `NetworkCategory` is cast `[string]` in PowerShell (it serialises as an
  integer). A single PowerShell object serialises as a scalar, none as empty.
- **Refused for a native Windows engine** (`host.backend` from the engine's own `/v1/info`):
  a portproxy to a process on the same port is a self-loop that exhausted ~15.5k of 16.4k
  ephemeral ports on 2026-09-17. No firewall row is added on that path either.
- `landoor.json` is durable intent written before mutation and updated to "rows exist" before
  the publish: a half-finished enable must overstate, never understate, because `disable`
  cleans up from the record. Rows persist until `lan disable` (no per-stop removal).
  `disable` withdraws the engine projection first and keeps the record if that fails.
- `reconcile` republishes addresses after a DHCP change and adds no row that exists.

## Elevation (UAC)

- Reads never elevate (`portproxy show`, `firewall show rule`, connection profiles, CIM
  features all answer a standard user). Only a change prompts.
- The host raises prompts (`wslstate.elevated_argv`, `Start-Process -Verb RunAs`); bootstrap
  only spells them. The tray is a process the person can see.
- Multiple elevated commands go under **one** prompt as `-EncodedCommand` (base64 UTF-16LE),
  never a temp script file (a user-writable file run as admin is an escalation surface).
- The elevated exit code is not authority (netsh reports only its last command; UAC can be
  dismissed): enable/disable re-read the machine and decide from that.
- Announce the prompt as it is raised. An already-elevated session (an administrator's SSH on
  Windows) gets no prompt at all. `--no-elevate` reports the argv and claims nothing.
- `make_private_argv` nests `powershell.exe -Command` because the elevation wrapper quotes
  every word and a quoted `-InterfaceIndex` is a string, not a parameter.

## Tailscale sharing (`sharing.py`)

The host owns `sharing.json` and the Serve entry; `tailscale_advertise` in the engine is a
projection. Same durable-intent and withdraw-first rules as the LAN door. `Engine.label`
prefixes refusals so a LAN failure is not reported as a sharing failure.

## Pairing (`pairing.py`, `connect.py`)

- `crucible://<name>@<host:port>/#<token>`. Not `http`: it carries a secret and must not be a
  link a browser follows. The name (contains `@`) and token are percent-encoded as RFC 3986
  userinfo; readers split on the **last** `@`. The `/` before `#` is part of the format.
  `@crucible/client`'s `parsePairing` is the other implementation, tested against the same line.
- `reachable_urls`: a wildcard bind becomes one URL per non-loopback IPv4 interface; a concrete
  bind (including `127.0.0.1`) is exactly that URL. `[server] advertise` entries are appended,
  never substituted, deduplicated in order.
- Pairing file: POSIX 0600 from creation under a 0700 home. Windows: `icacls /inheritance:r
  /grant:r <user>:(R,W)`; **if the ACL fails the file is deleted** (absent is handled; readable
  by all is a leak). `USERNAME` is read, never assembled.
- Pairing requests: the app polls with a separate high-entropy secret; requests are bounded,
  expire in 5 minutes and vanish on restart (flood guards, not auth). `PairingRequests` takes
  `open_pairing` with no default; the default lives only in `config._open_pairing`. With open
  pairing anyone who reaches the port gets the token, which also allows spending configured
  upstream accounts (keys are never returned).

## Interfaces (`interfaces.py`)

Addresses come from `getifaddrs(3)` via ctypes on Linux/macOS, and `Get-NetIPAddress` on
Windows. Not `gethostbyname(gethostname())` (resolver guesses), not a UDP connect trick (one
interface only), not psutil (a dependency). IPv4 only (IPv6 is a contract change), excluding
loopback, link-local and down interfaces. `sockaddr`'s family is a `u16` at offset 0 on Linux
but `sa_len, sa_family` bytes on BSD/macOS; the address is at offset 4 on both. A failed read
raises (`503 interfaces_unreadable`), never an empty list.

## Local lifecycle (`local.py`)

- `release_order` compares numerically (`1.0.10 > 1.0.2`); unparseable versions are refused,
  never assumed older. `installation.json` owns `release`.
- Loopback HTTP calls bypass proxy environment variables. Liveness probes time out at 3 s;
  `/v1/info` gets longer (it enumerates the catalogue). Exception text is wrapped with
  endpoint context.
- The started-engine version check applies only to an engine **this installation owns**. An
  installation with no `[server]` section runs no engine; an unreadable config is a refusal,
  not "ours". An engine that reports no version is not stale.
- On Windows a 401 during start is waited out: the tray rewrites the pairing file from the
  guest within seconds. `door_call` re-reads the pairing file, then asks the guest for its
  token (Crucible's distro, then the consented one).
- **Upgrade shutdown**: only a `child` engine is stopped; a guest (`wsl-unit` or `found`) keeps
  serving through the Windows swap and is carried by the new tray. The quit sends the handover
  header. A controller is gone when connects are **refused or reset** (ECONNRESET/WinError
  10054 is a normal exit) and its pid is not alive. Nothing installed is a successful stop.
- A refusal's own JSON body is surfaced, not `HTTP Error 409: Conflict`.
- `publish_installation` keeps `venv/bin/python` unresolved (resolving the symlink selects the
  base interpreter). `pythonw` is only for launching UI, never for JSON on a pipe.

## Services (`service.py`)

- `cuda-linux` → systemd (user unit, or system unit in WSL); `mlx-darwin` → launchd agent.
  Any other backend is refused.
- **Run the `crucible` console script, never `python -m crucible`**: a unit without a
  `WorkingDirectory` starts in `$HOME`, and a checkout named `crucible` there shadowed the
  package as a namespace package. `WorkingDirectory=CRUCIBLE_HOME` as well.
- **Record PATH and `CRUCIBLE_HOME`** in the definition. Services start with a bare
  `/usr/bin:/bin:/usr/sbin:/sbin`; the installing shell's PATH is the one tools were proven on.
  The server's own `bin/` is **appended**, never prepended (it holds `python3`, `uvicorn` and
  other scripts). Without `CRUCIBLE_HOME` a service serves `~/.crucible`, a different server
  with a different token.
- `Environment=` values are **double-quoted** (the directive is space-separated; WSL PATHs
  contain `/mnt/c/Program Files/...`). A value containing `"` or `\` or a newline is refused.
  `%` is doubled for systemd specifiers, and undone when reading back.
- **`Restart=always`** for systemd (ruling 2026-09-14): `systemctl stop` is still honoured
  (Restart= is not consulted for a unit systemd stopped), and a clean exit from SIGTERM must not
  leave the engine down. `RestartSec=2` is also the wait after "Restart engine". launchd keeps
  `KeepAlive: {SuccessfulExit: false}` because nothing else watches the Mac and `launchctl
  stop` must work.
- `enable --now` does nothing to a running unit, so `write_definition` reports whether the
  file was **replaced** and `install` restarts; a first install does not restart. Staged unit
  files use a unique `/tmp` name.
- Linger is reported, never enabled (it is the operator's decision); it says nothing about a
  system unit. Status fields are tri-state: `None` means nobody could be asked.
- launchd: create the `StandardOutPath` directory first (launchd fails opaquely otherwise);
  check load state before `bootstrap`; stop is `bootout` (SIGTERM would be restarted by
  KeepAlive). Read the plist with `plistlib`. Parse `systemctl show`, not `status`.

## Uninstall (`uninstall.py`)

- Order is the install order reversed: stop the engine (a unit whose `ExecStart` is gone is
  restarted every 2 s) → guest (`--wsl-too`) → on Windows, end the controller → service → envs
  → pairing file (a copy of config facts, so before the config) → config → working state →
  weights → packs (named, kept) → strangers (kept) → home if empty. "Empty" is computed from
  the plan, so a dry run agrees with the run.
- Stop, controller end and sharing withdrawal are prerequisites (`FATAL_BEFORE_REMOVAL`):
  failure there aborts before deleting files a live engine uses. Independent file failures
  continue.
- Weights are kept unless `--purge-weights`; the kept size is always reported. The weight
  directories are keyed exactly by `catalog.KINDS`.
- Deletes only what it can name, contained in `CRUCIBLE_HOME`; unknown entries are reported and
  kept. Never deletes the pack it runs from (`<home>/server`, `<home>/host`); the wrapper does.
- No prompts; `--dry-run` is the same plan never run; `--json` for apps.
- Supervisor keyed by platform, not backend (the config may already be gone; detecting a
  backend would probe a card).
- **On Windows nothing is force-killed.** The tray's tree holds the `wsl.exe … sleep infinity`
  session that keeps the guest's distro up and, on a native PC, the engine and its
  llama-server grandchild; a `taskkill /T /F` there (what 1.0.51 did, before the guest step)
  could idle-stop the distro under a running GPU job. The order is now: `stop-engine` runs
  `crucible local stop` through the live controller (pid from `host.pid`, never an image
  name), which stops the guest's unit or the Windows child cooperatively and pauses the
  watcher so it does not restart it; then the guest's own uninstall; then `stop-controller`
  runs `crucible local shutdown` (the orderly `POST /quit`). A controller that survives that is
  a fatal refusal naming its pid, `host.log` and the next command; the plan stops before any
  file is removed. A controller with no `pairing` file owns no engine, so `stop-engine` has
  nothing to ask and says so.

## Process groups and locks

- `procgroup.own_group`: POSIX `start_new_session=True`; win32 `CREATE_NEW_PROCESS_GROUP`
  (`start_new_session` is silently ignored there). `os.killpg` does not exist on win32.
- `ask_to_stop`: POSIX SIGTERM to the group; win32 `CTRL_BREAK_EVENT` to the child's group.
  A detached host has no console, so the break fails and the caller goes straight to
  `terminate_tree` (waited on well inside residency's 30 s clearance margin).
- Unknown platforms are refused by name.
- `processlock`: the kernel lock is the mutex; pid files are informational only (pid reuse
  cannot block it).
- Owned engines (`child_lifecycle.py`) get a private stdin pipe from the controller: any byte
  or EOF means graceful ASGI shutdown, so controller death stops them too. Read with `os.read`
  (a buffered stdin lock in a daemon thread can deadlock interpreter shutdown). A stop requested
  before uvicorn finishes starting is held until it is ready (early `should_exit` skips
  lifespan shutdown). Owned engines use the same keep-alive constant as `crucible serve`.
- `desktop.close_tray` waits for the tray process to exit: a Windows pack cannot be replaced
  while it runs.

## Host tools (`hosttools.py`)

`which()` searches `<home>/tools/bin` first, then PATH (minus Windows drives in WSL). Pinned
ffmpeg per platform in `FFMPEG_BUILDS` (linux-x86_64: BtbN static LGPL n8.1.3, rehosted on our
`tools` release; glibc ≥ 2.28; audio needs no GPL codecs). darwin-arm64 has no row and uses
Homebrew on PATH. ffprobe is placed beside ffmpeg. The digest is checked before anything is
placed; files are written under temporary names and renamed. Worker PATH puts the tools dir
first so libraries (urvc's `static_ffmpeg`) never download their own. Every "tool missing"
names the directories and PATH searched (`searched_note`).

## Logs holding many runs (`logtail.py`)

Engine and worker logs append, with each run starting `=== crucible `. Tails are read
backwards and stop at the last run header, so a previous run's fatal line (e.g. OOM) cannot
refuse the next start.

## CLI launcher (`launcher.py`)

Windows: `<home>/bin` plus a user PATH entry (broadcast so Explorer-launched processes see
it). POSIX: `~/.local/bin`, with an instruction rather than a profile rewrite. An existing
launcher not recorded as ours is replaced only if it runs `crucible.cli` out of this home
(older releases, hand shims); anything else is refused.
