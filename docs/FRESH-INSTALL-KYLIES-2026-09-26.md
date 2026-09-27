# Fresh install on kylies-pc, 2026-09-26: every snag, hangup and hand step

Owen, 2026-09-26: *"this is a good opportunity to see how crucible does when it's installed
on a fresh machine. it should be idiot proof. if we hit any snags or hangups, or we have to
do any configuration at all, write it down so we can address it when we're done setting it
up."*

The machine: kylies-pc (DESKTOP-U4AF49N), Windows 10 Home 19045, GTX 1660 SUPER 6 GB (driver
560.94), 16 GB RAM, one volume C: (283 GB free), LAN 192.168.68.88, Tailscale 100.64.0.7. The
box had just been switched on, with nothing but Claude open. No git, python, conda, ffmpeg or
node. The goal was a Crucible that serves `rvc` with the `sigma` model to owens-pc over the
LAN, for the deathstalker v13 sigma conversion.

The install was driven by kylies-pc-1 (the Claude session on the box), with steps from the
owens-pc Crucible session. Every entry says what happened, what a person had to do, and what
the product should do instead.

## Snags

### 1. "latest" is not the build you need
- **What:** `releases/latest` serves the last PROMOTED release, 1.0.40. The fixes this machine
  needs (the gcc build fix in 1.0.44, the CUDA 12 onnxruntime pin in 1.0.43) were in candidates
  that weren't promoted yet.
- **Hand step:** the install command had to name `-Release 1.0.44`.
- **Should be:** promotion keeps pace with what is deployed, or the installer takes the newest
  release that is installed somewhere. A fresh machine should never need a version number.

### 2. Job envs are not installed by the install
- **What:** the Windows install ends at a running guest engine with no job envs. PHASE19 2.12
  says "the app's coordinate step installs them on first connect". With no app on this box, the
  rvc env, its base assets and the model were all a command typed inside the guest:
  `wsl -d crucible -- bash -lc "~/.crucible/server/bin/crucible install rvc && ... rvc pull-base && ... rvc pull sigma"`.
- **Should be:** a client that submits an `rvc` job to a server without the env gets it
  installed (or a one-click in the operator console), not a shell command. PHASE19's own rule:
  "Nobody is ever shown a command."

### 3. LAN access is a separate, elevated step
- **What:** serving other machines on the LAN needed `crucible lan enable` on the Windows host
  (a WSL port forward plus a firewall rule, behind a UAC prompt).
- **Should be:** decided in the install or offered in the console. A box installed to serve other
  machines shouldn't need a CLI verb to be reachable.

### 4. The pairing line is fetched by hand
- **What:** registering kylies-pc on owens-pc needed `crucible token --url` run in the guest, and
  the line copied across.
- **Should be:** the LAN discovery or connect-request flow (PHASE19 section 4/5), so owens-pc sees
  kylies-pc and asks to pair, with no token handled by a person.

### 5. Long files must be split by the caller (not an install snag, found planning the run)
- **What:** the `rvc` job converts each input whole, and a 10-minute file grows the process by
  about 1.5 GB. So a 7-12 h master must be cut into 10-minute pieces and rejoined by the client.
- **Should be:** the job chunks and stitches long inputs itself, sample-exactly, the way the asr
  job cuts pieces. Owed: a measurement of whether urvc's output length equals its input's.

## Step 1: the host install (kylies-pc-1's report)

Timeline, local time:
- 21:29:48: install.ps1 downloaded.
- 21:29:50 to 21:30:59: the installer ran, about 70 s (python-build-standalone 45.7 MB at 40 MB/s,
  the 1.3 MB wheel, pip).
- 21:30:55: the tray started.
- 21:31:14: "engine: the move is started".
- 21:31:26: `wsl_reboot_required`, and everything stopped there.

### 6. The install ends needing a reboot, and nothing on the console says so
- **What:** the installer's last lines were "Crucible is ready in your notification area. It is
  setting up its Linux engine now; the app you installed from will show its progress", then exit
  0. Twelve seconds later the tray logged `wsl_reboot_required`. The only place that said "you
  must restart" was `host.log`, which kylies-pc-1 found by digging.
- **Hand step:** a person had to find out about, approve and perform a reboot.
- **Should be:** the installer waits until the move reaches a terminal state (done, a reboot
  needed, cannot) and prints it plainly, reboot included, on the console.

### 7. "the app you installed from will show its progress" is wrong for a hand install
- **What:** a PowerShell install has no app.
- **Should be:** the message depends on who called it; from a console, the console IS the app.

### 8. After the reboot a person must LOG IN at the console
- **What:** the tray resumes from a Startup-folder shortcut, which runs only at an interactive
  login. A headless or remote-only box sits there until someone signs in.
- **Should be:** a resume that doesn't need an interactive login (a scheduled task at logon or at
  boot), or the console message says plainly that someone must sign in.

### 9. `wsl-outcome.json` was not there when first looked for (CORRECTED)
- **What:** kylies-pc-1 found no `wsl-outcome.json` at about 21:32. After the reboot, the file
  existed with `"state": "reboot-pending"` stamped `21:31:26`. Either it's written later than its
  own stamp says, or first somewhere else, or the first look was wrong.
- **Should be:** check the write path. The record must exist the moment the reboot stop is
  reported.

### 10. The presence probe logs the whole `wsl.exe` usage text
- **What:** before WSL is live, `wsl -l -v` prints wsl.exe's usage screen, and the tray logged all
  of it as the "presence" value.
- **Should be:** recognise the inbox stub and log "WSL not installed yet".

### 11. It is unknown whether enabling WSL raised a UAC prompt
- **What:** after the install, Microsoft-Windows-Subsystem-Linux and VirtualMachinePlatform were
  enabled. Either they were already on (kylies-pc had WSL enabled on 2026-09-07) or they were
  enabled silently; `host.log` doesn't say which.
- **Should be:** `host.log` records whether each feature was already on, enabled, or needed
  elevation.

### 12. Cosmetic noise
- pip printed 11 "script ... is not on PATH" warnings for `host\Scripts`; pass
  `--no-warn-script-location`.
- `host.log` is UTF-8, and its em dashes read back as mojibake in PowerShell 5.1's `Get-Content`
  (the default encoding). Either write the log ASCII-only, or tell readers `-Encoding utf8`.

### 13. Observations to check, not yet snags
- `config.toml` has every `[jobs] enable_*` false, `enable_rvc` included; `install rvc` in step 2
  should flip it (on the PC, `crucible install rvc` wrote `[jobs] enable_rvc = true`).
- The auth token sits in plain text under `[auth]` with `open_pairing = true`; that's by design in
  PHASE19 (the token is an identifier, and the firewall protects the port).
- headscale lists kylies-pc as offline for 19 days while `tailscale status` on the box says online;
  that's not Crucible's, but it matters if training-pc-1 pairs over the tailnet address.

## Step 1, continued: the restart that didn't take

Timeline, local time:
- 21:32:59: Restart from the Start menu (User32 1074, RuntimeBroker, a restart).
- 21:33:23: cold boot (Kernel-Boot 27, type 0x0, not Fast Startup).
- 21:34:55: the tray resumed at login; someone logged in about 1 minute after the boot.
- 21:35:14: "engine: the move is resumed".
- 21:35:18: `wsl_reboot_again`, outcome `cannot`, and Crucible stopped.

### 14. A plain Restart does not commit the WSL feature while Windows Update has a restart pending
- **What:** the features were enabled with DISM on 2026-09-07 ("a reboot is necessary") and never
  committed. Tonight's earlier boot was a Fast Startup (hybrid) boot, which never commits servicing.
  Windows Update staged KB5126256 at 21:30 (also "reboot necessary"). CBS.log for the restart:
  "Deferring shutdown processing at users request", then "Deferring startup processing ... Aborted
  processing startup actions ... Reboot mark set". So the plain Restart committed NOTHING. Then
  Crucible's own DISM/WMI call at 21:35:15 re-pended the transaction.
- **Hand step:** a SECOND restart. CORRECTED by what happened next: Owen restarted again at 21:42:58
  with the same plain Start-menu Restart, and this time CBS ran servicing ("Begin shutdown
  processing", poqexec S_OK at 21:43:02-04, startup processing at 21:44:33). The Setup log says
  Subsystem-Linux and VirtualMachinePlatform were "successfully turned on" and KB5126256 was
  "Installed". So the FIRST reboot after the enable gets deferred (a WU package was staged minutes
  earlier) and the second commits. Crucible's `wsl_reboot_again` after one restart was premature:
  one more restart was the answer, and the message sent the person looking for help instead.
- **Should be:** Crucible performs the reboot itself (`shutdown /r /t 0` runs servicing), or tells
  the person to pick "Update and restart" when offered and never Shut down. On a fresh home PC,
  pending Windows updates are the normal case, not the exception.

- **Owen's ruling, 2026-09-26:** *"we should also make it clear that the user has to reboot and
  update, not just reboot. a lot of people avoid hitting update because its a pain in the ass, but
  the update logic is the route through which wsl installs and is necessary."* So every message
  that asks for a restart says **"Update and restart"** by name, and says why: the WSL feature is
  installed by Windows' servicing step, and a restart that skips or defers pending updates doesn't
  install it. That covers the installer's console line, the tray, `wsl_reboot_required` and
  `wsl_reboot_again`. If Crucible triggers the restart itself, it must be one that runs servicing.

### 15. The probe trusts `Win32_OptionalFeature InstallState = 1`, which is not "live"
- **What:** both features reported InstallState=1 (enabled) while CBS had
  Microsoft-Windows-Lxss-Package at "current: Install Pending", with no lxss.sys, LxssManager.dll or
  hypervisor. A probe keyed on InstallState (or DISM "Enabled") decides "features done, just
  reboot" and loops.
- **Should be:** key on what is live. `wsl.exe` answers `WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED` (a
  clean, machine-readable code), and CBS `RebootPending` / `PendingFileRenameOperations` say a
  restart is still owed.

### 16. `wsl_reboot_again` gives no action, and only an app can retry
- **What:** the sentence ends "this is a machine somebody has to look at". The tray never retries
  a `cannot` on its own (by design, PHASE19 2.5), and Try again is a button in an app. With no app
  on the box, the only retry was `POST http://127.0.0.1:7101/install` with `{"target":"wsl"}` and
  the engine's bearer, which a person would never find.
- **Should be:** the sentence names the likely fix (#14); Try again exists in the tray menu and as
  a CLI verb; and a `cannot` whose cause is plausibly transient (a restart still pending) is
  re-probed at the next login instead of recorded as permanent.

### 17. The presence line logs multi-line `wsl.exe` output again
- Now it's the `WSL_E_WSL_OPTIONAL_COMPONENT_REQUIRED` text; see #10. Key on the error code.

## Step 1, resolved: the second restart

- **Cost so far:** about 12 minutes from `wsl_reboot_required` (21:31:26) to servicing done
  (21:44:33), two restarts and two console logins, one of them "pointless" by the product's own
  lights.
- **After:** `wsl --status` exits 0 (Default Version 2), the hypervisor is present, lxss.sys is
  there, CBS RebootPending is cleared, and `wsl -l -v` says "no installed distributions".

### 18. With the machine fixed, the tray does NOT resume the move
- **What:** at 21:45:07 the tray started, logged the multi-line "no installed distributions" text
  as presence, and did nothing, because `wsl-outcome.json` still said `cannot` /
  `wsl_reboot_again`. Nothing tells the person they can proceed, and the retry is the
  `POST /install` a person would never find (#16).
- **Should be:** at every start the tray re-probes a `cannot` whose cause can change without a
  person (a restart owed, WSL just committed), and resumes the move when the probe now says it can.
  A `cannot` is permanent only for causes that really are (virtualization off in firmware).

### 19. `wsl_reboot_again` should allow at least one more restart
- **What:** Windows servicing can defer the first restart after a feature enable (#14). One restart
  isn't evidence that restarting won't help.
- **Should be:** allow two or three restarts, each probed by what is live (#15), before `cannot`.

## Step 1, the retry: a fresh machine cannot import the distro (A CRUCIBLE BUG, fixed in 1.0.45)

### 20. `wsl_read_failed` at step 2 of 11 on every fresh machine
- **What:** the Try again started at 21:45:21 and failed in under a second. Step 1 (`wsl-state`)
  read the machine correctly as `no_crucible_distro`. Step 2 (`import-distro`) ran `wsl -l -v`,
  which on a machine with WSL live and ZERO distributions prints "has no installed distributions"
  and exits non-zero, and it raised `wsl_read_failed` on the non-zero exit. The state probe had
  only survived the same answer by ignoring every failure, which would also hide a real one.
  owens-pc always had a distro, so this path had never run. Every first install on a clean
  Windows box would stop here.
- **And the message was wrong:** it showed wsl.exe's own text, telling the person to install a
  distribution themselves.
- **Fixed (1.0.45):** one reader for both steps, `presence.read_wsl_distros`. A failed listing is an
  empty list exactly when WSL has nothing registered under
  `HKCU\Software\Microsoft\Windows\CurrentVersion\Lxss` (the registry WSL itself uses, not
  wsl.exe's localised prose), and any other failure is still `wsl_read_failed`. kylies-pc-1 did
  NOT work around it (no throwaway distro), so the machine stayed a clean test.

## The 1.0.45 retry: the move fails at migrate-config (A CRUCIBLE BUG, fixed in 1.0.46)

Timeline, local time:
- 21:48:30 to 21:48:58: upgrading to 1.0.45 (28 s).
- 21:49:10: the retry started, and step 2 passed (#20's fix confirmed).
- 21:49:10 to 21:49:18: the Canonical rootfs downloaded, 357 MB in 8 s.
- 21:49:18 to 21:49:46: 28 s with no progress line while the distro imported.
- 21:49:51 to 21:50:01: install.sh ran in the guest.
- 21:50:03: migrate-config restarted the guest engine.
- 21:50:34: `guest_authentication_failed`.

### 21. The host misreads its own server's catalog, and blames the token
- **What:** the check after the guest restart polls `GET /v1/catalog`. The guest answered **200**
  every 2 s from 21:50:06 (its journal shows each one), but the host's `parse_catalog` read the key
  `subjects` and the server answers `rows`. So every 200 raised `catalog_unreadable`, and the
  step's single fixed message blamed the token. The tokens were identical (hash 9A735F51374A on both
  sides). kylies-pc-1 first suspected the Windows engine still holding 127.0.0.1:7100; the guest's
  journal shows the checks reached the guest, so that wasn't it here, but the next step, switching
  the pairing, deserves a look for the same reason.
- **Fixed (1.0.46):** the host reads `rows`; the check allows 180 s instead of 30; and it reports the
  last error by its own code: `guest_not_answering` with the real message, and
  `guest_authentication_failed` only for the server's `unauthorized`.

### 22. Canonical's WSL image runs cloud-init on every boot: 39 s before anything starts
- **What:** `systemd-analyze blame` in the guest: `cloud-init-local.service` 38.95 s. systemd sits
  at "initializing", and `crucible.service` (after network-online.target) waits behind it on every
  boot of the distro.
- **Fixed (1.0.46), for new imports:** the finish script writes `/etc/cloud/cloud-init.disabled`.
  kylies-pc's distro was imported before the fix and keeps cloud-init; the 180 s budget covers it.

### 23. The distro stops when idle
- **What:** after the failed move, the `crucible` distro was Stopped; each probe booted it, and WSL
  stopped it again about 25 s after the last session ended (the journal: started 21:58:35, stopped
  21:59:00). A guest engine only serves while something keeps the distro running.
- **To check once the move completes:** that the orchestrator holds it up.

### 24. The engine runs as root in the guest
- **What:** install.sh ran as root: `/root/.crucible`, a system unit with `User=root`, linger
  "granted to root". The distro's default user `crucible` has an empty home. Is that intended?
  PHASE19's vocabulary says the finish step creates a `crucible` user; nothing then uses it.

### 25. ffmpeg: not in the guest, and the rvc env's comes from a third party
- **What:** install.sh's prerequisites said "NO ffmpeg on PATH ... tts, asr, align, rvc and denoise
  will refuse until it is there". A fresh Ubuntu has none, and nobody would know to `apt install` it.
  The rvc env pins `static_ffmpeg==3.0`, a pip package that (to confirm) fetches ffmpeg binaries
  from its author's GitHub on first use; the other jobs look on the system PATH.
- **Owen's ruling, 2026-09-26:** *"that should be part of the rvc environment if we need it ... the
  environment is stored on gh releases right? ... downloading from gh releases is probably the most
  trustworthy/correct way."* So: one pinned static ffmpeg build, hosted as an asset on Crucible's
  own GitHub release with a digest, placed by `crucible install` into `~/.crucible/tools/`, and found
  by `hosttools` for every job type. No system package and no third-party download at run time.

### 26. Smaller things from the same run
- The installer's final console line still printed the raw "no installed distributions" wsl text
  in 1.0.45.
- 28 s with no progress line during the distro import.
- "954 GiB free at /root/.crucible" is the ext4.vhdx virtual maximum; C: has about 283 GB.
- The upgrade logged `claim: release did not land: peer_unreachable` against an engine it had just
  stopped itself (harmless, noisy).
