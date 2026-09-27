# Fixing the fresh-install snags: the plan

This plan covers every open snag in `FRESH-INSTALL-KYLIES-2026-09-26.md`, grouped into packages. The packages are split by the files they touch, so they can run in parallel. Owen asked for subagents on 2026-09-26 ("lets write a plan for fixing the snags ... and hand some off to subagents"). Each package runs in its own worktree, and the owens-pc Crucible session merges them.

## Already fixed (not in any package)

| Snag | What it was | Fixed in |
|---|---|---|
| #20 | empty WSL | 1.0.45 |
| #21 | the catalog key | 1.0.46 |
| #22 | cloud-init | 1.0.46; committed, unreleased |
| #24, #27 root cause, #32 root cause | root, then a relocated home with a new token (`guest_argv`) | 1.0.48 |
| #30 | no compiler (the `wheels` release) | 1.0.47 |
| #36 | truncated errors | 1.0.49 |
| #37 | `sudo -n` root | 1.0.49 |
| #38 | the WSLInterop binfmt.d entry | committed, unreleased |

Not ours, or by design:
- #13b: the plain-text token is by design.
- #13c: headscale's staleness.

## Rules for every package

- **No test suites.** Owen, 2026-09-26: tests run only when debugging. Don't run them, and don't write new ones unless a fix can't be checked any other way. Check the work by reading it and by running the code path by hand where that's possible on this box.
- **No deploys, no `ship.sh`, no releases cut.**
  - Don't touch kylies-pc or the Mac. kylies-pc and the Mac are running training-pc-1's RVC job.
  - Don't touch owens-pc's running Crucible either.
  - The one outward action allowed: package D may upload a pinned ffmpeg asset to a non-latest GitHub release on this repo (Owen's ruling on #25).
- **Commit on the worktree's branch** with `git commit -F <message file>`, ending with the session attribution lines. Don't merge or push.
- **Write like the surrounding code.** Docstrings carry the ruling and the date that caused them. Error codes are snake_case, each with one plain sentence.
- **Don't edit `FRESH-INSTALL-KYLIES-2026-09-26.md`.** Report back instead: per snag, what changed (file:line), what was checked, and what's left.

## The rulings that bind these packages

- **"Update and restart".** Every message that asks for a restart says **"Update and restart"** by name, and says why. WSL is installed by Windows' servicing step, and a restart that skips or defers pending updates doesn't install it. A restart Crucible triggers itself must run servicing.
- **ffmpeg.** One pinned static build, as an asset on Crucible's own GitHub release with a sha256, placed into `~/.crucible/tools/` by `crucible install` and found by `hosttools` for every job type. No system package, and no third-party download at run time.
- **Source-only packages** get prebuilt wheels on the `wheels` release, which recipes find through `--find-links`.
- **"Nobody is ever shown a command"** (PHASE19). A hand step is a bug.

## Packages

### A1. The WSL state machine: probe, outcome, resume
Files: `crucible/host/presence.py`, `wslstate.py`, `outcome.py`, `startup.py`, `tray.py`, `menu.py`, the restart budget in `installer.py`, and a `crucible` CLI verb for Try again.

| Snag | The fix |
|---|---|
| #15 | Key "is WSL live" on what's live: wsl.exe's `WSL_E_*` codes, and CBS `RebootPending` / `PendingFileRenameOperations`. Never on `InstallState` alone. |
| #10, #17 | Recognise the inbox stub and the `WSL_E_*` answers; log a one-line state, not wsl.exe's text. |
| #11 | `host.log` says, per feature, whether it was already on, enabled now, or needed elevation. |
| #19 | Allow 2–3 restarts, each judged by the live probe, before `cannot`. |
| #16, #18 | At every tray start, re-probe a `cannot` whose cause can change without a person (a restart still owed, WSL just committed), and resume when it can. A `cannot` is permanent only when its cause really is (virtualization off in firmware). Add Try again to the tray menu and as a CLI verb. |
| #8 | Resume without an interactive login (a scheduled task at logon or boot), or say plainly that someone must sign in. |
| #9 | Make sure `wsl-outcome.json` is written before the reboot stop is reported. |

The message TEXT belongs to A2. A1 only chooses which code to raise.

### A2. What the Windows installer says and prints
Files: `sdk/bootstrap/scripts/install.ps1`, the host message table (`crucible/host/errors.py`), the installer's console lines and progress in `installer.py`, and `host/log.py`.

| Snag | The fix |
|---|---|
| #6 | install.ps1 waits for the move to reach a terminal state (done, a restart owed, or cannot) and prints it plainly. |
| #7 | The wording depends on the caller; from a console, the console is the app. |
| #14 | "Update and restart" wording in every restart message: installer console, tray, `wsl_reboot_required`, `wsl_reboot_again`. `wsl_reboot_again` names the likely fix. |
| #28 | Don't print a stale outcome as current: stamp it with its `at`/`release`, or clear it when a new install starts. |
| #26 | No raw wsl.exe text on the console; a progress line during the 28 s distro import; free space reported for the real volume, not the vhdx maximum; no `peer_unreachable` noise against an engine it just stopped. |
| #12 | Pass `--no-warn-script-location`; write `host.log` ASCII-only. |
| #34 | No bare `RemoteException` lines. |
| #31 | The install error names the package that failed and why, in one line. The `jobenv` pip runner is D's file, so A2 coordinates the shape only if it's shared; otherwise #31 is D's. |

### B. Upgrades, relocation, and the guest
Files: `sdk/bootstrap/scripts/install.sh`, `sdk/bootstrap/src/*` (regenerate with `npm run gen:install`), `crucible/service.py`, `crucible/local.py`, the host's upgrade and hold code (touch only the parts of `tray.py` that A1 isn't touching), and `init` in `cli.py`.

| Snag | The fix |
|---|---|
| #39 | The shutdown before an upgrade uses the NEW code's door (stage the wheel first, or stop the unit with the new binary). |
| #35 | An upgrade hands the distro hold across; no ~50 s with nothing on :7100. |
| #23 | Confirm the orchestrator holds the distro up once the move is done. |
| #27 remainder | The tray reads the guest's pairing and token from wherever the guest lives now. An abandoned `/root/.crucible` is reported, or removed. |
| #32 remainder | The door or installer that finds the engine token rotated says so, and names the recovery (`init --force --config-from`, done by the product itself). |
| #29 | A Windows-side `crucible` shim forwards into the guest as the right user, so nobody spells a guest path. |
| #33 | `init` doesn't print the token. |

### C. LAN and pairing
Files: `crucible/lan.py`, `crucible/host/landoor.py`, `crucible/pairing.py`, `connect.py`, `sharing.py`, and `token --url` in `cli.py` / `apiclient.py`.

| Snag | The fix |
|---|---|
| #46 | `lan enable` checks the network profile of every interface it opens. It covers a Public profile or says plainly to switch it to Private. It tests reachability from outside instead of reporting `not_tested`. |
| #47 | Offer only addresses another machine can reach: no WSL vEthernet or NAT addresses. |
| #44 | Say "waiting for the administrator prompt on this PC's screen" the moment UAC is raised, and say plainly when there's no remote-only way. |
| #3 | LAN access is offered in the install or the console, not only through a CLI verb. |
| #4 | Report what PHASE19 §4/5 discovery and connect-request already has, and build the smallest piece that removes copying a token by hand. If that's large, write it up rather than half-build it. |

### D. Job envs and tools: install, ffmpeg, doctor
Files: `crucible/jobenv.py`, `hosttools.py`, `workerenv.py`, `envs/*`, `install` / `doctor` in `cli.py`, the `job_type_disabled` refusal in `api.py`, and `modules.py` / `weights.py` where pulls live.

| Snag | The fix |
|---|---|
| #25, #43 | Our own pinned static ffmpeg (linux x86_64; say what the Mac does), with a sha256, on a GitHub release of this repo (never "latest"). `crucible install` places it in `~/.crucible/tools/`, and `hosttools` finds it first for every job type. `/mnt/c/...*.exe` on the guest PATH is never used. Drop `static_ffmpeg` from the rvc recipe if nothing needs it. |
| #42 | `install` makes the running server re-read its job types, or restarts it. The refusal tells "not installed" apart from "installed, server not restarted". |
| #40 | A job type that's enabled but has no weights is a note in `doctor`, not unhealthy; or install doesn't enable denoise without a separator. |
| #41 | Suppress the HF "unauthenticated requests" warning. |
| #31 | Name the failing package and the reason, in one line. |
| #2 | A client that submits a job whose env is missing gets it installed (or a one-click in the operator console), not a shell command. Report what's there, and build the smallest honest piece. |
| #13a | Confirm `install rvc` flips `enable_rvc`. |

### E. The rvc job
Files: `crucible/jobs/rvc/*` and the rvc parameters in `api.py` / the SDK types. Coordinate with D only on `api.py` hunks.

| Snag | The fix |
|---|---|
| #45 | When an input NAME has no extension, take it from the uploaded file. Otherwise the error says it's the NAME that needs one. |
| #5 | The job chunks long inputs itself (quiet-point cuts, like asr's pieces) and stitches them sample-exactly. The output has the input's exact frame count: resample, then trim or pad; urvc runs about 20 ms short per 60 s and writes 48 kHz 16-bit. State the output format. Bounded memory for a 7–12 h master. |

### G. What a card can do (#48), and `crucible capability`
Files: `crucible/capability.py`, `accelerator.py` / `backend.py` for the probe, manifest fields in `manifests.py` / `models/*.toml`, and a proposal doc.

- Today `crucible capability` is arithmetic: total memory against the declared estimates. Add the declared half:
  - The probe reads compute capability (`nvidia-smi --query-gpu=compute_cap`) and derives the features it implies: bf16 ≥ sm_80, FlashAttention 2 ≥ sm_80, fp8 ≥ sm_89.
  - A manifest may declare what it needs; add fields only where an engine really needs them, like the Qwen ASR BF16 ruling on a Turing card.
  - Capability refuses with the number that stopped it, and `capability` / `doctor` show the card's facts.
- Write `docs/PROPOSAL-GPU-LADDER.md`: a measurement ladder run at install, covering what it runs, how long it takes, what it records (MEASUREMENTS.md basis `measured`) and how it feeds `capability`. The proposal only; Owen decides before it's built.

## What stays with the owens-pc session

- **#1 ("latest" isn't the build you need).** A process fix: promote each release once it's installed and smoke-passed somewhere, so `releases/latest` never lags what's deployed. No code, unless the merge shows the promote script needs a change.
- **Merging, conflicts, one release, and the kylies-pc / Mac deploys.** Only when their jobs are done, coordinated with training-pc-1.
- **Updating the snag log** with each package's report.

## Status, 2026-09-27 (everything merged to main, nothing released)

| Package | Merge | Snags |
|---|---|---|
| C LAN, pairing | b7e1360 | #44, #46, #47, #4 (`crucible pair`); #3 tray item left to Owen |
| A1 WSL state machine | 656e217 | #9, #10/#17, #11, #15, #16/#18, #19; #8 not built (sign-in stated in every message) |
| A2 installer words | bdfc57e, then bd9ac08 cherry-picked | #6, #7, #12, #14, #26, #28, #34; `Die` under `irm \| iex` (04ec29f) |
| D envs, ffmpeg | 1f30b0b | #25/#43 (`tools` release, ffmpeg n8.1.3), #31, #40, #41, #42, #13a |
| G card facts, ladder | 7217ae0, 98913ae | #48; quantize to 4-bit floor; install modal; `crucible ladder` |
| B upgrades, guest | 728db69 | #23, #27, #29 (`crucible guest`), #32, #33, #35, #39 |
| E rvc | bd9c18f, 08934f3 | #5, #45; recycle by memory; partial results kept; 48 kHz floor; input channels |
| desktop reserve | 53dc173 | measured at init, capped at 3 GiB; existing configs untouched |
| install on submit | 8a5d8de | #2 |
| virtualization re-check | c6375f6 | BIOS fix resumes by itself |

Owed before a release:
- `tests/test_lan.py` (C) and `test_tts_api.py` (a 16 GiB card now fits Higgs).
- An autouse patch of `ladder.measure_desktop_reserve` for tests that run `init` on a GPU box.

Owed on hardware:
- rvc seams on real speech, and urvc's memory growth on CUDA.
- The ladder's GPU rungs.
- Higgs quantized, and on Turing.
- The asr narrow widths.
- `capability --measure-desktop` on kylies-pc.

#1 stays a process fix: promote each release once it's installed and smoke-passed somewhere.
