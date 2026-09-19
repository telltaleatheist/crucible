# Phase 19 — The Linux engine arrives by itself: automatic WSL, no commands, no tokens

**Owen, 2026-09-18:** *"this should be idiot proof. we should assume the user doesn't know
how to do it, and we shouldn't offer to let them do it themselves. if they want to install
crucible on windows and then wsl, and use the windows crucible to point to the WSL version as
the engine, they can. but we should assume they have no idea how to do it and it should do it
automatically."* And: *"we removed tokens. this system is supposed to work like ollama, which
doesn't require a token request/approval to connect. its protection is the system firewall."*

This document is the CONTRACT for the phase (ARCHITECTURE.md R1). It is a PLAN: nothing in
it is built. **Revised 2026-09-19 for PHASE20 (code, not environments)** — the draft of
2026-09-18 was written against envpacks and a rootfs off our own releases, both of which
PHASE20 deleted that night. Section 2.12 lists what moved; every section it touches is
edited in place. Three builds read it: the orchestrator (section 2), BookForge (section 4) and
Foundry (section 5). Owen executes it when he is ready, through one Opus subagent per repo,
on a branch, and tests the result in-app himself before anything is merged.

## 0. What is decided, and what it corrects

**The topology does not change.** PHASE17 stands whole: one engine per machine on `:7100`,
one orchestrator on `:7101` that carries no data, apps dial the engine directly, and on the
same machine WSL2's own localhost forwarding is the whole of the Windows→guest path. The
"router" shape — a Windows server that proxies to the guest and serves itself when the guest
is down — was considered on 2026-09-18 and REJECTED for the reasons PHASE17 §0 already gives
(a byte-copying hop on every stream, two versions, two health states) plus one more: the
fallback it would offer is not capability-equivalent, since native Windows serves the llm
classes and `pages` only, so "the Windows copy handles it when WSL is down" can never mean
more than "text keeps working, slowly, on different weights, without saying so". That is a
silent fallback and the no-fallbacks rule forbids it.

**What changes is the DEFAULT.** `INTENT.md` says WSL is "an incentive but never a
requirement" and PHASE15's 09-16 amendment made native Windows the default with WSL an opt-in
button. Both stay true in letter and change in effect: **on every Windows machine that can
host WSL2, the orchestrator installs the Linux engine automatically, as the last step of the
same install, without a person choosing it.** Native Windows is the OUTCOME on a machine that
cannot (virtualization off, no disk, WSL1-only), and the app says so in one sentence. Nothing
is required of the user beyond the UAC prompt Windows itself raises and, when Windows demands
it, a restart.

**Nobody is ever shown a command.** Not `irm … | iex`, not `wsl --install`, not
`loginctl`. A command a person could run is a step the app should be running. Where the
app truly cannot (a BIOS setting), the sentence says what to change and where, and that is
the whole of it.

**Nobody is ever shown a token, and no app has a field for one.** The bearer still exists on
the wire and `open_pairing = true` hands it to whoever reaches the port
(`crucible/connect.py`); it is an identifier. The apps carry it inside the pairing line they
read from the pairing file or receive from a connect-by-address, and never render it, never
ask for it, never let it be pasted. What protects the port is the machine's firewall and the
operator's network, as with Ollama. Every remaining "access key", "connect code", "paste"
and "approve this code" surface in either app is deleted by this phase (sections 4 and 5).

**One mechanism for first install, reboot resume and later upgrade.** The orchestrator
decides at EVERY start whether this machine should be moving to the Linux engine, from facts
on disk and the state table, and acts. A fresh install, a tray coming back after the reboot
`wsl --install` demanded, and an old native install being upgraded are the same decision
seen three times. This is the fork PHASE15 4.3 left to Owen, now ruled: **the tray resumes,
not the app**, because the tray is the process that is there at login and already owns the
sequence, and an app that merely reads the outcome cannot double-run it.

## 1. Vocabulary

| word | meaning |
|---|---|
| **the move** | PHASE15 4.7's engine task, unchanged in shape: state table → `wsl --install` → import Canonical's Ubuntu 24.04 WSL image as the `crucible` distro and finish it (user, sudo, marker, systemd) → `install.sh` in the guest (pinned CPython from python-build-standalone, the release wheel, PyPI deps) → migrate config → migrate weights → switch pairing. **Job envs are NOT part of the move** (2.12): the app's coordinate step installs them on first connect to the guest. |
| **outcome** | a file in the host home, `wsl-outcome.json`, the ONE owner of "what happened to the move on this machine". Shape in 2.2. |
| **can / cannot** | the state table's verdict, partitioned: a code whose `action` is `run`, `run-elevated` or `reboot` is *can* (the tray carries it); a code whose action is `instruct` or `link` is *cannot* (a person must change something first). `wsl_ready` is *can* and trivially so. The partition is DATA in `wsl-states.ts` (2.1), not a list spelled in Python. |
| **declined** | `[orchestrator] wsl = "never"` in `%LOCALAPPDATA%\Crucible\config.toml`. The one way to keep a machine native on purpose. It is Crucible's setting, written by hand or by the page; neither app offers it, because the apps' setup has no choice to make. |

## 2. The orchestrator

### 2.1 The state table gains a partition

`sdk/bootstrap/src/wsl-states.ts` already carries `code`, probe, `sentence` and `action`
per state. Each row gains `automatic: boolean` — true when the tray can carry the machine
past this state without a person (today that is every row whose action kind is `run`,
`run-elevated` or `reboot`; false for `virtualization_disabled`, `pack_disk`,
`foreign_distro_not_systemd`, `guest_root_unreachable` and the rest of the instruct/link
rows). It crosses into `crucible/host/wsl_states.py` by the existing generator
(`scripts/gen-install-scripts.ts`; `npm run gen:install -- --check` refuses drift), and
the existing equality test between `wsl_states.py` and `wslstate.py` covers it.

### 2.2 The outcome file

`%LOCALAPPDATA%\Crucible\wsl-outcome.json`, written by the installer at every terminal
point and read by the tray at start and by the door on request:

```json
{ "state": "done" | "reboot-pending" | "cannot" | "failed" | "declined",
  "code": "<state-table code or task failure code, or null for done/declined>",
  "sentence": "<the 4c sentence verbatim, or null>",
  "at": "<ISO-8601>",
  "release": "<crucible release the move was for>" }
```

- `reboot-pending` REPLACES the bare `wsl-reboot-pending` marker
  (`crucible/host/installer.py:70`); the marker is deleted with it, one owner.
- `cannot` is terminal for the tray: it does not retry on its own. The apps show the
  sentence and a **Try again** control (2.5), because the person may have changed the BIOS.
- `failed` (a download died, the import timed out, `install.sh` failed in the guest) is
  retried by the tray at its next start, once; a second consecutive `failed` stays
  `failed` until a person presses Try again. The count lives in the file (`attempts`).
- `done` is terminal and the file stays as the record; `owner: wsl-unit` on presence is
  the live fact, the file is the history.

### 2.3 The decision, at every tray start

In the thread `main` already starts at the end (`Host.carry_guest_to_this_release`), after
`_presence_settled` — the owner is a MEASUREMENT the watch loop's first pass makes, and
FIX-35 (1.0.3) is what happens to a decision taken before it. One thread, one wait, two
branches: `wsl-unit` → the carry, as today; `child` → `Host.decide_engine()`:

```
owner is wsl-unit or found          → nothing (already there, or not ours: PHASE17 stays)
config says wsl = "never"           → write outcome declined if absent; nothing
outcome.state is cannot             → nothing (a person changes something, then Try again)
outcome.state is failed, attempts≥2 → nothing
outcome.state is reboot-pending     → the move, resumed (2.4)
otherwise (no outcome, or failed×1) → probe the state table:
     automatic == false             → write outcome cannot + code + sentence; nothing
     automatic == true              → the move, started
```

The move runs through the SAME `_sequence` the door's `POST /install` runs, under the same
`host._operation` lock, emitting the same events; the tray is simply the first caller. A
`POST /install` that arrives while the tray's own run is in flight gets the existing
`host_install_running` 409 and ATTACHES (2.6) instead of failing.

**The reboot is never taken by Crucible.** A `reboot` state writes `reboot-pending`, emits
the sentence, and stops. The app offers **Restart now** (4.2, 5.2), which runs
`shutdown.exe /r /t 5` as the interactive user — no elevation — only when a person presses
it. The Startup item brings the tray back; 2.3 sees `reboot-pending` and resumes.

### 2.4 Resume is idempotent because every step already is

Import is "present is a no-op" (`_import_distro`), the Ubuntu image is checked against
Canonical's own `SHA256SUMS` by filename, the guest install is the existing `install.sh`
(interpreter + wheel, PHASE20 §3, ~30 MB + 1 MB + PyPI), migrate-config refuses a file
without a token and is otherwise a re-write of the same three keys, migrate-weights pulls
before it deletes. Resume therefore means: run the sequence from the top. The one thing
resume adds is that `wsl --status` after the reboot must not report the same reboot state
again; if it does, the outcome becomes `cannot` with a sentence that says Windows asked for a
restart twice, which is a machine a person has to look at.

### 2.5 Try again

`POST /install` on the door, exactly as today. The apps' one WSL control is this call; its
label is **Try again** and it is shown ONLY when the outcome is `cannot` or `failed`. On a
`done` machine there is no control at all (the reverse move is still not in scope, PHASE15
4.7). On a native machine with no outcome yet the app shows the move's progress (2.6), not
a button.

### 2.6 The door can be watched, not only driven

`GET /install` on `:7101` returns `{ "running": bool, "outcome": <2.2 or null>,
"presence": <the tray's presence> }`. `GET /install/events` attaches to the running move's
ndjson stream from its current step (a ring of the last 200 events is kept so a late
attacher sees the step it joined at). `POST /install` while running answers 409
`host_install_running` and the client then attaches. `@crucible/bootstrap` gains
`installStatus()` and `watchInstall()` over these; `install()` on win32 becomes: run
`install.ps1` when the host pack is absent, then `watchInstall()` — it never posts the move
itself, because the tray already has.

### 2.7 `install.ps1` ends differently

Its section 7 starts the tray, starts the native engine and prints *"The Windows engine works now; the optional Linux engine is
available from its console."* The tray's own 2.3 now starts the move, so the script's
closing sentence becomes *"Crucible is installed. It is setting up its Linux engine now;
the app you installed from will show its progress."* on a *can* machine and the `cannot`
sentence otherwise — the script reads the outcome the tray writes within a few seconds, and
says which. The script gains no logic of its own; it stays generated.

### 2.8 The native engine still starts first

Nothing here delays the first answer on `:7100`. The native engine is up within seconds of
install as today, and the move runs behind it. Two consequences the apps must honour:

- **The app coordinates ONLY with the terminal engine.** Coordinate-on-connect (PHASE14
  4a) is what installs job envs and pulls weights, and under PHASE20 those are gigabytes
  from the mirrors and from Hugging Face. Run against the native engine on a machine
  that is mid-move, they land on Windows and make migrate-weights expensive for nothing.
  So the app's setup face stays on the install step until the outcome is terminal
  (`done`, `cannot`, `reboot-pending`), and coordinate runs once, against whichever engine
  is left standing: the guest on `done`, the native one on `cannot`. The engine is not
  asked to refuse anything — the app asked for the install, the app waits. The coordinate
  step's own `install` tasks (one per job type) are the LAST rows of the progress list
  (3.1), not something that happens after the page says done.
- **The engine switch mid-session stays invisible** as PHASE17 §6 promised: same port,
  same token, the SDK re-reads `/v1/info` after a refused connection.

### 2.9 The LAN door must know which engine it is opening

`crucible lan enable` adds `0.0.0.0:7100 → 127.0.0.1:7100` without checking the engine
(`crucible/lan.py`, `crucible/host/landoor.py`). On a native machine that row is the
self-loop that consumed 15.5k ephemeral ports on 2026-09-17. With native becoming a shape
real machines stay in, `lan enable` on `owner: child` refuses `lan_native_engine` and says
the native engine binds the LAN address itself when asked (`[server] host`), and the
firewall rule alone is what it adds. A test asserts the refusal.

### 2.10 Where the distro lives — RULING NEEDED

`wsl --import` lands the vhdx under `%LOCALAPPDATA%\Crucible\wsl` on C:, and under
PHASE20 that vhdx holds the Ubuntu image, the guest interpreter, EVERY job env (tts 5.3 GB
as an archive, more unpacked; llm 3.3; rvc 3.3; align 2.9; asr 1.3, `INSTALL-UNINSTALL.md`)
and the weights. On Owen's PC the fused-checkpoint fills traced to exactly this kind of
vhdx growth. Options: (a) keep C: and
let `pack_disk` refuse by name with the drive and the bytes — nothing to build; (b) let
`CRUCIBLE_HOME` on another drive move it — already true, undocumented, no UI; (c) a drive
picker in the app's install face. Recommendation: (a) for this phase, with the sentence
naming the drive, and (b) written into INSTALL-UNINSTALL.md. (c) is a choice, and the
install face is supposed to have none.

### 2.11 Tests, orchestrator side

`tests/test_host.py`: the decision table in 2.3, every row, with a fake state table and a
fake outcome file; a `found` engine is never moved; `declined` is honoured; `cannot` is not
retried; `failed` is retried once. `tests/test_installer.py`: the outcome file at every
terminal point; resume from `reboot-pending` re-runs the sequence; a second reboot demand
becomes `cannot`. Door: `GET /install`, attach mid-run, 409-then-attach. `test_lan.py`: 2.9.
Bootstrap: `installStatus` / `watchInstall` against a fake door.

**Acceptance, which no test replaces:** the move has never run end to end on a real machine
(PHASE15 7b.6, PHASE17 §8). It needs a Windows machine with no WSL — a Hyper-V VM on Owen's
PC with nested virtualization enabled can host WSL2 — and a fresh `install.ps1` there must
end with `Engine: WSL2` after the one reboot, watched from BookForge's setup page. That run
is the phase's gate and it is Owen's to schedule (GPU and machine use need his go).

### 2.12 What PHASE20 changed under this plan (2026-09-19)

PHASE20 (`docs/PHASE20-CODE-NOT-ENVIRONMENTS.md`, built and deployed as 1.0.3–1.0.5) made
a release carry code only. Everything else is fetched from its publisher at install: CPython
from python-build-standalone, the server wheel from our release, its deps from PyPI, the WSL
image from Canonical, job envs by `pip install -r <recipe>` from PyPI and the PyTorch and
SGLang indexes, weights from Hugging Face. What that changes here:

- **The move is small; the job envs are not, and they are not in the move.**
  `_install_job_types` installs nothing (the list is the coordinate records', which the
  Windows server has no door onto), so a finished move is a guest with an interpreter, a
  wheel and no job type. The gigabytes arrive when the app coordinates (2.8). Measured
  2026-09-19: host install ~70 s, carry 19 s, the PC's tts env 217 s from the mirrors. A
  fresh machine on a slow line is tens of minutes end to end, and the progress list (3.1)
  says so rather than looking stuck.
- **The decision lives in the carry thread** (2.3). PHASE20 gave the tray a start-time
  thread that waits for presence and acts on the guest; a second thread asking the same
  question is two owners of "what does this tray do at start".
- **Two downloads emit no bytes today.** `_import_distro` runs `curl.exe -o` blocking for
  up to an hour and the guest's interpreter fetch is inside `install.sh`; neither reaches
  the event stream as `bytes_done`/`bytes_total`, which `pull` already emits
  (`tasks.py`). The 340 MB image and the 30 MB interpreter get the pull shape. **pip has
  no byte total** — an `install` task streams pip's own lines (`_run_install_process`),
  and the app shows the line, honestly, not an invented bar.
- **The network probe proves one route and the install needs five.** `guest_no_network`
  fetches the release wheel off GitHub. A VPN or proxy that passes GitHub and blocks
  PyPI, `download.pytorch.org`, the SGLang index or Hugging Face passes the probe and
  fails minutes later inside pip. Either the probe touches each index a recipe names
  (one `curl -I` each, cheap), or a pip failure is parsed for the index it could not reach
  and named. Recommendation: the probe, because the sentence arrives before the download.
- **`pack_disk` never fires** — `_guest_ready` deliberately passes no `required_bytes`
  because the move itself is ~31 MB. The disk question moves to where the gigabytes are:
  `crucible install <type>` and `pull`. pip cannot size a recipe ahead of time, but the
  archive sizes PHASE20 MEASURED are cited numbers: `crucible install <type>` refuses
  `env_disk` by name when the guest has less free than the recipe's measured archive size,
  recorded in the recipe file's header (one owner, next to the pins). Ruling 6 in §7.
- **Canonical's `current/` is a moving pointer.** The sums file is matched by filename and
  a rename refuses; the bytes themselves change when Canonical republishes. That is
  Canonical's reproducibility, not ours, and the finish script writes what we need
  regardless. Recorded so nobody pins an image digest that will go stale in a month.
- **The bootstrap SDK is already a release asset** (PHASE20 §1). Step 6.1 no longer cuts
  anything for the apps; how the apps pin it is FIX-32 (PHASE20 §9), queued and separate.

## 3. What the apps share

Both apps' Crucible setup becomes the same three faces, drawn from the same SDK calls:

1. **Local**: found on this machine (pairing file) → connected, nothing to do. Not found →
   one button, **Install Crucible**, which runs the whole of section 2 and shows it as one
   progress list: *Installing Crucible* → *Starting the Windows engine* → *Setting up the
   Linux engine* (bytes for the Ubuntu image and the interpreter, 2.12) → one of: *Restart
   Windows to finish* + **Restart now** / *This computer can't run the Linux engine:
   <sentence>* + **Try again** / on to → *Installing what BookForge needs* (one row per job
   type the module asked for, pip's own lines underneath, no bar because pip has no total)
   → *Downloading models* (bytes, as `pull` already reports) → *Done — running on the
   Linux engine* (or *on the Windows engine* after a `cannot`). The list says at its top
   that a first install downloads several gigabytes and can take a while; it never looks
   stuck. No "Show what it does" with a command list; the steps ARE the progress list.
2. **Another computer**: one field, the address; one button, **Connect**. The app calls
   `/v1/connect` and, because pairing is open, is connected in two seconds. When an operator
   has set `open_pairing = false` on that engine, the existing code prompt appears — that is
   the only time a code is ever shown, and it is theirs to have chosen.
3. **Remove**: unchanged from today in both apps.

Gone from both: any field or paste box for a `crucible://` line or an access key, "Copy
connect code", "Test"/"Add" manual rows, the elevated-commands list, explainer copy that
says WSL is optional or "available afterward", and any button whose label is "Enable / Set
up WSL acceleration". Kept in both: the uninstall's WSL checkbox (it names a real fact).

## 4. BookForge

Files, by name, with what changes:

- `src/app/features/settings/components/crucible-doors.component.ts` — delete the paste
  disclosure (`:424-442`), the operator triple (`:467-479`), Test/Add (`:493-498`), the
  "Show what it does" block (`:591-609`), "Commands BookForge cannot run for you"
  (`:629`), and the "Match this code … Waiting for approval" copy (`:418-421`) in favour of
  Foundry's wording (5). The adopt-local door drops its name field: the engine's own
  name from `/v1/info` is the name. Add the progress list of 3.1 fed by
  `crucible:install-progress`, and the two controls **Restart now** / **Try again**.
- `src/app/features/settings/components/crucible-servers-panel.component.ts` — delete
  "Advanced manual connection" (`:434-466`) and "Copy connect code" (`:180`).
- `src/app/features/settings/components/crucible-engine-controls.component.ts` — the
  "Enable WSL acceleration" control becomes the outcome readout: nothing on `done`, the
  sentence plus **Try again** on `cannot`/`failed`, **Restart now** on `reboot-pending`.
- `electron/crucible/install.ts` — `driveCrucibleInstall` on win32: run `install.ps1` as
  today, then `watchInstall()` and relay until the outcome is terminal; the machine verdict
  sentence (`:729`) and the step list (`:777-782`) are rewritten to 3.1; the elevated list
  (`:810-815`) goes. Model pulls after connect are deferred until the outcome is terminal
  (2.8) — find the coordinate-on-connect caller and gate it on `installStatus()`.
- `electron/crucible/auto-connect.ts` — the two "Open Crucible…" sentences (`:21`, `:35`)
  say what to press in BookForge instead; there is no Crucible page to open from here
  (ruled 2026-09-17).
- `src/app/features/first-run-setup/first-run-setup.component.ts` — step copy at
  `:794-806` loses "skip this step and connect an engine later" only if Owen wants the step
  mandatory; otherwise unchanged. RULING: recommend keep skippable.
- Keeper: a `tools/test-crucible-setup-surface.js` that fails if `dist/renderer` contains
  any of: `Access key`, `Paste a connect code`, `crucible://name@host`, `Enable WSL
  acceleration`, `irm https://`, `Show what it does`. Derived from this section, so a
  reintroduced field fails by name.

## 5. Foundry

- `app/src/app/components/crucible-doors/crucible-doors.component.ts` — delete the
  Windows install copy that names `install.ps1` and shows the `irm` line (`:257-261`,
  `:286`) and the Mac copy that names `install.sh` (`:263-267`); the heading "The steps, in
  order. Nothing is installed without you." becomes the progress list of 3.1. The address
  door (`:144-194`) is already right and is the model BookForge copies.
- `app/electron/crucible-install.ts` — the copyable command constant (`:35-36`) goes, as
  does its surfacing; the step list (`:197-210`) becomes 3.1's; `elevated: []` stays empty
  and the renderer stops drawing the list. Same `watchInstall()` relay as BookForge.
- `app/src/app/components/setup-wizard/setup-wizard.component.ts` — the engine card's
  "Set up WSL acceleration" block (`:350-364`) and the progress lines (`:988-993`) become
  the outcome readout of 4's engine-controls; the `llama-windows` condition stays, since
  that is the only backend an outcome applies to.
- `app/src/app/pages/settings/ai-pane.component.ts` — the "Optional: use WSL
  acceleration" block (`:292-298`) goes; the readout lives on the servers card.
- `app/src/app/pages/settings/servers-card.component.ts` — "Advanced engine console"
  (`:195-196`): RULING NEEDED. BookForge removed its equivalent on 2026-09-17;
  `INTENT.md` allows occasional access to Crucible's UI. Recommendation: keep it in Foundry
  under Settings only, never in the wizard, so the two apps differ on purpose rather than
  by drift; or remove for parity. Owen's call.
- The hosted-in-BookForge guard ("Install Crucible from BookForge.") stays.
- Keeper: the same surface test as 4, over Foundry's renderer output.

## 6. Order of work, and who does it

One Opus subagent per repo, each on a branch in a worktree, Crucible first because the
apps' SDK calls (2.6) must exist before the apps can be built against them:

1. **crucible** — `feat/phase19-automatic-wsl`: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.9,
   2.11, and 2.12's four builds (bytes on the two downloads, the per-index probe,
   `env_disk`, the decision in the carry thread); SDK `installStatus`/`watchInstall`.
   Gate: `tests.sh --changed` (PHASE20 §7; one suite at a time, the flock rule), `npm run
   gen:install -- --check`, `npm test` in `sdk/bootstrap`. Ship it as a patch through
   `ship.sh`; nothing here is an environment.
2. **bookforge** — `feat/phase19-setup-surface`: section 4, pin the new bootstrap. Gate:
   `npx tsc`, `ng build`, the keeper, `dist` fresh. No packaging.
3. **foundry** — `feat/phase19-setup-surface`: section 5, same gate.
4. **Owen**: the acceptance run of 2.11 on a machine without WSL, then his in-app pass in
   both apps, then merges. Reviews and further keepers come after his pass, not before.

**Foundry lands in BookForge only by re-vendor.** `foundry-app/` is a mechanical copy
(`foundry-app/VENDORED.md`), and it carries Foundry's `crucible-install.ts` with the
`irm … | iex` constant. Step 3 therefore finishes before BookForge's keeper (section 4)
can pass: Foundry's setup surface is fixed on Foundry's branch, re-vendored into BookForge
at that sha, and only then does BookForge's renderer scan come up clean. Editing
`foundry-app/` directly is the thing VENDORED.md forbids. The hosted install door is not
reachable from BookForge anyway (the "Install Crucible from BookForge." guard), so the
vendored strings are a keeper matter, not a user-facing one.

Nothing in 1–3 touches the wire between apps and the ENGINE; `/v1/*` is unchanged. The only
new surface is the orchestrator door's `GET /install` and `GET /install/events`.

## 7. Rulings needed from Owen before step 1 starts

1. **Restart now** as a button the app offers (runs `shutdown.exe /r /t 5` as the user,
   only when pressed). Recommendation: yes.
2. **Where the distro lives** (2.10). Recommendation: C: with the drive named in the
   sentence; document `CRUCIBLE_HOME`.
3. **Foundry's "Advanced engine console"** (5). Recommendation: keep, Settings only.
4. **The acceptance machine** for a no-WSL run (2.11): a Hyper-V VM on the PC, or another
   box.
5. **The first-run step stays skippable** in BookForge (4). Recommendation: yes.
6. **`env_disk` floors from the measured archive sizes** written into each recipe's
   header (2.12). Recommendation: yes; it is the one cited number there is, and a floor
   that is too low still beats a pip that dies at 4.9 GB.

Everything else in this document is taken as ruled by the 2026-09-18 conversation and
needs no further word.
