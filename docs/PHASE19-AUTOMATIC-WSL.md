# Phase 19 — The Linux engine arrives by itself: automatic WSL, no commands, no tokens

**Owen, 2026-09-18:** *"this should be idiot proof. we should assume the user doesn't know
how to do it, and we shouldn't offer to let them do it themselves. if they want to install
crucible on windows and then wsl, and use the windows crucible to point to the WSL version as
the engine, they can. but we should assume they have no idea how to do it and it should do it
automatically."* And: *"we removed tokens. this system is supposed to work like ollama, which
doesn't require a token request/approval to connect. its protection is the system firewall."*

This document is the CONTRACT for the phase (ARCHITECTURE.md R1). It is a PLAN: nothing in
it is built. Three builds read it: the orchestrator (section 2), BookForge (section 4) and
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
| **the move** | PHASE15 4.7's engine task, unchanged: state table → `wsl --install` → import the `crucible` distro → server pack → migrate config → migrate weights → switch pairing. |
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

After `host.start()` and `_write_pairing`, before the door opens, `Host.decide_engine()`:

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

Import is "present is a no-op" (`installer.py:561`), the rootfs download is checksummed,
the pack install in the guest is the existing `install.sh`, migrate-config refuses a file
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

Its last lines (`sdk/bootstrap/scripts/install.ps1:253-256`) start the tray, start the
native engine and print *"The Windows engine works now; the optional Linux engine is
available from its console."* The tray's own 2.3 now starts the move, so the script's
closing sentence becomes *"Crucible is installed. It is setting up its Linux engine now;
the app you installed from will show its progress."* on a *can* machine and the `cannot`
sentence otherwise — the script reads the outcome the tray writes within a few seconds, and
says which. The script gains no logic of its own; it stays generated.

### 2.8 The native engine still starts first

Nothing here delays the first answer on `:7100`. The native engine is up within seconds of
install as today, and the move runs behind it. Two consequences the apps must honour:

- **The app's install is not finished until the outcome is terminal** (`done`, `cannot`,
  `reboot-pending`). An app that connected to the native engine and immediately pulled 20
  GB of weights onto Windows made migrate-weights expensive for nothing; so the app's setup
  face stays on the install step until then, and the engine's coordinate-on-connect model
  pulls are deferred by the APP until the outcome is terminal. The engine is not asked to
  refuse pulls — the app asked for the install, the app waits.
- **The engine switch mid-session stays invisible** as PHASE17 §6 promised: same port,
  same token, the SDK re-reads `/v1/info` after a refused connection.

### 2.9 The LAN door must know which engine it is opening

`crucible lan enable` adds `0.0.0.0:7100 → 127.0.0.1:7100` without checking the engine
(`crucible/lan.py`, `crucible/host/landoor.py`). On a native machine that row is the
self-loop that consumed 15.5k ephemeral ports on 2026-09-17. With native becoming a shape
real machines stay in, `lan enable` on `owner: child` refuses `lan_native_engine` and says
the native engine binds the LAN address itself when asked (`[server] host`), and the
firewall rule alone is what it adds. A test asserts the refusal.

**AMENDED 2026-09-18.** The first sentence above is no longer true of the code: `lan enable`
adds one row per enumerated address and refuses `portproxy_self_loop` for any listen set that
covers the connect address, so the self-loop is closed on a native machine and on WSL alike by
one rule about the listen set (PHASE15-HOST.md 4.1, amendment of 2026-09-18). What is still open
for this phase is the SECOND half — whether a native machine should get a portproxy at all, or
only the firewall rule and a wider `[server] host`. That is a question about which door is the
tidier one; the port exhaustion it was raised over is fixed.

### 2.10 Where the distro lives — RULING NEEDED

`wsl --import` lands the vhdx under `%LOCALAPPDATA%\Crucible` on C:. On Owen's PC the
fused-checkpoint fills traced to exactly this kind of vhdx growth. Options: (a) keep C: and
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

## 3. What the apps share

Both apps' Crucible setup becomes the same three faces, drawn from the same SDK calls:

1. **Local**: found on this machine (pairing file) → connected, nothing to do. Not found →
   one button, **Install Crucible**, which runs the whole of section 2 and shows it as one
   progress list: *Installing Crucible* → *Starting the Windows engine* → *Setting up the
   Linux engine* (bytes while the rootfs and packs download) → one of: *Done — running on
   the Linux engine* / *Restart Windows to finish* + **Restart now** / *Running on the
   Windows engine. This computer can't run the Linux engine: <sentence>* + **Try again**.
   No "Show what it does" with a command list; the steps ARE the progress list.
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
   2.11; SDK `installStatus`/`watchInstall`; a release tarball of `@crucible/bootstrap`
   for the apps to pin. Gate: `pytest` (one suite at a time, the flock rule), `npm run
   gen:install -- --check`, `npm test` in `sdk/bootstrap`.
2. **bookforge** — `feat/phase19-setup-surface`: section 4, pin the new bootstrap. Gate:
   `npx tsc`, `ng build`, the keeper, `dist` fresh. No packaging.
3. **foundry** — `feat/phase19-setup-surface`: section 5, same gate.
4. **Owen**: the acceptance run of 2.11 on a machine without WSL, then his in-app pass in
   both apps, then merges. Reviews and further keepers come after his pass, not before.

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

Everything else in this document is taken as ruled by the 2026-09-18 conversation and
needs no further word.
