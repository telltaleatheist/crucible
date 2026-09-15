# Phase 17 — orchestrator and engine: the relation, made explicit

> **Owen, 2026-09-15 ~01:30:** *"I had considered making it so a second copy of crucible
> installs on wsl and is managed by the parent windows install. If we did it that way, we
> could build a windows version, a Linux version, and a Mac version… Create a
> relationship/handshake between crucible installs where one is the [orchestrator] and one
> the [worker]… The [orchestrator] is a hollow orchestrator, the [worker] does the heavy
> lifting… If there's no wsl, the windows copy is the [worker] and does the processing
> instead of orchestration. So we'd basically just add an orchestration layer to crucible."*
>
> And the names, ruled the same night: **"orchestrator and engine. we'll go with that."**
> Never master/slave. **engine** is the contract's existing word for the thing that serves
> job types on a backend; **orchestrator** is the new word for the thing that keeps time and
> plays nothing.

## 0. What this phase is, and what it is not

**It is not new machinery.** `crucible host` (PHASE15 section 4) has been an orchestrator
since the day it shipped: it decides which server this machine runs, boots it, holds its
distro open, watches its `/v1/ping`, writes its pairing file, and hands it the engine move.
What it has never had is a **name for the relation**, so nothing on the wire says which of
two Crucible processes is which, and no app, page or test can ask. Tonight the relation is
named, put on `/v1/info`, and made claimable.

**It is not a relay, and it never will be.** Owen, 2026-09-14: *"the windows side will
always configure the WSL side"*, and PHASE15 section 0: **control is Windows's, data is the
card's**. An orchestrator installs, starts, stops, restarts and reconfigures its engine. It
does not sit in the request path. No job, no chat completion, no TTS stream, not one byte of
a render ever passes through an orchestrator — a hop that copies bytes for nothing is two
versions and two health states on every stream, for something no app can see. **The apps
keep ONE address per machine and that address is the ENGINE's.**

**It is not a second install on Windows by default.** PHASE15 section 0's *one server per
machine* stands, unamended: one thing answers `:7100`. What Phase 17 adds is that the
process which is NOT that thing has a role of its own, with a door of its own, on 7101.

## 1. Vocabulary

| word | meaning |
|---|---|
| **role** | `engine` or `orchestrator`. A property of a **process**, never of an install: on a Windows machine with no WSL, one install runs both, as two processes. |
| **engine** | a Crucible process that serves job types on a backend (`cuda-linux`, `mlx-darwin`, `llama-windows`). Everything that exists today is one. It answers `:7100`. |
| **orchestrator** | a Crucible process with backend kind `orchestrator`, **zero job types**, which manages **exactly one** engine. Today that is `crucible orchestrator` (the Windows tray, formerly and still `crucible host`), answering its loopback door on `:7101`. |
| **claim** | the act by which an orchestrator tells an engine it manages it. `POST /v1/peer/claim`. |
| **`managed_by`** | on an ENGINE: which orchestrator claimed it, or `null`. |
| **owner** | PHASE15 4.1a's word for *how* the orchestrator holds its engine: `wsl-unit`, `child`, `found`. It decides what the orchestrator may DO to it. A distro Crucible did not import can reach `wsl-unit` only by CONSENT (2.5) plus a unit that answers. |

**`orchestrator` is a backend kind on the wire and nowhere else.** `detect_backend()` never
returns it, `crucible init --backend` never accepts it, and it is absent from
`crucible/backend.py`'s `BACKEND_KINDS`, which is the list of backends a *server* can be
configured as. It is a literal in the orchestrator's own `/v1/info`, because a client reading
`host.backend` must get an answer that is true, and "what accelerator does the thing that
plays nothing have" has exactly one honest answer.

## 2. The relation

### 2.1 An orchestrator CLAIMS its engine

```
POST /v1/peer/claim          (on the ENGINE)
Authorization: Bearer <the shared token>
X-Crucible-Api: 1

{"orchestrator": {"name": "crucible-orchestrator@owens-pc", "url": "http://127.0.0.1:7101", "version": "0.6.0"}}
```

Answer `200`:

```json
{"role": "engine",
 "managed_by": {"name": "crucible-orchestrator@owens-pc", "url": "http://127.0.0.1:7101"},
 "claimed": "2026-09-15T02:04:11Z"}
```

The engine records it and answers its own `/v1/info` with it. That is the whole of what a
claim does: **it is a statement of fact by the one process that knows it**, not a grant of
permission. An engine does not check the claim before doing anything, because there is
nothing an orchestrator asks an engine to do that an app may not also ask.

**The bearer is the SHARED token** — the engine's own, which the orchestrator already holds
(it reads it from the guest's pairing line, or it is the token in the config it wrote). There
is no second credential. PHASE15 4.7 already states the rule for the door in the other
direction: *"a caller that can reach the engine can reach this and nothing else can."*

**Refusals, and they are the relation's own names:**

| code | status | when |
|---|---|---|
| `peer_token_mismatch` | 401 | the bearer is absent, malformed, or not this engine's token |
| `peer_version_incompatible` | 426 | `X-Crucible-Api`'s major differs from the engine's, or is absent |
| `peer_already_managed` | 409 | a DIFFERENT orchestrator url holds the claim, and `force` was not sent |

The first two are `unauthorized` and `api_version_mismatch` with the relation's name on them,
and that is deliberate. The caller here is not an app: it is an orchestrator that has just
booted an engine and is telling it so. Told `unauthorized`, an orchestrator cannot tell *"the
token I copied out of the guest's pairing file is stale"* from *"some app's token is wrong"* —
one is its own bug and the other is not its business. One name per relation, so a log line
says which handshake failed.

**Re-claiming by the SAME url is success, not a conflict**, and it must be: an orchestrator
re-claims on every down→up edge of its watch (2.4), and an engine that restarted has
forgotten (2.3). Same url, new claim time, `200`.

**`force` overrides `peer_already_managed`, and only a person may send it.** Body
`{"orchestrator": {...}, "force": true}`. Two orchestrators claiming one engine is a machine
misconfigured — two trays booting one distro, or a second install nobody remembers — and the
right first answer is a refusal that names the other one, not a silent steal. `force` exists
because the person looking at the operator page can see both and decide. **No orchestrator
sends `force` on its own**, ever, on any code path; `crucible/host/app.py`'s claim passes it
as `False` and there is no setting that changes that.

### 2.2 `DELETE /v1/peer/claim` releases

Same auth, same version check. Body optional; when present it is the same `{orchestrator:
{...}}`.

- Held by this url, or no body → released, `200 {"role": "engine", "managed_by": null}`.
- Nothing claimed → `200` with `managed_by: null`. A release is a release; there is no
  `peer_not_managed`, because "there is no claim" is the state the caller asked for.
- Held by a DIFFERENT url, without `force` → `409 peer_already_managed`. You do not release
  somebody else's claim by accident.

The orchestrator releases on Quit. A tray that exits leaving `managed_by` pointing at a door
that no longer answers would be PHASE15 3.6's *"a file that exists and disagrees is worse
than none"*, one layer up.

### 2.3 A claim is LIVE state, never persisted

`managed_by` lives in the engine process and dies with it. It is not written to
`config.toml`, not to a sidecar, not anywhere. The reason is the one this system keeps
finding: **a fact with two owners and nothing comparing them** (`docs/ARCHITECTURE.md`). A
claim on disk survives the orchestrator that made it — uninstall the tray, reboot, and the
engine still says it is managed by a door that will never answer again. So the relation is
re-asserted rather than remembered: the orchestrator claims at presence-detection and on
every down→up edge, and an engine that restarts comes up unmanaged and is claimed again
within one watch tick (15 s).

An engine nobody claims is a complete, correct Crucible. `managed_by: null` is the Mac, the
droplet, and any `crucible serve` run by hand.

### 2.4 Health flows ONE way

The orchestrator polls. The engine learns nothing it does not need.

- `GET /v1/ping` — unchanged, unauthenticated, every 15 s (PHASE15 4.1's `WATCH_SECONDS`).
- `GET /v1/peer` — new, authenticated, the relation's own read:

```json
{"role": "engine", "managed_by": {"name": "…", "url": "…"}, "uptime_s": 8134.2}
```

`uptime_s` is from the engine's monotonic `started_at`, so it survives an NTP correction —
and it is what tells the orchestrator that an engine which is answering again is a NEW
process rather than the one it claimed. That is the signal that a re-claim is owed.

**There are no engine→orchestrator callbacks, and there will not be.** An engine that phoned
home would need to know its orchestrator's address, keep it fresh across restarts, and behave
when it is wrong — three facts to own for a push that a 15-second poll already delivers. An
engine's only knowledge of its orchestrator is the two strings it was handed.

### 2.5 An orchestrator may be GIVEN a distro — CONSENT (ruled 2026-09-15)

> **Owen, 2026-09-15 morning:** *"THE CLAIM MUST LAND ON THIS MACHINE, by
> CONSENT rather than by distro name."*

4.1a's `found` rule is right, and on the machine it was written for it is also
useless. Owen's PC runs its engine inside **`Ubuntu`**, installed by hand years
before Crucible existed. The orchestrator can see that engine, read its pairing
line, and name the unit that would restart it — and refuses all three, because
the only thing it knows about `Ubuntu` is that Crucible did not import it, which
is exactly what it knows about a stranger's distro. A rule that cannot tell
*"the distro this machine's owner keeps his engine in"* from *"a distro nobody
has spoken about"* is a rule missing an input, and the missing input is a
**person saying so**.

**The setting.** In the Windows home's own config, `%LOCALAPPDATA%\Crucible\config.toml`:

```toml
[orchestrator]
distro = "Ubuntu"
```

`[orchestrator]` because that is what the process reading it IS, and `distro`
because the value is a distro's name as `wsl -l -v` spells it. It is read by
`crucible/host/app.py`'s `consented_distro()` with `tomllib`, from the same
document `read_token()` reads, at startup and before the watcher is built —
because consent decides which distro the watcher is *about*.

**What consent does, and it is three things:**

| | with no setting | with `distro = "Ubuntu"` |
|---|---|---|
| `probe_distro()` | `absent` — no distro is NAMED `crucible` | `present` — the named one is looked for instead |
| owner of a running engine there | `found` | `wsl-unit`, **if the unit probe answers** |
| the claim (2.1) | never made | made, and true |
| `engine-restart` (4.2) | `engine_not_ours` | `user-unit-restart`, then 4.1's recipes |
| `user-bus-restart` | n/a | **still refused** |

**The unit is PROBED, never assumed.** `systemctl --user is-enabled crucible.service`
inside the named distro. A unit that answers makes the owner `wsl-unit`: the
claim is then a statement that is true, and `engine-restart` has a door. A unit
that cannot be read leaves the owner `found`, with the reason in the log, and
the machine behaves exactly as it did before anybody wrote the setting.
**Consent is permission, not evidence** — an orchestrator that read the
permission as the fact would claim an engine it has no way to restart, which is
2.1's `managed_by` naming a door that refuses every verb the field implies.

Two details the probe earns its keep on. **The exit code is not the answer:**
`is-enabled` exits non-zero for a unit that is merely `disabled`, and a disabled
unit is still a unit `systemctl --user restart` starts, so what it PRINTED is
read against `UNIT_STATES` and `not-found` is the one answer that means there is
nothing to manage. And **there is one probe, not two** — 7b.8 measured the root
door onto the same manager (`systemctl --user -M <user>@`) failing on the same
machine in the same minute, and the paragraph below shows the plain call
ANSWERING on that same distro once it is given the runtime directory. One
probe, and it is the one that works. What it *says* when it does not answer
goes in the log verbatim.

**THE BUS NEEDS `XDG_RUNTIME_DIR`, AND THAT IS THE WHOLE OF THE PROBE — measured
2026-09-15, 07:34-07:35.** `systemctl restart user@1000` as root created
`/run/user/1000/bus`, so the socket 7b.4c and 7b.8 went looking for now exists —
and a `wsl.exe --exec` session **still could not reach it**. Such a session gets
no logind seat, so it has no `XDG_RUNTIME_DIR`, and systemctl looks for the bus
at `$XDG_RUNTIME_DIR/bus` and nowhere else. From the same kind of session,
`XDG_RUNTIME_DIR=/run/user/1000 systemctl --user is-active crucible.service`
answered `active`.

So every user-manager call the orchestrator makes — the probe, `user-unit-start`,
`user-unit-restart`, and the menu's Stop — is built by ONE function,
`user_systemctl_argv()`:

```
wsl.exe -d <distro> --exec env XDG_RUNTIME_DIR=/run/user/<uid> systemctl --user <verb> crucible.service
```

`env VAR=value cmd` under `--exec`, because `--exec` is what stops wsl.exe
pre-expanding the variable on the WINDOWS side, where it is empty, and `env` is
how a value reaches a process with no shell to set it.

**The uid is READ, once, and 1000 is never assumed** — `wsl -d <distro> --exec id -u`,
cached on success only. A distro a person installed years ago can run Crucible
as any uid, and `/run/user/1001` is not `/run/user/1000`. `id` needs no bus, no
session and no unit, so it answers in exactly the state where every
`systemctl --user` call does not, which is what makes it the right thing to ask
first. When it cannot be read, nothing is built out of a guess: the probe
answers not-readable (so the owner stays `found`), a recovery that needs it is
logged `NOT RUN` and skipped, and Stop refuses and touches nothing.

`user-bus-restart` keeps its literal `user@1000` and takes no uid. It is a
SYSTEM-manager call, and it can only ever run in the distro Crucible IMPORTED,
whose rootfs 4b builds with exactly one non-root user — so 1000 there is a fact
about a rootfs this project makes, not an assumption about somebody's machine.

**This is a correction to 7b.8's reading, and worth keeping as one.** 7b.8 saw
`Failed to connect to bus: No such file or directory` from three different
doors and concluded the socket was missing. A missing socket and a missing
`XDG_RUNTIME_DIR` produce the *identical* sentence, so the message could not
tell them apart and re-reading it never would have. It was found by setting the
variable. The sentence still appears, verbatim, in the log when the socket is
genuinely absent — and now it means one thing.

**CONSENT NEVER WIDENS DESTRUCTION, and that is the half of this ruling that
matters most.** `user-bus-restart` (`systemctl restart user@1000` as root) stays
refused in any distro Crucible did not import, consented or not, by name —
`orchestrator_recipe_not_ours`, written into the log at the point the recipe
would have run rather than drawn as a disabled menu item. 4.1a's reason is
unchanged and is not a matter of permission: that command kills every process
uid 1000 owns in the distro, and on the night the rule was found that was a
five-thousand-step LoRA trainer. A person granting consent is naming a distro;
they are not enumerating what is running inside it at the moment a recovery
fires, and no wording of a setting could make them. So the predicate for a
destructive recipe is the ROOTFS (`distro == "crucible"`), never the flag.

There is nothing else on that list, and it was checked rather than assumed when
consent was built: `--terminate` and `--unregister` appear on no branch the
orchestrator can reach with a distro name it was *given* — the one
`wsl --terminate` in the generated `wsl_states.py` is 4c's `distro_not_systemd`
row and is hardcoded to `crucible`, `foreign_distro_not_systemd` is `instruct`
with no argv at all, and `crucible/uninstall.py` refuses `--unregister` by
ruling. `install-engine` needs no guard either: a consented distro reads
`present`, and 4.2 offers that item only when the distro is absent or unknown.

**A setting that is present and unusable is REFUSED, not ignored**
(`orchestrator_distro_invalid`). A person who wrote `distro = 4` meant to grant
something; an orchestrator that shrugged at it would quietly be the unconsented
one while its config said otherwise — `docs/ARCHITECTURE.md`'s one shape, a fact
with two owners and nothing comparing them. Absent is the only quiet answer. The
refusal is a log line and the tray still runs, unconsented, because a
malformed setting is a reason to manage nothing, not a reason to have no tray.

**What consent is NOT.** It is not a second address, not a second engine, and
not an install: the 4.3 sequence still imports and configures `crucible` and
nothing else, and a consented distro is never written to. It is one sentence
from the person who owns the machine, about which of the Linuxes on it this
orchestrator is allowed to treat as its own.

## 3. `/v1/info` gains the role

### 3.1 On an engine

Two additive top-level fields:

```json
{"server": {…}, "host": {…}, "job_types": […], "capabilities": […], "pages_engine": {…},
 "role": "engine",
 "managed_by": {"name": "crucible-orchestrator@owens-pc", "url": "http://127.0.0.1:7101"}}
```

`managed_by` is `null` when unclaimed. Nothing else about `/v1/info` changes.

### 3.2 On an orchestrator

The orchestrator answers `GET /v1/info` on **its own door** (`127.0.0.1:7101`), with the same
bearer as everything else on that door — the engine's token.

```json
{"server": {"name": "crucible-orchestrator@owens-pc", "version": "0.6.0", "api_version": 1},
 "host": {"platform": "win32", "arch": "AMD64", "backend": "orchestrator",
          "gpu": {"vendor": "none", "name": "", "vram_bytes": 0}},
 "role": "orchestrator",
 "job_types": [],
 "engine": {"name": "crucible@owens-pc-wsl", "url": "http://127.0.0.1:7100",
            "backend": "cuda-linux", "owner": "wsl-unit"},
 "capabilities": [ … the ENGINE's, verbatim … ]}
```

- **`job_types` is `[]` and that is the definition.** An orchestrator serves none. A client
  that POSTs a job to an orchestrator is refused by the absence of the route, not by a list.
- **`engine` is `null`** when there is no engine on this machine (PHASE15 4.1's `owner:
  none`). `owner` is 4.1a's word, spelled `wsl-unit` | `child` | `found` — `child` rather than
  4.1a's internal `host-child`, because on the wire the word "host" is the thing this phase
  renames and the orchestrator is the only possible parent.
- **`capabilities` is READ THROUGH, not copied.** The orchestrator re-reads the engine's
  `/v1/info` on every request and returns its `capabilities` block verbatim. It caches
  nothing. A cached capability list is the same defect in a third place: the engine pulls a
  model, the orchestrator keeps answering yesterday's list, and a client picks a model the
  engine has and is told it does not. When the engine cannot be read, `capabilities` is `[]`
  and `engine.url` still names where it should be — the orchestrator does not invent an
  answer for a server that did not give one.
- **`GET /v1/ping`** is on the orchestrator's door too, answering
  `{"crucible": true, "name": …, "api_version": 1, "role": "orchestrator"}`, so the
  "is this a Crucible" probe works on either address.

Nothing else is on the orchestrator's door. It is not a second API surface: it has `/install`
and `/restart` (section 4), `/v1/ping` and `/v1/info`, and an app that wants anything at all
follows `engine.url`.

### 3.3 How a client reads `role` — all-or-nothing, like `route`

PHASE15 3.3's reading rule, applied again and for the same reason. **A `/v1/info` document
with no `role` field comes from a server that predates this phase, and such a server IS an
engine with `managed_by: null`** — a fact the document states by its vintage, not a default
the client fills. A document that carries `role` and omits `managed_by` (on an engine) or
`engine` (on an orchestrator) is a defect and is refused by name. A `role` that is present and
is neither `engine` nor `orchestrator` is refused. **API version stays 1**: every field this
phase adds is additive.

## 4. The tasks that belong to the relation

A task about the RELATION runs on the orchestrator, because the orchestrator is the only
process that can run `wsl.exe`, prompt for administrator, respawn a child and survive the
engine going away. The engine still holds the task's **door** — an app has one address —
and relays, exactly as PHASE15 4.7 built it.

### 4.1 `engine` — the move. Unchanged.

`POST /v1/tasks {"type": "engine", "target": "wsl"}` → the engine hands it to
`$CRUCIBLE_HOST_DOOR/install` and relays the ndjson. Every word of PHASE15 4.7 stands,
including `engine_move_needs_host`, `host_unreachable` and `host_install_failed`. The
environment variable keeps its name (`CRUCIBLE_HOST_DOOR`) so that tonight's installed tray
and the servers it has already started keep working; the doc word for what sets it is now
"the orchestrator".

### 4.2 `engine-restart` — NEW

`POST /v1/tasks {"type": "engine-restart"}`. No fields: there is exactly one engine to
restart and the orchestrator knows which. The engine relays to `$CRUCIBLE_HOST_DOOR/restart`.

The orchestrator restarts **by the owner-appropriate means**, and the owner is 4.1a's:

| `owner` | what it does |
|---|---|
| `wsl-unit` | `user-unit-restart` — `systemctl --user restart crucible` in the distro, then wait for `/v1/ping`; on failure, PHASE15 4.1's two recovery recipes in 4.1's order (`user-unit-start`, then `user-bus-restart`). **`user-unit-restart` is NOT in `RECIPES`**: those are recoveries for an engine that should be up, and a restart built out of `boot()` would ping a running engine, succeed and change nothing — a button that did nothing precisely when it was most obviously pressed |
| `child` | terminate the child and spawn `crucible serve` again (`host-mode-respawn`) |
| `found` | **refused `engine_not_ours`**. A machine whose engine is in a distro Crucible did not import reaches `wsl-unit` — and therefore this restart — by CONSENT (2.5), never by the orchestrator deciding on its own |

**`engine_not_ours` (409) is 4.1a's rule with a wire name.** *"A `found` engine is watched
and never acted on."* The orchestrator did not start it, has no unit it may name and no child
it may kill, and on the machine this rule was found on, `user-bus-restart` would have killed a
five-thousand-step LoRA trainer. The refusal lives at the door, not only in the menu: a
disabled tray item is a drawing, and the thing that must not happen is the act.

`engine_restart_needs_orchestrator` (409) is the POST-time refusal when `$CRUCIBLE_HOST_DOOR`
is not set — the same fact `engine_move_needs_host` names, under a name that does not say
"move", because a server refusing a restart with a sentence about moving to WSL2 is a sentence
that sends a person to the wrong page.

**The task's last event may never arrive, and the client must expect that.** The relay runs
in the process being restarted. PHASE15 4.7 already set this precedent for the move (*"the
page, which lost its server for a few seconds at the switch-over, re-reads `/v1/info`"*), and
a restart is the same shape in less time: the client re-reads `/v1/info` and believes the
server, not the stream. A stream that ends with no terminal event is still reported as
`host_install_failed` by 4.7's table — that name is now slightly wrong for a restart, and it
is kept rather than forked, because one relay with one set of endings is worth more than a
second table.

### 4.3 `uninstall` does NOT go through the relation, and that is ruled

`docs/INSTALL-UNINSTALL.md` §6.2 already decided it, and Phase 17 does not reopen it: the
orchestrator's door **is not extended with an uninstall**, for two reasons that are still
true — the door exists so one implementation of the INSTALL sequence serves every caller, and
an uninstall has no second implementation to unify; and **a door served BY the orchestrator
cannot survive the act of stopping the orchestrator**, so its last event would never arrive
(and unlike a restart, there is nothing left afterwards to re-read).

The flag that reaches the engine is **`--wsl-too`** on `crucible uninstall`
(`crucible/uninstall.py`, step `wsl-guest`), and it runs the guest's own
`crucible uninstall` through `wsl.exe -d crucible --exec` directly. So the relation's side of
uninstall is one sentence: **the orchestrator's claim is not consulted and not required.**
`--wsl-too` acts on the distro Crucible owns by name, which is a narrower thing than
"whatever this orchestrator claimed" — and narrower is what an irreversible verb wants. An
engine the orchestrator merely `found` is never uninstalled by it, from the same rule as
4.2's.

### 4.4 OWED: the orchestrator can only be stopped by its own menu

Found 2026-09-15, upgrading the installed tray to this phase's pack. **There is
no way to stop a running orchestrator except clicking Quit in the notification
area.** `quit()` — which releases the claim (2.2), lets the held distro go
(PHASE15 7b.4c) and takes the host-mode child down with it — is reachable from
`menu.py`'s `quit` item and from nowhere else. The door serves `/install`,
`/restart`, `/v1/ping` and `/v1/info`; no signal handler is installed; `run()`
ends when `icon.run()` returns.

So every non-interactive stop is `taskkill`, which runs none of that: the claim
is left standing on an engine whose orchestrator is gone, the `wsl.exe
--exec sleep infinity` hold is orphaned as a session nothing owns, and
`host.pid` is stale (harmless — `acquire()` checks liveness). A tray installed
by a script, upgraded by a script and started by a script should be stoppable by
one.

**Owed, and deliberately not built tonight:** a `POST /quit` on the loopback
door with the same bearer as everything else on it — it is the transport this
process already has, it authenticates the way 4.7 already decided, and it can
run the whole of `quit()` before the process ends. A `SIGTERM`/`CTRL_CLOSE`
handler is the alternative and is worse here: Windows gives a console-less
`pythonw` no reliable console-control event, and `taskkill /F` delivers nothing
at all, so the handler would cover the case that already works and miss the one
that does not. The last event of a `POST /quit` never arrives, for 4.3's reason
— a door served BY the orchestrator cannot survive stopping the orchestrator —
and unlike a restart there is nothing left afterwards to re-read, which is why
this is a verb with an empty answer rather than a task.

## 5. The shapes a machine can be

| machine | processes | roles |
|---|---|---|
| Windows + WSL2 (Owen's PC, the intended shape) | the tray; the guest's `crucible serve` | orchestrator (`:7101`) + engine `cuda-linux` (`:7100`), `owner: wsl-unit` |
| Windows, no WSL | the tray; its child `crucible serve` | orchestrator (`:7101`) + engine `llama-windows` (`:7100`), `owner: child`. **ONE install, TWO processes** — `role` is per process, never per install. |
| Windows, an engine installed by hand (Owen's PC today) | the tray; somebody's `crucible serve` | orchestrator + engine, `owner: found`, **no claim**, nothing acted on |
| Mac Studio | `crucible serve` under launchd | engine `mlx-darwin`, `managed_by: null` |
| a rented Linux GPU box | `crucible serve` under systemd | engine `cuda-linux`, `managed_by: null` |

**A remote orchestrator is allowed by the protocol and is not used.** Nothing in `POST
/v1/peer/claim` requires the orchestrator's url to be loopback, so a Mac could in principle
manage a Windows engine across a tailnet. No code does it, no page offers it, and the one
thing that would have to be built first is a way for an orchestrator to run `wsl.exe` on a
machine it is not on — which is the whole of what an orchestrator does.

## 6. What an app does

**Connect exactly as today.** PHASE15 5.1's three ways are unchanged, and the pairing line an
app reads is **the ENGINE's** — on Windows the orchestrator writes the guest's line verbatim
(3.6, 4.1a), and 4.7's move keeps the port across the switch-over, which is precisely what
makes the switch invisible to an app that connected before it.

**The SDK's rule, written once so both apps read it the same way:**

- `info()` gains `role`, `managedBy` and `engine`. A document with no `role` reads as
  `role: 'engine'`, `managedBy: null`, `engine: null` (3.3's all-or-nothing rule).
- `engineOf(info)` is the helper:
  - `role === 'engine'` → `null`, meaning **you are already there, talk to this address**.
  - `role === 'orchestrator'` and `engine !== null` → the engine ref. **Follow `engine.url`
    ONCE, with the SAME token**, and talk to the engine for everything after that.
  - `role === 'orchestrator'` and `engine === null` → throws `orchestrator_has_no_engine`.
    A machine whose orchestrator has no engine has nothing to ask; that is a fact to show a
    person, next to the button that installs one.
- **Once, and never a chain.** An app follows one hop and no more. An orchestrator whose
  `engine.url` names another orchestrator is a misconfiguration, and a client that followed
  it would loop; the second document's `role` is checked and anything but `engine` is refused
  rather than followed.
- Nothing else about the SDK changes. `engineOf` is the only new function and every new field
  is optional-by-vintage, not optional-by-shape.

## 7. What `crucible host` is now

`crucible orchestrator`, with **`host` kept as an alias**. Both spellings run the same verb
tonight, and `host` is **deprecated in this doc** rather than in code: the Startup shortcut
installed on Owen's PC on 2026-09-15 has `-m crucible.cli host` baked into it, and
`--install-startup` still writes exactly that string, so a pack rebuilt after tonight starts
the tray the shortcut already points at. The alias is removed only when a release changes the
shortcut, and that is not tonight.

**The Python package is still `crucible/host/`**, and it is not renamed. It is imported by
`cli.py`, `api.py`, `tasks.py`, `uninstall.py`, `envpack`, `install.ps1`'s expectations and
`tests/test_host.py`, with two other agents committing into three of those files on the night
this was written. A package rename that touches every one of them, to change a word that no
wire and no operator sees, is the opposite of cheap. The module is the orchestrator; the
directory is called `host`; this sentence is the mapping.

## 8. Tests

`tests/test_peer.py` — the claim, the release, the three refusals, `force`, same-url re-claim,
`GET /v1/peer`, `role` and `managed_by` on `/v1/info`, and a pre-Phase-17 document reading as
an engine. `tests/test_engine_restart.py` — `engine-restart` per owner, `engine_not_ours` for
`found`, `engine_restart_needs_orchestrator` with no door. `tests/test_host.py` — the
orchestrator's `/v1/info`, read-through capability including the unreadable-engine case, the
claim performed at presence-detection for `wsl-unit` and `child`, **and never for `found`**,
and the release on Quit. And 2.5's consent: absent means unchanged, a named distro
with a readable unit means `wsl-unit` and a claim, a named distro whose unit cannot be read
means `found` with the reason, and `user-bus-restart` refused in a distro Crucible did not
import whatever the setting says. `sdk/ts/test/unit-phase17-orchestrator.test.ts` — the three fields,
the vintage rule, and `engineOf`'s three answers.
