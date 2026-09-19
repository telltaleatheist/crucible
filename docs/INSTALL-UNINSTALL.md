# Install and uninstall — the four machines, and the shape an app reads

> **Owen, 2026-09-14:** *"make sure there's an uninstall route for crucible as well. Once
> everything is working end to end I'll probably uninstall it on the pc (wsl and windows) and
> fully reinstall end to end as a test. Same with Mac. And it should have a cli install route as
> well, so I can install it on a digital ocean rented Linux droplet with a powerful gpu."*

This file owns the SEQUENCES. What each step does belongs to the thing that does it —
`sdk/bootstrap/src/steps.ts` for the install list, `crucible/uninstall.py` for its inverse,
`crucible/service.py` for the unit and the plist, `crucible/host/` for the tray — and
neither of those is restated here. What is here is the order, per machine, in both
directions, and the contract an app builds against.

Four machines, because Windows is two: the host that runs on it, and the WSL2 guest that
holds the engine.

| machine | backend | supervisor | `CRUCIBLE_HOME` |
|---|---|---|---|
| Owen's PC, Windows side | `llama-windows` | the Startup shortcut + the tray | `%LOCALAPPDATA%\Crucible` |
| Owen's PC, WSL2 guest (`crucible` distro) | `cuda-linux` | systemd user unit + linger | `~/.crucible` |
| the Mac Studio | `mlx-darwin` | launchd agent `com.crucible.serve` | `~/.crucible` |
| a rented Linux GPU box | `cuda-linux` | systemd user unit + linger | `~/.crucible` |

---

## 1. The install sequence, once, for every machine that is not Windows

`install.sh` is GENERATED from `sdk/bootstrap/src/steps.ts` (PHASE14-ENVPACKS.md 4a: a hand
install and an app-driven install "cannot differ"), so this list is the same one
`@crucible/bootstrap`'s `install()` walks:

```
host-facts        CRUCIBLE_HOME, the user, free disk, curl/tar/zstd
prerequisites     (install.sh only) the card, the driver, ffmpeg, the disk
server-pack       download + verify + unpack into <home>/server — the interpreter comes WITH it
init              crucible init --token … [--host …] [--port …]      (SKIPPED when a config exists)
install-<type>    crucible install <type>                            (only what --install named)
service-install   crucible service install
linger            loginctl enable-linger <user>                      (systemd only)
capability-write  crucible capability --write
done              crucible token --url  →  the crucible:// pairing line
```

`init` is skipped when `<home>/config.toml` exists, **and its token is kept**. That is what
makes a reinstall over an existing home leave every paired app paired.

WHICH RELEASE those packs come from is decided before the first step and is §6.5's: the
channel's `releases/latest`, unless an operator named an exact version. Nothing in the list
above defaults to the version the installer was built beside.

## 2. The uninstall sequence — the same list, read upwards

`crucible uninstall` (`crucible/uninstall.py`) performs it, and the wrapper removes the one
thing the verb cannot:

```
stop-engine       systemctl --user stop | launchctl bootout | taskkill /PID <tray> /T /F
wsl-guest         (--wsl-too, win32) the guest's own `crucible uninstall`, same flags
remove-service    the unit + daemon-reload | the plist + bootout | the Startup .lnk
remove-envs       <home>/envs
remove-pairing    <home>/pairing
remove-config     <home>/config.toml            ← the bearer token goes with it
remove-logs/jobs/uploads/downloads, host.pid, narrator-higgs-voices.json, narrator-reference.wav
weights:*         models, voices, rvc, rvc-base, denoise-models, engines
                  KEPT, with their size said out loud, unless --purge-weights
pack:server|host  KEPT always — the interpreter this command is running from
remove-home       <home>, only when the steps above left it EMPTY
--- then the wrapper ---
server-pack       rm -rf <home>/server <home>/downloads        (install.sh --uninstall)
host-pack         Remove-Item <root>\host, <root>\downloads    (install.ps1 -Uninstall)
home              rmdir <home>, only if it is then empty
```

Three rules the order comes from:

1. **Stop before you delete.** `service.py`'s unit is `Restart=always` with `RestartSec=2`
   (PHASE15-HOST.md 4.1's ruling), so a unit whose `ExecStart` has been removed is a unit
   that crash-loops every two seconds.
2. **The guest before the tray's service, and after the tray itself.** The tray BOOTS and
   WATCHES the guest (4.1). Remove the guest's unit while the tray is alive and its two
   recovery recipes fire against a Crucible that is being deleted.
3. **The pack last, and by the wrapper.** `<home>/server` holds the relocatable interpreter
   whose `crucible` the verb IS. Unlinking a live interpreter's own `site-packages` is
   undefined on POSIX and refused outright by Windows, so the verb names that directory,
   keeps it, and the script that unpacked it removes it afterwards.

> **THE TOKEN ALWAYS GOES, on every uninstall, including the default one.** `config.toml`
> holds it and `remove-config` is unconditional — an uninstall that left a bearer token on
> disk would be an uninstall that left a credential behind. So a reinstall mints a new one
> and **every paired app has to be re-paired**, which is not true of a reinstall over a home
> that was never uninstalled (there `init` is skipped and the token is kept). If the point of
> the exercise is to move a server rather than remove one, copy `config.toml` out first and
> `crucible init --config-from` it back — that is the same flag the Windows→WSL migration uses
> (PHASE15-HOST.md 4.3), and it carries the token, `[routes]` and `[upstreams]` and nothing
> else.

And two things it deliberately does not do:

- **It never clears `<home>`.** It dismantles it, entry by named entry, and removes the
  directory only when that leaves it empty. An entry Crucible did not write — the Mac's
  `~/.crucible/hf-token.txt`, which Crucible never read either — is reported as kept.
- **It never unregisters a WSL distro.** `--wsl-too` runs the guest's own uninstall inside
  the `crucible` distro and stops there; `wsl --unregister crucible` destroys that distro's
  whole ext4 and is irreversible, so it is printed as a line for the operator to run. Every
  other distro on the machine — Owen's Ubuntu — is never named at all.

---

## 3. The Mac Studio

Install (`mlx-darwin`, launchd, no host):

```bash
curl -fsSL https://github.com/telltaleatheist/crucible/releases/latest/download/install.sh | sh
crucible install llm
crucible install tts --narrator-engine higgs-v3
crucible install rvc                 # denoise shares this env
crucible service install             # FROM A LOGIN SHELL — see below
crucible capability --write
crucible doctor
```

> **`service install` from a LOGIN shell.** The unit and the plist record the INSTALLING
> shell's PATH (`crucible/service.py`'s header). A non-login `ssh mac '<cmd>'` has
> `/usr/bin:/bin:/usr/sbin:/sbin` and no `/opt/homebrew/bin`, so a service installed from
> one has no ffmpeg and refuses every `tts` job. `ssh mac -t 'bash -lc "…"'`.

Uninstall, weights kept:

```bash
~/.crucible/server/bin/crucible uninstall --dry-run     # read it first
curl -fsSL .../install.sh | sh -s -- --uninstall
```

This is `docs/MAC-PARITY-AUDIT-2026-09-14.md` §3's upgrade checklist reversed, and its
warning still applies in this direction: the Mac's job envs were venvs PARENTED on a conda
env. If that conda env still exists on the machine, removing it before `crucible uninstall`
kills `<home>/envs` out from under the step that would have reported them.

Survives a default uninstall: `models/` (33 GB), `voices/` (57 GB, 7 voices), `rvc/`,
`rvc-base/`, `denoise-models/`. A reinstall finds all of it and re-pulls nothing. The TOKEN
does not survive — see the box in §2 — so the apps get a new pairing line afterwards.

## 4. Owen's PC — the two halves, in this order

The Windows host manages the guest, so the Windows side goes **last** in both directions.

Install:

```powershell
irm https://github.com/telltaleatheist/crucible/releases/latest/download/install.ps1 | iex
# the tray appears; its menu's "Install the WSL2 engine…" runs 4.3, which
# imports the distro and runs install.sh INSIDE it. One sequence, the host's.
```

Uninstall — everything, weights kept, from an ordinary (non-admin) PowerShell:

```powershell
irm https://github.com/telltaleatheist/crucible/releases/latest/download/install.ps1 -OutFile install.ps1
.\install.ps1 -Uninstall -WslToo
# then, only if you want the distro itself gone (this destroys its ext4):
wsl --unregister crucible
```

`-WslToo` makes it: stop the tray and its child → the guest's own `crucible uninstall`
through `wsl.exe -d crucible` → the Startup shortcut → `%LOCALAPPDATA%\Crucible`'s contents
→ the host pack. Without it, only the Windows half goes and the guest keeps serving on
`127.0.0.1:7100`, which is a legitimate thing to want and is why it is a flag.

> **The tray gets no Uninstall item, and that is 4.2 rather than an omission.** The menu is
> a pure function of `(distro, engine)` with six ids — `open-console`, `install-engine`,
> `restart-engine`, `stop-engine`, `open-log`, `quit` — and a seventh that removes the
> program drawing the menu is a click that cannot report its own outcome: the tray is the
> first thing `crucible uninstall` stops. The door is the CLI and the operator page.

To uninstall only the guest, from inside it:

```bash
wsl -d crucible --exec bash -lc '~/.crucible/server/bin/crucible uninstall --dry-run'
```

> **The measured gotcha, 2026-09-14 (PHASE15 7b.4c).** The WSL user bus is unreachable from
> a `wsl.exe --exec` session, and linger was off. `crucible uninstall` inside the guest
> stops the unit through `systemctl --user`, so run it from a shell that has a user bus —
> `wsl -d crucible` interactively, or through the tray — and if `systemctl --user` cannot be
> reached the stop step is refused `stop_failed` by name and the other nine steps still run.

## 5. A rented Linux GPU box (a DigitalOcean droplet)

One line, and it is the whole install:

```bash
curl -fsSL https://github.com/telltaleatheist/crucible/releases/latest/download/install.sh \
  | sh -s -- --token "$CRUCIBLE_TOKEN" --host 0.0.0.0 --install llm --min-free-gib 60
```

It ends by printing the `crucible://` lines — one per address the box is reachable on —
and that line is what gets pasted into an app's Connect door. There is **no pairing file
across a network**: `<home>/pairing` answers *"an app on THIS machine wants in"* (3.6) and
holds the loopback line; a remote server is connected by the connect code and nothing else.

**What a droplet needs from Owen, and nothing here can supply:**

| | |
|---|---|
| **a token** | `--token` (or let it mint one and read it back with `crucible token --url`). Pick it before you start if two apps on two machines are going to use it. |
| **the bind address** | `--host 0.0.0.0`. The default is `127.0.0.1` and a loopback-bound droplet is a droplet nothing can reach. `crucible serve` prints "bound beyond loopback: the bearer token is the only lock." |
| **the port** | `7100` unless `--port` says otherwise. |
| **the firewall** | Crucible opens nothing. The droplet's cloud firewall AND `ufw` (if the image enables it) both have to allow inbound `7100` from wherever the apps are. Prefer a tailnet address over the public IP; the bearer is the only other lock. |
| **linger** | the script tries itself, then `sudo -n`, then prints `sudo loginctl enable-linger <user>`. Without it the systemd USER unit dies when the installing SSH session ends. On a droplet you have root, so this always succeeds — but check the line. |

Prerequisites are checked by name before anything is downloaded, and each is a refusal
rather than a guess:

| refusal | what it means |
|---|---|
| `no_nvidia_smi` | no NVIDIA driver on PATH. `cuda-linux` is vLLM, SGLang and torch on an NVIDIA card. |
| `no_nvidia_driver` | `nvidia-smi` is there and named no driver — usually an image with the CUDA userland and no kernel module. |
| `no_cuda_arch` | the driver would not report a compute capability. |
| `cuda_arch_too_old` | below 7.0, which is the floor: vLLM and SGLang ship no kernels under it. |
| `no_ffmpeg` | `--install` named `tts`, `asr`, `align`, `rvc` or `denoise` and there is no ffmpeg. Required only then; otherwise reported, because a bare `llm` droplet does not need one. |
| `disk_too_small` | less free than `--min-free-gib` stated. The pack's own size is checked separately (`pack_disk`); weights have no size until somebody names a model, so the only honest disk rule here is the operator's. |

### Installing a branch rather than a release

```bash
curl -fsSL .../install.sh | sh -s -- --from-source feat/phase6-remote-render --token "$T" --host 0.0.0.0
```

A separate ROUTE and never a fallback: a download that failed is still
`pack_download_failed`. It needs `git` and a `python3` on the box (the published pack brings
its own interpreter; a source build cannot) and refuses both by name.

### Service: user unit + linger, not a system unit

`crucible/service.py` writes a systemd **user** unit and only that, deliberately — a system
unit needs root to install, runs as a user with no HuggingFace cache and no env, and puts a
server holding one operator's models outside that operator's control. A droplet has a root
shell, which changes none of those three; what it changes is that `loginctl enable-linger`
always succeeds, which is the only thing a user unit was missing. A system unit is not
implemented and is refused by absence rather than half-supported.

### Taking the droplet down

```bash
~/.crucible/server/bin/crucible uninstall --dry-run
curl -fsSL .../install.sh | sh -s -- --uninstall --purge-weights
```

`--purge-weights` on a droplet, because a rented box is destroyed afterwards and the weights
on it were never the copy that mattered.

---

## 6. The apps' surface — RULED

### 6.1 The Uninstall door is for a LOCAL server, only, ever

**An app may offer Uninstall for a server it can prove is THIS machine's, and for no other.**
Concretely, one of:

- the server this app installed in this session through `@crucible/bootstrap`'s `install()`; or
- the server named by this machine's pairing file (`$CRUCIBLE_HOME/pairing`, else
  `%LOCALAPPDATA%\Crucible\pairing` on win32, else `~/.crucible/pairing` — 3.6's reader
  order); or
- on Windows, the host this app can reach at `127.0.0.1:7101`.

**Never for a registry entry.** A `crucible://` connect code in an app's server list says
where a server is and what its token is. It does not say whose machine it is on, and a
loopback-looking address proves nothing — a tailnet, a port-forward or an SSH tunnel all put
`127.0.0.1:7100` in front of somebody else's card. An app that offered Uninstall on a
registry row would eventually offer to delete a colleague's engine, and there is no
confirmation dialog that makes that safe. The rule is a rule about which rows get the button,
not about what the button asks.

There is also no remote uninstall route to misuse: `crucible uninstall` is a CLI verb and
`DELETE /v1/…` does not exist for it. An operator with a shell on the box is the whole
mechanism, which is the correct amount of ceremony for a command that deletes a service.

### 6.2 What an app INVOKES, per OS, verbatim

There is one surface and it is the CLI. The host's loopback door (`POST /install` on
`127.0.0.1:7101`, 4.3) is **not** extended with an uninstall, for two reasons: it exists so
that one implementation of the INSTALL sequence serves every caller, and an uninstall has no
second implementation to unify; and a door served BY the host cannot survive the act of
stopping the host, so its last event would never arrive.

**Windows** — the host pack's relocatable entry point. Not `crucible.exe`: pip's
`Scripts\*.exe` launchers bake the build tree's interpreter path into the binary and do not
survive the move to `%LOCALAPPDATA%` (PHASE15 4.4), so the pack ships `crucible.cmd` and that
is what runs.

```
%LOCALAPPDATA%\Crucible\host\crucible.cmd  uninstall --json --dry-run
%LOCALAPPDATA%\Crucible\host\crucible.cmd  uninstall --json [--purge-weights] [--wsl-too]
```

`LOCALAPPDATA` is read from the environment and never assembled from a username. The
existence of that `crucible.cmd` is what "the host is installed" means
(`crucible/host/paths.py`).

**macOS and Linux** — the server pack's console script, under the home the operator set:

```
${CRUCIBLE_HOME:-$HOME/.crucible}/server/bin/crucible  uninstall --json --dry-run
${CRUCIBLE_HOME:-$HOME/.crucible}/server/bin/crucible  uninstall --json [--purge-weights]
```

An app that wants the wrapper's half too (the pack, and the home) runs `install.sh
--uninstall` / `install.ps1 -Uninstall` instead — but those print prose, not JSON. **An app
that needs a machine-readable answer calls the verb; an app that wants the machine clean runs
the wrapper.** Doing both, in that order, is fine and is what the sections above do.

Exit codes: `0` when nothing failed, `1` when a step did (`ok: false` in the JSON names
which), `2` usage.

### 6.3 The JSON, field by field

`--json` prints one document. `--dry-run` produces the SAME document with `dry_run: true`
and every `done` false — it is literally the same plan object, unperformed, which is why
there is no second description of the work to drift.

```jsonc
{
  "dry_run": true,              // bool. true = nothing was touched.
  "home": "/home/telltale/.crucible",   // string. The CRUCIBLE_HOME this plan is against.
  "platform": "linux",          // "linux" | "darwin" | "win32".
  "mechanism": "systemd",       // "systemd" | "launchd" | "startup" (win32's Startup item + tray).
  "backend_kind": "cuda-linux", // string | null. READ from config.toml's [backend] kind,
                                // never detected — an uninstall does not probe a card, and
                                // null is the honest answer on a home with no readable config.
  "purge_weights": false,       // bool. The flag, echoed.
  "wsl_too": false,             // bool. The flag, echoed.
  "steps": [ /* see below */ ],
  "kept": {
    "weights_bytes": 18353126657,   // int. The sum of the KEPT weights:* steps.
    "paths": [                      // string[]. Every path this plan leaves behind.
      "/home/telltale/.crucible/models",
      "/home/telltale/.crucible/server"
    ]
  },
  "removed_bytes": 0,           // int. The sum of the remove steps that ran. 0 on a dry run.
  "ok": true                    // bool. false when any step's refusal was fatal.
}
```

One step:

```jsonc
{
  "name": "weights:voice",      // string, unique in the list. Stable; an app may switch on it.
                                // Shapes: "stop-engine", "wsl-guest", "remove-service",
                                // "remove-envs", "remove-pairing", "remove-config",
                                // "remove-<state>", "weights:<catalog kind>", "pack:<server|host>",
                                // "keep-unknown:<name>", "remove-home".
  "what": "voice subjects — KEPT (57.02 GiB). …",  // string. One sentence, for a person.
  "action": "keep",             // "remove" | "stop" | "keep". Three words and no fourth:
                                // `keep` is a RESULT this command decided, not the absence of one.
  "target": "/home/telltale/.crucible/voices",  // string. The path, unit name, label, pid or distro.
  "bytes": 61225693184,         // int, OPTIONAL. Disk the target holds. Absent when the target
                                // is not a path (a unit, a pid, a distro) — which is not zero.
  "done": false,                // bool, ALWAYS present. Did THIS run perform the step? For a
                                // `keep`, performing it means leaving the path and pricing it.
                                // Always false after --dry-run.
  "refused": {                  // object, OPTIONAL. Present when the step did not happen.
    "code": "weights_absent",   // string. Searchable, switchable.
    "message": "this server holds no voice subjects (… is not there)",
    "fatal": false              // bool. false = "there was nothing there", an ordinary outcome
                                // of uninstalling a half-clean machine. true = it could not be
                                // done, and `ok` is false.
  },
  "detail": ["removed directory /home/…"]  // string[], OPTIONAL. Lines the act printed.
}
```

Refusal codes an app will see:

| code | fatal | means |
|---|---|---|
| `service_not_installed` | no | no unit / plist at that path; nothing to stop or remove |
| `engine_not_running` | no | win32: no live pid in `<home>/host.pid` |
| `envs_absent`, `pairing_absent`, `config_absent`, `<dir>_absent`, `file_absent` | no | not there |
| `weights_absent` | no | this server holds no subjects of that kind |
| `home_not_empty` | no | `<home>` kept; the message names what is in it |
| `home_absent` | no | there is no `<home>` |
| `wsl_distro_absent` | **yes** | `--wsl-too` and no `crucible` distro. Names what `wsl -l -q` did list. |
| `wsl_not_here` | **yes** | `--wsl-too` off win32 |
| `wsl_unreadable` | **yes** | `wsl.exe` could not be asked |
| `wsl_uninstall_failed` | **yes** | the guest's own uninstall exited non-zero |
| `stop_failed` | **yes** | the tray or the unit would not stop |
| `host_no_localappdata` | **yes** | win32 with `%LOCALAPPDATA%` unset |
| `unsafe_target`, `step_failed` | **yes** | a removal refused, naming the path |

A fatal step does **not** stop the run (ARCHITECTURE.md R6): the other steps still happen,
their work is on disk, and `ok: false` plus the refusal is how an app learns which single
thing is left. A UI that shows the list and marks one row red is exactly right; a UI that
says "uninstall failed" and implies nothing happened is wrong.

### 6.4 What an app should draw

1. Call with `--dry-run --json`. Show `steps` as a list, `kept.weights_bytes` as a headline
   ("57.0 GB of weights will be kept"), and a checkbox for `--purge-weights` that re-runs
   the dry run so the number moves.
2. On confirm, call again without `--dry-run`, same flags, and show the same rows with
   `done` filling in.
3. Then run the wrapper if the app wants the pack and the home gone too, and say so — the
   verb's own output already names `pack:server` as kept and why.

### 6.5 WHICH release an app installs — the channel, and never an older one

> **Owen, 2026-09-18:** *"It shouldn't install an older Crucible. Maybe it should
> pull 'latest' and latest should be the latest build. Like how WordPress does it."*

Until this section an app installed the release its VENDORED bootstrap tarball
was cut at — BookForge 1.0.1, Foundry 1.0.0, against a server already at 1.0.2 —
and `install.ts` read `GET /v1/info`'s `server.version` only to print it
AFTERWARDS. One Crucible per machine is the ruling (§6.1), so either app's set-up
button could put the other app's engine back a version and say nothing. Five
rules replace that, and the first owns the other four.

#### 6.5.1 "Latest" has ONE owner, and it is the release channel

The channel is this repository's GitHub releases and the pointer is
`releases/latest` — the one `scripts/promote_release.py --publish` moves, with
`gh release edit --prerelease=false --latest=true`, after a candidate's packs,
its assets and a fresh-install smoke have been verified. Every cut starts life
`--prerelease --latest=false` (`scripts/release.sh`), so a tag that exists is not
yet a release anybody should install, and the promotion is the single act that
says otherwise.

One reader, one URL, spelled once in `@crucible/bootstrap`
(`sdk/bootstrap/src/channel.ts`) and nowhere else:

```
https://api.github.com/repos/telltaleatheist/crucible/releases/latest  →  tag_name "v1.0.2"
```

`releases?per_page=1` is NOT that pointer, and the generated installers read it
until today. It answers *"the newest tag created"*, which on any day between a
cut and its promotion is the unverified candidate the promotion gate exists to
keep off people's machines. `sdk/bootstrap/scripts/gen-install-scripts.ts` is the
one owner of `install.sh` and `install.ps1`, and both now read `releases/latest`.

#### 6.5.2 There is NO offline fallback; the one door is an exact version

The `crucible-bootstrap-<v>.tgz` in an app's `vendor/` is a LIBRARY pin — the
code that performs an install — and it has stopped being the thing that gets
installed. The version inside it says what the bootstrapper is, never which
release the server should be.

An unreachable channel is a refusal by name,

```
release_channel_unreadable: could not read the release channel at
  https://api.github.com/repos/telltaleatheist/crucible/releases/latest: <what failed>
```

and never a reason to install out of a cache, a vendored tarball, or the
bootstrapper's own `BOOTSTRAP_VERSION`. The ONE override is an operator naming
the release themselves — `install.sh --release <version>`,
`CRUCIBLE_RELEASE=<version>`, `install.ps1 -Release <version>`, or
`install({ release })`. Nothing chooses an older release silently, because a
silent downgrade is the defect this section is about and a second source would
be a second answer to "which Crucible is this".

What that override does is take the CHOICE off the channel; it does not take the
install off the network. `--release 1.0.1` still fetches `envpacks.json` and the
pack's parts from `releases/download/v1.0.1/`, so it is the door for an operator
whose channel read was refused rather than absent — a rate limit, a proxy that
allows `github.com` and not `api.github.com`, or a deliberate pin to a release
that is not latest. **A machine with no route to GitHub at all has no install
door today**, and naming that here is the point: the `-From <exact file>` an
operator would need — a local pack archive plus the manifest that says what its
sha256 must be — is a second install source and therefore a design of its own,
not a flag to be added in passing. Until it is designed, "offline" is a state
this installer refuses in, loudly, rather than one it half-serves out of a
cache.

#### 6.5.3 NEVER OLDER: the app asks the running engine BEFORE it spawns anything

An app about to install asks the engine on this machine what it is — `GET
/v1/info`, `server.version`, the call it already made for the report — and
compares it with the channel's latest before any process starts:

| running here | channel's latest | what happens |
|---|---|---|
| nothing | 1.0.2 | installs 1.0.2 |
| 1.0.2 | 1.0.3 | installs 1.0.3 |
| 1.0.2 | 1.0.2 | `crucible_already_latest` — "crucible 1.0.2 is running on this computer and the release channel's latest is 1.0.2 — there is nothing to install" |
| 1.0.2 | 1.0.1 | `install_older_than_running` — "the release channel's latest is 1.0.1 and crucible 1.0.2 is running on this computer; refusing to install an older engine over it" |

There is no `--force` beside it. A machine whose engine is newer than the channel
is a machine somebody put something on deliberately, and the answer to that is
6.5.2's exact-version door, not a flag on a button an app draws.

**On Windows the release an app resolved is not the pin, and that is worth
knowing before reading a Windows install log.** Three readers of the one pointer
stand between the gate above and a pack on the disk: the number the app resolved
picks WHICH `install.ps1` runs (`hostInstallCommand` builds
`irm <that release>/install.ps1 | iex`, and `iex` takes no arguments); that
script reads `releases/latest` for itself; and the host pack it lays down
installs the guest at the host build's OWN version, because the door accepts
`release` without reading it (`crucible/host/app.py`'s `DEFAULT_RELEASE`,
PHASE15 7b.5). They agree — all three are reading the promoted release — and the
gate above still runs first and still refuses. What is not true on Windows is
that the app's number PINS anything, and making it do so means
`hostInstallCommand` handing `-Release` to a script block instead of piping to
`iex`, and the host reading the field it already accepts. Written down because a
section that implied the pin held everywhere would be wrong about the one
platform where it does not.

#### 6.5.4 The bootstrapper refuses to install over a NEWER pack, and rolls back only to an exact version

`<CRUCIBLE_HOME>/server/.pack` records `release=`, so the bootstrapper knows what
is on the disk it is about to write over. `installPack` refuses
`install_would_downgrade` when that stamp is newer than the release being
installed. The ONE legitimate downgrade is an operator rollback and it is spelled
as one: `install({ release: '1.0.1', rollbackTo: '1.0.1' })`, which installs
exactly 1.0.1 over a newer pack and nothing else. A `rollbackTo` that does not
equal the release being installed is `rollback_version_mismatch` — there is no
"downgrade anyway" and no range, because the point of the flag is that the
operator names the version they are going back to. On win32 the same call is
`host_rollback_unsupported` and says so by that name: the host owns the install
sequence there (PHASE15 4.3) and its door carries release, job types, home and
bind and no rollback, so the refusal hands back the line that does it —
`install.ps1 -Release 1.0.1 -RollbackTo 1.0.1` — rather than walking an ordinary
install the host would then refuse from inside a process the caller cannot see.

This is 6.5.3's gate one layer down, and both exist on purpose: the app's gate is
about the engine ANSWERING on this machine, the bootstrapper's is about the PACK
ON THIS DISK. On a machine whose engine is stopped those are different facts, and
they are read by different code at different moments.

#### 6.5.5 SDK ↔ server compatibility rides on the API version, not on the release

`X-Crucible-Api` is the contract — `crucible/__init__.py`'s `API_VERSION`, which
is 1 — and `GET /v1/ping` and `GET /v1/info` both report `api_version`. The server
already refuses a client that sends none (`api_version_required`) or a different
one (`api_version_mismatch`). The SDK now refuses in the same direction at
connect and by the same name:

```
api_version_mismatch: this Crucible speaks API version 2 and this client speaks 1
```

rather than folding it into `not_crucible`, which said *"this address is not a
compatible Crucible engine"* about a Crucible and told nobody which half was
wrong.

So `@crucible/client`'s `SDK_VERSION` is a BUILD LABEL — what goes in the
User-Agent, what `scripts/release.sh` holds to the server's version at a cut —
and never a compatibility statement. BookForge's `package.json` said *"the service
installer and SDK must name the same release"*; that sentence is retired. The two
vendored tarballs are pinned together because they are ONE library release cut
together, and what makes a client able to talk to a server is `api_version` and
nothing else.

---

## 7. Round trip — the test Owen described

```powershell
# PC, both halves, weights kept
.\install.ps1 -Uninstall -WslToo -DryRun      # read it
.\install.ps1 -Uninstall -WslToo
# back again
irm .../install.ps1 | iex                      # tray, then its menu's WSL install
```

```bash
# Mac, weights kept
~/.crucible/server/bin/crucible uninstall --dry-run
curl -fsSL .../install.sh | sh -s -- --uninstall
curl -fsSL .../install.sh | sh              # back; a NEW token, because the old config went
crucible install llm && crucible install tts --narrator-engine higgs-v3 && crucible install rvc
crucible token --url                         # re-pair the apps with this line
```

The property being tested is that the second install finds `models/` and `voices/` already
there and pulls nothing — 90 GB on the Mac that does not move. What does NOT come back is
the bearer token: `remove-config` is unconditional, so every uninstall is also a rotation
and every app is re-paired from the new `crucible token --url` line. Budget one paste per
app per round trip, and read §2's box before doing this to a machine other people's apps
are pointed at.
