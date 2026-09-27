# Phase 11: `crucible service`

The verb behind PHASE5-APPS.md section 6.0's ruling. **Owen, 2026-09-13, asked whether a
local Crucible is a machine service or an app's child process: "service".** This is the
half of that ruling that lives in this repo; the bootstrapper half (`@crucible/bootstrap`,
whose `ensureRunning()` is `systemctl --user start` / `launchctl kickstart` and not a
spawn) is still phase 5's.

Short by design. The unit and the plist are the artefact, and they are in
`crucible/service.py` as pure functions with byte-for-byte tests, because nobody sees a
service definition until the machine reboots.

## 1. The verbs

```
crucible service install     write this host's definition, enable it, start it
crucible service uninstall   stop it, forget it, remove the definition
crucible service start       make sure it is running
crucible service stop        stop it without forgetting it
crucible service status      running or not, with the pid and the definition's path
```

Every one is idempotent. `status` **exits 0 only when the service is running**, so a
script can gate on it the way it gates on `crucible doctor`; a service that is installed
and stopped is a server nothing can reach, and reporting that as success would make the
verb useless to the only thing that would automate it.

## 2. Two mechanisms, one per backend, and no third

| Backend | Mechanism | Definition |
|---|---|---|
| `cuda-linux` | systemd **user** unit | `~/.config/systemd/user/crucible.service` |
| `mlx-darwin` | launchd **agent** | `~/Library/LaunchAgents/com.crucible.serve.plist` |

A backend with no mechanism is refused by name. There is no generic "write an init
script" path: DESIGN.md section 2's list of backends is short and explicit, and so is
this one.

User-level on both, deliberately. A system unit would need root to install, would run as
a user with no conda env and no HuggingFace cache, and would put a server holding one
operator's models outside that operator's control. The cost is `linger`, section 4.

`ExecStart` is `<env>/bin/crucible serve --host <h> --port <p>` — **the console script,
never `python -m crucible`**; section 3a is the measurement that settled that. The host
and port are **read from `config.toml` at install time**. A config edited afterwards is
not what the service serves until `install` is run again, and `install` says so when it
finishes. The alternative — a unit that re-reads the config — is a unit whose behaviour
changes without anybody installing anything.

`WorkingDirectory` is `CRUCIBLE_HOME` on both mechanisms. A user unit otherwise starts in
`$HOME` and a launchd agent in `/`; a server's cwd should be its own state directory,
which is the only directory it owns and where every relative path it writes belongs.

`Restart=on-failure` (and launchd's `KeepAlive: {SuccessfulExit: false}`) rather than
`always`: a server that exited 0 was stopped on purpose, and restarting it would make
`crucible service stop` a thing that does not work. On launchd that is also why `stop` is
a `bootout` and not a signal — an agent killed with SIGTERM exited unsuccessfully, so
launchd would start it straight back up. The plist stays, so `RunAtLoad` brings it back
at the next login and `start` brings it back now.

## 3a. `python -m crucible` crash-looped the first real install

**Measured on Owen's PC, 2026-09-13, by installing the unit this doc's first version
described.** It ran `python -m crucible serve`, and the service came up and died, over and
over:

```
ImportError: cannot import name 'load_all_voices' from 'crucible.voices' (unknown location)
```

A systemd user unit with no `WorkingDirectory` starts in `$HOME`. `$HOME` on that box
holds the Linux checkout — a directory named `crucible`. And `python -m` puts the cwd on
`sys.path`, so `crucible.voices` resolved to `~/crucible/voices/`, the **manifest
directory**, as a namespace package with `__file__ is None`, instead of to
`crucible/voices.py`. Reproduced exactly: from `$HOME`, `import crucible.voices` gives
`__file__ None`; from `/`, the real module.

Two fixes, both applied, because they answer two different questions:

1. **`ExecStart` runs the console script.** A console script's `sys.path[0]` is the
   script's own directory, never the cwd, so what it imports does not depend on where it
   was started. `crucible/service.py` derives it from `sys.executable`'s directory and
   **refuses by name when it is not there** — a fallback to `python -m` would reinstate
   the bug on the one machine that has it.
2. **`WorkingDirectory=<CRUCIBLE_HOME>`**, on the unit and on the plist.

## 3. The PATH is recorded, and this is the bug that made it necessary

Measured on Owen's Mac on 2026-09-13, against a real `crucible doctor` over a **non-login**
shell, on a host where every env was installed:

```
job tts: NOT READY — there is no ffmpeg on PATH
```

ffmpeg was at `/opt/homebrew/bin/ffmpeg` the whole time. The PATH that shell searched was
`/usr/bin:/bin:/usr/sbin:/sbin` — which is the same bare PATH **a launchd agent and a
systemd user unit are started with**. A service installed without a PATH would have failed
in exactly that way, and the failure would have arrived as a refused job rather than as
anything an operator could see at install time.

Two things follow, and both are built:

- **`install` records the installing shell's PATH** into the definition
  (`Environment=PATH=…` / `EnvironmentVariables PATH`). That shell is the one the operator
  proved the tools in — they just ran `crucible doctor` in it. Hardcoding `/opt/homebrew/bin`
  would fix one Mac and would be a second owner of a fact the environment already holds.
- **Every "tool missing" refusal names the PATH it searched.** `asr`, `align` and `tts` all
  refuse `ffmpeg_missing`, and all three now carry `(PATH searched: …)` plus a `path` in
  the error's details. "There is no ffmpeg on PATH" is true and useless; the reader's next
  question is always *which PATH*, and on a host with two shells and a service manager
  that is a real question with three answers.

`crucible/hosttools.py` is the one owner of both — the PATH the server searched, and the
sentence that names it.

**On the PC the recorded PATH is about 2 KB of Windows**, dozens of `/mnt/c/...` entries
arriving through WSL interop. That is correct and it is not to be "cleaned": it is the
PATH the operator's shell actually had when `crucible doctor` passed in it, and trimming
it to the entries that look Linux-shaped would be this server deciding which of somebody's
tools count.

**`CRUCIBLE_HOME` is recorded too, and it is not optional.** A service started without it
serves `~/.crucible`, which on a host where the operator set `CRUCIBLE_HOME` is a
*different server with a different token*. The value written is the one the installing
process resolved, so the service and the shell that installed it can never be looking at
two configs.

## 4. Linger is reported, never assumed

A systemd **user** service stops when that user's last session ends, and does not start at
boot, unless `loginctl enable-linger <user>` has been granted. That is a change to somebody's
machine and needs root, so Crucible does not make it. It asks:

- `status` prints `linger: on` / `OFF` / `UNKNOWN`.
- `install` prints the exact command when it is off:
  `sudo loginctl enable-linger <user>`.
- **`UNKNOWN` is a real third answer**, not a soft no. A container with no `loginctl`, or a
  host where the call fails, has not said no — and recording it as "off" would be this
  server inventing a fact about a machine it could not read.

launchd has no equivalent question, so `linger` is `null` there.

## 5. How the tests can be trusted without a machine

Every `systemctl` and `launchctl` call goes through one injectable `Runner`, so the suite
asserts on the argv that *would* have been run, in order, and on the definition text byte
for byte. `user_home()` is a module-level probe the tests replace, so no run can touch a
real `~/.config/systemd` or `~/Library/LaunchAgents`.

What that does **not** prove is that systemd and launchd accept these files. Nobody has
installed either one yet.

### Ruling owed

- **Should `crucible install <type>` end by offering the service?** Today they are separate
  verbs, and an operator who installs three envs and never runs `crucible service install`
  has a Crucible that only exists while a terminal is open. The argument against coupling
  them is that installing an env is not consent to run a daemon; the argument for is that
  every other path leaves the machine in a state nobody asked for. **Proceeding on the
  careful assumption: separate verbs, and `crucible doctor` does not yet mention the
  service at all.** Owen to rule.
- **Should `status` be part of `crucible doctor`?** It is a fact about this host that
  `doctor` could report, and `doctor`'s exit code is already a health gate. Not done,
  because `doctor` currently probes only what a *running* server would need, and making it
  report a stopped service would change what its non-zero exit means.
