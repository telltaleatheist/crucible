# Phase 12: `@crucible/bootstrap`

> **AMENDED 2026-09-14 by PHASE14-ENVPACKS.md section 4.** The conda half of this document
> is HISTORY: `DEFAULT_CONDA_ROOTS`, `SERVER_ENV_NAME`, `probeInterpreter`, `no_conda`,
> `no_python` and `InstallOptions.wheel` are deleted, and the sequence is the one in
> section 1a below. What stays exactly as written: the five verbs, the injectable `Runner`,
> the named refusals, the WSL transport facts in section 2, the token discipline, and
> section 6's linger ruling. PHASE14 7b is what was built; this file is why the package is
> shaped the way it is.

The app-side half of PHASE5-APPS.md section 6.0's ruling — **a local Crucible is a
SERVICE, and the bootstrapper ships WITH Crucible** — built 2026-09-14 as `sdk/bootstrap/`,
a sibling of `sdk/ts/`, released as the fourth asset of every release at the server's
version. PHASE11-SERVICE.md is the server half (`crucible service …`); this is the thing
that drives it from an Electron app, and it is the one piece of Crucible that still crosses
the Windows/WSL boundary.

## 1. What it is, in one paragraph

Five idempotent verbs and nothing else: `detectHost()`, `install()`, `ensureRunning()`,
`readLocalConfig()`, `health()`. Every one takes an injectable `Runner` (spawn + fs), so
the whole suite runs against scripted argv without a guest, a service manager or a card.
Every missing prerequisite is a **named refusal carrying the exact command the host must
run** — elevation, reboots and sudo passwords are the app's, never this package's. Nothing
is ever spawned as a child that should be a service: `ensureRunning()` is `crucible service
start`, which is `systemctl --user start` / `launchctl kickstart`. The token is never
logged or printed. `sdk/bootstrap/README.md` is the surface; this file is the reasoning.

## 1a. The sequence, as of PHASE14

```
host-facts       one guest script: CRUCIBLE_HOME, the user, free disk, curl/tar/zstd, and
                 the server pack already there with its stamp
server-pack      the release's crucible-env-server-<backend>-<release>.tar.zst, fetched by
                 the GUEST's curl into the guest's filesystem, sha256'd in the guest,
                 unpacked to <dest>.partial and renamed. A matching stamp is a skip.
init             <CRUCIBLE_HOME>/server/bin/crucible init --token … (skipped when a config exists)
install-<type>   <…>/bin/crucible install <type> — now a download, not minutes of pip
service-install  <…>/bin/crucible service install
linger           win32, as root (section 6's ruling, unchanged)
capability-write <…>/bin/crucible capability --write
```

Three things this moved that are worth stating in this file, because they change what the
package IS rather than what it runs:

- **There is no interpreter to find.** The pack carries one. "The server interpreter is
  found, never made" (section 3) has become "the server interpreter arrives with the
  server", and the whole `no_conda` / `no_python` hand-over — which fired on every fresh
  machine, which is to say on every machine this package exists for — is gone with it.
- **The step list is DATA** (`src/steps.ts`), because `install.sh` and `install.ps1` are
  generated from it. `install()` walks the list; the generator reads the same list.
- **Which WSL distro is `local` has one rule** (`src/distro.ts`), because there is now a
  distro Crucible owns. See PHASE14 4b; `readLocalConfig`, `ensureRunning` and `install`
  all go through `resolveDistro()`.

## 2. Where each of Foundry's facts landed

FROM-FOUNDRY-WSL-VLLM.md section 2 listed the Windows-side facts that belonged to the
unbuilt bootstrapper. Each now has a home and a test:

| Fact | Where | Test |
|---|---|---|
| `wsl.exe -d <distro> --exec …`, never the implicit shell | `wsl.ts` `wslArgv` — the only builder; there is no `bash -lc` and no `--` anywhere | `unit-wsl` |
| wsl.exe's own output is UTF-16LE with a BOM, the guest's is UTF-8, same handles; decide per chunk by NULs | `runner.ts` `decodeWslBytes`; `stream()` decodes each chunk before splitting | `unit-runner`, including a real child emitting UTF-16 between UTF-8 lines |
| every one-shot call has a timeout | `RunOptions.timeoutMs` is required; the fake runner refuses a call without one; a timeout is `failure`, never a hang | `unit-runner` (a child that never exits) |
| argument arrays, never shell strings; backslash-halving | `spawn(argv[0], argv.slice(1))` only; `wslArgv` doubles `\` once | `unit-wsl` |
| `toWslPath` refuses UNC; `realpath.native` catches mapped drives | `toWslPath` (pure) + `networkPathBehind` + `guestPathFor`; `install()` runs the wheel through it | `unit-wsl`, `unit-install` |
| ~~conda by `test -x` at the three roots~~ → **the guest is asked ONE script for what it has** | `pack.ts` `guestProbeScript` — exit 0 always, `key=value` lines out. Same shape, no conda | `unit-host`, `unit-pack` assert the script text |
| prebuilt env tarballs unpacked by the distro's own tar, never through `\\wsl$` | `pack.ts` `installPack` — and the archive never crosses `/mnt/c` at all now: the GUEST downloads it | `unit-pack`, `unit-install` |
| pip's `\r`-repainted bar split into lines | `runner.ts` `splitLines` | `unit-runner` |

BookForge's `electron/crucible/local.ts` (the config.toml read rule and the bind→connect
mapping) is now `config.ts` here, function for function — `localConfigPath`,
`connectHost`, `parseLocalConfig`, `readLocalConfig` — with the same five named states.
BookForge's copy is the one to delete when it takes the dependency; until then it is a
derived copy and the two say the same thing (ARCHITECTURE.md R1).

## 3. Decisions taken while building, each with its reason

**~~The server interpreter is found, never made.~~ SUPERSEDED 2026-09-14 (PHASE14 0).** It
was `<conda>/envs/crucible/bin/python`, a 3.11, refused by name as `no_conda` or
`no_python` when absent — and absent is what it is on every machine that does not already
have a Crucible, which is every machine this verb is for. The reasoning was sound and the
premise was wrong: the right move was not a better refusal but removing the prerequisite.
The server pack carries its own CPython, and `pack.ts`'s `requirePack()` is all that is
left: either `<CRUCIBLE_HOME>/server/bin/crucible` is there or `install()` downloads it.

**On win32 the distro is required for every verb except `detectHost`.** `local.ts`'s rule,
kept: there is no default distro on purpose. `detectHost()` is the exception because it is
how the app learns which distros exist; it reads the guest's facts through the one wsl.exe
marks default when none is named, and **records which** in `wsl.probed`. *(PHASE14 4b adds
one layer above this: when a `crucible` distro exists it wins, and a machine with both it
and a config in the app's distro is `two_local_crucibles`. The rule below it is unchanged.)*

**`detectHost()` returns nulls AND the refusals for them.** The ruled shape was `{platform,
wsl, gpu, python}` with nulls; the ruled behaviour was refusals by name. Both, then: every
null has a `BootstrapRefusal` in `refusals` beside it, and `install()` throws the first.
A caller that only wants facts reads the fields; a caller that needs the card gets the
refusal without re-deriving it.

**The token is minted on the client and handed to `crucible init --token`.** The one
Python change this phase makes: `cmd_init` takes `--token` (refusing blank or spaced
values) and `write_config` is unchanged. `init` is **skipped** when a config already
exists — its token is kept, because `crucible init --force` mints a new one and every
client would need it — and a config that exists but is broken is refused by name with
`crucible init --force` as the command, never re-initialised over.

**`ensureRunning()` goes through `crucible service start` and `status --json`.** The
ruling names `systemctl --user start` / `launchctl kickstart`; those are exactly what
`crucible/service.py` runs, and the unit name, launchd label and `gui/<uid>` domain are
its facts. Spelling them a second time in TypeScript would be the drift this package
exists to remove. `status`'s exit code is not read — it is 1 for "installed and stopped",
and the JSON says which.

**`health()` maps the SDK's four transport-level errors and rethrows everything else.**
`CrucibleUnreachable` → `unreachable` (command: start it), `CrucibleAuthError` →
`wrong_token` (the file and the running server disagree; command: `crucible service
install`), `CrucibleNotACrucible` → `not_a_crucible`, `CrucibleVersionError` →
`version_mismatch`. A `CrucibleProtocolError` is **not** mapped: it means the server and
the SDK this package was built against disagree about the wire, which is the SDK's own,
already-named error and not a fact about the host.

**Zero runtime dependencies, with one peer.** `@crucible/client` is a `peerDependency`
pinned to the exact version (the app installs both tarballs); the TOML reader is ~200
lines of this package's own, reading exactly the dialect `tomli_w` writes and refusing the
rest by name. `release.sh` refuses to cut when the package version, the peer pin, the
`BOOTSTRAP_VERSION` literal and `crucible/__init__.py` disagree.

**A step's recorded argv is the target's, not the transport's.** `InstallStep.argv` is
`[crucible, 'init', '--token', '<redacted>', …]` as the guest sees it; the `wsl.exe -d …
--exec` wrapping is the runner's business and is asserted in the tests, not shown to the
app.

## 4. Measured on Owen's machines, 2026-09-14 (read-only)

`detectHost()` on the PC, live through the compiled `dist/esm`:

```
platform win32; wsl: Ubuntu (Running, v2, default), probed Ubuntu
gpu: nvidia "NVIDIA GeForce RTX 3090 Ti" 25757220864 bytes
python: /home/telltale/anaconda3/envs/crucible/bin/python 3.11.16; conda /home/telltale/anaconda3
refusals: []
```

`readLocalConfig({distro:'Ubuntu'})` → `crucible@owens-pc-wsl`, `http://127.0.0.1:7100`,
via wsl. `ensureRunning` → running, systemd, linger on, `started: false`. `health()` →
the SDK checkout's `CrucibleProtocolError: activity has no field "lease"` — the SDK in
this branch is ahead of the server build that is running, exactly the case the
non-mapping above is for.

**The Mac is a finding.** Its `crucible` env is at
`/opt/homebrew/Caskroom/miniconda/base/envs/crucible` (the plist's `ProgramArguments`
says so), which is not one of the three ruled conda roots; `~/miniforge3` exists there
too, without a `crucible` env. With the default roots `detectHost()` on that Mac would
answer `no_python` naming `~/miniforge3/bin/conda create …`, which would build a second
server interpreter beside the one the service already runs. `condaRoots` overrides the
list per call, and that is what an app on that Mac must pass until section 6's ruling.

## 5. What is deliberately not here

- **No pulls.** Weights are the app's, later, per model (`crucible models pull` and its
  siblings, through the Servers row).
- **No `ensureResident(id)`.** PHASE5-APPS.md section 6.0 sketched it; PHASE7-LANES.md
  section 9.2 owns the residency question and the SDK already has `loadModel` /
  `loadVoice`. A bootstrapper that loads models is an app deciding what is resident, which
  is the operator's decision.
- **No adoption.** A registry entry whose URL answers `ping` is a server, whoever started
  it; this package starts one only through its service, and only when asked.
- **No Docker image.** Section 6.1's friend path is still owed.

## 6. Rulings owed

1. ~~**The Mac's conda root.**~~ **MOOT, 2026-09-14 (PHASE14 0).** There are no conda roots
   to rank. The Mac's `crucible` env at `/opt/homebrew/Caskroom/miniconda/base/envs/crucible`
   is the OLD server's and stays whatever it is; a Crucible installed from now on lives at
   `~/.crucible/server` on that Mac like everywhere else. The interesting half of the
   question — should "which interpreter" be read from the installed service definition? —
   is answered differently and better: the definition is written FROM the pack, so the pack
   path is the one owner and nothing has to read the plist back.
2. ~~**Should `install()` create the `crucible` conda env?**~~ **MOOT.** It creates nothing
   on somebody's machine except `<CRUCIBLE_HOME>`, which is the directory the thing being
   installed lives in.
3. ~~**Prebuilt environments.**~~ **BUILT, 2026-09-14 — this is PHASE14.** "Env archives as
   release assets, sha256 per part, stamp file" needed "a catalog and a downloader"; the
   catalog is `envpacks.json` and the downloader is `pack.ts`. `guestUnpackArgv` is gone
   with the rest of the `/mnt/c` route: the archive never reaches a Windows path at all,
   because the guest fetches it.
4. ~~**`linger` off after `install()`.**~~ **RULED, 2026-09-14, by measurement.** The
   ruling had been framed as "whether the app shows the sudo line once, every launch, or
   offers to run it with elevation", which assumed there was elevation to obtain. On
   win32 there is not. Verified on Owen's PC:

       wsl.exe -d Ubuntu -u root --exec id -u                          → 0
       wsl.exe -d Ubuntu -u root --exec loginctl show-user telltale -p Linger
                                                                       → Linger=yes

   `-u root` is **which user wsl.exe starts the guest as** — no password, no sudo, no
   prompt — so the thing being handed over was one idempotent command this package can
   run itself, on a machine the app was already asked to install a server on. Handing a
   person a `sudo` line for a command that needs no sudo is not caution; it is a step
   somebody skips, and the consequence of skipping it is a Crucible that vanishes at the
   next logout.

   So on win32 `install()` and `ensureRunning()` read linger through `-u root` and grant
   it when it is off, as a named step carrying the argv (`sdk/bootstrap/src/linger.ts`).
   `enableLinger` is always null there and the elevated hand-over line is gone.
   `ensureRunning()` asks **even when the service is already up**, because "running" and
   "will still be running after this person logs out" are different facts and only the
   second is what that verb is for.

   **macOS and native Linux are untouched.** launchd has no linger question; on native
   Linux `sudo` is real elevation, the host app is the one that can obtain it, and
   PHASE11's reporting is still the right shape.

   **The one hand-over that remains** is a guest that will not give root — WSL1, or a
   distro with the root account disabled — which is `linger_unreadable` with the command,
   never a guess in either direction.
