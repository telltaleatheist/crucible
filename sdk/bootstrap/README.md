# `@crucible/bootstrap`

The app-side installer and ensurer for a **local** [Crucible](https://github.com/telltaleatheist/crucible).
A local Crucible is a *service* on the machine and no app owns it (Owen, 2026-09-13,
`docs/PHASE5-APPS.md` section 6.0); an app's whole job is to make sure this machine has
one and make sure it is running. That is this package's whole surface:

```ts
import { detectHost, install, ensureRunning, readLocalConfig, health } from '@crucible/bootstrap';
```

It ships **with** the server, from this repo, at the server's version, cut by the same
`scripts/release.sh` — so a bootstrapper is never paired with a server nobody tested it
against. It peer-depends on `@crucible/client` at that exact version and on nothing else.
Node 20+, and the Electron main process. ESM and CommonJS builds ship side by side.

## Install

From the GitHub Release, beside the client (there is no npm registry publish):

```bash
npm install https://github.com/telltaleatheist/crucible/releases/download/v0.6.0/crucible-client-0.6.0.tgz
npm install https://github.com/telltaleatheist/crucible/releases/download/v0.6.0/crucible-bootstrap-0.6.0.tgz
```

Pin both URLs in `package.json`. Only the *app* half of an Electron product consumes this.
An engine that runs where there is no app — Foundry's compiled binary — consumes an
endpoint and never learns what a Crucible is.

## Where the server lives

Windows is never a backend. On win32 every verb reaches into a WSL2 distro through
`wsl.exe -d <distro> --exec …`, and **which distro has one rule** (`docs/PHASE14-ENVPACKS.md`
4b): the distro Crucible owns — `crucible`, imported by `ensureDistro()` — when there is
one, else the distro the app names, else `no_wsl_distro`. There is still no "the default
distro": that is whatever `wsl --set-default` last said, and a server read from the wrong
guest is the wrong server. A machine with a `crucible` distro AND a config in the app's
distro is `two_local_crucibles`, refused rather than guessed; `{exact: true}` is the way
out. `detectHost()` lists the distros so the app can choose. On macOS and Linux the verbs
run on the machine itself. Anything else is refused.

**There is no conda and no Python prerequisite.** The server arrives as a *pack* — a
relocatable CPython with the crucible wheel already installed — downloaded from the
release into `<CRUCIBLE_HOME>/server`, and every verb runs
`<CRUCIBLE_HOME>/server/bin/crucible`.

## The surface

| Verb | Does | Refuses by name |
|---|---|---|
| `detectHost({distro?, home?, release?})` | what this host (or its guest) has: `{platform, backend, wsl, wslState, gpu, guest, server, refusals}` | `wsl_missing` (with `wsl --install --no-distribution`), `no_wsl_distro`, `wsl_read_failed`, `unsupported_platform`; and, as entries in `refusals` beside each null: `no_nvidia_driver`, `not_apple_silicon`, `guest_missing_tool` |
| `install({distro?, exact?, jobTypes, home?, release?, onLine, onStep?, bind?})` | host-facts → server-pack → `crucible init --token …` → `crucible install <type>`… → `crucible service install` → (win32) `loginctl enable-linger` as root → `crucible capability --write` | every `detectHost` refusal, `bad_job_type`, `two_local_crucibles`, the pack refusals (`pack_manifest_unreadable`, `pack_not_published`, `pack_download_failed`, `pack_sha_mismatch`, `pack_disk`, `pack_unpack_failed`), `config_unreadable`, `config_missing_key`, and `step_failed` (a `BootstrapStepFailed` with the step, exit code, tail and the steps that finished) |
| `ensureDistro({release, installDir?, downloadDir?, rootfsUrl?, onLine?})` | win32: the `crucible` distro exists, runs systemd and came from our rootfs — idempotent | `distro_unmarked`, `distro_import_failed`, `pack_download_failed`, `pack_sha_mismatch`, `wsl1_only`, `unsupported_platform` |
| `detectWslState({release, appDistro?, requiredBytes?, checkNetwork?, guestUser?})` | win32: the first row of PHASE14 4c that matches, with the sentence and the action | nothing — every state IS an answer, `wsl_ready` included |
| `ensureRunning({distro?, exact?, home?})` | `{running: true, pid, mechanism, definition, linger, enableLinger, lingerStep, started}` — a no-op when it already is, except that linger is asked (and on win32 granted) either way | `no_server_pack`, `service_not_installed`, `service_failed` (with the status output and where the logs are), `no_local_config`, `linger_unreadable`, `linger_failed` |
| `readLocalConfig({distro?, exact?, home?})` | `{name, url, token, configPath, via}` from the server's own `config.toml` | `no_local_config`, `no_wsl_distro`, `two_local_crucibles`, `wsl_read_failed`, `config_unreadable`, `config_missing_key` |
| `health({distro?, exact?, home?, clientName?})` | the SDK's `Activity` from `GET /v1/activity` | `unreachable`, `wrong_token`, `not_a_crucible`, `version_mismatch`, plus everything `readLocalConfig` refuses |

Every verb is idempotent. Every function takes an optional second argument, a `Runner`,
which is the one door to the machine (`spawn`, `fs`); the tests script it and assert on the
exact argv that would have run, and the real one is `processRunner()`.

### What a refusal carries

```ts
try {
  await ensureRunning({ distro: 'Ubuntu' });
} catch (err) {
  if (err instanceof BootstrapRefusal) {
    err.code;     // 'service_not_installed'
    err.message;  // the sentence
    err.command;  // '/home/owen/.crucible/server/bin/crucible service install' — or null
    err.detail;   // verbatim evidence (a status page, a stderr tail) — or null
  }
}
```

**A missing prerequisite is a named refusal carrying the exact command the host must run.**
Elevation, a reboot, a sudo password — those are the app's to obtain. This package never
attempts them, never falls back past them, and never guesses a value it could not read.

### `install()`

```ts
await install({
  distro: 'Ubuntu',
  jobTypes: ['llm', { type: 'tts', narratorEngine: 'higgs-v3' }, 'asr', 'rvc', 'denoise'],
  onLine: (line, stream, step) => log.append(`[${step}] ${line}`),
  onStep: (step) => ui.setStep(step.name, step.status),
});
```

- **WHICH RELEASE IS THE CHANNEL'S ANSWER, AND AN APP ASKS IT** (`docs/INSTALL-UNINSTALL.md`
  §6.5). `latestRelease()` reads `releases/latest` — the release `promote_release.py`
  promoted, not the newest tag — and an app passes what it says as `release`. It is NOT
  this package's own version: a vendored 1.0.1 bootstrapper installing 1.0.1 over a running
  1.0.2 is the defect that section is about. A channel that will not answer is
  `release_channel_unreadable`, never a quieter older install.
- **Never over a newer pack.** `<home>/server/.pack` records `release=`, so `install()`
  refuses `install_would_downgrade` before anything is downloaded when this disk already
  holds a newer Crucible. The one way down is `rollbackTo`, naming the exact version being
  installed; anything else is `rollback_version_mismatch`.
- **The interpreter arrives with the server.** `release` names a Crucible version and
  defaults to this package's own, which is the hand-install case and nothing else. The server pack is fetched
  INSIDE the guest with the guest's `curl`, never through `/mnt/c`; parts are appended and
  deleted one at a time (peak extra disk is one part), the sha256 is computed in the guest
  and compared here, and the unpack is renamed into place only after the tree runs its own
  `--version`. A stamp that already matches the manifest is a skip.
- **The token is minted here** and handed to `crucible init --token`, so the app already
  holds what `readLocalConfig()` would read back. It is never logged: the step's recorded
  argv spells it `<redacted>`, and `onLine` only sees what the step printed. `init` is
  **skipped** when a config already exists, and that config's token is kept.
- **Pulls are not part of install.** Weights are the app's, later, per model.
- `home` is `CRUCIBLE_HOME` for every verb, spelled as the target spells it (a guest path
  on win32). `bind` is `crucible init`'s `--host`/`--port`.
- A failing step throws `BootstrapStepFailed`: `step`, `exitCode`, `tail` (the last 40
  lines, stderr prefixed `! `), `stepsDone`. What finished stays on disk.

### `ensureRunning()`

Starting is `systemctl --user start` on cuda-linux and `launchctl kickstart` on
mlx-darwin — reached through `crucible service start`, because the unit name and the
launchd label are facts `crucible/service.py` owns. Nothing is ever spawned as a child.
**Linger is GRANTED on win32 and reported everywhere else.** A systemd user unit dies
with the user's last session without it, so `running: true` about a non-lingering server
is a promise that expires at the next logout. Inside WSL there is no elevation to hand
over — `wsl.exe -u root` is how the guest is entered, not an escalation performed in it
(measured 2026-09-14) — so this package reads linger as root and turns it on when it is
off, reported as `lingerStep` with the exact argv. On native Linux `sudo` really is
elevation and the host app really is the one that can obtain it, so there the fact is
still reported and `enableLinger` is the line for the app to show. On macOS launchd has
no linger question at all, and `lingerStep` is null.

A guest that will not give root — WSL1, or a distro with the root account disabled — is
the one hand-over that remains: `linger_unreadable`, with the command. Neither answer is
guessed, because "off" would grant something nobody asked for and "on" would promise a
server that dies with the next logout.

### `readLocalConfig()`

BookForge's `electron/crucible/local.ts` rule, now owned here: the local server has ONE
owner, `<CRUCIBLE_HOME>/config.toml`, the file the server itself reads. `$CRUCIBLE_HOME`,
else `~/.crucible`, resolved *inside the guest* on win32. The connect address is derived
from the bind address — `0.0.0.0` and `::` are reached at `127.0.0.1`, anything else as
written. A missing file is `no_local_config`, a state the app shows, not an empty client.

## The Windows/WSL facts, each with a test

From Foundry's deleted launcher (`docs/FROM-FOUNDRY-WSL-VLLM.md` section 2), carried
verbatim:

- always `wsl.exe -d <distro> --exec …`, never the implicit shell (it pre-expands `$var`);
- wsl.exe's own output is UTF-16LE with a BOM, the guest's is UTF-8, on the same handles,
  and BOTH CAN ARRIVE IN ONE CHUNK — so the decode is per run of bytes, not per chunk
  (`segmentWslBytes`), and a chunk that ends mid-character waits for the next one
  (`incompleteTailBytes`);
- every one-shot call has its own timeout, and a timeout is a reported failure;
- argument arrays, never shell strings; backslashes doubled once, because wsl.exe halves
  them once before bash exists (`wslArgv`);
- `toWslPath` maps `C:\a\b` → `/mnt/c/a/b`, refuses UNC, and `realpath.native` catches
  mapped drives (`guestPathFor`);
- a prebuilt environment never crosses `/mnt/c` at all: the guest downloads it with its
  own `curl` and unpacks it with its own `tar --zstd` (`pack.ts`), which is faster than
  the 9P mount and keeps the permission bits a Python tree needs;
- pip's `\r`-repainted progress is split into lines (`splitLines`).

## The standalone installer

The same sequence, without an app (`docs/PHASE14-ENVPACKS.md` 4a):

```bash
curl -fsSL https://github.com/telltaleatheist/crucible/releases/latest/download/install.sh | sh
```

`scripts/install.sh` and `scripts/install.ps1` are **generated** from `src/steps.ts` — the
same list `install()` walks — by `scripts/gen-install-scripts.ts`, and a test asserts that
the files on disk are what the generator writes. An app-driven install and a hand install
cannot describe different installs. `npm run gen:install` regenerates them. They carry no
job types and no weights: a bare Crucible that serves nothing until an app asks.

On Windows `install.ps1` walks every WSL state in `src/wsl-states.ts`, imports the distro
Crucible owns, and then runs the same `install.sh` inside it.
`scripts/build-rootfs.sh` builds the image it imports; that runs in CI on Linux and cannot
run on Windows.

## Tests

```bash
npm run test:unit
```

Unit tests only, found rather than listed (`scripts/unit.mjs`). Every branch and every
refusal runs against a scripted `Runner` and asserts the argv verbatim; the real runner's
decoding, streaming, timeout and spawn-error paths run against `node` itself. Nothing in
the suite touches a guest, a service manager or a GPU.
