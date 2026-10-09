# @crucible/bootstrap internals

`sdk/bootstrap` is the app-side installer and ensurer for a local Crucible:
`detectHost`, `install`, `ensureRunning`, `readLocalConfig`, `health`. Every verb
is idempotent, every missing prerequisite is a named `BootstrapRefusal` carrying
the command the host must run, nothing is spawned as a child that should be a
service, and the token is never logged. This page holds the design constraints
the code does not say by itself.

## Refusals

- A refusal is always one of the `BootstrapRefusalCode` names, with `command`
  set whenever a person can type something to clear it. Elevation, a reboot and
  a sudo password are the host app's to obtain, never this package's to attempt.
- `BootstrapStepFailed` carries the step, its exit code (or why there is none),
  the output tail, and every step that finished before it. Partial work stays on
  disk.
- The host door's `failed` codes (the WSL state codes) are rethrown verbatim,
  never wrapped: an app picks its next move from the code. They are not checked
  against `BootstrapRefusalCode` at runtime; the Python host gets them by
  generation and `gen:install --check` keeps the two sides tied.
- `no_local_config` and `no_server_runtime` are states, not bugs.

## Release channel (`channel.ts`, `release.ts`)

- "Latest" is GitHub's `releases/latest`, never `releases?per_page=1`. A release
  is cut `--prerelease --latest=false` by `scripts/release.sh` and becomes latest
  only when `scripts/promote_release.py --publish` flips it after its assets and a
  fresh-install smoke are verified; `per_page=1` answers "newest tag", an
  unverified candidate.
- There is no fallback under the channel. An unreadable channel is
  `release_channel_unreadable`; it is never a reason to install the version this
  package was built at (that is how 1.0.1 got installed over a running 1.0.2).
  The one override is an operator naming an exact release, which still downloads
  from GitHub. There is no offline install.
- Versions compare number by number (`compareReleases`): a string compare puts
  1.0.10 before 1.0.2. A string that is not three numbers is refused rather than
  sorted.
- A release carries only our code: the wheel (`py3-none-any`, one file for every
  backend), the sdist, the SDK tarballs and the generated installers. The
  interpreter comes from python-build-standalone, environments from PyPI, the
  WSL image from Canonical.
- The wheel's digest is a sibling asset `<wheel>.sha256` (one line, digest
  first). The installer runs before any Crucible exists to ask, so a sibling file
  is the only thing it can read, and a digest carried inside the bytes it
  describes would not be a check.

## Never older (`runtime.ts`, `install.ts`, generated installers)

- `<CRUCIBLE_HOME>/server/.crucible` holds `python_sha256=`, `python_version=`
  and `release=`. Installing an older release over it is
  `install_would_downgrade`, checked before any download. An unstamped tree is
  not "older"; it is reinstalled whole.
- The one way down is an operator rollback naming the exact version:
  `rollbackTo` must equal `release` (else `rollback_version_mismatch`).
  `RuntimeInstallOptions.rollbackTo` is required-and-nullable on purpose so a
  caller has to decide.
- On win32 a rollback cannot pass through the host door (the request body has
  no field for it), so `install()` refuses it up front with the `install.ps1`
  line that performs it.

## Server runtime (`interpreter.ts`, `runtime.ts`, `steps.ts`)

- The interpreter is python-build-standalone `install_only` (one relocatable
  `python/` directory, no baked absolute paths), pinned by release and sha256 read
  from that release's `SHA256SUMS`. The Windows archive has no `bin/`, so layouts
  are never spelled inline.
- `crucible/interpreter.py` holds the same pins because the server downloads
  interpreters for recipes too. Neither side can import the other (this runs
  before Python exists), so `tests/test_interpreter.py` reads `interpreter.ts` as
  text and asserts every row agrees. This table carries only the server's 3.11
  rows; recipe-only interpreters (the 3.12 Higgs row) are the server's.
- `DESKTOP_PACKAGES` (pystray, pillow) are installed after the wheel on desktop
  platforms and are deliberately not wheel dependencies, so a headless Linux
  server never carries a GUI toolkit.
- Everything happens inside the guest with the guest's `curl`, `tar` and `pip`,
  never across `/mnt/c` (slow, and it loses permission bits).
- The interpreter is fetched only when the stamped digest differs; the wheel
  always installs (it is the deploy, about 1 MB, and re-running it repairs a
  half-finished install).
- Every download is digest-checked before use; a mismatch deletes the file and
  refuses `runtime_sha_mismatch`. Downloads are fetched whole, never resumed
  (`CURL_ARGS` has no `--continue-at`), so a proxy that ignored a range cannot
  produce a corrupt archive.
- The unpack goes to `<dest>.partial` and is swapped in only after `tar` exits
  0. The `install_only` archive's top-level `python/` directory is what moves.
- Before `site-packages` is rewritten the running server is stopped with the NEW
  release's code (`crucible_quiesce` in `install.sh`, the same staging in
  `install.ps1`): the verified wheel is unpacked with `pip install --target
  --no-deps` beside the tree and its `local shutdown` runs on the installed
  interpreter. Otherwise a fix to how Crucible stops itself could never help the
  upgrade that ships it. If neither the new nor the installed code could stop it,
  the freshly installed binary is asked once more (`crucible_quiesce_after`) and a
  failure is refused by name; `|| true` is never used.
- The wheel is therefore fetched before the interpreter half runs.
- The guest probe checks `REQUIRED_TOOLS` (curl, tar) before anything is
  fetched. macOS has no `sha256sum`, so `shaArgv` uses `shasum -a 256` there.

## One step list (`steps.ts`, `scripts/gen-install-scripts.ts`)

- An app-driven install and a hand install "cannot differ": `install()` walks
  `installSteps()` and the generator writes `install.sh` from the same list.
  Step names, order, skip rules (an existing config keeps its token; a matching
  interpreter digest is not re-fetched), argv, paths, curl/tar flags, pins and URL
  shapes are shared. `renderArgv` and `renderSh` produce the spawned array and
  the shell line from the same words.
- A `{ sh }` word exists only for the generated script (typed flags like `$BIND`,
  a job type from `$1`). `renderArgv` refuses it: there is no shell on the
  TypeScript side, and `$BIND` would reach `spawn()` as a literal.
- The three program steps (host probe, server runtime, linger) carry their
  shell beside the TypeScript; the iteration is written twice, the facts once.
- `env-patch-llm` runs `crucible env patch llm` while the server is down, because
  an upgrade installs a new wheel without rebuilding an unchanged env and so skips
  `install_env`'s own patch step. It reports `not_applicable` where there is
  nothing to patch.
- A start request accepted by launchd/systemd does not prove the API is up; the
  final `local-start` step waits for the authenticated identity before success.
- The token is minted on the client unless the operator brought one (the droplet
  case): install.sh `--token-env` reads it from `$CRUCIBLE_INIT_TOKEN`, then unsets
  that so nothing else the script starts inherits it (`--token <t>` still works, on
  sh's argv). Either way it reaches `crucible init --token-env` through
  `$CRUCIBLE_INIT_TOKEN` in that one command's environment, never its argv.
- The hand installer's progress lines use `crucible/interpreter.py`'s progress
  wire (written in `steps.ts`, parsed there by `parse_progress_line`, tied by
  `tests/test_host.py`) rather than curl's CR-separated meter. `bytes_total` is
  null when a HEAD gives no `Content-Length`.
- `install.sh --uninstall` runs `crucible uninstall` first and then removes the
  interpreter itself: the verb cannot unlink the interpreter it runs from, and
  this script unpacked it. `--dry-run` turns the removal into a sentence.
- Hand-install flags: every variable starts empty or 0 and empty means the bare
  behaviour; an unknown flag is refused (a typo in `--purge-weights` must not run
  the keep-weights path). The only disk guard is the operator's `--min-free-gib`;
  inside WSL `df` measures the ext4.vhdx's virtual ceiling, not the Windows drive.
  On Linux ffmpeg is Crucible's own pinned build placed by `crucible install`; on
  the Mac it is required from Homebrew only when an audio job type is asked for.
- `--from-source <ref>` replaces the wheel half only (same pinned interpreter,
  `pip install <checkout>`), is never a fallback for a failed download, and stamps
  the commit as the release.
- Linger: native-Linux user units die with the last session without it. The
  script tries as itself, then `sudo -n` (never prompts), and otherwise prints the
  one line to run. Inside WSL the server is a system unit and has no linger
  question.
- The launcher is not a step of its own: `local-install-desktop` (`crucible local
  install-desktop`) writes the Start Menu item on Windows and `~/Applications/Crucible.app` on
  the Mac. Only the generated scripts then OPEN it, and only on a fresh install typed at the
  machine: `install.sh` when `config.toml` did not exist before `init`, on Darwin, with stdout
  a terminal and no `SSH_CONNECTION`; `install.ps1` when `crucible.cmd` did not exist, the
  session is interactive, not ssh, and not driven by an app (`$FromApp`). `install()` never
  opens a window: an app driving the install is already the UI. `install.sh`'s last line stays
  the pairing line.
- `install.ps1 -PythonArchive <file> -WheelFile <file> -WheelSha <hex>` take already-downloaded
  files instead of curl, for the Windows setup (`docs/internals/scripts.md`). Every check
  still runs on them: the archive against the pinned `$PySha`, the wheel against `-WheelSha`.
  Without the flags nothing changes.

## Generated outputs

- `npm run gen:install` writes `scripts/install.sh`, `scripts/install.ps1` and
  `crucible/platform/wsl_table.py`; `--check` (asserted by
  `test/unit-gen-install.test.ts`) fails on drift.
- `install.ps1` must be ASCII. Windows PowerShell 5.1 reads a BOM-less `.ps1` in
  the ANSI code page, and the generator writes BOM-less UTF-8 (a BOM breaks
  `irm | iex`). An em dash inside a string made `[Parser]::ParseFile` fail
  (measured 2026-09-15). `asciiOnly` transliterates via `ASCII_FOR` and refuses
  any other non-ASCII character rather than letting it degrade.
- `platform/wsl_table.py` carries the WSL state table as data for the Windows host. The
  `means` predicates are code and live in `crucible/host/wslstate.py`, one per
  code; a pytest asserts the two sets are equal. Sentences are functions, so the
  generator calls them with sentinel evidence and swaps sentinels for
  `{said}`/`{app_distro}`/`{release}`/`{required}`/`{free}`, asserting each swap
  fired and no sentinel survived. `optional` is read from the rows disabled when
  nothing is measured. The generator asserts `automatic` equals "action is `run`
  or `run-elevated`" for every row except `wsl_ready`.
- `install.ps1` specifics: `$ErrorActionPreference = "Continue"` because 5.1 turns
  native stderr into a terminating error under Stop; exit codes are checked
  explicitly. `Die` uses `exit 1` from a file and `throw` when piped through
  `irm | iex` (where `exit` closes the window). `Native` turns ErrorRecords from
  redirected native stderr into plain text. `tar` is `%SystemRoot%\System32\tar.exe`
  by full path: PATH is the caller's, and from Git Bash it found GNU tar. The
  `crucible.cmd` shim is written CRLF (cmd.exe can swallow the last line of an
  LF-only batch file) and quotes `%~dp0` because the user's name can contain
  spaces. pip runs with `--quiet --no-warn-script-location`. The script knows it
  was run by an app when it is saved as `crucible-install.ps1` (`runInstallPs1`'s
  name) with output redirected; then the install watcher gets `--brief`.
- `install.ps1` installs the Windows host only and stops; the host owns the WSL
  sequence. It needs no admin.

## Windows: the host door (`hostdoor.ts`, `install.ts`)

- On win32, `install()` does two things: run `install.ps1` if the host is absent
  (per-user, no elevation, saved to a file and run with `-Release` because a piped
  script cannot take a switch), then `watchInstall()`. It never POSTs the move:
  the tray decides at every start (fresh install, after the reboot
  `wsl --install` needs, upgrading a native install) and a second caller would
  race it. Linux and macOS walk the steps locally.
- The door is `http://127.0.0.1:7101` (not `localhost`, which may resolve to
  `::1`). Its other end is `crucible/host/controller_door.py`; change both together. The
  envelope is `crucible/tasks.py`'s `append_event` (`{id, event, data}`) so the
  Windows server can relay it under a task id without reshaping it.
- Event kinds beyond tasks.py's: `state` (a WSL table row, and how a caller learns
  a UAC prompt is coming) and `line` (a step's output). `done` carries the install
  result, unlike tasks.py's empty `done`, because the guest's name/url/config
  path cannot be read any other way from Windows.
- A `line` belongs to the last announced `step`. `progress` is only delivered to
  `onEvent`; no line is invented for it. Unknown event kinds reach `onEvent` and
  nothing else; a line that is not JSON, a non-string event name or a missing
  `data` object is a broken stream (`host_install_failed`), as is a stream with no
  `done`.
- The ndjson is parsed incrementally; a line split across chunks is held for the
  next chunk. The poster and the watcher share `parseEventLine`/`handleEvent`.
  For a watcher, `failed` is an event about someone else's move, so the stream is
  read to its end and the outcome file says how it ended.
- The door's bearer is the engine token from `%LOCALAPPDATA%\Crucible\config.toml`;
  the host-mode token and the guest token are the same token. No token yet is
  `host_no_token`. A host that is installed (`crucible.cmd` present: a `.cmd` shim
  because pip's `.exe` launchers bake an absolute interpreter path) and not
  answering is `host_unreachable`, distinct from `host_not_installed`.
- `wsl-outcome.json` (owner `crucible/host/outcome.py`) has five states; a sixth
  is refused. Terminal for an app: `done`, `cannot`, `reboot-pending`,
  `declined`. `failed` is not terminal because the tray retries once, and the
  watcher follows the retry.
- `DECISION_WAIT_MS` (195 s) is the tray's own presence-settle ceiling
  (`crucible/host/app.py` `PRESENCE_SETTLE_CEILING_SECONDS`: `WATCH_SECONDS` 15 +
  2 x (`RECIPE_TIMEOUT_SECONDS` 60 + `BOOT_WAIT_SECONDS` 30)). Until the tray
  decides, "nothing running and nothing recorded" is a normal answer; waiting
  past the ceiling waits for something that gave up.
- Bootstrap never elevates. `run-elevated` is data; `elevatedArgv` builds the
  `Start-Process -Verb RunAs` argv for an app to run when it chooses.

## Windows: WSL facts (`runner.ts`, `wsl.ts`, `target.ts`)

- Every process and file read goes through a `Runner`; tests script one.
  Argument arrays only, `windowsHide`, never a joined command line. Every call
  has a required timeout; a timeout is a reported failure. A timed-out child is
  killed with `taskkill /t /f`, because `child.kill()` reaches wsl.exe and leaves
  the guest-side process running.
- wsl.exe's own messages are UTF-16LE (with BOM); output from inside the distro is
  UTF-8, on the same handle and possibly in the same chunk. Decoding is per run
  of bytes (`segmentWslBytes`): UTF-8 never contains NUL, Latin UTF-16LE has a NUL
  high byte in nearly every unit; a run continues while this pair or the next has
  a NUL high byte, so a lone `—` does not end it. A BOM is stripped from each run.
  A chunk ending mid-character (odd UTF-16 byte, incomplete UTF-8 sequence) holds
  those bytes for the next chunk; nothing else is held.
- Lines split on `\n`, `\r\n` and bare `\r`, because pip repaints progress with
  `\r`.
- Always `wsl.exe -d <distro> --exec`: without `--exec` the distro's default shell
  pre-expands `$var` and `$(...)`; `--` does not help. Guest scripts therefore
  carry no host-side `$` and no backslash.
- wsl.exe halves backslashes once, quote-blind; `wslArgv` doubles them.
- `-u root` (`wslRootArgv`) selects the user wsl.exe starts; it is not elevation.
- `toWslPath` maps `C:\a\b` to `/mnt/c/a/b` and refuses UNC paths. Mapped drives
  pass every string test but WSL2 does not automount them; `guestPathFor` asks
  `realpath.native`, and "could not tell" is not "network drive".
- On win32 every command needs a named distro; there is no "default distro",
  because a server read from the wrong guest is the wrong server.
- `wsl -l -v`'s header is skipped by content, not position (some builds print a
  blank line first).

## Windows: the `crucible` distro (`distro.ts`)

- A Windows Crucible lives in its own distro named `crucible`, imported with
  `wsl --import` from Canonical's WSL image (series `24.04`, `current/`). Reasons:
  no interactive first-run prompt, no writes to a person's own distro, systemd is
  written rather than probed, and the GPU comes from the Windows driver.
- Canonical's `SHA256SUMS` beside the image is the digest's only owner; we store
  none. The sums file lists every image, so the row is found by filename, and a
  sums file that does not name our download (what `current/` moving looks like)
  is refused by name. A caller-supplied `rootfsUrl` has no sums file; the caller
  vouches for it. The image is downloaded on the Windows side (`curl.exe`,
  verified with `certutil -hashfile`), because `wsl --import` reads a Windows path.
- `finishImportScript` does what the image lacks, as one idempotent root script
  that is also emitted into `platform/wsl_table.py` for the host's importer: the
  `crucible` user, passwordless sudo, `/etc/cloud/cloud-init.disabled` (Canonical's
  image ran cloud-init at every boot, which held systemd and `crucible.service`
  for ~39 s looking for a datasource), a `WSLInterop` binfmt entry (with
  `systemd=true`, systemd-binfmt drops WSL's registration and every `.exe` from
  the guest fails "Exec format error"), and `/etc/wsl.conf`.
- `# crucible-rootfs` in `/etc/wsl.conf` marks our distro. A `crucible` distro
  without it and without a config is a partial import (unregistered and redone);
  with a config it is someone else's (`distro_unmarked`).
- `resolveDistro`: `exact` wins; else the `crucible` distro; else the app's
  setting; else `no_wsl_distro`. A `crucible` distro plus a config in the app's
  distro is `two_local_crucibles`. A distro that does not answer is not "a distro
  with a config". `detectHost` uses a looser pick and says which guest it asked.
- Never `wsl --shutdown` (stops every distro); only `wsl --terminate crucible`.
- `wsl-outcome.json` lives under `%LOCALAPPDATA%\Crucible\`; its name is spelled
  in `distro.ts` and generated into the Python host and `install.ps1`.
  `%LOCALAPPDATA%` is always read from the environment, never built from a
  username.

## WSL state table (`wsl-states.ts`)

- Each row is probe, code, sentence and action (`run`, `run-elevated`,
  `instruct`, `link`). The first matching row wins, so rows are ordered deepest
  cause first: `virtualization_disabled` precedes "WSL missing" because
  `wsl --status` also fails when the feature is off.
- Probes are keyed and cached; a healthy machine costs `--status`, `-l -v`, one
  `cat`, one `curl`, one `df`, one `id`. A disabled row's probe is not run, so
  reading a machine's facts never reaches the internet (`checkNetwork` is off by
  default).
- The network row HEADs every place the install downloads from and names the
  first unreachable one on stderr. The URL list (`{indexes}`) is filled by
  `crucible/host/wslstate.py` from the recipe files, the interpreter pin and the
  release wheel; it is not written down here because a list would drift the first
  time a recipe gains an index. `-I` because index pages are large; `-L` because
  release assets redirect to the CDN a proxy may block.
- A distro the person chose that lacks systemd is reported and asked about, not
  repaired: writing their `/etc/wsl.conf` changes their machine.
- `guest_root_unreachable` is the one hand-over left: the guest server is a system
  unit, so root is what installs it.
- `automatic` is data, not derived from the action kind, because `wsl_ready`'s
  action is `instruct` "Nothing to do." yet it is the most automatic state.
- The last row matches everything; a table whose last row stops being total is a
  bug.

## Local config (`config.ts`, `toml.ts`)

- The local server has one owner: `<CRUCIBLE_HOME>/config.toml`, the file
  `crucible/config.py` reads (`$CRUCIBLE_HOME`, default `~/.crucible`). On win32 it
  is read inside the guest through `wsl.exe --exec bash -c`. Missing is
  `no_local_config`; the guest script signals it with a dedicated exit code.
- Connect address is derived from bind: `0.0.0.0`/`::` connect on `127.0.0.1`.
- `parseLocalConfig` requires the keys `load_config` requires; the server would
  refuse to start on a file missing one.
- `toml.ts` reads only the dialect `tomli_w` writes from Crucible's config and
  refuses anything else (inline tables, multi-line strings, dates) with a
  `TomlError`, which becomes `config_unreadable`. Zero runtime dependencies is a
  package rule.

## Service and health (`service.ts`, `health.ts`)

- `ensureRunning()` goes through `crucible service start|status --json`, because
  unit names, launchd labels and domains are `crucible/service.py`'s. The status
  exit code is not read (1 means "installed and stopped"); the JSON says why.
- The binary is always `<CRUCIBLE_HOME>/server/bin/crucible`; a host without it is
  `no_server_runtime` rather than a hunt for an interpreter.
- `health()` maps SDK errors: unreachable means `ensureRunning()`, wrong token means
  config and server disagree, not-a-Crucible means something else holds the port.

## Versioning

- `BOOTSTRAP_VERSION` is a literal so ESM, CJS and bundlers agree without reading
  `package.json`. `test/unit-version.test.ts` fails on drift from `package.json`
  or the `@crucible/client` peer pin, and `scripts/release.sh` refuses a cut that
  drifts from `crucible/__init__.py`.
