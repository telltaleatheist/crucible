# The `crucible` command line (`crucible/cli/`)

Every verb in this package acts on **this machine's** installation and takes no address.
The client half (`crucible api …`, `crucible pair`) lives in `crucible/apiclient.py` and
speaks HTTP to a server that may be local, in WSL, or across the network (see
`docs/API-CLI.md`).

Exit codes: `0` success, `1` refused (a named reason on stderr, via `common._fail`), `2`
usage (argparse's own).

## Layout

| module | verbs |
|---|---|
| `__init__` | `build_parser`, `main`, and the names other modules and tests import |
| `__main__` | `python -m crucible.cli` |
| `common` | exit codes, `_fail`, `_backend_mismatch`, `_env_spec`, `_models_config`; the one place `detect_backend` and `load_config` are reached from |
| `init` | `init`, `--config-from` carrying, the desktop-reserve decision |
| `install` | `install`, `INSTALLABLE_JOB_TYPES`, `INSTALLER_FOR`, `SMOKE_IMPORT` |
| `capability` | `capability`, `ladder`, and the measure and capability steps `install` runs |
| `weights` | `remove`, `models`, `rvc`, `denoise` |
| `voices` | `voices list/pull/pin/check/card/export` |
| `orchestrator` | `orchestrator`, `guest` |
| `serve` | `serve` |
| `service_cmd` | `service install/uninstall/start/stop/status` |
| `doctor` | `doctor`, `env patch` |
| `uninstall_cmd` | `uninstall` |
| `token` | `token`, and the pairing-file helpers `init`, `serve` and `service install` share |

Each module exposes `add_parser(subparsers)` (`weights` has `add_model_parsers` and
`add_rvc_denoise_parsers`, because `voices` sits between them). `build_parser` calls them
in the order `--help` lists the verbs. `local`, `sharing`, `lan` and `apiclient` are
imported inside `build_parser` rather than at module scope, because a CLI's import time is
its `--help` time. For the same reason `serve` imports `crucible.api` only when it runs.

### Constraints on the layout

- **`-m crucible.cli` is baked into installed machines**: `launcher.py`'s shims,
  `local.py`, `desktop.py`, `host/app.py`, `host/runner.py` (which compares argv against
  `["-m", "crucible.cli", "serve", "--controller-stdin"]`), `uninstall.py`, the
  bootstrapper's install scripts, and `host/startup.py`, whose Startup item runs
  `runpy.run_module('crucible.cli', run_name='__main__')`. So `crucible/cli/__main__.py`
  must exist and the package must keep that name. `pyproject.toml`'s console script is
  `crucible.cli:main`, and `python -m crucible` imports `main` from here too.
- **Submodules call `common.detect_backend()` and `common.load_config()`**, never their
  own imported copies. Tests replace those two names on `crucible.cli.common`, and that is
  only one patch if every caller looks them up there. `serve` calls
  `token._sync_pairing_file` through the module for the same reason.

## Backends: no platform gate

`main` has no platform check. Windows is a backend (`llama-windows`), and every verb runs
on win32. Whether this machine can run what the config says is a question about the
backend, not the platform, and it is asked where a backend is read: `_backend_mismatch`
(from `init --backend`, `serve`, `service install`, `install`, `remove`, `env patch`)
refuses `backend_not_here` and names both kinds. It is the one wording for that fact. It
appends `WINDOWS_REFUSAL` only for a `cuda-linux` config found on a Windows host, because
that config is right about wanting vLLM and wrong about where it runs (inside the WSL2
guest).

`crucible orchestrator` still refuses off win32 (`host_windows_only`). This is a feature
check, not a platform check: on Linux and macOS the service manager already supervises the
server, and a tray would be a second owner of presence.

`init --backend` states what the caller expects and is checked against what is detected.
It never chooses a backend. `crucible orchestrator` passes `--backend llama-windows`.

## `init`

- **Token**: minted, or taken from `--token` (for `@crucible/bootstrap`, which mints on its
  own side), or carried by `--config-from`. `--token` and `--config-from` together are
  refused. A blank or whitespace token is refused.
- **`--config-from`** (written at 0600 by the host when it moves a Windows Crucible into
  the WSL guest) carries the token, `[routes]`, `[upstreams]` and the `[accelerator]`
  desktop reserve with its basis and note. Host, port, name, backend and job flags belong
  to the machine being initialised. A guest that inherited `backend = "llama-windows"`
  would refuse to serve on its own card. The reserve is carried because the Windows server
  and the guest share one card and one desktop: re-deciding it would re-measure a stated
  reserve or lower it on a quiet minute.
- **Desktop reserve** (`_decide_reserve`). The first answer wins:
  1. `--desktop-allowance-bytes`: stated. This always wins.
  2. `--config-from`'s reserve, with its value and basis unchanged.
  3. `--force` over a config whose reserve is **stated**: kept (`_existing_stated_reserve`).
  4. An NVIDIA card with nothing of Crucible's on it: **measured**
     (`ladder.measure_desktop_reserve`).
  5. Otherwise the backend's declared default, with the reason it was not measured in the
     note. A Mac always lands here: its reserve is a share of unified memory.

  It is decided before the config is written, so a measurement never sees a half-written
  home. Owen, 2026-09-26: *an existing reserve is never changed automatically*. A re-init
  is not a request to lower a reserve someone stated (owens-pc keeps 3 GiB for streaming).
  `crucible capability --measure-desktop` is the only deliberate replacement.
- **`[tts.<engine>]` footprints** are written by `init` from `declared_tts_footprints`. A
  voice from its own repo carries no machine facts, so without this table a server
  refuses such a voice with `engine_footprint_unset`.
- The `--desktop-allowance-bytes` help text writes `%%`, because argparse formats help
  with `%` and a bare `25% of` crashes `init --help`.

## Pairing file and the pairing line

- `<CRUCIBLE_HOME>/pairing` holds the **loopback** line whatever the server is bound to.
  It answers "an app on this machine wants in". `pairing.write_pairing_file` is the one
  writer, and it sets the Windows ACL.
- It is written by `init`, `service install` (with the same token, so installing a unit
  unpairs nothing) and `serve` at startup (`_sync_pairing_file`). `serve` rewrites it when
  it is absent or its line differs from the config, because a rotated token, a renamed
  server or a moved port leave a file that is worse than none. It uses the **config's**
  host and port, not the run's `--host`/`--port`, so a developer's `serve --port 7999`
  does not repoint every app on the box. A write failure there is printed and is not
  fatal: the server works without the file.
- `init` and `service install` do **not** print the line (`PAIRING_NOT_PRINTED`,
  fresh-install #33). Both run inside every install, so printing it put the token into
  every install log. Only `crucible token --url` prints it. That verb needs no `--show`:
  its name says it prints the door.
- `_pairing_lines` puts the loopback line first, always. On a wildcard bind
  `reachable_urls` has no loopback entry, and an app on the server's own machine would
  otherwise get whichever interface the OS listed first.
- `_pairing_permission` says "ACL: this user only" on win32. A Windows file has no POSIX
  mode, and `stat` reports 0o666 whatever the ACL says.

## `serve`

- `app.state.bind_host`/`bind_port` hold where it really listens. `GET /v1/setup` builds
  pairing lines from the bind address, so a `--host 0.0.0.0` override must reach it.
- `uvicorn.run(timeout_keep_alive=KEEP_ALIVE_SECONDS)` is set on purpose. uvicorn's
  default of 5 s against Node undici's pooled ~4 s meant a client's next request could
  land on a socket being closed and read ECONNRESET while the server was fine.
- `--controller-stdin` (hidden) hands the server to `host.child_lifecycle.run_owned_server`.

## `service`

- The backend is detected and compared with the config (`_service_context`), because
  `serve` refuses a mismatch and a unit that cannot start is worse than none.
- Host and port are read from `config.toml` at install time and baked into the unit. After
  changing either, re-run `service install`. A unit that re-read the config would change
  behaviour without anyone installing anything.
- `service status` exits 0 only when the service is running, so scripts can gate on it.
  `running` is three-state: `None` means no manager could be asked, and is not reported as
  "no". On systemd, linger is reported and never assumed.

## Capability, the ladder, and `install`

- **`_decide_here`** sizes on `backend.gpu.vram_bytes`, never on free VRAM. Capability is
  a fact about the host. Free VRAM is a fact about this second, and a browser open during
  an install must not permanently disable TTS. The runtime guard answers the free-memory
  question at load time (`insufficient_memory`). It also passes `gpu_vendor` (which pool
  this is on `llama-windows`), the config's `local_models` selections, and
  `ladder.card_for` (card generation and measured facts: same bytes, different precision).
- **`_write_capability` rewrites the whole `config.toml`** through `write_config`. So it
  must carry forward everything it is not changing: token, reserve with basis and note,
  `retention_days`, `tts_engines`, `routes`, `upstreams`, and the advertise lists. Each
  omission silently resets something (an unrouted server, a 7-day retention, every repo
  voice refusing `engine_footprint_unset`, a measured reserve turning "stated").
  `init --force` mints a new token and must never become the repair for a capability
  decision.
- **`crucible capability`** exists apart from `install` because card, reserve and
  manifests change independently of envs. Re-deciding should not mean rebuilding a venv.
  It is a dry run by default. `--write` may only turn a flag **off**: a flag means "this
  server offers this type", which needs the env too, and only `install` knows that.
  `--measure-desktop` re-measures the reserve, refuses by name while anything of
  Crucible's is on the card, prints old and new, and writes.
- **`install` order**: build the env from its recipe (the only path; there are no packs)
  → `_smoke_import` → `_ensure_tools` → `_measure_step` → `_capability_step`.
  - `SMOKE_IMPORT` holds import names, not distribution names (`mlx-lm` → `mlx_lm`,
    `faster-whisper` → `faster_whisper`, …). `asr` differs by backend (`faster_whisper`
    vs `mlx_whisper`). The tts key is the env directory's name, which carries the engine
    on cuda-linux (`tts-higgs-v3`). An env that cannot import its library is not
    installed, whatever pip said.
  - `_ensure_tools`: Owen, 2026-09-26 (fresh-install #25). ffmpeg comes from Crucible's
    own `tools` release, pinned by sha256, and every `install` places it. It runs before
    the capability step, so no flag turns on without ffmpeg. Re-running retries only
    ffmpeg. The silero VAD for `asr`'s `speech_only` is placed too, but its failure is not
    a refusal: a `speech_only` job fetches it itself.
  - `_measure_step`: Owen, 2026-09-26: *"our measurement tool should determine how much
    space is available, whether tensors are available, cuda graphs, vllm, etc. and install
    the best the user can use"*. It measures first, then decides. It never refuses the
    install. A busy card waits, and unknown refuses nothing. `--no-gpu-measure` limits it
    to nvidia-smi. `crucible ladder` runs the same thing on demand.
  - `_capability_step` may turn flags **on** (this door just proved the env exists). It
    writes the record and flags **then** refuses if any type came out disabled, so the
    reason survives (R6). One env can serve several types (`rvc` also serves `denoise`),
    and each gets its own verdict. It prints the same `capability.install_plan` lines the
    operator page shows.
- **`INSTALLER_FOR`** maps a job type to the `install` verb that builds its env: `denoise`
  → `rvc` (shared env), `pages` → `llm` (`pages` is a capability class served by the llm
  proxy, not a job type). `doctor`'s "could enable" notes read it, which keeps `echo`
  (no installer) out and points `denoise` at `install rvc`.
- **`_env_spec`**: `tts` needs `--narrator-engine` on cuda-linux (one venv per engine; two
  engines cannot share a torch). It is refused for `llm`, which has one env.
- **`llama-windows` has no env**. `install llm` (and so `pages`) fetches the pinned
  llama.cpp release. `tts`, `asr`, `align`, `rvc`, `denoise` are refused `needs_wsl` with
  `capability.NEEDS_WSL_REASON`.

## Weights verbs

- `models` is one namespace over `models/`, `asr/` and `align/` manifests
  (`_all_manifests`). An id collision is refused, not settled by read order. `rvc` and
  `denoise` have their own verbs and namespaces because their weights are not repo
  snapshots (one archive, or two named files), and so an RVC `sigma` cannot collide with a
  narrator voice `sigma`.
- `remove` asks the door's questions in the door's order (unknown, not installed, in
  use), except whether a running server holds the subject, which only the server process
  knows. With a server up, `DELETE /v1/catalog/{kind}/{id}` is the one to use.
- `denoise list` reports "stamped" and "present" separately. Two hand-placed files are
  usable but unstamped (`--force` pins them). `rvc pull-base` and `denoise pull` re-check
  the tree after the pull and refuse if a file is missing.
- `voices pull` prints the path for a local (`source = local`) block. Local voices are
  not pullable, and `voices list` does not offer a pull for them.

## Voices

- `voices check` parses a `crucible-voice.toml` exactly as the loader would, merged with
  **this machine's** `[tts.<engine>]` table, because the question is "would a server here
  load it". A local file has no pin, so a stand-in repo/sha is used. Those play no part
  in the checks.
- `voices card` renders README.md from the manifest. For a local file it prints the two
  blocks it owns (frontmatter and limits). `--upload` needs a repo reference.
- `voices export` prints the rows the repo schema does not carry on stderr. They move;
  they are not dropped silently.
- `voices pin` loads the pin (`voice_for_pin`) before writing it, as
  `PUT /v1/voices/{id}` does.
- The training ladder parses `voices pull`'s `"<id>: <repo>@<sha12> for <arm>"` and
  `"<id>: N GB at <path>"` lines. Do not change them.

## `doctor`

- `problems` make the host unhealthy. `notes` never do. These are notes, not problems:
  stranded weights (`catalog.stranded_weights`), an unmeasured reserve that the ladder saw
  the desktop fit under (`desktop_reserve_unmeasured`), a type waiting only for weights
  (`awaiting_weights`, fresh-install #40: `install rvc` turns on `denoise` without
  pulling a separator), and a type the card could hold that is off (`could_enable`).
- **Capability**: a record is stale when its backend or pool size differs from this host.
  The check compares numbers, not dates. The only capability problem is a flag that is on
  while every class behind it is refused (the first request would OOM).
- **PATH** (Mac audit, 2026-09-14): `doctor` in a non-login shell said "no ffmpeg on
  PATH" while the service was fine. The service's recorded PATH
  (`service.read_recorded_path`) is shown beside the shell's. `agree` is three-state
  (`null` when there is nothing to compare). Differing is normal, and every other line is
  measured in the shell.
- `config_permissions` is skipped on win32, where the mode is not what restricts the file.
- `llama-windows`'s llm row is the llama.cpp engine at its pinned tag
  (`_llama_engine_report`), shaped like an env row. Asking `jobenv` for an llm env there
  crashes. Its patch checks use a path that never exists (`_no_python_env_dir`), so they
  read `not_applicable`.
- **Provenance** names two drifts apart: `env_recipe_drift` (a `pip install -r` into the
  venv) and `narrator_sha_drift` (one `pip install --no-deps`). The verdict is
  `jobenv.plan_install`'s, so doctor's sentence and the installer's remedy are one. A
  planner refusal is a doctor problem, not a crash.
- CUDA toolkit links are checked only when the tts env exists. Otherwise the env row
  already states the one cause. An llm patch in `no_env` is not a second problem for the
  same reason.

## `env patch`

The installer's `env-patch-llm` step. An upgrade installs a new wheel and never runs
`crucible install`, so patches for an unchanged recipe would never be applied. It exits 0
only when every row is `applied` or `not_applicable`. No env installed is not a failure.

## `uninstall`

`crucible/uninstall.py` owns every decision. The verb builds the plan, runs it unless
`--dry-run`, and prints it. The home is always `crucible_home()` and never a flag. On a
command that deletes directories there must be one answer to "where". The backend is read
from the config, never detected, because an uninstall must work with a missing driver, a
half-removed config, or a busy GPU.

## `orchestrator` and `guest`

`orchestrator --install-startup` / `--remove-startup` manage the Startup item and exit.
`--try-again` runs the Linux-engine move once more for a machine with no app, then prints
the outcome's sentence. Bare, it runs the tray. `--headless` runs the controller with no
tray. `crucible guest <words…>` forwards a command to the `crucible` inside the WSL engine
as the engine's user (`host/guestcli.py`).
