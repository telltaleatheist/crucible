# scripts/ internals

What the repo's operator scripts depend on that their code does not say by itself.

## Releasing

### The seven version places (`bump.py`, `release.sh`)

The version is stated in seven places and `release.sh` refuses a cut unless all agree:
`crucible/__init__.py`, `pyproject.toml`, `sdk/ts/package.json`, `sdk/ts/src/version.ts`
(the SDK's User-Agent), `sdk/bootstrap/package.json`, its `@crucible/client` peer pin, and
`sdk/bootstrap/src/version.ts`. `bump.py` owns the list (`PLACES`).

- **Never search-and-replace a version.** Each place has its own anchored pattern that must
  match exactly once; a file that changed shape refuses the bump. Other files name old
  versions on purpose (e.g. `docs/PHASE14-ENVPACKS.md` records a measurement taken on
  v0.6.6 -> v0.6.7), and rewriting them would turn a fact into a lie.
- **Generated files carry the version too.** `modules/*.module.json` names it as
  `<version>+<content hash>` and `docs/API.md` is built from the app, so `bump.py`
  regenerates both (`GENERATORS`) and `release.sh` checks each with `--check`. v0.6.3 shipped
  stale module files because nothing compared them.
- The standalone installers contain no version: they resolve the release at run time.
- The proof a bump worked is `release.sh --dry-run`, the same gate the cut uses.

### What a release carries (`release.sh`)

Our code and nothing published elsewhere (PHASE20-CODE-NOT-ENVIRONMENTS.md section 1), under
one tag `v<version>`: the sdist, the `py3-none-any` wheel, `<wheel>.sha256`, the client
tarball, the bootstrap tarball, `install.sh` and `install.ps1`. The interpreter
(python-build-standalone, pinned by digest in `crucible/interpreter.py` and
`sdk/bootstrap/src/interpreter.ts`), job environments (PyPI, via
`crucible/envs/<type>/<recipe>.txt`) and the WSL image (Canonical) are not release assets; the
release notes say so explicitly for readers of old releases.

- **The digest is required.** Both installers fetch `<wheel>.sha256` and compare it before pip
  sees the download. One line, digest first, the shape `sha256sum` writes and
  `awk '{print $1}'` reads; macOS uses `shasum -a 256`, which writes the same columns.
- **The installers are generated** from bootstrap's own step list, so an app-driven install
  and a hand install cannot differ; a stale committed installer refuses the cut.
- The bootstrap's dev dependency on the client is `file:../ts`, so the client is built before
  the bootstrap's `npm ci`.
- **An existing tag (local, remote, or as a release) refuses the cut.** Re-cutting a version
  is how two sets of bytes end up with one name.
- `--branch` exists for one case: the first release of a feature branch about to merge, so
  its tarball URL exists before the merge. The branch is named in the release notes.

### Promotion (`promote_release.py`)

Read-only by default. `--publish` moves `releases/latest` and also requires
`--confirmed-install-smoke`, an attestation that a person installed the candidate; release
metadata cannot prove that, so nothing automates it (`ship.sh` only prints the command).

- **Asset names are read from `release.sh`**, from its `gh release create` invocation, never
  listed here: gh's contract is `create <tag> [files...]`, so the first positional is the tag
  and a `"$VAR"` after a `--flag` is that flag's value. `NAME="value"` assignments are expanded
  (bounded, so a cycle fails instead of hanging).
- Every asset must carry this version; a leftover from another release is refused, because
  promotion would make installers fetch bytes that are not this candidate's.

### One command (`ship.sh`)

Order: clean and pushed tree; `bump.py`; commit and push; `release.sh`; optionally `deploy.sh`
(`--deploy`); print the promote command; print a timing table.

- **No test step, and no flag to add one.** Owen, 2026-09-18 (PHASE20 section 7): *"Normal
  deploy does not need 25 minutes worth of tests. We should run one or two focused tests on
  the area of code we changed before we reach the deploy stage. By the time we reach deploy,
  we should know it's going to work already."* Tests run on the branch via `tests.sh`; the
  release script deliberately names neither the selector nor the suite.
- **No dry-run gate before the commit**: `release.sh --dry-run` refuses a dirty tree, which a
  just-bumped tree is. `release.sh` checks and builds everything before creating anything, so a
  failure after the commit leaves only a bump commit, which `ship.sh --no-bump` resumes.
- **The timing table prints from the EXIT trap**, on failure as well as success. `step` is
  both banner and clock; the phase whose banner was last on a non-zero exit is the one that
  failed. The trap is armed after argument parsing.
- `--deploy` passes `--yes` to `deploy.sh`: nobody is at the keyboard. Per-machine times are
  read back from `deploy.sh`'s own output (only it knows them; timing the call learns only
  the slowest machine).

## Deploying (`deploy.sh`)

The fleet is `pc` (Windows host driving the WSL guest `Ubuntu`) and `mac` (an ssh alias). A
different operator has a different list.

- **What a machine runs is read from the machine**: `<CRUCIBLE_HOME>/installation.json`,
  before (skip if already there; `--force` overrides) and after (an install whose record does
  not name the release is a failure). Readers print a version, `none`, or `unreachable`;
  unreachable is never reported as up to date.
- **The PC has two records and one install.** Only `install.ps1` runs; the host carries the
  guest (PHASE15-HOST.md 4.3/4.4). Owen, 2026-09-18: *"windows is the driver; the thing moving
  wsl forward. use the established, installed, functional system to drive the new one."* A
  deploy must never be a second driver of the guest. The PC's reading collapses to a bare
  version only when host and guest agree, otherwise `host:<a> guest:<b>`, which cannot equal
  the release.
- **The installer comes from the tag being installed**, not `releases/latest` (which only
  moves on promotion).
- **Fetch to a file, `sh -n` it, then run it; never `curl | sh`.** A pipe discards curl's
  status, and a truncated transfer (measured 2026-09-17) ran half an installer that had
  already written the new version into `installation.json`.
- The remote payload's format string is single-quoted so `$(mktemp)` and `$f` reach the far
  machine as text. On the Mac it runs under `"$SHELL" -lc`, expanded remotely: a bare ssh
  command gets a non-login PATH without Homebrew, and `bash -lc` reads the wrong profile for a
  zsh account.
- `install_pc` returns the installer's status explicitly: `set -e` is disabled inside a
  function called as an `if` condition.
- **The record may lag the installer**: it is published when the runtime starts, and on the PC
  the guest is carried after `install.ps1` returns. `await_release` polls every 2 s up to 150
  times, sized from the host's `PRESENCE_SETTLE_CEILING_SECONDS` plus the guest install and
  restart. It returns the moment the record matches.
- **Busy servers are not restarted.** Before a restart the machine's own `GET /v1/activity` is
  asked over loopback; `running`, `queued`, `streaming`, a chat in flight or a lease refuses
  by name, `--interrupt` overrides. A resident model with nothing using it is not busy, and an
  unreachable server is not busy (reported as `idle(<why>)`). A deploy on 2026-09-20 restarted
  the PC six minutes into a 128-chunk render and lost all of it.
- **Machines install in parallel**, each in a subshell whose verdict is a file
  (`<machine>.seconds` always, `<machine>.why` on failure) because a background status cannot
  name a machine. Every line is prefixed with its machine and stderr is merged into stdout so
  lines do not interleave mid-line. The confirmation is asked once, before the fan-out, since
  the subshells share stdin; `|| answer=""` turns a closed stdin into a refusal.
- `--only` names are validated; the retired names `wsl` and `windows` are refused with what
  replaced them.

## Tests (`tests.sh`)

Owen, 2026-09-18: *"If we change crucible's handshake logic, we don't need to re-run the GPU
test. We can test the handshake logic we just built and assume the GPU works since it did last
time we changed anything."*

- **Selection**: a changed `tests/test_x.py` runs itself; any other `.py`, `.ts` or `.sh`
  runs the test files that name it, trying its repo path, then its module stem
  (`crucible/*.py`), then its basename; anything else selects nothing. Every uncertainty
  resolves to FEWER tests; `--all` is the proof. A file no test names is reported, not widened.
- A name that matches nearly every test file (e.g. the stem `tests`) is discarded and the next
  name tried.
- A file whose only change is its version literal (`crucible/__init__.py`, `pyproject.toml`
  on every release) selects nothing.
- `grep --include` must come before `--`; after it, it is a file operand and `.pyc` caches get
  handed to pytest.
- The base is "since the last tag plus the working tree". Tags are created server-side by
  `gh release create --target`, so local tags are usually stale; an untrustworthy base is the
  one case that widens.
- Live keepers are shell scripts and are never selected.
- On Windows pytest runs inside WSL2 (the CLI refuses Windows), reusing `CRUCIBLE_WSL_DISTRO`
  and `CRUCIBLE_WSL_ENV`. It refuses while `train_lora.py` is running (13 GB guest, one card),
  and `flock` serialises concurrent runs.

## The PHASE15 button (`testrun-phase15.sh`)

Owen, 2026-09-14: *"get everything ready so we can just hit a button and have the tests run."*
Stages run in order, stop at the first failure, and write
`C:\tmp\phase15-testrun\<timestamp>\report.md` as they go. It never asks a question: anything
optional (API key, page, BookForge checkout) is looked for by name and reported SKIPPED.

- Every command goes through `run`, so `--dry-run` has no second code path.
- The trainer guard greps for `[t]rain_lora.py` so the pattern does not match itself.
- The Anthropic key goes into a 0600 file inside the guest and is deleted in the same
  command; it never reaches argv, a log, or the report. The config is restored before the
  assertion.
- Only T6/T7 need the card. T9 (the Mac) and staging are deliberately not in this script.
- T10 probes the host door with a GET, never a POST (a POST starts a real WSL install); the
  staged server uses 7102 because the door is on 7101. A POST with `CRUCIBLE_HOST_DOOR` set is
  a 202 with a task id; failures are read from the task, not the POST body.

## Page and chunk probes (`read_one_page.py`, `one_cleanup_chunk.py`)

- **The caller owns residency.** Crucible never loads a model to answer a chat
  (`model_not_resident`), so these submit `load-model`, ask, then `unload-model`; a
  llama-server left holding the card breaks the next stage.
- **The measurement is recorded before the unload.** A `finally` around the read let the
  unload's refusal replace a page that had been read. An unload refused `model_not_resident`
  is success: the card is already clear of the model.
- Error codes are read from `error.code`, never matched in prose.
- `thinking: false` is sent as `chat_template_kwargs`, as BookForge does; otherwise Qwen3.5
  spends the budget on reasoning and returns no content.
- `read_one_page.py` builds its request from `crucible/pages.py` (prompt, dpi, pixel budget).
  It writes `answer.json` and `shape.json`; T7 compares the shape (block keys and categories in
  order), never the text, since two engines may disagree about a character but not the dialect.
  Rasterising is the app's work; a PDF needs `pypdfium2`, and without it the script exits 2
  ("could not try", pytest's collection-error code) so a caller can tell SKIP from FAIL.
- The cleanup chunk is invented text with the two defects the pass fixes, so no test run
  carries a copyrighted page.
- `task_field.py` prints nothing (exit 0) for a document that does not parse or lacks the
  path, because its caller polls and makes its own assertion.

## Generators

- **`gen-api-docs.py`** writes `docs/API.md` from the FastAPI app with every job type
  enabled. Auth scopes are read from each route's resolved dependency tree, descending into
  `_IncludedRouter.original_router` (fastapi 0.141.1 no longer splices included routes) and
  carrying router-level dependencies down. Routes are keyed by `path_format`, the spelling the
  OpenAPI document uses (`{subject_id:path}` vs `{subject_id}`). A path the walk did not reach
  is an error, never defaulted to "open". Union types are joined with ` or ` because a bare `|`
  ends a table cell.
- **`gen-foundry-lineup.py`** and **`gen-modules.py`** put this checkout first on `sys.path`
  and verify it won: manifests are found relative to the package, and importing another
  checkout's package would write its catalog into this one. Output is LF on every platform,
  since the files are vendored into other repos and compared by content. `--check` runs in CI
  and `tests/test_lineup.py`/`tests/test_modules.py` assert the same. The lineup check ignores
  `generated_from`; the module check ignores nothing (the version is a content hash).
- A vendored `<app>.module.json` is never edited in the app: changes start in
  `modules/<app>.toml`. An empty `modules/` is a broken checkout.
- `gen-modules.py` does not touch `foundry-lineup.json`; each has its own generator and guard.

## Voice bands (`check-voice-bands.py`)

BookForge's `electron/data/higgs-safe-bands.json` is the authority for safe bands (620 renders
at n=32/rung, pooled by the packer's character counts). Crucible advertises the band and the
client packs to it, so a wrong band silently damages someone else's audio. It is a script and
not a test because Crucible must run where BookForge does not exist. The overlay is sparse:
a voice it does not mention is fine, and non-band keys (`_README`) are skipped.

## Live keepers and measurement

- **Never SIGKILL in WSL2.** Servers are stopped with SIGTERM; the server's shutdown stops its
  engine. A hard-killed GPU process wedges the distro until Windows reboots.
- `keeper-llm-live.sh` and `keeper-tts-live.sh` have a local mode (throwaway server against
  the real `~/.crucible`, refusing by name if env or weights are missing) and a remote mode
  (both `CRUCIBLE_URL` and `CRUCIBLE_TOKEN`; one without the other is refused).
- On Git Bash `python3` is a Store stub; the keepers find a real interpreter and name it.
- A chat before any load is `409 model_not_resident`; a tts render loads its own voice.
- `keeper-tts-live.sh` refuses if anything other than this server holds more than the desktop
  allowance on the card, samples the card during the render (the peak is the figure), and
  prints manifest lines to paste.
- **`measure-llm-memory.sh`** measures memory used minus memory used before the engine
  started, on both backends. Not "available" (macOS reclaims inactive pages) and not RSS:
  MLX memory-maps weights, and on `qwen3.8-27b-4bit` RSS read 14,643 MiB against a 32,116 MiB
  delta. The context prompt is sized with the model's own tokenizer in one encode and a slice
  (re-encoding per word is O(n^2)). On failure the throwaway home's engine log is kept.
- **`calibrate-kv.sh`** runs the engine's real argv with `VLLM_LOGGING_LEVEL=DEBUG`, so vLLM's
  profiler prints every term (`requested = total x util`, `non_kv = total_consumed +
  transient_peak_headroom`, `available_kv = requested - non_kv - cudagraph_estimate`);
  `GPU KV cache size` over `Available KV cache memory` is bytes per token. `KV_BYTES` sets
  `--kv-cache-memory-bytes`, which ignores `gpu_memory_utilization` and skips profiling.
  nvidia-smi is sampled at 1 Hz because `total_consumed` is a whole-card delta. The engine env
  comes from Crucible (without `VLLM_WSL2_ENABLE_PIN_MEMORY` the load dies with
  `UVA is not available`).

## End to end

- `e2e.sh` runs the SDK e2e suite against a throwaway server (own `CRUCIBLE_HOME`, free port,
  echo enabled) on Linux/macOS.
- `e2e-from-windows.sh` keeps the client native and the server in WSL2 (Windows localhost is
  forwarded into the guest). Git Bash path conversion is disabled because every path is a
  guest path, which also means `>/dev/null` must be a shell redirect, not `curl -o`. `wsl.exe
  --exec` avoids a Windows-side shell pre-expanding `$vars`.
- `wsl-serve.sh` detaches the guest server with `setsid`, since `wsl.exe` tears down its
  session on exit and a `nohup ... &` dies with it. The child writes its own pid file.
