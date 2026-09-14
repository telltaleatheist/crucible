# What came home from Foundry's WSL vLLM launcher

Owen, 2026-09-13: *"make sure you relay specifics about how vllm in wsl worked so bookforge
can recreate it in crucible… we're going to want to keep the logic. its just shifting from
the foundry codebase to the crucible codebase."* Foundry deleted its launcher that night
(`app/electron/vllm-server.ts`, `wsl.ts`, `backend-setup.ts`, `env-install.ts`,
`env-catalog.ts`, all at foundry `323dc63`; read them through git, they are gone from the
tree). Foundry-pc-1 wrote the load-bearing facts down before deleting; this file is where
each of them lives now — **already in Crucible**, **belongs to the bootstrapper**, or
**a ruling Owen owes** — so that nothing is lost and nothing is duplicated.

The one structural difference that changes most of the list: Foundry's launcher was a
WINDOWS process reaching into the guest with `wsl.exe`. Crucible IS the guest process —
`crucible serve` runs inside WSL2 as a systemd user unit (PHASE11-SERVICE.md) and starts
vLLM as its own child. Everything about crossing the Windows/WSL boundary therefore moves
to the one thing that still crosses it: the app-side bootstrapper (`@crucible/bootstrap`,
PHASE5-APPS.md section 6), and BookForge's `electron/crucible/local.ts`, which reads the
guest's `config.toml` through `wsl.exe`.

## 1. Already in Crucible, with where

| Foundry's fact | Where it lives here |
|---|---|
| `vllm serve <model> --trust-remote-code --host 0.0.0.0 --port … --gpu-memory-utilization 0.5 --max-model-len 32768` for dots.ocr; `--trust-remote-code` mandatory (pydantic ValidationError without it) | `models/dots-ocr.toml` `engine_args` carries exactly those four, plus `--max-num-seqs 16`; `crucible/engines/vllm.py` composes `--model/--served-model-name/--host/--port` and appends the manifest's args. Numbers are the manifest's, per model, never a launcher default. |
| GPU utilisation 0.5 flat under WSL's dxg layer (every reserved GiB is also committed host RAM; reserve for the model's need, not the card's emptiness) | per-model `engine_args`; the reason is recorded here because the manifest only carries the number. The desktop reserve (3 GiB on cuda-linux, PHASE9) is the same instinct applied to admission. |
| A first serve downloads ~6 GB from HF, so readiness must wait up to 15 min | Does not arise: `crucible models pull` fetches weights at the pinned revision BEFORE anything serves (`crucible/weights.py`); a serve that finds no weights refuses by name and never downloads. |
| Readiness by polling `GET /v1/models` | `crucible/engines/base.py` — readiness is the engine answering its own door, with a SILENCE timeout that any log line resets (`LOAD_SILENCE_TIMEOUT_SECONDS`), so a slow load is not a dead one. |
| Failure = fatal log patterns (`CUDA out of memory`, `ValidationError`, `ModuleNotFoundError`, `Address already in use`, HF fetch errors), EXCLUDING lines prefixed `INFO|WARNING|DEBUG` because a healthy vLLM warns `No module named 'vllm._C'` | **Deliberately different, and R4 is why.** Crucible fails a start on the FACT — the process exited, or the door never answered — and hands back the log tail (`LOG_TAIL_LINES`) as the message. A pattern that KILLS a start is a log line made load-bearing (ARCHITECTURE.md R4). Foundry's patterns are worth keeping as *explanations*: the next improvement to `base.py` is to classify a dead engine's tail with them and put the classification in the refusal beside the tail, never to act on a pattern while the process lives. Their INFO/WARNING exclusion is the part that makes that classification safe. |
| Port held → refuse, not move | `served_common._refuse_if_misbound` / `_BOUND_PORTS_SCAN` (narrator), `llama-model-server.ts` `assertPortFree` (BookForge): a stranger on the port is a refusal by name. |
| One in-flight start promise | `Residency` holds one engine per card and serialises loads on the lane; a second load is `409 server_busy` naming the first. |
| Stop from INSIDE the distro (`pkill -TERM … --port 8000`), wait up to 60 s for CUDA to release, only then a Windows-side kill; never SIGKILL a CUDA holder | Crucible is inside the distro: `stop()` sends the engine's own quit, waits `QUIT_GRACE_SECONDS`, then SIGTERM; there is no Windows-side kill in the design at all. (memory: `wsl-wedge-proofing`.) |
| Scope the kill by port so a vLLM on another port is never touched | Crucible only ever signals children it spawned; it has no `pkill -f`. |
| `checkVllm` via `importlib.find_spec`, not `import vllm` (tens of seconds) | `crucible doctor` reads `pip list` from the env (`jobenv.installed_packages`) and never imports the engine. |
| Env name distinct from hand-made envs (`foundry-vllm`) so a rebuild can never hit the user's | `~/.crucible/envs/<type>/` — Crucible's own tree, never a conda env of the user's. |
| Never fall back from the route the user chose; idempotent installs; talks the whole time | `crucible install <type>`: venv from the server interpreter only; a half-built env (no `crucible-env.json`) is "did not finish, re-run", never silently reused; `--verbose` streams pip. |
| Measured environments pinned | every `envs/*/*.txt` is a resolved set from a real install as of 2026-09-13 (`1ec936d`, `b751319`), with the interpreter and package count stamped in `crucible-env.json`. |
| Adopt a server already on the port, never own it | **Not carried, on purpose.** Crucible refuses a stranger on its port. A Crucible has no business driving an engine it did not start: it cannot know the weights, the args or the owner (`model-identity-belongs-to-crucible`). The adopt case Foundry had — a user's hand-run vLLM — is what `--endpoint` on Foundry's own door is for; it never goes through a Crucible. |

## 2. Belongs to the bootstrapper (`@crucible/bootstrap`, unbuilt) and to `local.ts`

These are Windows-side facts about driving the guest, and they are exactly the machinery
`ensureRunning()` / `install()` will need. Carry them verbatim:

- **Always `wsl.exe -d <distro> --exec …`, never the implicit shell** (it pre-expands `$var`
  on the host; memory `wsl-exe-implicit-shell-trap`). Foundry used `-e bash -lc`; same rule.
- **wsl.exe's OWN output is UTF-16LE with a BOM; output from inside the distro is UTF-8, on
  the same handles.** Decide per chunk by looking for interleaved NULs (`decodeWslBytes`).
  `local.ts` today reads a small TOML through `spawnSync(..., encoding: 'utf-8')` and has
  not hit this because the guest's `cat` output is what it reads; the moment the
  bootstrapper reads wsl.exe's own messages (a distro that is not installed, a VM that is
  booting) it will.
- **Every one-shot call gets its own timeout**: a booting distro or a blocking profile makes
  wsl.exe never return. The cold-VM `exit −1` the Servers row met tonight
  (CRUCIBLE_ROLLOUT_PLAN.md 2.2) is the same animal and is owed a reproduction.
- **Argument arrays, never shell strings; wsl.exe halves backslashes once before bash
  exists** — anything carrying backslashes is doubled or sent on stdin.
- **`toWslPath` maps `C:\a\b` → `/mnt/c/a/b`, refuses UNC, and `realpath.native` catches
  MAPPED drives WSL2 does not automount** (memory `wsl-cannot-see-network-drives`).
- **Conda found by `test -x` at `~/anaconda3 | ~/miniconda3 | ~/miniforge3`, never `which`**,
  so the conda whose `envs/` the interpreter lands in is the one used. Crucible's server
  interpreter is what `crucible install` builds venvs from; the bootstrapper's job is only to
  find or create THAT one interpreter.
- **Prebuilt environments as release assets**: the environments Foundry was measured with
  were pinned and shipped (`foundry-env-wsl-x64-v1.tar.gz`, three parts, sha256 per part,
  release tag `env-v1`), downloaded on the Windows side, handed across as `/mnt/c/…`, and
  unpacked by the DISTRO's own tar — **never through `\\wsl$`**, whose 9P redirector
  flattens the symlink thicket a Python install is; a null sha256 is a refusal; an install
  replaces only a directory carrying its own stamp file. This is the friend's install path
  for Crucible's `~/.crucible/envs/*` (DESIGN.md section 5 forbids GitHub Releases for
  WEIGHTS, not for environments; BookForge already ships envs this way — memory
  `env-hosted-on-github-releases`, the 2 GiB split).
- **`streamInDistro` forwards pip's stdout AND stderr line by line, splitting on `\r`** as
  well, because pip repaints its bar.

## 3. Rulings Owen owes (already listed in PLAN.md; restated with Foundry's evidence)

- **Drain, not between jobs.** Foundry kept the server up while the queue had work and
  brought it down when the queue DRAINED, with an optional keep-warm window
  (`keepServerWarmMinutes`, default 0) that a new job cancels; three books loaded the model
  once; an empty queue had no claim on half the card; bringing it back cost ~44 s. Crucible
  today never unloads on its own (PHASE5-APPS.md section 7, proposed default "no idle
  unload; the Servers row shows what is resident and for how long"). Tonight's rollout met
  the consequence: a 9 B sat resident after the Foundry proof until a person unloaded it,
  and Owen saw "something loaded and nothing happening". Foundry's shape — up while a
  client has work, down on drain, keep-warm as a config key — is the alternative, and the
  model lease (CRUCIBLE_ROLLOUT_PLAN.md, rulings owed) is the fact it would key on.
- **Only a reading waits for the server; a rendering replays the bank and reads nothing.**
  Foundry's rule; Crucible's equivalent is that a client with a bank asks for nothing —
  nothing to build, recorded so the queue never pins a server for a step that does not
  need one.
