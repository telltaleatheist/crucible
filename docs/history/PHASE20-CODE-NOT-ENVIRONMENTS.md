# PHASE 20 — Code, not environments (Owen, 2026-09-18 22:30–22:55)

**The ruling.** A deploy ships CODE. An environment is downloaded ONCE, at install, from
wherever its bytes are published, and is never rebuilt, re-hosted or re-downloaded because
code changed. A normal deploy runs NO tests. This is the BookForge pattern
(`electron/tools-env-bootstrap.ts`: the env is a tarball on GitHub releases, fetched on
first run, repaired by the doctor), and it is the pattern for everything.

**Why it exists.** On 2026-09-18 a narrator one-liner cost a cutover that ran two hours.
Measured (C:\tmp\bughunt\cutover-progress.log, the envpacks runs, the v1.0.2 asset list):

| | |
|---|---|
| minutes before the first byte was deployed | 39, of which ~25 were suites that had already passed at merge |
| releases cut in the preceding 26 hours | 12 — the release train was the only way to move code |
| uploaded per release for a 1 MB wheel | ~190 MB: three interpreter packs (46/55/59 MB) + a 30 MB WSL rootfs, all rebuilt every tag |
| unpacked per machine per patch | 220–325 MB |
| third-party wheels re-hosted on our releases | ~17 GB (tts 5.3 GB, llm 3.3, rvc 3.3, align 2.9, asr 1.3 on cuda-linux; ~0.9 GB mlx) |
| the tts pack, rebuilt because narrator's git sha sits INSIDE its recipe | 13.3 min CI + a 5.3 GB download per machine |

Owen: *"we built crucible from the ground up the other day, and we got a case of runaway
tests / runaway repackaging / runaway zipping without thinking through how it should
operate efficiently. This is literally taking longer than it takes to TTS an entire 17-hour
book just to deploy a very simple, basic API."* And: *"the unit of deployment is the code."*

## 1. What a release IS

A GitHub release carries OUR code and nothing that is published elsewhere:

| asset | what | size today |
|---|---|---|
| `crucible-<v>-py3-none-any.whl` (+ sdist) | the server | 1 MB |
| `crucible-client-<v>.tgz`, `crucible-bootstrap-<v>.tgz` | the SDKs | <1 MB |
| `install.sh`, `install.ps1` | the installers, generated | — |

Deleted from the release: every `crucible-env-*` pack, `crucible-rootfs-*`, `envpacks.json`.

## 2. Where everything else comes from — its publisher, pinned by version and digest

| thing | publisher | pinned where |
|---|---|---|
| CPython (3.11 for the server, 3.12 where a recipe wants it) | astral-sh/python-build-standalone releases | the version+digest table that lives in `envpack.py` today, moved to one owner |
| crucible's own deps (fastapi, uvicorn, …) | PyPI | `pyproject.toml` |
| a job env (torch, vLLM, SGLang, whisper, …) | PyPI / PyTorch index / SGLang index | `crucible/envs/<type>/<recipe>.txt`, exact pins, unchanged |
| narrator | bookforge git at a sha | the recipe's direct-reference line, unchanged in form |
| the WSL distro | Canonical, `cloud-images.ubuntu.com/wsl/releases/24.04/current/` + `SHA256SUMS` (checked 2026-09-18: 200, 340 MB) | the installer |
| `llama-server` | ggml-org releases (already) | unchanged |
| weights | Hugging Face (already) | unchanged |

Nothing above is stored on our releases, ever again.

## 3. What an install does (first run, once)

1. Download the pinned CPython from python-build-standalone into `<home>/server/`; verify the digest.
2. `pip install crucible-<v>.whl` from the release into it (deps from PyPI). `installation.json` records the release — the one owner of "what is on disk", unchanged.
3. Windows only: import the Canonical rootfs as the distro named `crucible`, verify against `SHA256SUMS`, then do INSIDE it what `build-rootfs.sh` used to bake: the `crucible` user, passwordless sudo, the `# crucible-rootfs` marker in `/etc/wsl.conf`, `[boot] systemd=true`. Then steps 1–2 inside the distro.
4. `crucible install <job>` = what `--build` does today: a venv from the pinned interpreter, `pip install -r <recipe>` from the mirrors, the stamp (`crucible-env.json`) recording the recipe hash. The `--build` flag is deleted because there is no other path. A recipe that names a CPython the server does not run gets that CPython from the same publisher; PATH is never searched.

## 4. What an upgrade does (every patch)

`pip install crucible-<v>.whl` into the interpreter that is already there; restart. pip fetches only what changed in `pyproject.toml`'s deps, which is normally nothing. Under a minute, in parallel across the fleet (`deploy.sh`).

A job env is touched ONLY when its recipe hash changed, and then by `pip install -r` into the existing venv (pip skips what is already satisfied) — never a delete-and-rebuild, never a tarball.

**narrator (the 1b case).** The stamp records the environment half (every line but narrator) and the narrator sha separately. When only the narrator sha moved, the env is left alone and the one line is reinstalled: `pip install --no-deps 'narrator @ git+…@<sha>#subdirectory=python'`. A narrator edit costs seconds, not 13 GB.

## 5. The doctor

`crucible doctor` already names a missing or drifted env; `crucible install <job>` already repairs one from its recipe; an app's doctor calls those. Nothing new. What changes is that "repair" means "pip from the recipe", which is the same thing a first install means — one path.

## 6. What is DELETED (the point)

- `envpack.py`'s pack build / manifest / download / carry-by-reference (keep only the interpreter table, relocated), `crucible envpack`, `scripts/plan_packs.py`, `scripts/release_packs.py`, `.github/workflows/envpacks.yml` and its 13-runner matrix, disk guards, part splitting and reassembly, `envpacks.json`.
- `sdk/bootstrap/scripts/build-rootfs.sh` and its CI job; the pack-installing half of `pack.ts` / `envpacks.ts` / `steps.ts` and the generated installers.
- `release.sh`'s envpacks dispatch; `ship.sh`'s wait for the pack run; the three-minute appearance poll.
- Every test that existed to keep the above honest.

`PHASE14-ENVPACKS.md` keeps its §0 ("what it deletes": no conda, no distro Python — still true, the interpreter still comes from python-build-standalone) and §4b/4c (the owned distro, the WSL state table — still true, the image just comes from Canonical). §1–3 and §7 are superseded by this document and say so at their top.

## 7. Tests and the release path

- **A normal deploy runs zero tests.** `ship.sh` has no test step. The tests that matter ran on the branch, focused on what changed, before the merge — "by the time we reach deploy, we should know it's going to work already."
- `tests.sh --changed` selects ONLY the test files that name the changed module. The WIDE list that fans a version literal, a recipe or a workflow out to the whole suite is deleted. `--list` says why each file was chosen. `--all` exists for "something is wrong or we're debugging."
- No live keeper and nothing that touches a GPU is in any release or deploy path. "If we change the handshake logic, we test the handshake logic and assume the GPU works since it did last time; if it breaks, we debug from there."

## 8. What ship and deploy look like after

`ship.sh patch`: bump (7 literals + generators) → commit → push → `release.sh` (tag, wheel, sdist, SDK tarballs, installers) → done. No CI to wait for. `ship.sh` prints a timing table per step so "why is it slow" is read, not guessed.
`deploy.sh --release <v>`: the three machines in PARALLEL, each `install.<sh|ps1>` at that tag, which is a wheel install and a restart; `installation.json` read before and after, as today.
Expected: a code patch ~3 min end to end through GitHub; an env change = a pip install from the mirrors per machine; an interpreter change = one ~30 MB download per machine.

## 9. The apps (FIX-32, queued, NOT this phase)

Both apps vendor the SDK tarballs by SERVER version and BookForge carries a 715 MB copy of Foundry's app source; a server-only patch rebuilds both apps for nothing. FIX-32: pin the SDK by content, and Foundry as a dependency the apps resolve from Foundry's own releases. Same ruling, app side.
