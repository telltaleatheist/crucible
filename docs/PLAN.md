# Crucible — build plan

Rule for every phase: BookForge changes little or nothing until Crucible is ready. The
first BookForge consumer is `bookforge-cli` (it drives the compiled pipeline, so it proves
the seam without touching the app UI).

## Phase 1: handshake (now)

**A1. Server skeleton** (`crucible/`, Python)
- package + `pyproject.toml`, CLI (`init`, `serve`, `doctor`, `token --show`)
- config + token, backend detection (`cuda-linux` via nvidia-smi / `mlx-darwin` via
  platform+arch+`mlx` import), refuse with a named reason otherwise
- API v1: `ping`, `info`, `health`, `uploads`, `jobs` (+events SSE, artifacts, cancel)
- job framework: plugin registry, queue with one exclusive lane, job scratch dirs,
  provenance sidecars
- `echo` job type behind `[jobs] enable_echo = true`
- pytest suite (in-process TestClient) + one live keeper (`scripts/keeper-live.sh`)
- verified on the PC's WSL (reports `cuda-linux`) and the Mac (reports `mlx-darwin`)

**A2. SDK** (`sdk/ts/`, `@crucible/client`)
- `CrucibleClient({url, token, clientName})`: `ping`, `info`, `health`, `upload`,
  `submit`, `job`, `events(jobId)` async iterator, `artifact`, `provenance`, `cancel`;
  `X-Crucible-Api: 1` and the bearer token on every call except `ping`
- typed errors: `CrucibleConfigError`, `CrucibleUnreachable`, `CrucibleNotACrucible`,
  `CrucibleAuthError`, `CrucibleVersionError`, `CrucibleRefused` (with the server's named
  reason), `CrucibleServerError`, `CrucibleProtocolError`
- runtime deps: none (fetch + ReadableStream; Node 20+, bun, Electron)
- ESM and CJS builds with `.d.ts` for both, so all three runtimes can import it
- `scripts/e2e.sh` starts a real server and runs the echo job end to end;
  `scripts/e2e-from-windows.sh` does it with the server in WSL2 and the client native
- `npm pack` → `crucible-client-<ver>.tgz`, released with the server (see A3)

**A3. Release plumbing** (`scripts/release.sh`)
- **one release per version, tagged `v<ver>`**, carrying all three artefacts:
  `crucible-<ver>.tar.gz`, `crucible-<ver>-py3-none-any.whl` and
  `crucible-client-<ver>.tgz`. The SDK is not released separately; the server and the
  client that speaks to it share a version so a client can never be paired with a server
  nobody tested it against.
- the version is read from `crucible/__init__.py`, `sdk/ts/package.json` and
  `sdk/ts/src/version.ts`, and a disagreement is a refusal
- `gh release create v<ver>` with the three assets and generated notes; an existing tag
  is refused rather than re-cut

**B. BookForge handshake** (after A2's tarball exists)
- `package.json` pins `@crucible/client` to the release tarball URL
  (`https://github.com/telltaleatheist/crucible/releases/download/v<ver>/crucible-client-<ver>.tgz`)
- `electron/crucible/servers.ts`: server registry in `<userData>/crucible-servers.json`
  (`{name, url, token}`), no UI yet
- `cli/bookforge-tts.py`: `--crucible-ping`, `--crucible-info`, `--crucible-echo <file>`
  driving the compiled dist. Nothing else in the app changes.

## Phase 2: `llm`
vLLM on `cuda-linux`, mlx-lm on `mlx-darwin`, behind `/v1/openai`. First real consumer:
BookForge clean-text (`qwen3.5:9b-bf16` today via Ollama) through the CLI. Then Foundry
cleanup / translate / simplify (27B fits the Mac Studio, not a 24 GB card; capability
advertisement does the routing).

## Phase 3: `tts` and `vlm-pages`
Higgs via narrator (SGLang on cuda-linux, MLX on darwin, cap certificate per pair).
dots 3B page reading for Foundry. This is where the WSL path-rewriting layer in
BookForge is deleted, not patched.

## Phase 4: `align`, `rvc`, bootstrapper, friends
Aligner and RVC job types. The in-app bootstrapper (detect host, install local server).
Docker image for `cuda-linux`. First friend gets the client + a tailnet invite against
Owen's server.

## Conventions
- Worktrees for agents: `C:\Users\tellt\Projects\crucible-worktrees\<branch>` (PC),
  `/Volumes/Callisto/Projects/crucible-worktrees/<branch>` (Mac). Never move `main`'s HEAD
  under another session.
- Sync between machines with git only, never cp/scp/rsync.
- Tests must exit non-zero on failure; keepers are a subset, trust the exit code.
- No fallbacks, no band-aids. A stopgap is labelled as one in the code and in the commit.
