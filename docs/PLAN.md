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

## Phase 2: `llm` (DONE 2026-09-13)
Contract: `docs/PHASE2-LLM.md`. vLLM on `cuda-linux`, mlx-lm on `mlx-darwin`, behind
`/v1/openai`. Verified live on both backends: `qwen3.5-9b`, `qwen3.8-27b` (Mac only, by
size), `qwen3.8-27b-4bit`, every estimate measured on the card it names. BookForge's
`crucible` provider runs cleanup and simplify through the CLI (`--ai-cleanup --provider
crucible --server <name> --model <id>`); the app UI is unchanged.

Owed inside phase 2, small: a `qwen3.8-27b-8bit` manifest for the Mac; a decision on the
9B's edit-list prompt (its few-shot block makes the model think out loud as `content`).

## What phases 3 and 4 are built from

`docs/CLIENT-SURFACES.md` section 10 is the ranked list of everything BookForge and
Foundry actually send a model. It is the source for the scope below, not DESIGN.md's
sketches. Two rulings from Owen shape it:

- **Division of knowledge** (DESIGN.md section 3.1): the client knows the order and the
  server; the server knows the engine. Engine tuning is Crucible config, never a wire
  field. Every tier-3 row in the audit is a server requirement.
- **Voices come from the server.** Crucible advertises a `tts` capability whose rows are
  voices (id, backend, caps); the order names the voice id the server returned. Voice
  catalogs and checkpoints move behind Crucible the way models did, pulled from
  HuggingFace by manifest.

## Phase 3a: `llm` finishing touches (small, first)
- `/v1/models` and `/v1/openai/models` report `max_model_len`, so Foundry's `capFor`
  clamps requests instead of getting a 400.
- Prove `response_format: {type: "json_schema", strict: true}` survives the proxy on
  both engines (Foundry's analyze and tag calls depend on it). `finish_reason` is never
  touched.
- A dropped client connection aborts the engine request; the proxy never lets a
  request run on after its caller is gone. This is the only cancel either app has for a
  chat, so it is the cancel path and not a nicety.
- The served name rule: Crucible's id plus the pinned revision is what a client records
  (`qwen3.5-9b@<sha>`); dtype is part of the id only when it is not bf16. Provenance
  carries the revision, which until now it did not.
- With these, Foundry's clean / translate / simplify point at Crucible by URL with no
  client change, and BookForge's `text-server.ts` (1,364 lines, already a small
  Crucible) is retired.

## Phase 3b: `tts` — contract in `docs/PHASE3-TTS.md`
The largest job type and the one that deletes the most: BookForge's WSL spawn, path
rewriting, per-engine sampling tables and VRAM arithmetic for TTS all go. Two doors — a
render job and a streaming WebSocket — because the extension's streaming is what Owen uses
every Sunday and it is not the render door with a smaller buffer. Voices are manifests the
server advertises; engine tuning never crosses the wire. narrator is the managed
subprocess, the way vLLM is, because it already holds the EOS logit surgery the audit calls
the hardest single item in the contract.

## Phase 3c: page reading — contract in `docs/PHASE3-VLM.md`
Much smaller than it looked. Both apps rasterise locally and send a chat completion with an
image content part, so this is the `llm` proxy plus `modalities` on a model row, a rule that
refuses `--skip-mm-profiling` on an image-capable manifest, and a dots.ocr manifest. Deletes
`vlm-page-server.ts` and the unarbitrated port-8000 route; keeps `mlx-local` until the Mac
backend serves it.

## Phase 4: `align`, `asr`, `rvc`, the probe — contract in `docs/PHASE4-AUDIO.md`
`align` (Qwen3-ForcedAligner, resident across a book), `asr` (faster-whisper, six sizes, no
default), `rvc` (ultimate-rvc, model identity becomes a manifest), and the
"who holds the accelerator" probe that replaces BookForge's three arbitration schemes and
the `external-gpu-job.lock` convention. Denoise, separation and Resemble are named and
deferred there, with the reason: their contract is sample-exact and cannot be asserted
against a fake engine.

## Phase 5: the apps, the bootstrapper, and friends
- A Servers settings row over the registry, a server column on queue rows (the operator or
  the queue picks the server), the `crucible` provider wired into book analysis and
  translation.
- BookForge's TTS WebSocket on 8766 becomes a relay to the chosen server, which is what
  keeps the browser extension working unchanged.
- The bootstrapper: detect the host, install a local server, start/stop it, report health.
  A Docker image for `cuda-linux`. Then a friend gets the client and a tailnet invite.

## Open questions (decided by default unless Owen says otherwise)
- Owen's Ollama-built LoRA adapters (footnotes, OCR, headline, blocks): served as
  adapters over the base from HuggingFace when Foundry needs them; Ollama builds are
  not a source.
- LLM cancel through the proxy: dropping the fetch is the client's move; the server
  rule above makes it sufficient.

## Conventions
- Worktrees for agents: `C:\Users\tellt\Projects\crucible-worktrees\<branch>` (PC),
  `/Volumes/Callisto/Projects/crucible-worktrees/<branch>` (Mac). Never move `main`'s HEAD
  under another session.
- Sync between machines with git only, never cp/scp/rsync.
- Tests must exit non-zero on failure; keepers are a subset, trust the exit code.
- No fallbacks, no band-aids. A stopgap is labelled as one in the code and in the commit.
