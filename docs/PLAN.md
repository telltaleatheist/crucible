# Crucible — build plan

Rule for every phase: BookForge changes little or nothing until Crucible is ready. The
first BookForge consumer is `bookforge-cli` (it drives the compiled pipeline, so it proves
the seam without touching the app UI).

**Where this stands, 2026-09-13.** Phases 1 through 4 are built and merged: every job type
in DESIGN.md's table exists, is tested, is documented in its own contract file, and is
reachable from `@crucible/client`. Two of them — `llm` and page reading — are verified on
real cards with measured numbers in their manifests. The other four were built against
fake engines on a night when both of Owen's cards were busy, and say so: every voice,
aligner and whisper manifest carries `estimate_basis = "declared"`.

So what is left is not code. It is **a card, and Owen's rulings on the six things below.**

### Owed, and only a free card discharges it
- `scripts/keeper-tts-live.sh` on the PC and the Mac: render a chapter, measure the peak,
  paste the printed lines into the voice manifest. Same for the aligner and a whisper size.
- dots.ocr: pull it, read one real page, compare the markup to what the `dots` env
  produces today, and measure the utilisation with multimodal profiling ON.
- The tts envs have never been installed anywhere, so their recipes are pins read off
  narrator's own `pyproject.toml` rather than a resolved set.

### Owed from Owen, and each blocks something
1. **Extract `narrator` into its own repo?** `telltaleatheist/bookforge` is private, so
   `crucible install tts` needs credentials for a repo that has nothing to do with
   inference. This is the single most load-bearing open question.
2. **`capped` on narrator's wire.** The frame cap never leaves the engine, so the `chunk`
   event cannot tell a long sentence from a runaway — the one thing it exists to tell.
   Not a two-line change: it touches the generation loop that renders his books.
3. **Where do urvc's base assets live?** `rvc` refuses by name rather than fetching them
   from a GitHub release, which DESIGN.md section 5 forbids as a source for weights.
4. **Publish the promoted fine-tune merges.** The HF revisions are older merges than the
   arms the catalog measured; the caps survive that gap, the pace bands do not.
5. **The take ladder.** Its steps are server config and its judgment is the client's — the
   one place the division of knowledge had a genuinely arguable alternative.
6. **Does a resident model ever unload itself?** Proposed: no, and the Servers row shows
   what is resident and for how long (PHASE5-APPS.md section 7).

## Phase 1: handshake (DONE)

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

## Phase 3a: `llm` finishing touches (server side DONE 2026-09-13)
Contract: `docs/PHASE2-LLM.md` section 5, amended for all four. Proved without an
accelerator, against `tests/fake_engine.py` and — for the disconnect half, which
`TestClient` cannot reach — a real uvicorn on a real socket (`tests/live_server.py`).
- `/v1/models` and `/v1/openai/models` report `max_model_len`, so Foundry's `capFor`
  clamps requests instead of getting a 400. It is a separate field from
  `context_default`, which stays the manifest's intent.
- `response_format: {type: "json_schema", strict: true}` survives the proxy, and is
  tested rather than assumed — along with `finish_reason` on `stop`, `length` and
  `tool_calls`, streamed and not, and an engine's own 400 being relayed rather than
  rewrapped. The request body is now literally verbatim where the engine answers to the
  Crucible id: it was being parsed and re-serialised, which is not the proxy's business
  to do to somebody else's grammar.
- A dropped client connection aborts the engine request. Streamed already worked by
  accident of which ASGI branch Starlette takes and now works on purpose; non-streamed
  did not work at all and now races the POST against the caller's socket.
- The served name rule is written down, and `fingerprint` (`<id>@<revision>`) is on both
  listings and in the provenance sidecar — whose `revision` was `null` on every artifact
  Crucible had ever written, which is fixed.
- **Still owed, client side:** Foundry's clean / translate / simplify pointed at Crucible
  by URL, and BookForge's `text-server.ts` (1,364 lines, already a small Crucible)
  retired. Nothing on the server blocks either now.

## Phase 3b: `tts` — contract in `docs/PHASE3-TTS.md`
The largest job type and the one that deletes the most: BookForge's WSL spawn, path
rewriting, per-engine sampling tables and VRAM arithmetic for TTS all go. Two doors — a
render job and a streaming WebSocket — because the extension's streaming is what Owen uses
every Sunday and it is not the render door with a smaller buffer. Voices are manifests the
server advertises; engine tuning never crosses the wire. narrator is the managed
subprocess, the way vLLM is, because it already holds the EOS logit surgery the audit calls
the hardest single item in the contract.

**Built as of 2026-09-13:** the voice manifests and the lifecycle pair, then
`crucible/engines/narrator.py`, the render door (`{"type": "tts"}`) and the `envs/tts/`
recipes. The streaming door is the remaining half, and the note above about it being a
WebSocket is superseded by PHASE3-TTS.md section 7: it is SSE plus posts, because Electron 33
bundles Node 20 and there is no global `WebSocket` in the main process.

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

**All three job types and the probe are built.** What is left of this phase, written down in
PHASE4-AUDIO.md where it belongs: **urvc's base assets have no source Crucible will pull
from** (BookForge hosts them on a GitHub release, which DESIGN.md section 5 refuses, and
urvc's own first-run downloader is what `URVC_SKIP_INIT` turns off), so `rvc` refuses by
name until somebody puts a models tree at `~/.crucible/rvc-base/`. Everything else is
measurement rather than construction: every `memory_bytes_estimate` in `align/` and `rvc/`
is COMPUTED and says so in the file, `envs/rvc/*.txt` are chosen rather than resolved sets,
and `envs/align/mlx-darwin.md` says exactly what would earn the Mac an align backend.

## Phase 5: the apps, the bootstrapper, and friends — contract in `docs/PHASE5-APPS.md`
Where "the app changes little or nothing" ends, one deletion at a time and each with a way
back. A Servers settings row over the registry (and the ability to pull, load and unload
from it, which no UI offers today), a server column on queue rows, BookForge's TTS
WebSocket on 8766 kept exactly where it is and turned into a relay so the extension never
notices, and the bootstrapper. The contract carries the retirement order, the two
behaviours of `text-server.ts` that Crucible deliberately does not have (yielding, and
adoption), and two decisions to make before building rather than during.

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
