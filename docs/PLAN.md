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

**Superseded in part, 2026-09-13 (later the same day).** Owen ruled that the guard belongs
to the model and that a connected server is queue capacity, which opened two more phases:

- **`PHASE6-REMOTE-RENDER.md`** — the guard moves into the engine and the chunks travel
  back over the wire. Its section 0 is a finding that reorders everything below: narrator
  has two rendering worlds and **Crucible drives the unguarded one**, so owed item 2 is not
  "`capped` is missing from the wire", it is "the render door has no guard at all".
- **`PHASE7-LANES.md`** — a server is a machine and a machine is a set of slots;
  `GET /v1/activity`; routing, affinity, and the ruling that a job is atomic. Owed ruling 5
  (the take ladder) is CLOSED by phase 6: the retake decision goes with the guard.

**Superseded again, 2026-09-13 (same day, later still).** *"a connected server is queue
capacity"* did not survive contact: Owen ruled that **queues belong to clients and
admission belongs to the server** — *"if the server is busy, it cant receive a new job"* —
and `docs/ARCHITECTURE.md` section 3 is the ruling, with 3.1 recording what shipped.
`POST /v1/jobs` now answers `409 server_busy` naming the holder instead of appending to
the deque. The lane, `position`, `queue_depth`, cancel and events are untouched; a server
is capacity for exactly one job, and which job is the client's decision to make.

**Not built, and deliberately (3.2): the "card is free" edge signal.** A lane-free signal
would lie while a streaming session holds the card, the claim is released off the event
loop with no publisher, and the consumer is the SDK. Clients poll `GET /v1/activity` until
those three are settled.

**Landed since, 2026-09-13 (overnight).** Four pieces, each in its own commit:

- **`crucible service install|uninstall|start|stop|status`** — PHASE5-APPS.md 6.0's ruling
  made real: a systemd user unit on `cuda-linux`, a launchd agent on `mlx-darwin`, every
  verb idempotent, linger REPORTED rather than assumed. The first real install found two
  things a doctor run could not: a unit gets a bare PATH (so `install` records the
  installing shell's, and every "tool missing" refusal now names the PATH it searched) and
  `python -m crucible` from `$HOME` imports the checkout's `crucible/` directory instead of
  the package (so `ExecStart` runs the console script, in `CRUCIBLE_HOME`).
  `docs/PHASE11-SERVICE.md`.
- **Per-model `[defaults]`, applied on the chat door** — `qwen3.5-9b` ships
  `thinking = false`, which both apps have been carrying in their own code. A field the
  request states wins, a field it omits takes the manifest's, and every response says
  which in `X-Crucible-Sampling`. PHASE2-LLM.md section 9 is the contract to hand Foundry.
- **The `denoise` job type** — audio-separator sharing the rvc env, one audio file in, its
  stems out, with the three invariants BookForge measured (native rate in, exactly one
  primary stem, same length sample for sample). PHASE4-AUDIO.md section 4.2.
- **urvc's base assets are pulled** — owed ruling 3 below, answered by reading the engine:
  its own first-run downloader names a HuggingFace repo. `crucible rvc pull-base`.
- **The separator checkpoint is pulled too** — `crucible denoise list` /
  `crucible denoise pull <id>`, which is what PHASE4-AUDIO.md section 4.2's own owed
  ruling asked for. Both files, both digests, one pinned revision, into the flat
  directory audio-separator reads by name — a layout `crucible/denoisemodels.py` owns
  and the job reads, so `crucible doctor`'s denoise row stops saying "Crucible does not
  fetch them".

**Landed 2026-09-14: the lease** (`PHASE7-LANES.md` section 5.2). A chat holds nothing, so
a server mid-way through a two-thousand-block translation reported itself idle between
blocks and BookForge's `load-voice` evicted Foundry's translator at block 400. A timer would
have been a fact standing in for a guess; the fact is that a client intends a run, so the
client says so — `POST /v1/models/{id}/lease`, heartbeat, `DELETE`. While one is open the
jobs that would take the leased thing off the card are refused `409 leased` at the door
(chats never are: they are what it protects). Expiry is read from the clock, never swept,
and a restart forgets. In the SDK as `lease`/`heartbeat`/`release`, `Activity.lease` and the
typed `CrucibleLeased`.

**Extended the same day: a lease names the RESIDENT THING of any kind.** Leasing only the
resident *model* turned the unload below into a regression it never asked for — a book
rendered CHAPTER BY CHAPTER, which is how the app works, paid a narrator load per chapter,
and a book aligned chapter by chapter paid an aligner load per chapter. The same route takes
a voice id or an aligner id, the server reads the kind off its own residency (one card, one
resident thing, so there is nothing for a client to disambiguate), and what a lease refuses
is **derived** from one `CARD_EFFECTS` table — which is what lets a `tts` render of the
leased voice, and an `align` on the leased aligner, be admitted rather than refused.

**Landed 2026-09-14: the unload** (`PHASE7-LANES.md` section 5.3). Owen: *"Models should
always be unloaded when we're done with them. Every time."* This **overrules
`PHASE5-APPS.md` section 7**, which had proposed no idle unload with the Servers row showing
what is resident and for how long. "Done" is read from four facts — no job on the lane, no
lease open, no streaming session holding the claim, no chat in flight — and the moment the
last goes false the resident model, voice or aligner is unloaded and says why, in the server
log and in a `note` event on the job that triggered it. No window, no timer and no config
key: a keep-warm minute is a fact standing in for a guess. What makes it safe rather than a
44-second reload between every book is the lease above — so **a client that does not lease
reloads its model between requests**, and BookForge's doors do not lease yet. A load is not
a holder letting go, which left four `# RULING OWED:` in `crucible/settle.py`. **The render
door's and the align door's are closed** by the extension above. The load door's is
sharpened and still open — half of it was wrong (a lease is another holder, so it would not
free a walked-away operator's card), and the half that remains is the window between a
load's `done` and its client's own lease, which needs a `lease` block on the JOB wire and so
**needs Owen**.

**Landed 2026-09-14: the local form** (`PHASE9-CAPABILITY.md` section 7). Owen, via
Foundry: these manifests are the catalog of record for Foundry's LOCAL lineup too, so "what
can this machine run" has one owner. A `[local]` table on a model manifest (an Ollama tag,
or a GGUF + its projector, with a memory figure and its basis), `scripts/gen-foundry-
lineup.py` writing `foundry-lineup.json` through the manifest loader and
`capability.CLASSES`, and a `--check` CI runs so the committed file cannot drift. Two
findings written into the manifests rather than around them: `qwen3.8:27b-24g` is a local
Modelfile (404 on ollama.com — labelled stopgap, ruling owed), and Foundry's page reader
pins a different GGUF projector than the one ruled.

**Landed, 2026-09-14: phase 13, the operator door** (`PHASE13-OPERATOR.md`). Owen:
*"not microservices. but crucible has its own ui. and it provides the token or whatever
else we need to set it up on foundry or bookforge."* Everything a person does to a
Crucible after it exists needed a shell on that machine; now it is five routes and a
page the server serves itself.

- **`GET /v1/setup`** — name, backend, every URL this server is reachable on
  (`getifaddrs(3)` through ctypes; never a hostname lookup, and an unreadable interface
  list is `503 interfaces_unreadable` rather than an empty `urls`), the token, and one
  `crucible://<name>@<host>:<port>/#<token>` **pairing line** per URL. The name is
  percent-encoded because a server name contains an `@`, and a line with two of them is
  refused rather than split at the likelier one.
- **`GET /v1/catalog`** — every pullable subject this backend can hold, derived from the
  stamp `weights.py` writes, the manifests, `lineup.floors()` and the residency. No new
  table: `crucible/catalog.py` is the one list, and the pull task reads the same one.
- **`POST /v1/tasks`** (+ list, read, SSE, cancel) — `pull`, `install`, `module`. One at
  a time; an `install` waits for the four facts and its `409 server_busy` names which of
  the four holds the card, in that holder's own fields. **A pull is genuinely
  cancellable**: the hook `huggingface_hub` already takes is made the cancel point, so a
  `DELETE` stops the download at its next chunk and removes the partial directory —
  rather than reporting "cancelled" while nineteen gigabytes went on arriving.
- **3.4, decided: an in-place reload.** Re-exec cannot deliver the `done` event it would
  take with it, and presumes a supervisor a foreground `crucible serve` does not have.
  The swap adopts a re-read of `config.toml` into the one `Config` object every route
  holds, rebuilds the registry with the same residency, and re-reads the four facts
  immediately before it — a holder found there fails the task `reload_refused` and
  leaves the env on disk (R6).
- **`GET /` and `/ui/*`**, public, serving `crucible/ui/` as package data. **The page
  itself landed 2026-09-14** (section 4.1): three vanilla files, no build step and no
  CDN, six sections each drawn from the read that owns it, every refusal shown with its
  code beside the control that earned it — `/` now 307s to `/ui/` so the page's relative
  references have one home, and `/v1/capability` grew a derived `job_types` (3.2a) so the
  page carries no table of its own.
- **`crucible token --url`**, and `init` / `service install` ending with the same lines.
  Two existing tests reverse deliberately: `init` used to assert the token never reached
  stdout, and it now rides in the pairing line in front of the person who just minted it.
- **The SDK** grows `setup`, `catalog`, `submitTask`, `task`, `tasks`, `taskEvents`,
  `cancelTask` and a pure `parsePairing`. The pairing rule has two implementations
  because one must run in TypeScript, so the seam is a literal line asserted from both
  ends.
- **Modules are GENERATED** (5.4, from Foundry's review): `modules/<app>.toml` declares
  job types, capability classes and named subjects; `scripts/gen-modules.py` resolves
  them against these manifests and writes the JSON each app vendors byte for byte, with
  a `--check` in CI beside the lineup's. A class with a floor resolves to it, a class
  with one candidate resolves to it, and `analysis` — two candidates, no floor — must be
  named in the declaration, because picking would be inventing a policy nobody wrote.

So what is left is not code. It is **a card, and Owen's rulings on the five things below.**

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
3. ~~**Where do urvc's base assets live?**~~ **ANSWERED, 2026-09-13, by reading the
   engine instead of guessing.** urvc's own first-run downloader (the one
   `URVC_SKIP_INIT` turns off) fetches them from `JackismyShephard/ultimate-rvc` on
   HuggingFace, which DESIGN.md section 5 allows. `rvcbase/ultimate-rvc.toml` pins that
   repo at a revision with a digest per file and `crucible rvc pull-base` places them.
   This was a question about the world, not a ruling — Owen can overturn the source, but
   nothing is waiting on him.
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
render job and a streaming session — because the extension's streaming is what Owen uses
every Sunday and it is not the render door with a smaller buffer. Voices are manifests the
server advertises; engine tuning never crosses the wire. narrator is the managed
subprocess, the way vLLM is, because it already holds the EOS logit surgery the audit calls
the hardest single item in the contract.

**Built, both doors, 2026-09-13:** the voice manifests and the lifecycle pair,
`crucible/engines/narrator.py`, the render door (`{"type": "tts"}`), the `envs/tts/` recipes,
and the streaming door — a session, an SSE stream and posted ops rather than the WebSocket
this paragraph used to promise, because Electron 33 bundles Node 20 and there is no global
`WebSocket` in the main process (PHASE3-TTS.md section 7 has the whole argument, and the
`Last-Event-ID` reattach it buys).

Building the second door found the hole the first one left: nothing said what happens when a
render job and a session both want the card, and narrator has **one stdin**, so two
conversations on it do not fail loudly — they read each other's replies. `Residency.claim`
refuses the second by name.

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

**All three job types and the probe are built, and `denoise` joined them on 2026-09-13**
(section 4.2: audio-separator in the rvc env, one audio file in, its stems out).
**urvc's base assets now have a source Crucible pulls from** — urvc's own downloader
names a HuggingFace repo, so `rvcbase/ultimate-rvc.toml` pins it and `crucible rvc
pull-base` places all four files, digests verified before any is placed. What is left of
this phase is measurement rather than construction: every `memory_bytes_estimate` in `align/` and `rvc/`
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
