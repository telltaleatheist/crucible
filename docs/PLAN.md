# Crucible — build plan

Product direction is defined by [the intent of Crucible](INTENT.md), recorded
2026-09-16. This historical implementation plan must be read subject to that
contract, especially native Windows independence, optional WSL, app-owned model
selection and avoiding duplicate downloads. Phase completion is not evidence that
the intent's end-to-end acceptance scenarios have passed.

Rule for every phase: BookForge changes little or nothing until Crucible is ready. The
first BookForge consumer is `bookforge-cli` (it drives the compiled pipeline, so it proves
the seam without touching the app UI).

**PLANNED, 2026-09-18 (not built): PHASE19 — the Linux engine arrives by itself.** The
topology stays PHASE17's (a router was considered and rejected: the native fallback is not
capability-equivalent, so it would be a silent one). What changes is the DEFAULT: on every
Windows machine that can host WSL2 the orchestrator runs the engine move automatically at
install, resumes it itself after the reboot Windows demands, and records the outcome in one
file; native Windows is the outcome only where the machine cannot. No app surface shows a
command or a token, ever — the bearer is an identifier the open door hands out, and the
firewall is the lock, as with Ollama. Both apps' setup pages collapse to three faces and lose
every paste/key/manual row. Five rulings are listed for Owen; Opus subagents build it when he
says go. Contract: `docs/PHASE19-AUTOMATIC-WSL.md`.

**Newest first, 2026-09-18: PHASE18 — uncertified voices. THE SOURCE AXIS IS BUILT; THE
REST IS A RULING.** A voice
may name its weights by PATH as well as by HF pin (the deploy problem has been blocked since
2026-09-15 by full HF private storage), and a voice's certificates — cap, pace band, serving
width — become facts a manifest may decline to state and a render may decline to be bound by,
stated as `"voice"` / a value / `null` and never omitted. Two findings reverse earlier
beliefs: the fine-tuning ladder needs NO `seed` wire field, because narrator's draw is a pure
function of `(index, take)` and is weights-independent; and the unguarded render arm already
exists in `narrator/serve/worker.py`, chosen today by engine capability rather than by the
caller. An uncertified voice cannot be guarded at all — narrator's guard IS the certificate,
so `guard: true` on a voice that states no pace triple and no cap is refused by name; and the
certificate is not merely absent there but UNKNOWABLE, since the pace a guard enforces is the
median of the voice's own clean renders, which is what the refused render would produce. Three
schema relaxations carry it: the source axis, `[voice.pace]` omissible in whole (absent, never
zeroed — an inherited pace is how deathstalker's 16.64 survived a promotion), and a numberless
rung above take 0. **The first of those three is built**: a backend block declares a pin or
a `path`, a local voice carries `source` and `identity_basis` on its `/v1/voices` row, and
Crucible refuses to fetch, stamp, delete or catalogue bytes it does not own. The other two
are not.
Contract: `docs/PHASE18-UNCERTIFIED.md`. Owen has not ruled on the rest of the build order.

**2026-09-15: PHASE17 — orchestrator and engine.** Every Crucible process now
has a `role` on `/v1/info`. `crucible host` was always an orchestrator and is now named one
(`crucible orchestrator`, `host` kept as an alias); it CLAIMS the one engine it manages,
restarts it by the owner-appropriate means, and still never carries a byte of data. Apps keep
ONE address per machine and it is the engine's — `engineOf(info)` is the whole of the new
client rule. Contract: `docs/PHASE17-ORCHESTRATOR.md`; the block is below, before Phase 1.

**Where this stands, 2026-09-13.** Phases 1 through 4 are built and merged: every job type
in DESIGN.md's table exists, is tested, is documented in its own contract file, and is
reachable from `@crucible/client`. Two of them — `llm` and page reading — are verified on
real cards with measured numbers in their manifests. The other four were built against
fake engines on a night when both of Owen's cards were busy, and say so: every voice,
aligner and whisper manifest carries `estimate_basis = "declared"`. (Partly discharged
2026-09-14: every `mlx-darwin` aligner and whisper figure is now MEASURED on the M1 Ultra
— see the Mac paragraph below. The cuda-linux blocks and every voice are still declared.)

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

> **REVERSED, 2026-09-18, by `PHASE20-CODE-NOT-ENVIRONMENTS.md`.** The two entries
> below record what phase 14 landed and are kept as that record. The environment
> packs, `envpacks.json`, `crucible envpack`, the rootfs asset and
> `crucible install --build` are all deleted: an interpreter comes from
> python-build-standalone, an environment from its recipe on the machine, and the
> WSL image from Canonical. A release carries our code and nothing else.

**Landed, 2026-09-14: phase 14's SERVER SIDE — an env comes off the release**
(`PHASE14-ENVPACKS.md` sections 0-3, 5 and 7). Owen: *"we should use gh releases to
download the environments we need to run it, just like we do in bookforge."*
`crucible install <type>` **downloads** now: `envpacks.json` from this version's release,
the parts joined and deleted one at a time, the sha256 of the reassembled whole, an unpack
into `<key>.partial` and an atomic rename. `--build` is still there and is an ARGUMENT —
nothing chooses it because a download failed, and every way a download can fail has a name
(`pack_not_published`, `pack_manifest_unreadable`, `pack_download_failed`,
`pack_sha_mismatch`, `pack_recipe_drift`, `pack_disk`, `pack_unpack_failed`). A pack is a
python-build-standalone CPython 3.11.16 with one recipe installed INTO it, pinned once per
backend with the sha256 read off that release's `SHA256SUMS`. `crucible envpack build` is
the producing half; `.github/workflows/envpacks.yml` runs it on the tag, ten packs across
two runners, and uploads the manifest LAST so a release whose manifest exists has every
pack it names. `crucible doctor` prints each env's pack sha beside its recipe hash.

- **Measured, on the card-free half of Owen's PC:** `asr`/cuda-linux is 2.86 GB unpacked,
  1.29 GB as one part, 69 s to build and **12 s to install**; `server`/cuda-linux is 199 MB
  unpacked, 55 MB packed, 18 s. Both smoke-tested by unpacking SOMEWHERE ELSE and running.
- **What the real build found:** python-build-standalone bakes no absolute paths and **pip
  does** — every console script's shebang is the installing interpreter's absolute path, so
  a moved tree answers `bin/crucible --version` with *ENOENT naming the script*. The same
  trap BookForge wrote up in `electron/rvc-bridge.ts`. Fixed rather than avoided (section 4
  needs `bin/crucible` to work): every `bin/` entry that names the build tree is rewritten
  to distlib's sh/Python polyglot resolved from `$0`.
- **Owed:** no pack has been built on the Mac — there is no Crucible checkout on it, and
  making one is a setup act rather than a test — so the four `mlx-darwin` packs and their
  interpreter pin are proved by the first `macos-14` CI job and not before. The four torch
  packs' runner-disk figures are labelled ESTIMATES in section 3.3.

**Landed, 2026-09-14: the Mac catches up** (`PHASE15-HOST.md` 4.6 and the record in 7c).
The read-only audit found three classes `mlx-darwin` did not serve; two of them do now, and
the third is the interesting one. **`align`** is the same engine on a different device —
Qwen3-ForcedAligner is plain torch, torch runs on Metal — so the whole change is a recipe
(the freeze of BookForge's `qwen-align` env), the identical repo and revision, and a
per-backend device table with no default. **`asr`** is a SECOND ENGINE, because CTranslate2
has no Metal backend: `mlx-whisper`, its own worker, and seven `mlx-whisper-*` ids the
loader will not let anyone confuse with the card's six, since a transcript records the id
and nothing else about the bytes. The wire is unchanged either way, so BookForge's readers
change nothing; `vad_filter: true` is refused BY NAME on mlx-whisper rather than ignored.
**Every estimate on those eight new blocks was MEASURED on the M1 Ultra** — none is a
declared allowance and none is a copy of the cuda figure, which would have been an
allowance for a CUDA context that does not exist. Also landed: `envs/rvc/mlx-darwin.txt`'s
owed freeze (from the env `crucible install rvc` actually built there), and the audit's one
owed improvement — `crucible doctor` now names the SERVICE's PATH beside the shell's, read
back out of the plist or unit Crucible wrote.

- **`pages` got the structure and not the capability, and that is a measurement.**
  `BACKEND_ENGINES` is one engine per (backend, class family) as 4.6 decided, the family
  derived from `modalities` rather than declared, and `crucible/engines/mlx_vlm.py` is the
  class. But mlx-vlm's OWN HTTP SERVER does not put the image into the prompt for dots.ocr:
  in process the page reads correctly in 16.32 s, over the server the same weights answer
  `[{"bbox": [1,0,1008,1008], "category": "Picture"}]` in 0.72 s, and the server logs
  `prompt_tokens=216` where `prepare_inputs` on that machine returns 3,464. Four request
  shapes, two versions. Shipping the manifest block would light `pages: yes` on every Mac
  and answer every page with one Picture, so it is not written and
  `models/dots-ocr.toml` carries the run, the pins and the estimate for the day it can be.
  Foundry's Mac route is unaffected — it calls `generate()` in process, the half that works,
  which is why nobody had found this.
- **Still unmeasured, and said so rather than guessed:** the aligner's MPS-vs-CUDA timestamp
  comparison, and mlx-whisper-vs-faster-whisper accuracy on a real book. Both are one book
  through two machines and neither has been run.

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

**Landed, 2026-09-14: Orpheus is removed.** Owen: *"orpheus is deprecated too but
hasnt been removed yet. higgs is the frontier"* — *"i guess we can remove it now."*
Crucible had named `orpheus` as a servable narrator engine since PHASE3-TTS.md, with a
recipe on disk (`envs/tts/orpheus-cuda-linux.txt`), an env key (`tts-orpheus`), a
sampling row, a streaming width and a `--narrator-engine` choice — and no Crucible was
ever going to serve it. The operator page built the day before made the cost visible:
it draws its `tts` engine picker from `/v1/capability`'s `narrator_engines`, so the page
offered an engine with no future, correctly, because the server said so. The engine is
out of every table.

Each of those tables **stays a table keyed by engine, with one row and the ruling beside
it**, because a second engine will come. What it has to add, in one place per fact: a
row in `voices.NARRATOR_ENGINE_SAMPLING` (its own sampling defaults, since that table is
the one list the CLI choices, the task door and the capability row all read), a row in
`ttsstream.STREAM_BATCH_WIDTH` (a MEASURED streaming width — `batch_width_for` refuses
an engine nobody has measured, and there is deliberately no default), a recipe per
backend under `envs/tts/`, a row in `jobenv.CUDA_LINUX_SERVING_STACK` if it starts a
server underneath narrator, and membership of `narratorvoices.DOCUMENT_READERS` if it
resolves a voice by name in a document. The drift guard that would have caught the
listed-but-unservable engine is
`tests/test_jobenv.py::test_the_engines_the_server_names_are_exactly_the_engines_with_a_recipe`,
which compares the named engines against the recipe files in both directions.

**Nothing under `~/.crucible` was touched.** The 7.1 GB `tts-orpheus` env on the PC's WSL
server is the operator's to remove, and this repo does not remove an operator's disk.

**Landed, 2026-09-14: phase 15's SETTINGS HALF — the engine holds the keys**
(`PHASE15-HOST.md` sections 2, 3.1–3.4, 3.6–3.8, and 3.10's foundation). Owen:
*"Settings live in the engine and nowhere else … If the user enters an anthropic api key,
it should pass through to crucible."* `config.toml` gains `[routes]` and
`[upstreams.*]`; `GET`/`PUT /v1/settings` is the one door that changes them, applied
whole or not at all, live without a restart, with a key that is write-only and reaches
no response, log line or activity row. `POST /v1/settings/upstreams/{name}/test` asks the
provider what it serves, because this server ships no cloud model list. The chat door
forks on one character — a `model` with a `/` goes to the named upstream, Anthropic
translated in both directions including a `response_format` schema as one forced tool —
and it never retries a request that may already be billed. `/v1/capability` rows say
`route`. The operator page gained a Settings panel that holds nothing, `@crucible/client`
gained `settings`, `putSettings`, `testUpstream` and `readPairingFile`, and
`<CRUCIBLE_HOME>/pairing` means an app on the server's own machine never asks anybody to
type a token.

**And the same day: Windows became a backend.** Owen: *"the windows side should still host
GPU jobs even if WSL isnt present/workable … just like it runs from the mac side."*
`llama-windows` is now a backend kind beside `cuda-linux` and `mlx-darwin` —
`llama-server` on GGUF, detected on win32 with a card or with the machine's RAM, refusing
nothing — with its catalog rows on the three models whose GGUF is published and a
capability answer in three parts: the llm classes and `pages` from the GGUF table, the
five Python job types off with one sentence about WSL, and `echo` always on. What is
still owed on it is the engine subject and the child: see PHASE15-HOST.md section 7,
which records the pinned llama.cpp tag, its three asset digests and every decision the
next build needs so nothing is derived twice.

**Landed, 2026-09-14 (evening): phase 15's HOST SIDE — Windows gets a presence.**
`PHASE15-HOST.md` section 4, and the pieces of 3.5/3.6 it needs to exist; section 7b of
that file is the record. WSL has no boot — nothing starts a distro at login — so the
engine was down after every reboot until an app happened to poke it, and a clean stop
that afternoon left it down at 16:10 with nobody noticing. The only process that can own
"the engine is running" is one that is itself running on Windows, and that is
`crucible host`: a tray icon, a login item, a 15 s watch, and the loopback door
(`POST /install` on 127.0.0.1:7101) that the operator page's engine switch and
`@crucible/bootstrap` both drive, so the Windows→WSL2 move has ONE implementation.

Four things in it are worth reading before the server half lands beside it:

- **The 4c state table crosses into Python by GENERATION.** `crucible host` walks the same
  ten rows `install.ps1` used to, and it is Python. Rather than a second hand-written copy,
  `sdk/bootstrap/scripts/gen-install-scripts.ts` gained a third output —
  `crucible/host/wsl_states.py` — and `npm run gen:install -- --check` refuses a drift in it
  exactly as it does for the two scripts. Only the `means` predicates are spelled twice,
  because they are code; a pytest asserts the two sets of codes are equal. It is the seam
  `envpack.SMOKE_IMPORT` already has with `cli.INSTALLABLE_JOB_TYPES`.
- **`install.ps1` stopped walking that table.** It installs the host and stops. The whole
  sequence — the states, the UAC prompts by name, the import, the config move, the weights
  rule, the switch-over — is the host's, which is what lets 4.7 make it a task the page
  drives and what lets a reboot state be true (the login item is what resumes it).
- **A third pack backend, `llama-windows`, and it was BUILT.** 64 s, 186 MB unpacked, 46 MB
  archived, one part, `crucible.cmd --version` passing from a directory the build never
  saw — while `Scripts\crucible.exe --version` from that same directory exits 1, which is
  the whole reason the `%~dp0` shim exists. pip bakes the build tree's interpreter path
  INTO the launcher binary, and unlike PHASE14 7.2a's POSIX form no shebang rewrite can
  reach it. The interpreter pin was read from python-build-standalone's own SHA256SUMS,
  which is also how the doc's `-shared-install_only` asset was found not to exist.
- **`service.py` moves to `Restart=always`, and the reason is this phase.** `on-failure`
  was chosen on the reading that restarting an exit-0 would break `crucible service stop`;
  systemd does not work that way, and what `on-failure` actually bought was the 16:10
  defect. The host now watches, and 4.1 says it must NOT reimplement the restart loop — so
  the unit is the half that is total. The launchd agent is deliberately NOT changed with
  it: there is no host on the Mac, so `launchctl stop` there is the only stop there is.

`sdk/bootstrap` is at 245 tests from 191; `tests/test_host.py` adds 69, every one of which
runs OFF Windows because the platform, the environment and every subprocess are injected —
a suite that skipped its subject on the machine it runs on would pin nothing.

**What the host side could not do, and is owed** (7b.6 is the list): no distro was imported
and no install ran end to end, because `crucible-rootfs-<version>.tar.zst` is on no release
yet; `install-job-types` and `migrate-weights` install and move NOTHING and say so on the
event stream, because their inputs are the Windows server's coordinate records and catalog,
which are the server half of this phase; there is no delete door for a Windows weights copy
and the LAN forward was detected but not added, because `netsh` needs administrator and
Owen's machine was to be read, not changed. **The delete door landed while this was being
written** (3.5a, `DELETE /v1/catalog/{kind}/{id}`), so `migrate-weights` is real: pull in the
guest, wait for the GUEST's catalog to say `installed`, then delete on Windows, re-diffing
both catalogs every round so a resume needs no state that survived the crash. A
`subject_in_use` is waited out with its holder named and then fails BY THAT NAME — never
skipped, because 3.5 says nothing is skipped and an unbounded wait would be a third ending.

**Landed, 2026-09-14 (late): phase 15 is CODE-COMPLETE, and the button is written.**
`PHASE15-HOST.md` section 7.6 is the record. The three branches are one — the host's
(section 4) and the Mac's (7c) merged into this one — and the seven things 7.4 handed
over are built: the `engine` subject at the pinned llama.cpp `b10970` with both CUDA
zips verified before either is unpacked; a file-aware `weights.pull` that fetches ONE
quantization out of a repo that holds twenty and calls a subject installed only when
every file it names is present; `LlamaServerEngine` with `--alias`, a fatal-line early
exit and a 30-second stop; `crucible install llm` fetching the engine and the five
Python job types refused `needs_wsl`; `doctor`'s engine row; `main()`'s win32 gate
DELETED rather than widened, with `backend_not_here` wired where a backend is actually
read; `DELETE /v1/catalog/{kind}/{id}` (3.5a) with `subject_in_use` naming the holder;
and the `engine` task (4.7), which the Windows server hands to the host's loopback door
and relays under its own id.

Four things arrived beside them, each because running the thing found it:

- **A wheel never carried the manifests.** `packages.find = ["crucible*"]` shipped the
  code and `crucible/ui/`; `models/ voices/ denoise/ rvc/ rvcbase/ align/ asr/ envs/`
  sat BESIDE the package, so a fresh install answered `catalog_unreadable` — measured
  on the Mac, and the phase-14 `server` pack unpacks that same wheel, so every install
  that was not a developer's `pip install -e .` was broken. The directories moved INSIDE
  the package, and `tests/test_wheel.py` builds a wheel, looks inside it, installs one
  in a throwaway venv and calls all eight loaders there.
- **ONE definition of a page request** (`crucible/pages.py`), published on
  `GET /v1/info`'s `pages_engine.request`. Page reading has no job type, so the CLIENT
  builds the chat completion — and it was building it out of constants pinned in
  Foundry's source. A prompt and a pixel budget are facts about the weights. The prompt
  is byte-identical to the handover copy and a test compares them.
- **A module names CLASSES and the server resolves them** (5.3a), because the generator
  resolving them meant posting the cuda-linux answer to a Mac. A class this engine has
  disabled is `unmet` on the task, never a refusal of the module.
- **`crucible serve` writes the pairing file**, so a server that existed before this
  phase stops telling an app on its own machine that there is no engine there.

**Staged and measured, 2026-09-14.** S2 ran end to end on the PC: the host pack built in
49 s (46 MB archived, 186 MB unpacked), the engine subject fetched in 19 s (645 MB of
zips, 1.17 GB unpacked), `dots-ocr` in 51 s (4.42 GB) and `qwen3.5-9b` in 97 s (9.53 GB,
one Q8 out of a repo of quantizations) — 227 s and 15 GB in total, into a temporary home
on port 7101 that touches nothing the machine already has. `crucible doctor` on it reads
**healthy**, with `load-model: ready — loadable: ['dots-ocr', 'qwen3.5-9b']`. S1 ran too:
the WSL server is at this branch's HEAD and answers `pages_engine` and eleven capability
rows carrying `route`.

**`scripts/testrun-phase15.sh` is the button** (section 8), with a `--dry-run` that
prints every command and runs none. What is still owed is the CARD: seconds per page
under llama.cpp and under vLLM, whether the Q8 answers in the parser's dialect, seconds
per cleanup chunk, and the three declared `memory_bytes_estimate` figures. Every one of
them is written in 7.6 as *unmeasured — tested once the GPU is free*, and the button's
report ends with the list so the numbers go back into the doc rather than into a chat.

So what is left is not code. It is **a card, and Owen's rulings on the five things below.**

### Owed, and only a free card discharges it
- `scripts/keeper-tts-live.sh` on the PC and the Mac: render a chapter, measure the peak,
  paste the printed lines into the voice manifest. Same for the aligner and a whisper size.
- dots.ocr: pull it, read one real page, compare the markup to what the `dots` env
  produces today, and measure the utilisation with multimodal profiling ON.
- The tts envs have never been installed anywhere, so their recipes are pins read off
  narrator's own `pyproject.toml` rather than a resolved set.
- `mlx-audio==0.4.8` on the Mac: the ceiling was measured against the engine removed on
  2026-09-14, and nobody has run 0.5.1 against `higgs-v3` alone. The pin stays where the
  measurement put it and the recipe header says the re-measurement is owed.

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
5. ~~**The take ladder.**~~ **RULED and BUILT, 2026-09-14** (PHASE3-TTS.md section 3). The
   steps stay server config and the judgment stays the client's. Owen: *"i just know if a
   sentence/chunk was problematic before, itll likely be problematic again with the same
   settings used to originally generate it"* — so the requirement is that a retake must not
   reuse the settings that produced the problem, and the spread IS the ladder. The five
   fine-tunes declare rung 1 (`temperature = 0.7`, with its measurement), the rung reaches
   narrator per item, and `sampling_not_wired` is deleted.
6. **Does a resident model ever unload itself?** Proposed: no, and the Servers row shows
   what is resident and for how long (PHASE5-APPS.md section 7).

**Landed, 2026-09-15: a machine can be taken OFF, and a rented one can be put ON.**
Two routes `docs/INSTALL-UNINSTALL.md` now owns end to end, both from Owen the night
before: *"make sure there's an uninstall route for crucible as well … I'll probably
uninstall it on the pc (wsl and windows) and fully reinstall end to end as a test. Same
with Mac. And it should have a cli install route as well, so I can install it on a digital
ocean rented Linux droplet with a powerful gpu."*

- **`crucible uninstall`** (`crucible/uninstall.py`) is `sdk/bootstrap/src/steps.ts` read
  upwards, step by named step, with `--dry-run`, `--json` and `--purge-weights`. Weights
  are KEPT by default with their size said out loud (3.5: they are the expensive part);
  it deletes only what it can NAME, so a stray under `CRUCIBLE_HOME` is reported rather
  than swept and `<home>` goes only when it is EMPTY; it never removes the relocatable
  interpreter it is running from, which is the wrapper's to remove afterwards. A step it
  cannot do is refused by name, and "there was nothing there" is not a failure —
  uninstalling a half-clean machine has to work. `--wsl-too` runs the guest's own
  uninstall inside the `crucible` distro and never unregisters it, and never names Owen's
  Ubuntu at all. **The bearer token always goes** (`config.toml` is removed
  unconditionally), so every round trip is also a rotation and every app re-pairs.
- **`install.sh --uninstall` / `install.ps1 -Uninstall`** call the verb and then remove
  the pack it deliberately cannot. **`install.sh` gains the droplet route**: `--token`,
  `--host`/`--port` (a rented box is reached over the network; the bearer is the lock),
  `--install <type>` from the published packs, `--from-source <ref>` as a ROUTE and never
  a fallback, and prerequisites refused BY NAME before anything is downloaded —
  `no_nvidia_smi`, `no_nvidia_driver`, `no_cuda_arch`, `cuda_arch_too_old` (7.0 floor),
  `no_ffmpeg` (only when a type that decodes audio was asked for), `disk_too_small`
  against the operator's own `--min-free-gib`. 4a still holds: a bare `curl … | sh`
  performs byte for byte the install it did before.
- **RULED for both apps** (§6.1): the Uninstall door exists for a server this app can
  prove is THIS machine's — one it installed, one this machine's pairing file names, or
  the Windows host on `127.0.0.1:7101` — and NEVER for a registry entry. A connect code
  says where a server is, not whose machine it is on, and a loopback-looking address
  proves nothing behind a tunnel. The surface is the CLI on every OS (`crucible.cmd` under
  `%LOCALAPPDATA%\Crucible\host\` on Windows, `<home>/server/bin/crucible` elsewhere); the
  host's `POST /install` door is NOT extended, because a door served by the host cannot
  survive stopping the host.
- **A defect found by running the parser rather than by reading it:** Windows PowerShell
  5.1 reads a BOM-less `.ps1` as the ANSI code page, so an em dash inside a `Die "…"`
  string broke `install.ps1`'s parse. The generator now emits ASCII and refuses any
  character it has no spelling for.

## Phase 17: orchestrator and engine — contract in `docs/PHASE17-ORCHESTRATOR.md`

**Built 2026-09-15.** Owen: *"Create a relationship/handshake between crucible installs
where one is the [orchestrator] and one the [worker]… The [orchestrator] is a hollow
orchestrator, the [worker] does the heavy lifting… If there's no wsl, the windows copy is
the [worker]"*, and the names, ruled the same night: **"orchestrator and engine. we'll go
with that."** Never master/slave.

It is not new machinery. `crucible host` has been an orchestrator since it shipped — it
decides which server a machine runs, boots it, holds its distro open, watches it, writes
its pairing file. What it never had was a NAME for the relation, so nothing on the wire
said which of two Crucible processes was which and no app, page or test could ask.

- **`role` is per PROCESS, never per install.** `engine` (serves job types on a backend;
  everything that exists today) or `orchestrator` (backend kind `orchestrator`, zero job
  types, manages exactly one engine). A Windows machine with no WSL runs BOTH from one
  install, as two processes.
- **The claim.** `POST /v1/peer/claim` on the engine, bearer = the shared token, and the
  engine answers its own `/v1/info` with `managed_by`. `DELETE` releases; `GET /v1/peer`
  is `{role, managed_by, uptime_s}`. Refusals are the relation's own —
  `peer_token_mismatch` (401), `peer_version_incompatible` (426), `peer_already_managed`
  (409, naming who holds it; `force` only through the page). A claim is a STATEMENT OF
  FACT, not a permission: nothing consults it before doing anything.
- **It is never persisted.** A claim on disk outlives the orchestrator that made it, which
  is ARCHITECTURE.md's one shape. The relation is re-asserted at presence-detection and on
  every down-to-up edge of the 15 s watch instead.
- **A `found` engine is never claimed**, from 4.1a's rule: the orchestrator did not start
  it, so `managed_by` would name a door that refuses every verb the field implies.
- **`engine-restart`** joins `engine` as a task on the engine's door that the ORCHESTRATOR
  runs — unit via `systemctl --user restart`, child by respawn, `found` refused
  `engine_not_ours`. Its last event may never arrive, because the relay runs in the
  process being restarted; the client re-reads `/v1/info`, as the move's switch-over
  already taught it to.
- **The apps keep ONE address per machine and it is the ENGINE's.** `info()` gains `role`,
  `managedBy` and `engine`; `engineOf(info)` is the whole of the new rule — null means
  "talk here", an `EngineRef` means "follow it ONCE with the same token", and an
  orchestrator with no engine throws `orchestrator_has_no_engine`. API version stays 1: a
  document with no `role` reads as an engine, which is PHASE15 3.3's vintage rule applied
  again.
- **Uninstall does NOT go through the relation**, and `docs/INSTALL-UNINSTALL.md` §6.2
  already ruled it: the flag is `--wsl-too`, it runs the guest's own uninstall through
  `wsl.exe` by distro name, and a door served BY the orchestrator cannot survive stopping
  the orchestrator.
- **NOT renamed:** the Python package is still `crucible/host/`. It is imported by cli,
  api, tasks, uninstall, envpack and the suite, and a rename touching all of them to change
  a word no wire and no operator sees is the opposite of cheap (PHASE17 section 7).

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
