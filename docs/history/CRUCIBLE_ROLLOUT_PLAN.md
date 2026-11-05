# Three apps, one server — the rollout plan

Written 2026-09-13 19:50, the night Owen asked for *"three fully functioning apps when I
get up"*. This is the list, the order, and the honest state. It is updated at each wake
(2 AM, 6 AM) and is the first thing to read in the morning.

## 0h. AFTERNOON 2026-09-15 — the Mac rendered 7x too slow, and it was one unset variable

Owen's `thirdreich` book on the Mac, through Crucible, ran at **2.9 chunks/min — "261 words/min ·
14 sent/min"** where that machine has done **~130 raw sent/min** since 2026-09-10 (BookForge
`626980a2`, deathstalker at MLX batch 62). Nothing was saturated: 66.5% of one core of twenty,
load 3.00, 9.45 GB resident on a 64 GB machine.

**ROOT CAUSE — the seam, not the machine.** narrator's in-process backend reads its batch ceiling
from `NARRATOR_HIGGS3_MLX_BATCH`, which **defaults to 1**; at 1 `render_many` takes
`_render_many_serial` and a book renders one chunk at a time. BookForge's own darwin worker has set
that variable since 2026-09-05 (`electron/higgs-spawn.ts:higgsMlxBatchEnv`, the Orpheus memory
tier's 64). **Crucible never learned to** — `crucible/engines/narrator.py:environment()` emitted the
three `HIGGS_*` variables on the served arm and nothing at all on `mlx-darwin`, because "reads none
of the three" had been read as "reads nothing". So the throughput did not regress on the Mac; it was
lost by every render that moved out of the app and into the server. The live process's environment
held exactly two variables, and the engine log carried zero `MLX batch generating` heartbeats.

**MEASURED on example-mac** (M1 Ultra, `thirdreich`, one resident load per run, 521-572-char
chunks inside the voice's 500-700 band, driven through `generate_batch` the way the render job does):

| width | chars/min | realtime | vs width 1 |
|---|---|---|---|
| unset → 1 | 1,799 | 1.99x | 1.00x |
| 16 | 6,749 | 7.43x | 3.75x |
| 32 | 9,662 | 10.78x | 5.37x |
| 64 | 12,579 | 13.97x | **6.99x** |

At its own width the Mac **beats** `cuda-linux`, which measures 11,387-11,584 chars/min at
`HIGGS_MAX_NUM_SEQS = 16`. Owen's live job measured 2.003x realtime off its own artifacts, flat from
a 4.9 s chunk to a 50.5 s one — so per-chunk overhead is nil and width was the only lever.

**FIXED** — crucible `d9fb696f`: `MLX_RENDER_WIDTH = {"higgs-v3": 64}`, a measured table keyed by
engine (`ttsstream.STREAM_BATCH_WIDTH`'s discipline — no default, an unmeasured engine refused by
name), emitted only on the arm that reads it. Four unit tests. **NOT deployed to the Mac** — that is
a `git pull` + `launchctl kickstart` on `/Volumes/Callisto/Projects/crucible`, and it is Owen's
in-app pass that proves it.

**Two suspects ruled OUT with evidence.** (1) The morning's patch-wipe defect cannot apply here:
`crucible/narratorpatches.py` says both patches are vLLM-stack edits and `envs/tts/mlx-darwin.txt`
pins neither — confirmed on the machine, which has no `vllm`/`vllm-omni` at all and matches its
recipe exactly (`mlx 0.32.2`, `mlx-audio 0.4.8`, `mlx-lm 0.31.3`). (2) Last night's `take` added to
`_mlx_batch_groups`' key does not over-split: the render door sends one take and one sampling for
the whole job, and at `BATCH_SIZE = 1` the grouper is never reached.

**OWED:** `NARRATOR_HIGGS3_MLX_MEM_BUDGET_GB` is still narrator's own 42 GB default — right for a
64 GB Mac and wrong for a smaller one. It wants deriving from the capability's `total_bytes` less
its desktop allowance, which is the absorption PHASE9-CAPABILITY.md section 5 already has queued.
Full write-up and the caveats (the seed regime changes with width; the `maxChars` certificate is
still single-row) in `crucible/docs/PHASE3-TTS.md` section 6b.

## 0h. THE RESERVED NAME `local` IS GONE (ruling 2026-09-15, BUILT)

Owen, verbatim:

> *"it shouldnt be named 'local' anywhere. it might not be local. a local
> crucible server shouldnt be treated any differently than a remote crucible
> server. it should all be entered the exact same way. if we have to change how
> the code works then we should do that. bookforge shouldnt even know if it's
> local because it doesnt mater"*

**A Crucible server is a registry entry and nothing else.** Same fields, same add
flow, same coordination, same bench row, whether it answers on `127.0.0.1` or
across the tailnet. What was deleted:

- `LOCAL_SERVER_NAME` and `electron/crucible/local.ts` (the file is now
  `discovery.ts`, and what it answers is an OFFER, not a server).
- `describeLocal()`, `ResolvedServer.source`/`.local`, `CrucibleServerListing.stale`,
  and the refusal codes `local_is_not_registered`, `stale_local_entry`,
  `loopback_duplicates_local`. **A loopback URL is added like any other now.**
- `HostCrucibleRegistrySnapshot.localAbsent`.
- `coordinateLocalOnStart()` + `readBenchFactsOnStart()` → one
  `coordinateServersOnStart()`: **every enabled server is coordinated and asked
  the bench questions the same way.** It stays cheap because coordination posts
  nothing when nothing is missing (ASK, THEN ACT).
- The Servers panel's "engine on this machine" card, its badge, and the row that
  could not be removed. One list, one kind of row, Remove on every one.

What replaced the convenience: `discovery.ts` still reads the pairing file (else
`config.toml`, on Windows through `wsl.exe`) and answers *what the entry for the
Crucible on this computer would look like*. That PREFILLS the add door —
`crucible:add-discovered` takes a NAME and main supplies the rest, because the
token may not cross the seam — and ends in the same `addServer` every other door
ends in. `node cli/crucible.js --discovered` / `--add-discovered --name <n>` is
the same thing headless.

**Names are free text**, so the rules are enforced at the point of entry
(`servers.ts validateServerName`) and refused by name: 1–48 characters, no
control characters, no `:` (the queue hangs `<server>:cloud` off a name), no `/`
or `\` (the name becomes a browser session partition), no double spaces, and not
one of the queue's own bench rows (`local-work`, `local-longform-align`,
`legacy-local-narrator`, `any`). Uniqueness is case-insensitive; **lookup stays
exact**. There is deliberately NO display label: one string, everywhere.

**Migration.** `electron/crucible/retire-reserved-name.ts` runs once at start,
above `startQueueEngine()` because it rewrites `queue-engine.json` on disk. It
adds the discovered engine to the registry under the name the engine calls itself
(or reuses the row that already has that URL) and rewrites every reference in
`crucible-routing.json`, `crucible-upstreams.json` and `queue-engine.json` in the
same pass. **If it cannot be done it refuses BY NAME and writes nothing at all** —
no Crucible here, an unusable name, that name taken by a different machine, or a
record that will not parse. `tools/test-crucible-retire-reserved-name.js` drives
it over a copy of Owen's own records and asserts every file is byte-identical
after each refusal.

**The one place "here" still means anything** is the scheduler's one-card rule:
BookForge still runs the long-form aligner on this machine's GPU itself, so a
render on a Crucible that answers on loopback is the same 3090 Ti.
`serversOnThisMachine()` answers that from the ADDRESS, it decides nothing about
how a server is added or drawn, and it goes away with §B7.

**Still standing, and it is Foundry's:** `foundry-app/electron/ipc.ts` registers
this machine's engine as `local`, arguing *"parity with BookForge"*. That
argument is now false. It is a vendored subtree with its own tsconfig — neither
`npx tsc -p tsconfig.electron.json` nor `node tools/run-keepers.js` compiles it —
so it must be changed upstream in the foundry repo and re-vendored.

## 0g. MIDDAY 2026-09-15 — what Owen's own in-app pass found, and what it cost

He ran the app. Every item below came from USING it, not from reading it, which is why they are
here and not in 0f.

**FIXED AND VERIFIED**
- **v0.6.0's release was BROKEN and is now whole.** The tag's CI run finished `failure`: the Windows
  host pack BUILT and then failed to upload, because the workflow selected the parts with
  PowerShell `-Filter "*.part[0-9][0-9]"` — `-Filter` goes to the Win32 matching API, whose only
  metacharacters are `*` and `?`, so a character class matched nothing and `gh release upload` got a
  tag and no files. Worse, the manifest job declared `needs: [pack, host]` with a comment saying a
  manifest without the host row leaves Windows refusing `pack_not_published` on a release that looks
  complete — and then carried `if: always()`, publishing exactly that (12 rows, no host). BOTH fixed
  in `.github/workflows/envpacks.yml` (crucible `1ddf6cb`): a `-match '\.part\d\d$'` regex plus an
  explicit empty-list refusal, and `always()` narrowed to `!cancelled()` with the UPLOAD gated on a
  merged-vs-`every_pack()` comparison. The pack was then built BY HAND from the tag's commit
  (`git archive v0.6.0`, NOT a worktree — a worktree here is CRLF and `recipe_sha256` at that commit
  hashes raw bytes, which would have disagreed with every other row) and uploaded; CI was NOT
  re-dispatched, because that would clobber twelve already-correct packs whose hashes would differ for
  tar-mtime reasons alone. VERIFIED INDEPENDENTLY: the published manifest now has **13 rows incl.
  `host`/`llama-windows`**, and the URL `install.ps1` fetches returns **HTTP 200, 46,343,744 bytes**
  matching the manifest's sha256. **The Windows "Set one up on this computer" one-liner now works.**
  The workflow fix is COMMITTED AND NOT PUSHED — an unpushed workflow fix does nothing and the next
  tag repeats the failure; it sits on top of crucible `3649c2c` + `205a3d7`.
- **The 811-char sentence that refused his book.** Two different causes, not one: chunk 21 was 801
  because the packer measured `spoken` (794) while the wire measures raw `len(text)` — the 7-char lead
  `[break]` (fixed, `268c26e7`); chunk 68 was a genuine ONE-SENTENCE 811 chars. Owen ruled: split at
  clause boundaries as a last resort. Built (`e5725653`): fires only for a single over-cap sentence,
  tiers `;` → `:` → em/en dash → `,`, cuts NEAREST THE MIDDLE, recurses, boundary char stays with the
  preceding piece, measured as the wire measures, printed at prep; no real boundary → refused
  `sentence_over_cap_unsplittable` naming index and length. **His chunk 68 now packs 405 + 412 at the
  third of its five semicolons**, and the real sentence (found in the Kershaw EPUB on `Z:`) is in the
  test verbatim. Normal packing pinned unchanged by hashing the Mutineer corpus at three caps.
- **Pack to the VENUE's band, never this machine's catalog** (`50b6747e`, `electron/crucible/voice-band.ts`):
  ceiling = `safe_max_chars` else `max_chars`, never above `max_chars`; target = server's, else the
  local catalog's CLAMPED; refusals `crucible_unknown_voice`, `crucible_voice_states_no_cap`,
  `crucible_voice_states_no_pace`, `crucible_chunk_over_venue_cap`. The "RULING OWED" in `stream.ts` is
  discharged. The arm question (Windows prep resolving `arm='wsl'` for a Mac render) no longer bites —
  the venue band REPLACES the caps outright — and the stale comment claiming a refusal that does not
  exist is corrected (`f1c11d4c`). `_targetCharsNote` → `_chunkLengthNote` on all four fine-tunes; the
  "MLX 900" prose that cost a wrong diagnosis is gone (there was never a `targetChars` for a fine-tune).
- **The bench's cloud lanes** (`8699d71d`): a server's `:cloud` lane is drawn only when that engine HAS
  an upstream configured (read from `/v1/settings`, chosen over per-class `route` because an upstream is
  a deliberate rare act and a route implies one); unknown keeps the lane.
- **Crucible `POST /quit`** (`205a3d70`) — the orchestrator can be stopped without `/F`. Measured this
  morning: `taskkill /PID` without `/F` reached NO quit path, wrote nothing, and the tray lived.
  One implementation shared with the tray menu; per owner: `wsl-unit` releases the claim and lets the
  hold go, `child` also stops its engine, `found` releases nothing. Answers BEFORE exiting (pinned).
- **`recipe_sha256` no longer depends on line endings** (`3649c2c`): normalise CRLF→LF, not the git blob,
  because recipes ship inside the wheel where there is no repo. It also collapsed a SECOND owner —
  `envpack`'s manifest column and `jobenv`'s env stamp hashed independently; now one function.
  v0.6.0's published manifest carries the LF digest and is unaffected, but BEFORE this fix a `--check`
  on the Windows desk would have REFUSED a correct published pack.

**FOUND BY HIM, BEING FIXED (agents in flight at compaction)**
- **A CANCELLED JOB KEEPS RENDERING — the worst of the day.** He stopped a narration; the Mac kept
  going. MEASURED: BookForge's remote-cancel fired correctly (`DELETE /v1/jobs/<id>` → `200 OK`); the
  job stayed `running`, progress `0.427 → 0.438 → 0.449`, artifacts `38 → 39`; a hand-issued DELETE
  answered `{"status": "cancelling"}` and ten seconds later it was still running. So the door ACCEPTS
  the cancel, RECORDS `cancelling`, and nothing acts on it: the loop never observes it, the claim is
  never released, the voice never comes off the card. It rendered 25 more chunks past his stop before
  I killed it with `launchctl kickstart -k gui/501/com.crucible.serve`. Agent is tracing where the flag
  is lost, making a cancel stop within one chunk, release the claim and settle the card the way a
  FINISHED job does, and checking whether `asr`/`align`/`rvc`/`denoise`/`pages`/`llm` and the streaming
  door share the hole. BookForge's side needs nothing — `render.ts`'s "A REMOTE RENDER IS CANCELLED ON
  THE SERVER, not by hanging up" is correct and worked.
- **THE MAC IS 6-10x SLOWER THAN IT WAS.** He measured 13-14 sentences/min where that machine has
  historically done **80-130** (his number, for the Mac, NOT the PC's batched figure — I had that
  backwards at first). So it is a REGRESSION, not an MLX ceiling. Symptoms: narrator at ~65% of ONE
  core, load average 3 on a 20-core M1 Ultra, nothing saturated. Two suspects, both self-inflicted last
  night: (1) the Mac's `install tts --build --force` rebuild at ~05:42 may have wiped site-packages
  patches exactly as the identical command did on the PC — and the re-apply fix (`958ddab`) landed at
  ~07:30, AFTER it; (2) we added `take` to `_mlx_batch_groups`' group key, so if anything now varies
  per row a batch of 16 becomes 16 batches of 1 while looking healthy. Agent measuring on the freed Mac.
- **Orpheus is still in the narration picker.** `shared/tts/engine-caps.ts:363` —
  `SELECTABLE_ORDER = ['orpheus', 'higgs']`, Orpheus FIRST. Ruled deprecated 2026-09-14, never removed
  from the one list that feeds `narrationEngineOrder()` → `selectableEngines()` → the modal's button
  group. Being retired the way XTTS was (the file's own `RetiredTtsEngine` machinery: a retired engine
  still PARSES and DISPLAYS so old records open, and can no longer be chosen), with the shipped presets
  that name `ttsEngine: 'orpheus'` corrected rather than left to resolve to nothing.
- **The bench's legacy GPU row.** Owen: one GPU slot per Crucible engine, two local CPU slots, nothing
  else. Both rows are CORRECT (both are registered engines, not in-app paths) — though the name
  `local` is no longer one anything here uses; see 0h. The offender is `the local narrator (legacy)`,
  which has THREE tenants,
  not one: `epub-align` (a local WhisperX spawn that auto-selects CUDA; Crucible has no long-form align
  job — the `align-longform` spec is in §B7, RULED 2026-09-15 and awaiting its build), `video-assembly` (declares `resource: 'gpu'`
  with no justification and no `machines()`), and any render while the legacy switch is on. Agent is
  making the row appear ONLY when something in the snapshot charges it, and classifying
  `video-assembly` honestly. Also fixing: `upstreamsByServer` is an IN-MEMORY Map, so every server reads
  `unknown` at app start and draws phantom lanes until coordinated — persist it and read it per enabled
  server at startup.

**MEASURED AND WORTH KEEPING**
- A narration run composes exactly `tts-conversion`, `rvc-enhancement`, `final-denoise`, `reassembly`.
  No `generate-sentences`, so epub-align never runs during narration. `slotSetForStep` tests
  `resource === 'cpu'` BEFORE travel, so a CPU step can never reach the legacy set; reassembly is CPU
  unless its config carries an inline RVC pass, which the narration composer never sets.
- **Prep always runs on THIS machine**, even for a Mac render (it reads the book off this disk), and it
  does NOT touch the GPU: the three modules the prep probe imports load with torch absent and no CUDA
  context. Owen's "is it using the local GPU" was the Windows desktop baseline (~2.6 GB: explorer,
  start menu, an Edge WebView, the NVIDIA overlay — BookForge's own Electron window is part of it).

**OWED FROM OWEN**
push the crucible branch (the workflow fix is inert until then) · does `video-assembly` need the card · the HF storage wall (two Orpheus voices still
only on WSL disk) · the linger ruling (a voice does not survive its render; ~30 s reload per standalone
render on the Mac, ~138-145 s per render on the PC) · regenerate the WSL unit from HIS login shell
(the on-disk one still has the unquoted PATH) · the in-app pass continues.

## 0f. THE MORNING OF 2026-09-15 — WHERE IT STANDS (read this first; 0e below is the evening before)

**Owen's ask overnight:** "fully functional by morning. Including the crucible servers set up and
configured on Mac, windows, and wsl"; an uninstall route (he will uninstall PC+WSL and Mac and
reinstall end to end as THE test); a CLI install route for a rented Linux GPU droplet; BookForge's
and Foundry's setup/settings pages able to install AND configure it; and Phase 17 — the explicit
**orchestrator / engine** relation (his words; never master/slave).

**Landed, all committed, nothing pushed except BookForge's branch (Owen authorised; `origin` at
e0c0ebe3+):**
- **The button ran, measured** (crucible PHASE15-HOST.md §7.3/7b/7c/7.6): T1 SDKs 245+299 · T2 full
  pytest green after 3 stale tests · T4 keepers green · **T6 dots under vLLM/WSL 73.1 s load, 4.5 s/page,
  11 blocks** · **T7 llama-windows: dots 6.2 s load, 3.7 s/page, 11 blocks byte-identical to vLLM's;
  qwen3.5-9b Q8 10.2 s load, 1.3 s/chunk** · T8 remove door · **T9 Mac asr 27 s, align 49 s** · T10
  engine task. T5 skipped (no `C:\tmp\phase15-testrun\anthropic-key.txt`). Seven defects found on the
  card and fixed the same night with tests (settlement race, Windows accelerator read, Windows guard
  asks for ROOM, projector pin → `ggml-org/dots.ocr-GGUF` Q8 pair, engine-task endings named, T6 stage,
  narrator HIGGS_ENV prefix regression).
- **Takes + zero-shot proven on the Mac's card** (MLX arm, tts env rebuilt on narrator 95d11238):
  take 1 ≠ take 0, take 0 == take 0 (seeded; per-take seed lane), take 2 refused `unknown_take`
  (what that run measured; the refusal was RETIRED on 2026-09-19 — a take past the ladder is a seed
  lane at the voice's own sampling, crucible docs/PHASE18-UNCERTIFIED.md §5);
  **first zero-shot render ever: 12.1 s audio in 9 s** with the owen-morgan clip + transcript. On the PC
  the same two checks are OWED once the card is free (Training pc holds it until ~06:30; the tts venv
  has narrator 95d11238 reinstalled `--no-deps`; the full `install tts --build --force` rebuild waits
  for the card window because of the 13 GB VM cap).
- **Servers:** WSL engine at crucible 53f40e2 on 7100 (`role: engine`), run tonight as a held process —
  the unit's user bus has NO socket in this boot, so `systemctl --user` fails from every session; the
  fix is `systemctl restart user@1000`, which would kill the trainer → done in the card window, then the
  unit owns the server again. Linger enabled (`loginctl enable-linger <your-wsl-user>`, root). The WSL distro
  terminates seconds after the last `wsl.exe` session even with units running → the orchestrator now
  holds it open (c59dc9d); tonight a held `sleep infinity` does. **Windows orchestrator installed for
  real**: `%LOCALAPPDATA%\Crucible\host\` (pack built locally — NO release publishes a host pack yet;
  `install.ps1` cannot run until the windows-latest CI row publishes one: OWEN's release), tray pid
  45504, one Startup `Crucible.lnk`, pairing file = the guest's line copied (§3.6). It runs the PRE-Phase-17
  pack; rebuild + tray restart in the card window. **Mac** at 53f40e2-era 3184860, tts env rebuilt,
  capability tts yes / pages no, doctor healthy, nothing resident.
- **Phase 17** (crucible b14ea44…53f40e2, `docs/PHASE17-ORCHESTRATOR.md`): `role` per process,
  `POST/DELETE /v1/peer/claim`, `GET /v1/peer`, `managed_by` on an engine, `engine{}` + capability
  read-through on the orchestrator's door (:7101 now answers `/v1/ping` + `/v1/info`), `engine-restart`
  task, `crucible host` = alias of `crucible orchestrator`; SDK `engineOf`; two defects fixed (the door's
  bearer was the host's token; tray Restart did nothing).
- **Uninstall + droplet install** (crucible `crucible uninstall --dry-run --json --purge-weights
  --wsl-too`, `install.ps1 -Uninstall`, `install.sh --uninstall` and the Linux branch;
  `docs/INSTALL-UNINSTALL.md`): weights KEPT by default; the token goes with config.toml so every app
  re-pairs after Owen's round-trip; never run live yet (dry-run on the staged home: 19 steps, 17 GiB kept).
- **BookForge doors** (0e3d9da6…e0c0ebe3 + the .cmd-spawn fix): "Set one up on this machine" was
  hardcoded DEAD on every machine — now live through the host's loopback door (`@crucible/bootstrap`
  vendored as a tarball); Settings → AI's write-through window shows refusal CODES; a fourth door in
  Crucible Servers "Remove the engine from this computer" (local-only, dry-run first, weights kept);
  `cli/pass.js` had been PERSISTING an Anthropic key in plaintext — refused by name. Foundry matched the
  door word for word (their main d9bfb4b+). Keepers 159 green. SDK re-vendored four times tonight, last
  `de666f5c` (sha 97da4118…).
- **Also landed:** Enhance page deleted; extension → Crucible direct with clip picker; Bible
  abbreviations; narrator sampling channel; coordinate shape matched with Foundry; Orpheus cleanup
  (26 GiB; two voices wait on the HF storage ruling).

**In the card window (Training pc's ONE "clear" message, ~06:30):** PC take-0/take-1 + zero-shot
render through the WSL engine; `crucible install tts --narrator-engine higgs-v3 --build --force` in WSL;
`systemctl restart user@1000` → unit owns the server; `envpack build host` → tray restart → the
orchestrator claims the engine (`managed_by` set); T2 full pytest; then hand back to whoever needs it.

**The card window, 07:25–08:xx (Training pc's clear message came at 07:25):**
- **PC takes proven on cuda-linux:** take 0 = 11.30 s / 264,174 B (byte-identical to last night's — seeded),
  take 1 = 12.02 s / 275,887 B (differs: the rung reaches SGLang), take 2 refused `unknown_take`
  (as measured that morning; that refusal is retired — see the note above).
  **Zero-shot load OK on the PC** (116 s, reference registered). The zero-shot RENDER on the resident voice
  refused `accelerator_busy` → root-fixed (crucible a87badab: a render on a resident voice never asks the card
  for room; on cuda-linux the guard owns narrator's whole process group).
- **A regression from the full `install tts --build --force`, found and fixed the same hour:** pip reinstalled
  upstream vllm/vllm-omni 0.28.0 pristine and WIPED the two site-packages patches narrator needs
  (`higgs-sentinel-filter`, `vllm-negative-token-id`) — every Higgs load then failed "sentinel proof holds no
  records"; nothing re-applied them. Now `narratorpatches.apply()` runs the vendored appliers after pip and the env
  is not stamped without the doctor's marker (958ddab / 7e29922). Rebuilt in 137 s; verified under the unit:
  `load-voice owen` 118 s, one render 3.9 s, 96 sentinel records. (There never was a 0.29.0 env — the banner reads
  0.28.0 on both the working and the failing runs.)
- **The WSL engine belongs to its systemd unit again:** `systemctl restart user@1000` as root (07:34) gave the
  user manager its bus; from a `wsl --exec` session the working form is
  `env XDG_RUNTIME_DIR=/run/user/1000 systemctl --user <verb> crucible.service` (the missing variable and a
  missing socket print the identical "Failed to connect to bus"). The unit file's `Environment=PATH` is still the
  unquoted one on disk (`service.py` now quotes it) — regenerate with `crucible service install` from Owen's OWN
  login shell, never from an agent's.
- **T2 full pytest PASS in 415 s** on the tip (seed lane, Phase 17, uninstall route, patches).
- **Windows orchestrator:** the Phase 17 pack rebuilt (sha 0da51e6d…) with a NEW consent setting —
  `%LOCALAPPDATA%\Crucible\config.toml` `[orchestrator] distro = "Ubuntu"` names the distro the orchestrator may
  manage (watch + claim + unit restart widen; destructive recipes stay refused by rootfs) — and the unit probe /
  start / restart / tray Stop all carry `XDG_RUNTIME_DIR`. **Tray swapped 08:24, pack 0da51e6d, pid 23052 — THE CLAIM LANDED:** 7101 `/v1/info` reads `engine.owner: wsl-unit`, 7100 `/v1/peer` reads `managed_by {crucible-orchestrator@example-pc, http://127.0.0.1:7101}`, pairing file byte-identical. Measured on the way: `taskkill /PID` without /F does not stop the tray (PHASE17 §4.4 `POST /quit` owed). The morning's figures are in crucible PHASE15 §7.3/§7.6/§7b and PHASE17 §8 (f300df8).
- Two things for later rulings: the tray has no shutdown route (PHASE17 §4.4 owed: `POST /quit`); `recipe_sha256` in
  envpacks.json is line-ending dependent (CRLF checkout vs LF) — one line to fix in release machinery.

**Owen, in the morning:** the in-app pass on both machines (first-run wizard, Settings → AI, Crucible
Servers doors, the extension's Options + Load voice, a Listen, a render, Correct Sentences); the
release that publishes the host pack (`./scripts/release.sh --branch feat/phase6-remote-render`,
after `envpacks.yml`'s windows row runs); the HF storage choice; the linger ruling (a voice does not
stay resident after a render — ~30 s reload per standalone render, measured on the Mac); then the
uninstall-and-reinstall round trip on PC and Mac as the acceptance test.

## 0e. WHERE IT STANDS AT 19:40, 2026-09-14 (0d below is the BookForge half; read this first)

**THE INSTALL/CONFIGURE/UNINSTALL PATH IS WALKED END TO END, 2026-09-15 (`0e3d9da6`,
`c255746d`, `3172156b`).** Owen: *"let's make sure bookforge's and foundry's settings and setup
pages are capable of installing and configuring it."* Step by step, with a keeper each.

**Step 1 — first run, no server.** Already true and now actually reachable. The setup step
(`app-crucible-doors mode="probing"`) shows ONE of three faces from `crucible:host-install-plan`;
the read of a Crucible on this computer is `discovery.ts`, pairing file FIRST. The gap was the third face: **the install
button was disabled on every machine** because `DRIVEN_INSTALL_AVAILABLE` was a hardcoded
`false` over a hand-transcribed `@crucible/bootstrap` surface. The package is now vendored
(`vendor/crucible-bootstrap-0.6.0.tgz`, `npm pack` of `sdk/bootstrap`, the same stopgap shape as
the client tarball, note in `package.json`), the types are IMPORTED — two had already drifted,
`install()` having gained `release`/`home`/`bind`/`onHostEvent`/`fetchImpl` and lost
`wheel`/`condaRoots` — and `loadBootstrap()` does the real `await import`. On Windows the door is
the HOST's and nothing else (§4.3): no host → `host_not_installed` carrying the package's own
`hostInstallCommand()`; a host → `POST /install` on 7101, events relayed. New IPC event
`crucible:install-progress` carries `step` / `progress` (bytes) / `state` (the 4c code BY NAME) /
`line` / `done` / `failed`, drawn as three signals in the doors component. The plan stops printing
a shell sequence — those eight conda-and-wheel lines were a second description of an install
PHASE14 §4a says cannot differ, and they HAD: `CRUCIBLE_WHEEL`, `MAC_CONDA_ROOTS` and the plan's
`wheel` field are deleted, `steps` is read from `installSteps()` and carries no commands.
Keeper: `test-crucible-install-seam` (42), rewritten to guard the opposite failure — a live
button over an absent installer — over a scripted runner and a scripted host door.

**Step 2 — Settings → Engine.** Already true (`e8ef901b`, `87770b0e`): the write-through window
is Settings → **AI** (`app-ai-setup-wizard`, section id `ai`), with **Where each job runs** (one
select per llm class) and **Accounts this engine can send work to** (three cards, key write-only,
`keyHint` verbatim, Test before Save), all through the real SDK. One gap, fixed: the refusals were
rendered as `message` alone, so `route_bad_model`, `route_upstream_unconfigured`, `upstream_in_use`,
`unknown_upstream` and `upstream_bad_field` reached a person with no name on them. The code is
drawn beside the sentence now, in all four places, as every other Crucible panel already does.
"Open Crucible" is **Open engine console** (crucible-words rule 3) and the servers panel's
Select/Remove/Test is rank-order + enable + the wait-for radio, deliberately.

**Step 3 — uninstall.** New. `crucible uninstall` is the ENGINE's verb; `electron/crucible/
uninstall.ts` finds the CLI that owns the local server, runs `--dry-run --json`, shows it, then
runs the same plan. **LOCAL ONLY** (ruling taken with Foundry so both apps draw one door):
`uninstall_not_local` for anything else, at the door as well as the button. Weights KEPT unless
asked; `keep` is a reported RESULT with its size. `--wsl-too` refused where it means nothing;
nothing defaulted out of the document; a CLI predating the verb is `uninstall_not_available`.
Settings → Crucible Servers gains door 4, **Remove the engine from this computer**, with
**Show me what would go** before **Remove it**. Keeper: `test-crucible-uninstall` (23).
TODO: built to `crucible/uninstall.py`'s `Plan.to_dict()` and its parser's flags, read while that
side was in flight — `docs/INSTALL-UNINSTALL.md` will own the contract.

**Step 4 — the Anthropic key passes THROUGH and is stored nowhere.** True in
`electron/`/`shared/`/`src/` since `ec24f361`. Not true in `cli/`, which
`test-no-cloud-doors.js` never walked: `ai-clean.js` built `{claude:{apiKey}}` configs,
`pass.js` wrote `claudeApiKey` into a PERSISTED pass record (queue.json and the book ledger — a
plaintext credential with no expiry), `bookforge-tts.py` exported `BOOKFORGE_AI_API_KEY` into the
child. All refused by name now, flags left declared so a stale script gets a sentence rather than
"unrecognized arguments". The keeper grows a `cli/` walker and six checks built on "looking is not
using": a read is banned when the value is BOUND, and the two files allowed to look must throw
about what they see.

**The `.cmd` defect, found by Foundry and reproduced here (`7cf49297`).** `spawn('…crucible.cmd', argv)` throws **EINVAL** on Electron 33's Node — the CVE-2024-27980 fix refuses a
`.cmd`/`.bat` without a shell — and the host pack's `crucible.cmd` is the ONLY Crucible CLI a
Windows machine has, so the first real press of the uninstall door would have failed with nothing
on screen about batch files. `electron/crucible/host-runner.ts` is BookForge's runner now: the
package's `processRunner()` everywhere else, and `cmd.exe /d /s /c ""<program>" "<arg>""` with
`windowsVerbatimArguments` for a batch target (`/d` skips AutoRun, whose output would land in the
JSON this app parses; `/s` makes the quoting one rule). NOT `shell: true` — that is the injection
hole the CVE is about. A token with `"` or `%` is `uninstall_bad_path`, refused not escaped: both
would turn a path into a different path and this command deletes directories. A space is fine.
Measured against a real scratch `.cmd` in a spaced directory: bare spawn EINVAL, hardened run exit
7 with stdout, stream three lines, non-`.cmd` untouched. The INSTALL door spawns no `.cmd` today
(win32 `install()` is HTTP plus `fileExists`/`readFile`) and shares the runner anyway, with the
no-spawn fact pinned. Keeper +6 (29). **Ruling, both apps: the uninstall checkboxes are
CLEAR-ONLY** — a change clears the plan and the person presses "Show me what would go" again; no
automatic re-run, because a dry run measures six weight directories and a plan that reappeared by
itself under a live "Remove it" button is somebody pressing Remove against numbers they had not
read.

**Gates:** `npx tsc -p tsconfig.electron.json` 0, `npx tsc --noEmit -p .` 0, `npx ng build` 0,
`node tools/run-keepers.js` **159 suites, 0 failing** (1 skipped: `test-quire`, no
`BOOKFORGE_KA_EPUB`). Every new row verified in `dist/renderer`.

**One live READ, nothing else.** `%LOCALAPPDATA%\Crucible\pairing` EXISTS — the Windows host
agent has written it — and `GET /v1/ping` on 7100 answers
`{"crucible":true,"name":"crucible@example-pc-wsl","api_version":1}`. So step 1's first door is live
on this machine right now. **No install was run**: it would fight the host agent and the WSL
server holding the card. Everything else was driven over injected runners and a fake host door.

**PHASE 16 STEPS 2-4, 2026-09-14 late evening.** Steps 2 and 3 are in
(`9d2a85f3`, `a7f12592`, `0d372bce`, `c6a7897e`, `db775aa5`, `a818d328`): the Listen text path
and the Listen client are `shared/listen-text/` and `shared/listen-client/`, compiled by the
main process, the renderer and the extension's esbuild alike, and the browser extension now
talks to a Crucible with **no BookForge in the path** — a registry of pasted connect codes with
one selected server, Load/Unload as `load-voice`/`unload-voice` jobs, and one
`POST /v1/tts/stream` session per read. `stream-scheduler.ts` and `electron/crucible/stream.ts`
are adapters onto that shared code with their exported surfaces unchanged, so the 8766 relay,
the reader bridge and the Play tab still run on it. Two keepers hold it:
`test-listen-text-one-source` compares the two bundles' function bodies byte for byte (a split
that differs by one character makes a resumed block splice one row's audio under another row's
text, and nothing throws), and `test-extension-option-columns` pins both columns of the plan's
table by name. **Two rulings came with it (Owen, late):** step 4 — **the Streaming tab STAYS in
the main process**, no bearer token in the Angular renderer, so `src/app/core/listen/` is not a
directory that is owed but one that is not wanted (the shared policy and the shared row layer
are already what the tab drives); and step 6 is **SPLIT** — the speak relay goes, the tab
recorder's endpoint on that same server stays, which is why the extension keeps its BookForge
host/port/token rows for good. **Still owed:** (a) **zero-shot is BLOCKED, not skipped** —
`load-voice` has no `reference` field in PHASE3-TTS.md or the 0.6.0 SDK, and a
`kind = "zeroshot"` voice is refused before any engine starts, at the load door as well as the
render door, so the extension's clip picker cannot be built until narrator's load message
carries clips. (b) **Nothing was run on a card** — both were off limits — so the extension has
never spoken: Owen's in-app pass is the first real read. (c) `test-crucible-module-file` is RED
and correctly so: the crucible checkout regenerated `modules/bookforge.module.json` at 19:31
(`subjects: [model qwen3.5-9b]` → `needs: [{class: "clean"}]`, the §5.3a classes-not-ids
change) and BookForge's vendored copy has not been re-vendored yet. That is the re-vendor
already on this section's "next" list; it was left red rather than silenced.

**Landed today, evening:** Crucible Phase 15 contract (`crucible/docs/PHASE15-HOST.md`, §0–§8 incl.
the AMENDED block: Windows IS the `llama-windows` backend; control is Windows's, data is the
card's; §3.5a remove door; §3.10 llama-server child; §4.7 engine switch on the page; §4.6 Mac;
§5.3a modules carry classes; §8 THE BUTTON). Crucible branch `feat/phase6-remote-render` has: the
host branch merged (tray, Startup item, install.ps1 → host, host pack built, migrate-weights real),
the Mac branch merged (align + asr on mlx-darwin, measured; engine per (backend, class-family);
NO dots mlx block — mlx-vlm's server drops the image, ruling owed), settings/routes/upstreams,
SDK with six fixes (0191715, b97c05f), the remove door (332116f), one page-request builder
(d028d56), manifests INSIDE the wheel (4473b53). BookForge: 21 commits to cef93ad4 + this plan
(`docs/EXTENSION-TO-CRUCIBLE-PLAN.md` = Phase 16, rulings: options carried, Enhance dropped,
zero-shot). Foundry main ecd03e3 (H–K merged; L gated). The MAC runs main HEAD (editable
checkout `/Volumes/Callisto/Projects/crucible`, deploy = git pull via bundle; NEVER a wheel until
4473b53 is proven). Live WSL server still on the env-packs build (493f040) — S1 owed.

**Running:** (1) the server finish agent in the Crucible checkout: engine task (in progress —
api.py/host/app.py/tasks.py dirty), then §5.3a class-resolving modules, SDK re-pack, S1 (WSL
clone → HEAD + restart), `scripts/testrun-phase15.sh` (+ `--dry-run`), docs §7.5. S2 staging is
DONE on disk: `C:	mp\phase15-testrun\{home,host,packs}` = 15 GB (engine zip, dots GGUF pair,
9B GGUF). (2) the Orpheus cleanup agent: verify-by-sha against HF, delete, fstrim; reports the
Ollama store (63 GB) without deleting it — Owen's call.

**Owen's standing orders tonight:** NO GPU on the PC; Mac's card allowed; full pytest suites run
ONLY when he says (single files under `/tmp/crucible-pytest.lock` with the `[t]rain_lora.py`
guard); a trainer runs in WSL (13 GB VM cap); all subagents Opus or lower.

**Next, in order:** the server agent's report → re-vendor the re-packed SDK into BookForge
(`npm install file:vendor/…tgz`, check the lock's integrity) and tell Foundry the sha; regenerate
+ re-vendor `bookforge.module.json` (classes, not ids) and tell Foundry; Mac staging (7c's
M-steps: install align|asr --build, pulls, capability write); run `testrun-phase15.sh --dry-run`;
tell Owen THE BUTTON is ready; he releases the card and says when; run it; paste the measured
figures back into §7/§7b/§7c. Then: Owen's in-app pass; Foundry's L + the single re-vendor;
publish v0.6.0 (`./scripts/release.sh --branch feat/phase6-remote-render`, Owen runs it); Phase
16 (the extension). **Rulings owed from Owen:** dots on the Mac through Crucible's own
mlx-vlm wrapper (recommended yes); Correct Sentences' spread = the take ladder (recommended
yes); the Ollama store.

## 0d. PHASE 15 — the engine is the one door, and the apps have no provider code (2026-09-14, evening)

Read `C:\Users\<user>\Projects\crucible\docs\PHASE15-HOST.md` — it is the CONTRACT, and it
carries the rulings this section only summarises. Owen, that evening: *"the user will be
installing bookforge/foundry (and by extension, crucible) on windows … one centralized
location that controls the GPU power … bookforge/foundry gain a simple contract: send
commands to the crucible server. period. they dont have ollama fallbacks or cloud anything at
all … one contract, one SDK, one API, one communication method."* And: *"Bookforge and
foundry setup/settings should be able to configure crucible settings. If the user enters an
anthropic api key, it should pass through to crucible … the user shouldn't have to interact
with crucible almost at all but should have access to it if they want to."*

### The ruling this OVERRULES

**§3's "CLOUD KEYS HAVE ONE OWNER: FOUNDRY'S CLOUD CARD" is overruled, 2026-09-14 evening,
by PHASE15 §0.** It was a default ruling taken that morning and Owen said he might overturn
it; he did. The keys move INTO the engine — `[upstreams.anthropic] key` in its `config.toml`,
mode 0600 beside the token it already holds — and Foundry's card becomes a WINDOW onto the
engine's settings, exactly as BookForge's AI section now is. Neither app stores a credential
anywhere. `electron/cloud-credentials.ts`, which read Foundry's `cloudProviders` record out
of `<userData>/app-settings.json`, is deleted with everything that called it.

### Two amendments the same evening, both from Owen

- **Windows IS a backend** (crucible `56cfe37`): *"the windows side should still host GPU jobs
  even if WSL isnt present/workable … we can still run dots, qwen 9b, or whatever else from
  the windows side. just like it runs from the mac side."* So `backend_kind = "llama-windows"`,
  structurally what `mlx-darwin` is, serving the llm classes and `pages`; there is no
  `backend_kind = "none"` and no "this machine has no accelerator" sentence. What WSL adds on
  top is vLLM/SGLang and the five Python job types (`tts asr align rvc denoise`), and those
  five answer `enabled: false` with ONE shared sentence so a screen says it once.
- **There will never be a local GPU slot** (crucible `aadd9ed`): *"there will never, ever be a
  local gpu configured. there simply wont be an outlet for it."* The end state of
  `shared/queue/slot-sets.ts` is one `[gpu]` per REGISTERED Crucible server, its `[cloud]`
  lane, and `local-work [cpu][cpu]` — CPU slots stay local. The legacy set and its GPU slot
  are deleted with the legacy spawn layer after Owen's in-app pass; that is one subtraction,
  and nothing phase 15 added depends on the legacy set existing.

### What landed in BookForge

| commit | what |
|---|---|
| `a1c99c41` | the settings seam (`electron/crucible/settings-wire.ts`) + the pairing file (`pairing-file.ts`), and `tools/test-crucible-settings-seam.js` |
| `ec24f361` | the deletion: two providers left, and the scheduler's `[cloud]` lane |
| `7ce4c137` | four keepers follow the deletion |
| `5b9818e5` | this section, the overruled ruling, the ledger's §11.5a, and the second Foundry tripwire |
| `73e2e3ad` | the setup step stops asking anybody to type a token |
| `3b88b1bf` | the SDK pin's note names what else waits on it |
| `e8ef901b` | Settings -> AI and the wizard's AI step become a WINDOW on the engine's settings document: write-through, key write-only, Test before Save, one PUT that configures an upstream AND routes to it |
| `571fc8f1` | every chat states its act, so `/v1/activity` stops saying "a chat" |
| `17b763d0` | the Phase 15 SDK vendored, and the npm cache trap that hides it (twice) |
| `87770b0e` | the seam DELETED - the settings door goes through the real SDK |

### THE WRITE-THROUGH PATH, RUN AGAINST A REAL PHASE 15 SERVER (2026-09-14, the Mac)

Owen: *"we can freely test the logic pathways anyway by using the mac."* The Mac Studio runs
the merged Phase 15 server (`mlx-darwin`, `192.0.2.79:7100`). Everything below went
through BookForge's OWN doors — `parsePairing` → `addServer` → `engine-settings.ts` — from a
node script against a TEMP userData, never curl and never the real registry. **The Mac's
settings were put back and the final GET is byte-identical to the first.**

| # | what ran | what came back |
|---|---|---|
| 0 | the paste box door: `parsePairing`, then `addServer` | registered; the listing carries `****…` and not the token |
| 1 | `crucibleEngineSettings` | all four llm classes present; three upstreams present, none configured; `backendKind: mlx-darwin` |
| 2 | `crucibleCapabilityWithRoutes` | `route` on EVERY row; `clean/translate/simplify/analysis/tts/rvc/denoise` all `local`; the record went `unknown` → `local`, which is what the scheduler asks synchronously |
| 3 | coordination's READS | job types `denoise, echo, llm, rvc, tts`; **missing: `asr`, `align`, `faster-whisper-large-v3`, `qwen3-aligner`**. NOT posted — see below |
| 4 | ONE `putSettings` with `{routes:{translate:"anthropic/claude-sonnet-5"}, upstreams:{anthropic:{key:…}}}` | `route: upstream`, `configured: true`, `keyHint: "…test"`; **no key anywhere in the document**; the route record re-recorded from the PUT's own answer with no second read |
| 5 | a RE-READ of capability | `translate enabled=true route=upstream selected=anthropic/claude-sonnet-5`, and the reason keeps the local answer: *"routed to anthropic; the local answer would be: qwen3.8-27b fits: it needs 51.7 GiB and there is 61.0 GiB available…"* |
| 6 | `testUpstream('anthropic', {key: 'sk-ant-test'})` | `{ok:false, refusal:{code:"upstream_rejected"}}` — an ANSWER, not a throw, reaching BookForge's projection intact |
| 7 | `putSettings {routes:{translate:"local"}, upstreams:{anthropic:null}}` | back to `local`; `configured: false` |
| 8 | final `crucibleEngineSettings` | **deep-equal to step 1** |

**Nothing was posted at step 3, deliberately.** Owen allowed the Mac's CARD — "a small llm
chat or a tts render" — and coordinating would have downloaded and built the `asr` and
`align` environments plus two model pulls on his Mac, which is a different magnitude of
thing. Every READ coordination makes was made; what it WOULD install is named above; the
press is Owen's. (Nothing needed a completion either, so the Mac's card was never used.)

**A fifth SDK defect, found here.** The `upstream_rejected` result's MESSAGE names the wrong
party: *"crucible refused the token (upstream_rejected): anthropic refused this server's
credentials with 401: API key is invalid."* The second half is right and is the SDK's own
`serverMessage`; the prefix is its `CrucibleAuthError` prose, because `testUpstream` catches
its own 401 and returns `message: error.message`. A person reading that would go and
re-paste their CONNECT CODE, when what was wrong is the Anthropic key they just typed.
BookForge passes it through verbatim — renaming another owner's refusal is the defect this
whole phase is against — so this one is theirs to fix.

**Nothing in phase 15 has met a card.** Owen's instruction while it was being built was to
stay off the GPU: no job against the live engine, no run that loads a model. The one thing
that touches the live server at all is a `GET /v1/capability` from `test-clean-step-door`,
and that is how the SDK's pre-phase-15 defect below was met for real rather than in theory.

**FIVE DEFECTS THE LANE ONLY HAD BECAUSE IT LANDED, all found by reading the
paths it touches rather than by running anything** — and each one is a place
where "this run holds no card" was true and something downstream still assumed
a card:

| commit | what it would have done |
|---|---|
| `4fc071d4` | a RESTART re-derived the resource from the module (`gpu`) while the venue persisted (`mac:cloud`), leaving a step on a lane with no gpu slot: never admitted again, nothing saying why |
| `9b75d093` | a routed run still took a model LEASE, which §3.4 refuses `lease_not_needed` — asking for a refusal and reporting it as a failure to clean a book |
| `ccdd0153` | the residency preflight asked `/v1/models` about an upstream id that is never in it, refusing `crucible_unknown_model` and telling somebody to `--crucible-load` a thing that cannot be loaded |
| `2e502feb` | the ENGINE door did the same, and silently ignored a `loadFirst` that cannot happen (now `crucible_upstream_not_loadable`) |
| `dc662e4c` | admission still ran the routed row past the engine's GPU slot, this machine's card and the training lock — so a translation on somebody's API waited for a narration, which is the exact thing the lane exists to stop |
| `f17d555c` | the bench told a row waiting for a full cloud lane it was "Waiting for a CPU slot", sending a person to look at their own processor |
| `28ad983f` | a removed server's route record outlived it, so a re-added name was answered from for one pump |

Three doors now read ONE discriminator for "is this an upstream model id"
(`isUpstreamModelId` in `crucible/text-acts.ts`, the contract's slash rule from
§1), which is why it was put beside the act names rather than written
`.includes('/')` three times.

**CONNECT (§5.1) — the pairing file is a second door to the Crucible on this computer, not a fallback.**
`discoverCrucible` asks `$CRUCIBLE_HOME/pairing` first (else
`%LOCALAPPDATA%\Crucible\pairing` on Windows, `~/.crucible/pairing` elsewhere — pinned in
PHASE15 §3.6, which this build asked for and the Crucible agent wrote), and only then reads
`config.toml` through `wsl.exe`. Two named artefacts written by different parts of the
system, and §3.6 dates the second: the WSL read *"is how the WSL server gets registered"*
until a host exists on the machine, *"and that door is deleted when the host lands"*. On
Owen's PC today there is no host, so door 2 is the live one. `LocalServer.via` gained
`'pairing'`. An absent file is `null`, and `null` is the FACT "no engine on this machine".

**THE MODEL IS THE SERVER'S ANSWER (§5.3).** A text door sends `capability.selected` for its
class and nothing else. `crucibleActModel` in `electron/crucible/text-venue.ts` is the one
owner of that read and of the stamp that memoises it for the run, so a translation making
three hundred batch calls asks once. `providerConfigOf` takes the ACT, required and never
guessed. `crucibleModelForAiStep` is gone: the id needs a server and a round trip and the scheduler's
question is synchronous, so all three lease hooks answered `null` by construction — the
argument `pass.ts` had already written for `narration-text`, true of every act. **And a hook
every module answers `null` for is a comparison that matches nothing**, which is how `pause()`
came to close the lease of a step that was still running (bug hunt 2026-09-19 §H). So
`StepModule.leasedModel` was REMOVED on 2026-09-19 and the carry-over compares
`StepModule.crucibleClass` on the row's server (`nextActWouldUseHeldCard`, queue-engine.ts):
one server maps one class to one model, and `withRowLease` still compares the real ids at the
act. The OWED async id-read is closed by that — not by a table on this side.

**THE `[cloud]` LANE (§5.3).** A row whose class routes `upstream` on its engine takes that
engine's `<server>:cloud` lane — one per server, `gpu: 0`, two wide — and no GPU slot and no
lease. `slotSetForStep` reads the venue BEFORE the resource so the pair lands there;
admission writes both at the one moment both facts exist. The route comes from
`electron/crucible/routes.ts`, filled by the capability read coordination now makes on every
connect (a third read beside `info` and `catalog`) and again out of every settings write's
own answer. **Nothing polls**, and an engine nobody has read yet is a WAIT with a sentence —
never a guess in either direction, because assuming `local` parks an upstream-routed class on
a card nothing runs on and assuming `upstream` does the mirror.

**THE SEAM, AND ITS EXPIRY.** `@crucible/client` is growing `settings()`, `putSettings()`,
`testUpstream()`, `readPairingFile()` and `CapabilityRow.route`; the pinned tarball has none
of them. `electron/crucible/settings-wire.ts` speaks the wire the contract specifies and
`tools/test-crucible-settings-seam.js`'s FIRST check asserts the SDK still lacks every one —
the day it goes red is the day the seam is deleted, and the failure prints the four steps.
`crucibleCapabilityWithRoutes` is in that seam for a reason worth stating: the SDK's own
parser builds a row from five named fields and DROPS `route`, so a route read through it
today is not missing, it is discarded silently.

**HOW AN ABSENT `route` IS READ** (crucible `eb59f7b`, settled because Foundry reads the same
document and the two of us were diverging): no row carrying `route` is a pre-phase-15 server
and every class IS local — a stated fact, not a filled default; SOME rows carrying it and one
not is refused `capability_route_missing` naming the row; a value that is neither is refused
`capability_route_unknown`. Owen's live WSL server answers the first way until the phase-15
branch is deployed onto it.

### The legacy local spawn layer keeps exactly two Ollama doors

Both named in `tools/test-no-cloud-doors.js` so a THIRD is a red test, and both die with that
layer after the in-app pass: `gpu-arbiter.ts`'s `unloadOllamaModel(s)` (VRAM eviction so the
legacy narrator spawn can have the card — it asks nothing of a model and gets no text back),
and `tts-number-normalizer-runner.ts`, whose request MOVED into it from `ai-bridge.ts` rather
than being kept alive in a bridge that no longer has providers.

### Foundry, and the single re-vendor

Foundry main is **`ecd03e3`** (pushed): their packages H (coordinate), I (settings window),
J (pairing file + connect code) and K (dispatch on the route) are merged, they read `route`
per `eb59f7b`, and every Crucible slot has a `<slot>:cloud` lane of width 2 on their side too
— the same shape, arrived at from the same contract. Their L (the deletions) is gated on a
`llama-windows` server on a clean box.

**The single re-vendor target is their sha AFTER L**, not `ecd03e3`. It carries what the plan
already lists — `RunOptions.waitFor`, the deletion of `hosted_placement_not_vendored`, the
`slots?()` removal — plus the cloud card's replacement (PHASE15 §5.4).
`tools/test-foundry-hosted-crucible-seam.js` grew a SECOND tripwire, written the same way
round as the first: it passes while `foundry-app/electron/cloud-providers.ts` and the
dispatcher's `slot.kind === 'cloud'` are still there, and goes red the day they are not —
which is the day the target is reachable, not the day something broke.

### TWO KNOWN SDK DEFECTS, and the one thing that is ours

Found 2026-09-14, when the Phase 15 SDK was vendored and BookForge's stand-in was deleted.
Numbers 2 and 3 are DEFECTS IN THE SDK, confirmed as such by the Crucible side the same
evening after Foundry measured both against Owen's live server; a re-packed tarball is
coming and **BookForge works around neither**. Number 1 is ours. Number 4 is a deliberate
divergence of ours, and small.

Everything here is written down rather than remembered because each one has a tripwire, and
a tripwire nobody can find the reason for gets deleted by the next person.

1. **The SDK's is ASYNC and BookForge's read is on a SYNCHRONOUS path.** The SDK says why in
   its own header, and the reason is a packaging rule rather than anything about the
   operation: a static `node:fs` import would break a bundler targeting a browser-ish
   runtime, so its imports are assembled at run time and a dynamic import is a promise —
   `cruciblePairingPath` is async too, so even the PATH cannot be had synchronously.
   BookForge's `discoverCrucible` is synchronous because the registry, `readRouting()` and the
   hosted-Foundry snapshot all are, and `readRouting()` runs inside the queue's synchronous
   pump. Ours, ours to fix, and not on the way past.
2. **THE SDK'S PATH RULE DOES NOT CARRY THE WINDOWS CASE §3.6 PINS.** The table in §3.6 says
   `%LOCALAPPDATA%\Crucible\pairing` on Windows, beside `wsl\`, `downloads\` and `host\`,
   because the thing that writes a Windows-side copy is `crucible host` and that is its
   per-machine root. `cruciblePairingPath` implements `$CRUCIBLE_HOME`, else
   `~/.crucible/pairing`, on **every** platform. That is one name with two owners.

   BookForge implements the DOC, because the doc is the owner (PHASE15's preamble). **It is
   not load-bearing today** — on Windows the writer is `crucible host`, which does not exist,
   so there is no file at either path and the `config.toml`-through-`wsl.exe` door is the live
   one. It becomes load-bearing the moment the host ships, and then whichever of the two is
   wrong finds nothing and offers to install a second engine over a running one.

   **Confirmed by the Crucible side the same evening as an SDK DEFECT, not a question about
   the contract** — Foundry measured the same thing against Owen's live server — and a
   re-packed tarball is coming. BookForge does NOT work around it: it follows the doc, and
   the instruction is to read the path from the SDK once it is fixed.
   `tools/test-crucible-pairing-file.js` goes red the day they agree and says what to delete.
3. **`client.capability()` THROWS on a pre-Phase-15 document** — one with no `route` on any
   row — which contradicts §3.3's rule that such a document means "this server predates the
   field and every class IS local". Also measured by Foundry against Owen's live server, also
   being fixed in the SDK, and also NOT worked around here: BookForge reads capability through
   the SDK, and the keeper that exercises a no-route document is marked as depending on the
   re-vendor. **Owen's live WSL server answers exactly that way until the phase-15 branch is
   deployed onto it**, so this one bites the moment anybody connects to it.
4. **`CrucibleClientOptions` has no timeout.** Found by the swap itself: the deleted seam
   put a 60 s `AbortController` on every settings call, and there is nowhere to put one on
   the client. A settings screen against a server that goes away mid-answer now waits on
   `fetch`'s own default. Small, real, and theirs to take back.
5. **`testUpstream`'s refusal MESSAGE names the wrong party** — found on the Mac, see the
   transcript above. `{ok:false, code:'upstream_rejected'}` is right; the sentence beside it
   is the SDK's `CrucibleAuthError` prose about OUR bearer token, not the upstream's key.
6. A sixth, small, and this one is OURS by choice: an EMPTY pairing file. Ours throws; the SDK answers `null`. We keep ours,
   applying the SDK's own argument — its header says a malformed file must throw "because a
   line somebody's installer wrote badly is a broken install, and answering 'there is no
   server here' would send the user to install a second one", and a zero-length file is the
   same broken install.

### What needs Owen

1. **The in-app pass, on a free card.** Everything in this section has met a fake server and
   nothing else. Owen's instruction while it was being built: *stay off the GPU* — no job
   against the live Crucible at `127.0.0.1:7100`, no run that loads a model. So the whole of
   phase 15 on this side is "built, keepers green, never run".
2. **Publish v0.6.0** (§0c item 1), still, and then the SDK's phase-15 methods — at which
   point `test-crucible-settings-seam.js`'s first check goes red on purpose and the seam is
   deleted.
3. The legacy spawn layer's deletion, which is what takes the last two Ollama doors, the
   legacy slot set and its GPU slot, `'local'` from `AIProvider`, and ~250 GB of envs.

## 0c. WHERE IT STANDS AT 17:30, 2026-09-14 (0d, above, is later — read it first)

**Rulings Owen made today, in order:** Orpheus is DEPRECATED and removed from Crucible; XTTS
fully removed; Higgs is the frontier · Crucible has its OWN UI (not microservices) and hands out
the token · CPU slots stay local · environments come from the GitHub release · Crucible installs
independently and every app coordinates with whatever is there · a Crucible-owned WSL distro,
"make it idiot proof" · NO consent step — coordination is automatic on every connected server ·
Foundry does the same simplification now.

**Landed today (all on the two feature branches, NOTHING pushed, nothing merged):**
- Crucible `feat/phase6-remote-render`: phase 13 (operator API, tasks, `/v1/setup` + pairing
  lines, the PAGE at `/ui/`, generated `modules/*.module.json`, SDK operator door + `parsePairing`
  + `ChatOptions.act`), Orpheus removed with a drift guard, bootstrap linger-as-root, phase 14
  host side (conda gone, server pack, own distro, WSL state table, `install.sh`/`install.ps1`
  generated, UTF-16 chunk bug fixed) — 1171 pytest / 235 sdk / 191 bootstrap. Phase 14 SERVER
  side (envpack model, `crucible install` downloads, CI matrix, rootfs job, release.sh assets) is
  the ONE agent still running at this writing; the real `asr` pack build in WSL is its proof.
- BookForge `feat/narrator-guarded-serve`: per-server slot sets; every GPU step travels; one
  lease per row (per MODEL — Foundry's catch); PASS steps travel through one mapping (a 4th and
  5th private provider copy deleted); hosted-Foundry seam (`FoundryHost.servers()`, refusal
  renamed truthfully, tripwire keeper); the 164-row settings audit implemented — 4-step wizard,
  11 settings sections, Orpheus/Higgs/RVC/STT sections deleted, dead controls deleted, XTTS
  remnants gone, cloud keys → Foundry's card, `crucible-models.json` gone; the operator door
  (connect code paste, Open engine console with the hardened window, vendored module + keeper);
  coordinate-on-connect (`electron/crucible/coordinate.ts`, catalog first, no button, no
  consent); SDK pinned to `vendor/crucible-client-0.6.0.tgz` (dated STOPGAP). 149 keepers.
- Foundry (theirs): e096734/990bd2e/33bb187/599d0c8/6a11661 — one registry seam hosted, waitFor
  crosses runJob, cloud editable hosted, two hosted guards, Open Crucible, tarball pinned;
  their settings audit is running. **The single re-vendor targets their sha AFTER their
  coordinate + wizard land** and carries: `RunOptions.waitFor`, deletion of
  `hosted_placement_not_vendored`, `slots?()` removal.

**Live servers:** the PC's WSL Crucible runs the PAGE build (2328562, restarted 16:10 via
`systemctl restart user@1000` as root after a clean SIGTERM left it down — `Restart=on-failure`
defect owed). The WSL clone `/home/<user>/crucible` is BEHIND HEAD; pull + restart once the
env-pack agent lands. The Mac still runs 22eccf0 (pre-page). Both hold nothing.

**Disk:** C: 5 GB → ~505 GB free. Root cause was the Higgs checkpoint screen's fused merges
(Training pc fixed the loop at 12:11). Deleted today: dead conda envs (Windows xtts/xtts_training/
clipforge-bench, WSL orpheus_ft/whisperx-cuda/headline27b), package caches, temp caches, WSL
pip cache, `tts-orpheus` Crucible env; OneDrive Documents+Projects marked online-only;
`$WINDOWS.~BT` = Owen's admin cleanup (takeown was hopeless; cleanmgr). LEGACY envs (63 GB) go
with the local spawn layer after the in-app pass.

**Needs Owen, in order:**
1. **Publish v0.6.0** once the env-pack agent reports: `cd /c/Users/<user>/Projects/crucible &&
   ./scripts/release.sh --branch feat/phase6-remote-render` — its CI run is the first real
   build of the packs and the rootfs; the agent's report says which packs fit a hosted runner.
   Then BookForge and Foundry swap the vendored tarball for the release URL and
   `DRIVEN_INSTALL_AVAILABLE` flips.
2. **The in-app pass on a free card** (Training's ladder is done; the card was idle at 16:00).
3. **Rulings:** which deathstalker merge is canonical (Crucible's HF pull ≠ local
   `ds_v7_930_prod`; §3); Resemble Enhance (job type or delete); a user's own RVC archive;
   `mlx-audio` 0.4.8 pin re-measurement; `Restart=always` for the unit.

## 0a. WHERE IT STANDS AT 06:00, 2026-09-14 — read this first

**Crucible 0.6.0 is prepared, pushed and RUNNING on both machines** (`22eccf0`). Both were
upgraded and restarted at 06:05 and both report nothing resident — the Mac dropped a model a
test had left loaded four hours earlier, which is Owen's unload ruling doing its job.

**What is proven on real hardware, four things:** the first Higgs render through Crucible
(the Mac, 8/8 keeper checks); Foundry's `clean-text` through the chat door (734 blocks,
78.5 s); a model lease held through a whole book with both a second lease and a competing
load refused by name (Foundry's drive, 29 blocks, 99 s); and BookForge's own render seam
reaching another machine (partial — it exposed two defects, both fixed).

**What is built and has never met a card:** every other door. Each has a keeper against a
fake server. Nothing in the app has rendered, cleaned, transcribed, aligned, converted,
denoised or read a page on a GPU.

**What needs Owen, in order:**

1. **Publish v0.6.0** — one line, and it is what makes the client library and the installer
   package installable and both apps' setup screens clickable:
   `cd /c/Users/<user>/Projects/crucible && ./scripts/release.sh --branch feat/phase6-remote-render`
2. **The in-app pass on a free card.** Settings → Crucible Servers, Test each, render a short
   chapter. The first render goes to `local`.
3. **Five rulings**, each blocking a build: narrator into its own repo (a friend cannot
   install the tts env without credentials to a private one); `higgs-default` on cuda-linux;
   zero-shot voices at the load door; the retake ladder's sampling channel; narrator's
   items-in door so a remote alignment can finish.

**Gates at 06:00:** Crucible 1054, client 184, bootstrap 117, narrator 1449, BookForge 142
keepers, tsc clean. Both branches pushed, trees clean, nothing merged.

## 0b. THE COMPLETE MAP OF WHAT IS NOT WORKED OUT — 2026-09-14, morning

Owen asked: *"is there anything else that we havent fully and completely worked out already?
does bookforge correctly add gpu slots for each crucible server its connected to? does it
remove its own local slots if theres a crucible server installed locally?"* Read from the
code, not from memory. Each item names the file that proves it.

**Owen, 2026-09-14 (after reading this map): *"well lets fix it. i did expect orpheus to be in
there. sounds like the job isn't done yet."*** Rulings taken from that, then CORRECTED an hour later — **Owen, 2026-09-14: *"xtts is
completely deprecated and removed as of when narrator was created. orpheus is deprecated too but
hasnt been removed yet. higgs is the frontier."*** So **B1 = Orpheus is DEPRECATED, not built into
Crucible**: it lives only on the legacy local path and is DELETED with that layer after the in-app
pass; the Orpheus wizard step and Settings section go now (audit `docs/SETUP-AND-SETTINGS-AROUND-CRUCIBLE.md`);
Crucible's stray `orpheus` entries (`ttsstream.py` batch width, jobenv's "two tts envs") are
removed so the server does not name an engine it will never serve. Higgs is the one narration engine; **D-registry = hosted Foundry reads BookForge's server
registry** (one owner; Foundry's own registry is standalone-only); A1, A3, A5 and the stale
hosted-env refusal are BUILT, not ruled. Agents: scheduler (BookForge), hosted-Foundry seam
(BookForge), Orpheus (Crucible, after the phase-13 server agent lands), then the page and the
apps' doors.

### A. The scheduler — the two questions, answered NO and NOT YET

- **A1. Per-server GPU slots — BUILT 2026-09-14 (`67139746`).** `RESOURCE_SLOTS` is gone;
  `shared/queue/slot-sets.ts` is the capacity model and the engine composes the set list from
  the routing record onto every snapshot. One `[gpu]` set per ENABLED server (`local`
  included), one for the legacy narrator spawn, and `local-work [cpu][cpu]` for what BookForge
  does itself. A GPU step counts against the venue it RESOLVED to, recorded on the step
  (`QueueStep.venue`); `any` takes the first server whose slot is FREE in rank order; a
  disabled server keeps its set, marked `retiring`, until its occupant lands (§4.3); and
  `thisMachinesCardHeldBy` states the thing the old global number prevented by accident — the
  legacy spawn and a local Crucible are two venues over one card. The bench draws a lane per
  machine and a row waiting for a card is told WHICH card. 26-check keeper
  `tools/test-queue-slot-sets.js`. Two rulings recorded in §3 (the server CPU number, and the
  legacy set's own slot).
- **A2. ~~Local slots are not "removed" when a local Crucible exists — they are RENAMED.~~ SUPERSEDED by 0h: there is no `local` row to rename.** Once
  `local` resolves, this machine's card IS the `local` server's GPU slot; there is no separate
  "BookForge's own GPU" slot to remove. What still existed beside it was the LEGACY SPAWN
  LAYER (WSL narrator, local text engines, local VLM/RVC/align spawns) behind the ONE switch
  `routing.legacyLocalRender`.
  **HALF DONE, 2026-09-15 — `docs/LEGACY-REMOVAL.md`.** The SWITCH and every DECISION behind
  it are deleted: `legacyLocalRender` is out of the record, the wire, the IPC and the Settings
  page (a record still carrying the key is stripped on read and said once, never honoured and
  never refused); `GenerationVenue`, `RunVenue`/`StepVenue`, `TextActVenue` and `VlmVenue` are
  one-member unions; every `legacyLocal:` arm on every Crucible door is gone; Listen is a
  Crucible session with no local backend; Orpheus is retired from the Listen picker as it was
  from narration's. Every act is now "a Crucible server, or a named refusal", and
  `tools/test-no-legacy-venue-doors.js` is what keeps it that way.
  **The SPAWN ITSELF is HELD, deliberately.** `orpheus-worker-pool.ts`, `narrator-spawn.ts`,
  `higgs-spawn.ts` and everything else that builds narrator's environment or argv is the
  SURVIVING RECORD of tuning measured over months — Crucible's `mlx-darwin` arm was found on
  2026-09-15 rendering one chunk at a time because nobody set `NARRATOR_HIGGS3_MLX_BATCH`,
  which BookForge's spawn has set to 64 since 2026-09-05 (7x, measured). Deleting those files
  now would destroy the only copy of knobs Crucible may still be missing. **DELETE once that
  audit says what is missing and it has been carried across** (~250 GB of envs with it).
  **CORRECTED 2026-09-15:** the switch was not the last tenant of the in-app GPU row.
  `generate-sentences` with `method: 'epub-align'` also charges it and did not go away with
  the switch — see **B7**, which is what actually empties the row. With the switch deleted
  that act is the row's ONLY tenant, so the row was RENAMED to say so:
  `LONGFORM_ALIGN_SET` / `local-longform-align` / "the local long-form aligner"
  (`legacySetCharged`/`legacyCharged` became `longformAlignCharged`/`alignerCharged`). An old
  `step.venue` spelling is read into the new set at one door, as a FILE migration. The row is
  NOT deleted: `slotsOf` answers 0 for a set that is not on the bench, so deleting it would
  leave `epub-align` charging nothing and never being launched.
  **AND THE ROW IS NOW CONDITIONAL (same day).** `slotSets` draws it only when something in
  the snapshot would charge it, computed with `slotSetForStep` over the snapshot's own steps
  (`SlotSetFacts.legacyCharged`, `legacySetCharged()`), so with nothing queued the bench is
  exactly Owen's sentence — one GPU slot per registered server plus `local-work [cpu][cpu]`.
  The old argument for drawing it unconditionally (a step charging a set with no slots is
  never launched) was about a step that EXISTS, so the row exists exactly then; the bench and
  the pump read one function, so they cannot disagree about it. `video-assembly` left the list
  the same day — see B7.
- **A6. The bench's cloud lanes — FIXED 2026-09-15 (`8699d71d`).** Owen read the bench and it
  did not match §0d's ruling: nine rows, four of them cloud lanes on engines that can never
  fill one (nothing routes upstream on either server today, and no upstream is configured).
  The lane was drawn unconditionally for every enabled server, and the argument for that was
  about the per-class ROUTE flickering between pumps — kept, and raised one level. A lane is
  now drawn when the ENGINE HAS AN UPSTREAM CONFIGURED AT ALL (`GET /v1/settings`'s
  `upstreams[*].configured`, read at coordination as a fourth read and again out of every
  settings write's own answer, held beside the routes in `electron/crucible/routes.ts`).
  Configuring an upstream is deliberate and rare, so it does not flicker; it is what the lane
  MEANS; and it is implied by any route at all, since the server refuses
  `route_upstream_unconfigured`. An engine that has not been asked, did not answer, or predates
  the settings door is `unknown` and KEEPS its lane — absence of knowledge is not absence of an
  upstream — and `SlotSetFacts.upstreams` refuses a server it was told nothing about by name.
  Bench today: one `[gpu]` per registered Crucible, no cloud lanes, `local-work [cpu][cpu]`,
  and the `local-longform-align [gpu]` row B7 is about ONLY while something charges it (A2).
- **A6b. The record was in memory only, so `unknown` was every launch — FIXED 2026-09-15.**
  Owen read the bench again and it still drew `mac — routed elsewhere · CPU ×2` while `local`'s
  lane was correctly gone. Measured: the Mac's `GET /v1/settings` says
  `upstreams {anthropic: false, openai: false, ollama: false}` with all four routes `local`, so
  the RULE was right and the RECORD was empty — `upstreamsByServer` lived in a module-level
  `Map`, `local` had been coordinated that session and the Mac had not, and `unknown` draws the
  lane. Phantom lanes on every server were the steady state of a fresh launch. Two halves:
  (1) the fact is written to `<userData>/crucible-upstreams.json` on every note and read back at
  start (`loadCrucibleUpstreams`, called from main before the first bench), pruned wherever
  routes are pruned so the two cannot diverge — its own file rather than `crucible-routing.json`
  because that one is the operator's PREFERENCES and is refused when corrupt, while this is a
  fact the app LEARNED and can re-learn by asking; (2) `readUpstreamsOnStart` asks every ENABLED
  server one `GET /v1/settings` at start (`local` excepted — its coordination reads it on the
  way past), which is the read coordination already makes as its fourth, at the one moment
  nothing else was making it. No timer, no poll, nothing posted. `unknown → draw the lane`
  stays the rule and is now momentary; the code says why that is conservatism rather than
  necessity (a RUNNING routed row cannot be stranded — its venue IS the cloud lane and occupied
  sets survive on the bench — and a QUEUED one is computable from the snapshot the way the
  legacy row is). Three checks in `tools/test-crucible-coordinate.js`.
- **A3. Only two step kinds travel.** `machines()` is declared by `tts-conversion.ts` (`any`)
  and `foundry-job.ts` (per config). Every other GPU step — `align.ts`, `rvc-enhancement.ts`,
  `final-denoise.ts`, `vlm-convert.ts`, `generate-sentences.ts` (asr), `translation.ts`,
  `ai-provider.ts`, `book-analysis.ts` — had a Crucible DOOR in `electron/crucible/` that the
  queue step had not been taught to use. **BUILT 2026-09-14 (`61238640`):** all eight declare
  `machines()` and take the run's venue through `runVenueOfRow` instead of deciding again.
  `rvc-enhancement` and `final-denoise` cross-check the row against the session's own record
  and refuse a disagreement by name; `vlm-convert`'s door had no venue-following path at all
  and now takes one; `generate-sentences` travels for `whisper` only; the two AI steps travel
  against `crucible` only, and each bridge gained a Crucible arm through one extracted
  `crucibleChatOnce`. `align` travels and then REFUSES BY NAME before submitting, because a
  remote alignment cannot finish until B5 — see §3. 12-check keeper
  `tools/test-queue-step-travel.js`. Still local: the PASS steps (§3).
- **A4. Double admission on the local card — RULED AND BUILT 2026-09-19.** Owen: *"Crucible
  is configured to be system agnostic. Doesn't matter if it's on this system or on a rented
  DigitalOcean GPU, it should effectively be treated the same locally or otherwise. Like
  Ollama — the user connects to it the same way whether local or remote."* So the queue has
  no notion of a server being here at all: `isLoopbackUrl`, `serversOnThisMachine`,
  `thisMachinesCardHeldBy`, `thisMachineSetId` and `SlotSet.onThisMachine` are deleted, and
  every registered server takes ONE road — the venue's slot, then `reserveBeforeLaunch`, then
  the launch. `external-gpu-job.lock` and the GPU arbiter are asked only for a non-travelling
  GPU step (`LONGFORM_ALIGN_SET`), which is the work this process runs itself. **The two
  consequences, deliberate:** a training chain holding the lock no longer holds back a
  Crucible render on the same box, and `acquireGpuForJob` no longer evicts the resident Ollama
  models before one — Crucible owns its card's memory. The bench draws the nvidia-smi reading
  on the in-app aligner's row alone, and the aligner keeps a lane of its own instead of being
  filed into a loopback server's slots. Keeper: `tools/test-queue-admission.js` §3a.
- **A5. One lease per row — BUILT 2026-09-14 (`52ed21c8`).** The scheduler runs every step
  inside a ROW SCOPE named by the run's id (`AsyncLocalStorage`, injected so the engine keeps
  its no-Electron property), and inside a scope `withCrucibleLease` hands its lease to the
  scope instead of releasing it. `settleStep` gives it back the moment nothing follows, which
  `StepModule.leasesModel` is what answers — so the lease is never held across an assembly or
  an hour of narration. Heartbeat and release semantics unchanged. 12-check keeper
  `tools/test-crucible-row-lease.js`. The act name it could not settle is a ruling in §3:
  Crucible's vocabulary has no name for "a row of acts", so the lease is stamped with the act
  that opened it while every request still names itself truthfully in `X-Crucible-Act`.

### B. Engines — one is missing entirely

- **B1. Orpheus is not in Crucible — and will not be (ruled deprecated 2026-09-14; RETIRED
  from both pickers 2026-09-15, narration in `c167d5ea` and Listen in the legacy-removal
  commit — a saved record still parses and displays as "Orpheus (retired)" and can no longer
  be chosen).** Kept for the record: Every voice manifest is `narrator_engine = "higgs-v3"`;
  `engines/narrator.py` knows Orpheus only as a comment. CLAUDE.md still calls Orpheus "the
  narration engine" and the wizard gives it a step of its own. So an Orpheus render or Listen
  can ONLY take the legacy WSL path, and deleting that layer (A2) deletes Orpheus. Ruling:
  Orpheus manifests + a `narrator_engine = "orpheus"` env in Crucible (vLLM 0.7.3, its own
  venv — the CLI already has one venv per narrator engine), or Orpheus retired in favour of
  Higgs. **RULING, then BUILD or DELETE.**
- **B2. `higgs-default` on cuda-linux** — the token voice exists as a manifest; whether the WSL
  server can serve it is the owed ruling from 06:00. **RULING.**
- **B3. Zero-shot voices — the upload question is SETTLED; the PUBLISHING question is the
  one still open.** Crucible's `zeroshot` takes clips in the request and
  `electron/crucible/voice-load.ts` sends them: `crucibleVoiceLoadFor` reads the wav off
  THIS machine's disk and uploads it base64, under the one server-side id `zeroshot`, with
  `name` saying which of the four is up. That works today.

  **What Owen expected, 2026-09-15:** *"zero shot works effectively identically to fine
  tuned models. it sends it through the base and appends the reference clip that's already
  present on the crucible server."* The first half is exact. The second describes the
  design Crucible is BUILT for and which nobody has finished — and the gap is why these
  four are special-cased everywhere they appear.

  `crucible/voices/zeroshot.toml` takes `clips` as EITHER a published list of
  `{file, transcript, seconds}` (the server holds the bytes — Owen's description) or the
  literal `"from-request"`. It is `"from-request"`, and says why: the four wavs *"are not
  published anywhere: they live in `<userData>/runtime/higgs-models/refs/`"*, and Crucible
  *"pulls a published artifact at a pinned revision and has no other way to get bytes onto
  a server."*

  **What it costs to close, and the manifest already gives the recipe:** join each voice's
  clips into one wav with 0.35 s of silence between, join the transcripts in the same
  order, measure `seconds`, publish the set, and write four manifests with real `clips`
  lists. Then `needsReference` goes false, the four become ordinary server-side voices,
  `placeVoices` picks them up with no change, and `placeCarriedVoices` —
  `electron/crucible/voice-inventory.ts`, written 2026-09-15 for exactly this case — has
  nothing left to place. The local disk stops being an authority about voices at all.

  **Until then the special case is CORRECT and must not be tidied away.** For those four,
  and only those four, this machine's disk is the right thing to ask, because this app is
  the only thing holding the bytes. `tools/test-voice-inventory.js` pins both halves so a
  later cleanup cannot merge two questions that have two answers. **RULING: publish the
  reference set, or keep carrying the clips?**
- **B4. The retake ladder's sampling channel** and the guard belonging to the model (the
  crucible-guard ruling, NOT STARTED). **RULING + BUILD.**
- **B5. Narrator's items-in door** — a remote ALIGN cannot finish without it. **BUILD (narrator).**
- **B6. Narrator into its own repo** — a friend cannot `crucible install tts` against a
  private BookForge sha. **RULING.**
- **B7. `align-longform` — the job type that empties the bench's last in-app GPU row.
  RULED 2026-09-15: BUILD IT.** Owen, asked which act this was: *"align longform, is that the
  generate-sentences logic? that should be a gpu job."* It is — and the identification is
  more precise than the question, because only ONE of that row's two methods is the subject.
  `method: 'whisper'` already travels (it is a Crucible `asr` job, taken through
  `transcribeAtVenue`); `method: 'epub-align'` is the one that spawns here, and
  `generate-sentences.ts`'s `machines()` is already asked of the CONFIG rather than answered
  unconditionally, so the build flips one ternary and deletes nothing else.

  **The whole step travels, CPU stages included** — the same ruling Owen made for TTS
  (*"the entire tts step goes to the other system. That includes anything the step needs to
  do even if it's cpu"*). `transcribe` and `coarse-align` are CPU and are most of the wall
  clock; they go with the job, and the job charges ONE slot, its venue's, for its whole
  duration. Slicing locally to send only the GPU stage is the thing this ruling forbids.

  **What it costs on the wire, since one input crosses.** The m4b as-is, not a wav: a 16 h
  audiobook is roughly 460 MB at 64 kbps, where the 16 kHz mono wav the script makes
  internally would be ~1.84 GB. Let the server do that conversion. One transfer per book,
  over LAN, against a stage that takes ~40 minutes — the ratio is not close.

  **What it empties.** Once both methods travel, `LONGFORM_ALIGN_SET` has no tenant left:
  `video-assembly` stopped being one when it was measured as CPU (below), and the legacy
  render venue is deleted (A2). The bench then draws NO GPU row that is not a registered
  Crucible server, ever — which is Owen's *"without a crucible server, there is no gpu
  slot"* reached completely rather than conditionally. The set and its migration must stay
  readable for queues written before the switch, which is what
  `tools/test-queue-slot-sets.js` pins.

  Raised 2026-09-15 when Owen read the bench: *"without a
  crucible server, there is no gpu slot, because bookforge shouldnt know how to drive gpu work
  in-app."* One step still does, and it is not the legacy narrator: `generate-sentences` with
  `method: 'epub-align'` (`electron/whisperx-align-bridge.ts` → `electron/scripts/align_audiobook.py`).

  **Why it cannot use the `align` job we already have.** That job is
  `{type:"align", model:"qwen3-aligner", params:{language, chunks:[{index,text}]},
  inputs:{"<index>.flac":{blob_id}}}` — one audio input PER chunk, matched by index, i.e. a
  caller who already knows which seconds of audio hold which sentences. True of a render
  (narrator wrote the chunks); it is precisely what this act must DISCOVER. The script's stages
  are `transcribe` (faster-whisper over the whole m4b, CPU env, ~40 min on a long book) →
  `coarse-align` (DTW of the ebook's sentences onto that rough transcript — this is what
  produces the chunk spans) → `align` (Qwen3 per chunk, on the card, 229–395x realtime) →
  whisper-authority gate + monotonic clamps + drift correction + silence snap → `write`. Only
  the third stage has the shape Crucible offers. Slicing locally and sending ~1,000 chunks of a
  16 h book would move the cheap stage and keep the expensive ones. `electron/crucible/align.ts`'s
  header already rules the same way from the other side: *"that bridge keeps its local CPU spawn
  until it is deleted, not moved."*

  **What the job takes** (written before the ruling, unchanged by it — the ruling says build
  this, not build something else):
  - `type: "align-longform"`, `model: "qwen3-aligner"` (the same weights; the difference is the
    orchestration, not the model).
  - `inputs`: ONE — the audiobook, `audio.m4b` (or a 16 kHz mono wav the client already has to
    make). No per-chunk inputs; the server slices. The EPUB never crosses: it is the client's
    book, and the sentences it extracted are text.
  - `params`: `{language, sentences:[{index, text, kind}], rough_model, chunk_s, hole_min_s,
    snap_silence_s, silence_source}` — `kind` because headings are already stamped at extraction
    and the classifier must not re-infer; the rest are the script's own arguments, which the app
    states rather than defaults today.
  - Result: the same VTT the script writes, plus its report — `alignment.vtt` and
    `align-report.json` as artifacts, with per-stage progress events (`transcribe`,
    `coarse-align`, `align`, `write`) so the stacked stage bars keep working.
  - Refusals it must make BY NAME, all of which the script already makes locally: a language
    qwen3 was not trained on (`qwen3_language_unsupported`, refused BEFORE the pool, not after a
    book's worth of bad cues); no sentences (`no_sentences`); a silence scan that produced
    nothing (`silence_map_empty` — the script treats this as a FAILED RUN, because edges placed
    with no silence map are the pre-2026-09-06 build under a new label); and an audio input the
    backend cannot decode.
  - It needs BOTH stacks on the server — `faster_whisper` for the rough stage and `qwen_asr` for
    the align stage — which on this PC are two separate envs today and would be one Crucible
    engine's problem instead of BookForge's.

  **Until it exists, the bench keeps one GPU row that is not a registered server — but only
  while a step of this kind is actually in the queue.** As of 2026-09-15 `epub-align` is its
  ONLY tenant (the legacy render venue is deleted, A2), so the row is named for it:
  `local-longform-align`, "the local long-form aligner". Deleting the row outright would leave
  such a step charging a set with no slots (`slotsOf` answers 0 for a set that is not on the
  bench) and the scheduler would never launch it; so as of 2026-09-15 the row is drawn exactly
  when something charges it, derived from the snapshot with `slotSetForStep` itself
  (`SlotSetFacts.alignerCharged`). With nothing queued there is NO in-app GPU row, which is
  Owen's sentence realised; an `epub-align` queued brings it back with its one card, and
  nothing else does any more. `tools/test-queue-slot-sets.js` pins both halves of that
  shape, and that every GPU row drawn is a registered server's bar this one named exception.

  **`video-assembly` was the third tenant and is not any more — MEASURED 2026-09-15.** It
  declared `resource: 'gpu'` with no comment since before the slot sets existed, so a video mux
  waited for the 3090 Ti and a render waited behind a video mux. Read end to end,
  `electron/video-assembly-bridge.ts` draws PNG frames in an offscreen BrowserWindow
  (`capturePage()`) and muxes them with `ffmpeg -f concat … -c:v libx264 -preset medium -crf 23
  -c:a aac` — a SOFTWARE x264 encode, no NVENC, no `-hwaccel`, no weights, no VRAM held across
  the run. Owen's boundary is MODEL INFERENCE vs DETERMINISTIC work, not GPU vs CPU, so it is
  `cpu` and charges `local-work` beside assembly and muxing. Rows queued before the change are
  re-asked on load (`reviveInterrupted` re-derives `resource` from the module), so nothing
  needs migrating. Pinned in `tools/test-queue-step-travel.js`.

### C. Install and stocking — phase 13, in flight

- **C1. Crucible's own page + operator API + generated module files** — being built now
  (`crucible/docs/PHASE13-OPERATOR.md`). Closes: no in-app model downloads, printed pull lists,
  two install stories, "how do I get the token".
- **C2. The wizard step probes on entry** (local found → connected; none and hostable →
  Install; not hostable → Connect only). §5.5 of the phase doc. **BUILT 2026-09-14
  (`f1139476`), then SUPERSEDED THE SAME DAY by coordination-on-connect** — crucible
  `docs/PHASE14-ENVPACKS.md` §4a, Owen: *"if its present, bookforge should coordinate with the
  installed crucible to make sure it has what it needs to run all of its features."* What that
  changed, and the three rulings inside it:
  - **The "Set up for BookForge" button is DELETED.** Presence of the app is the request.
    `electron/crucible/coordinate.ts` is the one owner and is called from four moments: app
    start for `local`, a server added, a server switched back ON, and the wizard's step landing
    on connected. A driven install is routed through the same function rather than having its
    own post. Never twice concurrently for one server; a task already running is FOLLOWED, not
    re-posted.
  - **ASK, THEN ACT** (crucible `cecfdd0`, from Foundry's review). Coordination READS
    `GET /v1/info` + `GET /v1/catalog` and compares the vendored module against them. Nothing
    missing = a read and NO POST. Posting an idempotent module on every connect would have been
    correct and still wrong: one task at a time means two apps arriving together collide on
    `task_busy`, and a book already rendering refuses its own app `server_busy` over a task
    whose every entry would have been `skipped`.
  - **NO CONSENT STEP — Owen's ruling, 2026-09-14** (*"lets make it as simple as possible"*,
    crucible `1a10cc8`), **not a default.** Coordination is automatic on EVERY connected
    server, `local` and every remote, however long ago it was registered. Foundry proposed a
    one-press consent on a remote registered in an earlier session; it was heard and overruled.
    Stated once so it is a choice: opening BookForge on a laptop connected to the Mac downloads
    onto the Mac whatever BookForge needs there and is not. **Disabling a server in Settings is
    the one way to say "not that one"** — a disabled server is asked nothing at all.
  - `server_busy` is a WAIT with the holder shown verbatim, retried when the server's own
    `slots.accelerated.accepts_work` says the card is free (20 s apart, 90 asks, then it stops
    with the holder still named and the next connect starts it again). A refusal about the
    REQUEST (`invalid_module`, `unknown_subject`) fails ONCE by name and is remembered for the
    session. 20-check keeper `tools/test-crucible-coordinate.js`.
- **C3. The wizard's Orpheus / Higgs / RVC / Tools steps still install LOCAL engines** through
  the component manager. They become "which server" rows against the catalog once A2's layer
  is deleted — and B1 decides whether the Orpheus step survives at all.
- **C4. The driven install is gated on a published release** (`DRIVEN_INSTALL_AVAILABLE=false`).
  Order: phase 13 lands → Owen publishes 0.6.0 → BookForge pins the tarball → flip.
- **C5. Prebuilt env packs on the release** — speed only; after C1.

### D. Foundry seam

- **D1. Hosted Foundry's text acts — WORKED OUT, and what is left is a RE-VENDOR.** *(done
  2026-09-14; BookForge `e284c7bb`, `b2c50d78`, `70d3b896`.)* The refusal's premise had gone stale: foundry `f300fc6` gave
  the vendored `runEngine` an `extraEnv` argument and the 2026-09-14 re-vendor brought it in at
  `e6d5424`, while the guard went on quoting `env: process.env` for ten hours. Reading the
  subtree rather than the comment found the REAL gap, one layer along: the seam BookForge calls
  is `runJob(request, {parentStep, signal, onProgress})`, which carries no environment, and the
  only thing that fills `extraEnv` is the vendored DISPATCHER's own placement
  (`crucible-dispatch.ts`) — which hosted resolves credentials from a registry that is always
  empty. So the act cannot be composed here and could not be composed there. **Foundry fixed
  their half at `e096734`** (below); BookForge's half is built. What ships when `foundry-app/`
  is re-vendored at or past that commit: the refusal, the reach value and the venue resolution
  in `queue-steps/foundry-job.ts` all go, and the act's endpoint, model, header map, residency
  and **lease** become the dispatcher's — a BookForge lease on that path would be refused as a
  second lease (crucible allows one per server), which is the opposite of what the old note in
  that file said. The refusal is now keyed to the VENDORED SUBTREE and read by
  `tools/test-foundry-hosted-crucible-seam.js`, whose tripwire goes red ON the re-vendor with
  the instructions on it; `FOUNDRY_VERSION_FOR_CRUCIBLE_TEXT` is deleted, because a version
  number standing in for a line of somebody else's code is how the stale guard survived.
- **D1-done. THE RE-VENDOR HAPPENED AND HOSTED TEXT ACTS RUN** *(2026-09-15, on Owen's ruling:
  "text acts from foundry through crucible should work, vendored or host … i.e. vllm")*. The
  bundling this plan assumed — *"their sha AFTER L"* — was **not reachable**: Foundry's package L
  (the PHASE15 §5.3 deletions) has not landed and is gated on a `llama-windows` server on a clean
  box, while translate and simplify did not run as queue rows at all. So the re-vendor was taken
  at the EARLIER target instead — **`1ce539a`**, main's tip, on Owen's ruling *"the latest, so we
  can vendor a fully functional and up to date version in"*. (`e096734` alone is not enough: the
  placement also needs `RunOptions.waitFor` to cross `runJob`, so `990bd2e` was the minimum and
  `ecd03e3` the first route-aware one. Every assertion BookForge's keepers make holds at all of
  them; the tip was taken for the two things below.) `npm install --no-audit --no-fund` +
  `npm rebuild electron` + `npm run build` ran in the subtree (`@crucible/client` 0.5.0 → 0.6.0);
  the `file:..` self-link junction was deleted non-recursively and the main `node_modules`
  verified intact either side. What shipped: `hostedCrucibleTextActNotVendored`
  and the `hosted_placement_not_vendored` code deleted; the `none` member of
  `EndpointHeaderReach` deleted with them (a caller that makes no spawn now has no value it could
  pass); `FoundryRunJobOptions.waitFor: string | null` required, `null` = *this kind does not
  travel*, translated to an absent key at the mount; `queue-steps/foundry-job.ts` composing
  nothing — no engine, no model, no endpoint, no lease, no local vLLM profile — and handing the
  request across verbatim. The ONE check this side keeps is the NAME, against the same snapshot
  it hands the window: their `slotNamed` matches exactly, and a miss is a `wait` a detached
  `runJob` retries for ever, so it is refused here (`hostedCrucibleServerNotOffered`). **Second
  effect:** a placement declares `door: 'openai'`, whose pool default is 12 against Ollama's 4 —
  before this, every hosted act fell through UNPLACED to the Ollama door, so placing them is also
  what gives them the pool vLLM batches. The keeper's FIRST tripwire is inverted (the hosted
  registry read is now the load-bearing property); the SECOND, on the cloud layer, is untouched
  and still fires on the L re-vendor.
- **D1-hop. AN ORCHESTRATOR IS NOT AN ENGINE, and hosted it is OUR entry being resolved**
  *(foundry `1ce539a`, PHASE17 §6)*. One resolver, `resolveEngine(entry)`: `info()` once, the
  SDK's `engineOf` for the rule, and for an orchestrator a client at `engine.url` with the SAME
  token — after reading THAT document and refusing `orchestrator_engine_is_not_an_engine` if its
  role is anything but engine, so *once, never a chain* is enforced rather than assumed. Cached
  on name+url and **never on the token**, 60 s, in-flight deduped, forgotten on a registry change
  BEFORE the capability cache. `placeOnCrucible` resolves once, so the request and the spawn
  cannot end at two different processes. **This is BookForge's exposure more than Foundry's**: the
  entry resolved hosted is the one `crucible/host-registry.ts` composed, so a person who registers
  the tray rather than the engine has every hosted text act depending on that hop — and a token
  rotated on THIS side must reach the window, which is why the cache key excluding the token is
  pinned. Foundry proved fifteen checks against the real `:7100`/`:7101`; the HOSTED case does not
  exist until the vendor, so `tools/test-foundry-hosted-crucible-seam.js` drives their compiled
  resolver and the real SDK over a stubbed `fetch` with rows of our shape — engine, orchestrator,
  chain, no-engine, rotated token — and pins that no refusal sentence carries the bearer.
  **Mirrored, not contradicted:** `crucible:open` and `adoptPairingFile` deliberately do NOT
  follow the hop (a placement asks a machine to work; a console is a person going to look at the
  process they named, and the orchestrator's console is the one with install/restart/quit).
- **D1-standing. THE PARKED-ROW DEFECT IS FIXED, and Foundry agreed it was theirs** *(foundry
  `1ce539a`)*. BookForge's hosted step hands work to a DETACHED `runJob`, which holds no pump slot,
  so `placeRun`'s `for(;;)` retried a wait for ever and the promise never settled. The wait arm now
  carries `standing`, **required at all sixteen sites via explicit constructors rather than an
  optional flag**, so *"nobody thought about it"* and *"this is transient"* cannot be the same
  value; a pinned slot answering with a standing wait REFUSES, carrying the server's reason and
  where to fix it. Standing: a disabled or empty capability row, `upstream_unconfigured`, both
  orchestrator refusals, a cloud slot stepped past by `any`. Transient: busy, engine in use, model
  leased, model not resident, unreachable, capability undecided, network refusals, a cancelled
  load, a slot switched off. A slot MISSING from the list stays transient by their deliberate
  choice — right for a row in their pump, fatal for a detached `runJob` — so BookForge's preflight
  now covers exactly that one case and nothing else.
- **D1-v2. RE-VENDORED AGAIN AT `40aaa42` (foundry 2.0.0), same day.** Ten files, no deletions.
  The binary is rebuilt clean from that tree, so the app sha and the engine sha AGREE again and
  `dist/foundry-windows-x64.exe` answers `foundry 2.0.0 (40aaa42)` — all hex in the parentheses,
  which is what `tools/test-foundry-clean-text-vendor.js`'s tier-2 regex needs to find a commit and
  verify instead of refusing. It carries `registerPairing()` as the ONE writer both pairing doors
  share, the connect-code door finally COORDINATING (it was the one road into their registry that
  skipped the moment `crucible:add` beside it took), and slot-name rules under one owner with the
  silent `.slice(0, 60)` deleted. **Three things were read rather than assumed**, because the
  commit is marked BREAKING and its subject is a reserved NAME while BookForge hands that window an
  entry literally called `local`: (1) `local` is still legal — the retired reserved name was the
  PAIRING door's, and `LOCAL_SLOT_NAME = 'This computer'` is their own GPU-slot label; (2) the new
  `tidySlotName` is NOT applied to host-supplied entries (`cleanHostServers` still uses a bare
  `.trim()`), which matters because it collapses whitespace runs — had it run there, a server named
  `my␣␣mac` would become `my␣mac` over there, our preflight would pass and their `slotNamed` would
  miss, which is the exact forever-park the preflight exists to prevent; (3) `slotNamed` is still
  `slot.name === name`, exact and case-sensitive, which is what our refusal sentence claims.
  **Nothing double-coordinates:** hosted, their registry is the host's and read-only, so neither
  write door can add a server at all; the startup sweep does run hosted by their explicit design,
  with each app posting its OWN module — *"the union of the two modules on one server is the
  contract"* — and only the drawing suppressed.
- **D1-cautions.** Three come with the tip and **Owen has accepted all three**: `CRUCIBLE_READS` is
  true (this seam sends `waitFor: null` for a `read`, so their own default decides and nothing
  changes); the old cloud card sits beside the engine settings card until a later package; the
  engine settings card is not hosted-gated.
- **D-registry. HOSTED FOUNDRY READS BOOKFORGE'S SERVER REGISTRY** (Owen's ruling, 2026-09-14 —
  one owner; Foundry's own registry is standalone-only). *Both halves built, neither wired:
  BookForge `b2c50d78`, foundry `e096734`.* Their `crucibleServers()` asks `FoundryHost.servers()`
  hosted and DERIVES the slot list from it, which closes a break nobody had pressed yet —
  `computeSlots()` took the host's `slots?()` list while `placeOnSlot` looked the credential up
  in a settings file that is empty hosted, so a row pinned to "mac" parked for ever on *"no
  longer registered"*. Ours: `electron/crucible/host-registry.ts`, offered at the mount in
  `main.ts`. Synchronous, because theirs is read while the queue page paints, so it answers from
  a SNAPSHOT taken at named moments (app start, every write to the registry or the rank record,
  and the Servers panel's read, which is where `local` is re-checked) — resolving `local` is a
  `wsl.exe` spawn and one per paint is a stuttering window. A call before the first reading is
  refused by name (`registry_snapshot_not_taken`) rather than answered with an empty list a
  window cannot tell from *"this machine has no servers"*. Priority order, disabled entries kept
  and marked, `local` present exactly when it resolves, a name that will not resolve omitted and
  RECORDED. We never offered `slots?()`, so there is nothing to delete on this side; their
  deletion of it lands with the re-vendor. **Owed: the re-vendor itself** (not built here), and
  the seam gap it exposes — see §3.
- **D2. Cloud slots** for translate/simplify on an underpowered machine — Foundry owns them
  (Owen's 01:40 ruling). BookForge's own AI providers already list models by key
  (`ai-bridge.ts:2148`, `2278`); the TILE rule (local floor OR an enabled cloud slot) is
  Foundry's. **Foundry, then re-vendor.**
- **D3. Lineup rulings:** `qwen3.8:27b-24g` is a local Modelfile not a published tag; the
  dots.ocr pin (Q8_0 vs F16) — one must move. **RULING.**
- **D4. `"foundry": "file:.."` devDependency** makes `npm ci` junction the subtree to the repo
  root. **RULING (Foundry drops it for the snapshot?).**

### E. Proven on a card: four things. Everything else: a keeper against a fake server.

Render (Mac), Foundry clean-text, a lease through a book, the render seam across machines.
Never met a card: cleanup, asr, align, rvc, denoise, pages, streaming, re-roll, every wizard
door. **Owen's in-app pass** is what turns the rest from "built" into "works".

### F. Small, known, owed

The `service install` over a loaded launchd agent (exit 5); a doctor verb comparing the running
version to the checkout (the stale-install 500s); the cold-VM `wsl.exe` exit −1 wanting a
reproduction; the ~34 remaining log-line contracts (R4); `orpheus-memory.ts` tier table; the
Mac's conda root not being one of the three ruled roots; `enableLinger` — what the app does
with it; `dots-ocr` `temperature = 0` server-side?; resume of a render across an app restart
(job id not persisted); re-pointing an assigned book is refused.

## 0. Where it actually stands tonight

**Crucible** (the server) is finished as a codebase: every job type exists, is tested, and
is on the wire as v0.5.0. But on the two machines that matter it is **deployed as an LLM
server only**:

| machine | version running | job types enabled | envs installed | models on disk |
|---|---|---|---|---|
| PC (WSL2, `crucible@example-pc-wsl`) | 0.5.0, **not running** tonight | echo, llm | `llm` only | qwen3.5-9b, qwen3.8-27b-4bit |
| Mac (`crucible@example-mac`) | **0.4.0**, running since 16:38 from the `main` checkout, started by hand | echo, llm | `llm` only | qwen3.5-9b, qwen3.8-27b-4bit |

No `tts`, `asr`, `align` or `rvc` env has ever been installed anywhere. No voice, whisper,
aligner or RVC weights have been pulled. Neither server is a service; both were started
from a terminal.

**BookForge** (the app) has **no Crucible integration reachable from its UI.** Audited
tonight, every GPU feature the app exposes still spawns Python itself:

| feature | today | Crucible-side ready? |
|---|---|---|
| audiobook render (narration modal) | WSL narrator spawn in `parallel-tts-bridge.ts` | yes: `tts` job, guard verdicts on the wire, `render-artifacts.ts` downloader built and tested, **nothing calls it** |
| Play tab / extension / Reader streaming | local `stream-scheduler.ts` → narrator | yes: streaming session (`/v1/tts/stream`) |
| correct-sentences re-roll | local | yes: `tts` render job |
| AI cleanup / edit-list | Ollama / llama-server; a `crucible` provider exists in `ai-bridge.ts` but the renderer's enum lacks it, so **Settings cannot select it** | yes: chat door |
| simplify / translate / analysis (vendored Foundry) | Foundry → Ollama | yes: chat door with `X-Crucible-Act` |
| VLM page reading | WSL conda spawn | yes: `pages` |
| ASR "Generate sentences" | local whisper | yes: `asr` |
| forced alignment | local qwen3 aligner / whisperx | yes: `align` |
| RVC enhancement | local | yes: `rvc` |
| denoise | local audio-separator | **no**: `denoise` job type not built |

The registry (`<userData>/crucible-servers.json`) holds `mac` and `wsl`; the `wsl` entry
is a copy of the local server's own token, which is the duplication defect (section 2.1).

**Foundry** has its Crucible contract built (PHASE7-LANES.md section 8.0) and four items
of its own outstanding: Ollama retirement, the `capFor` 128-token floor, the `translate:`
log prefix on 43 lines, and the stamp decision.

## 1. What "fully functional" means, and what one night can deliver

Fully functional = *every GPU feature in every app goes through one Crucible per machine,
the old spawn layers are deleted, and Owen has run each feature in-app.* That is a
multi-day migration with an in-app pass between steps, and nothing tonight changes that.
The GPU is owned by fine-tuning all night, so **nothing that loads a model can be verified
before morning.**

What tonight delivers, in three tiers. Everything in tier 1 and 2 is committed on its
branch with tests; nothing is merged, because Owen tests in-app first.

### Tier 1 — Crucible becomes the full server on both machines (no GPU needed)
- [x] PC: enable `tts`, `asr`, `align`, `rvc` in config.toml (add the keys; never `init --force`) *19:45*
- [x] PC: `crucible install tts --narrator-engine higgs-v3`, `… orpheus`, `install asr`, `install align`, `install rvc` *done 20:17; asr's recipe had pinned numpy 2.5.3 (Python ≥3.12 only) — pinned 2.4.6, and all four chosen-set recipes were regenerated from the real installs' `pip freeze` (crucible `1ec936d`)*
- [x] PC: re-apply the two site-packages patches to the Higgs env; `crucible doctor` reports both `applied` *20:01*
- [x] PC: pull the weights BookForge uses (all pulled by 20:10): voices `deathstalker`, `mistborn`, `owen`, `sigma`, `thirdreich`, `zeroshot`, `higgs-default`; `faster-whisper-large-v3`; `qwen3-aligner`; rvc `sigma`, `deathstalker-rvc-v3`
- [x] PC: `crucible capability --write`; `crucible doctor` — every env ready, both patches applied, every type ready except `rvc` (base assets, see below) *20:25*
- [x] Mac: checkout `feat/phase6-remote-render` (v0.5.0), `pip install -e .`, same enables, installs and pulls *done 19:55; asr/align have no mlx-darwin recipe and `capability --write` recorded them off, correctly; doctor: `tts` NOT READY only because ffmpeg is off the non-login PATH, `rvc` NOT READY for the base assets*
- [x] Mac: restart the server *done 22:15 once Owen's stream ended and nothing was resident: checkout at the pushed branch, `pip install -e .`, `host = "0.0.0.0"` in its config, hand-started process stopped by Owen, `crucible service install` → launchd agent running v0.5.0, reachable from the PC; rvc base assets pulled; rvc env rebuilt with the separator pin; doctor: tts + rvc ready, `denoise` NOT READY until a pull verb exists — then `crucible denoise pull` landed (`83e3b86`) and both machines report `job denoise: ready`, installed `denoise-roformer` (0.91 GB, digests verified; one stamp per model; the puller and the job read one layout function). Crucible: 877 tests green.*
- [x] `crucible service install` — a systemd user unit in WSL, a launchd agent on the Mac (PHASE5-APPS.md section 6.0 ruled a local Crucible is a *service*). *Built (crucible `a838101`); installed on the PC 20:30 — the unit crash-looped because `python -m crucible` from systemd's $HOME cwd resolved `crucible.voices` to the checkout's `voices/` manifest directory; hand-fixed, then the generator fix landed (crucible `b40de27`: the unit runs the console script in `WorkingDirectory=<CRUCIBLE_HOME>`) and `service install` was re-run on the PC at 22:35. **The WSL server is up as a service now** (`{"crucible":true,"name":"crucible@example-pc-wsl"}`), no model resident, no VRAM taken. Needs Owen: `sudo loginctl enable-linger <your-wsl-user>` so it survives the last shell. Mac: at the 2 AM wake.*
- [x] Owed to Foundry: per-model sampling and thinking defaults in manifests, applied server-side; then tell Foundry-pc-1. *Done 22:40 (crucible `835628e`, PHASE2-LLM.md §9): request states it → wins; else manifest; else engine; `X-Crucible-Sampling` header names the source per key; `qwen3.5-9b` ships `thinking = false`. Foundry told. Ruling owed: should `dots-ocr` state `temperature = 0` server-side?*
- [x] `denoise` job type (shares the RVC env; `audio-separator` pinned by name) *done (crucible `380e9d2`: roformer pass in the rvc env, `audio-separator==0.31.1` compatible-not-resolved until the first real install; `enable_denoise` flag; capability class). Also `5593320`: the urvc base assets are pulled from HF by pinned digest (`crucible rvc pull-base`) — the upstream is the installed fork's own `JackismyShephard/ultimate-rvc`, not the ancestral repos, and the embedder's `config.json` joined the list (without it transformers will not load the directory). Crucible: 859 tests green.*

### Tier 2 — BookForge gets its doors (built + tested, unverified on a card)
- [x] ~~**2.1 Local discovery, registry holds remotes only**~~ **SUPERSEDED by 0h (2026-09-15)** — the reserved name and the loopback refusal are deleted. As delivered on 2026-09-13 (the fix Owen said go on): `electron/crucible/local.ts` reads the local server's own `config.toml` (`$CRUCIBLE_HOME`, on Windows through `wsl.exe -d <distro> --exec`); the reserved server name `local` resolves to it; `addServer` refuses loopback URLs by name; the stale `wsl` entry is refused at use with the fix in the message. *Done 20:05 — 20-check keeper `tools/test-crucible-servers.js`; live: `--list` shows `local` read through WSL, `--server wsl` refused as stale, `--server mac` healthy; the stale `wsl` entry was removed from the real registry with the CLI's own repair door.*
- [x] **2.2 Servers settings row** (PHASE5-APPS §2, PHASE7-LANES §4.2.2): the local server first, then remotes; add/remove remote with **Test** (ping, then info); drag to rank; enable switch per server; **New jobs wait for: top-ranked / Any**. IPC + preload + renderer. *Done 20:55 (`f6ef8d76`, `ea39bc58`): **Settings → Crucible Servers**; routing record `<userData>/crucible-routing.json` (`electron/crucible/routing.ts`, 19-check keeper); live read-only probe of the Mac OK; Load/Unload buttons wired and never pressed. Open: the first `wsl.exe` read against a cold VM returned exit −1 and the second succeeded — the card has a Re-check button, and the root cause is owed a reproduction.*
- [x] **2.3 The `crucible` AI provider reachable from Settings**: renderer enum gains `crucible`, the AI setup picks a server (from 2.2's list) and a *resident* model; `checkProviderConnection` finally receives its server parameter. *Done 21:00 (`3a98ff70`): **Settings → AI → Crucible** card; a queue row cannot yet name a server (that is 2.5's `waitFor`), so that door refuses `crucible` by name.*
- [x] **2.4 Audiobook render through Crucible** *(built 21:25, `944d1f02`: `electron/crucible/render.ts`, one `tts` job per book with `take: 0`, voices checked against `/v1/voices` before the POST, 26-check keeper against a fake Crucible; three rulings owed in the file: zero-shot voices, whether a remote render holds this machine's GPU lease, resume across an app restart. Follow-up done 21:50, `dbc5633b`: `decideWhereGenerationRuns` in `electron/crucible/generation-venue.ts` — caller's name wins, else the legacy switch, else the top-ranked server (not pinged: a named machine is an instruction), else for `any` the first enabled server whose ping answers; refusals `no_enabled_server` / `no_reachable_server`, never a silent drop to local. The resolved venue is written onto the run's saved state so Continue goes back to the same machine. On this PC the default venue is `local`, the WSL service. The legacy switch is in Settings → Crucible Servers: "Render audiobooks with the local narrator instead (legacy — removed after the in-app pass)".)*: the narration modal's generation step submits a `tts` job (chunks up front, so there is a percentage), streams `chunk` events into the guard ledger, downloads artifacts into `sentencesDir` with the existing downloader, and assembly runs locally as it does today. The WSL narrator spawn stays **until Owen's in-app pass**, then is deleted in a commit he approves — a dated stopgap, not a fallback: the app takes the Crucible path whenever the selected server is reachable and refuses by name when it is not.
- [x] **2.5 Per-row `waitFor`** on queue items (PHASE7 §4.2.1) — the default written from the 2.2 setting; disabling a server surfaces the rows that name it. *Done 23:10 (`e91d2a41`): `shared/queue/wait-for.ts` holds the one pure decision and every hold sentence; `waitFor` on the run (one book = one GPU), `waitForResolved` written once at first admission; a row composed before this build carries NO instruction and holds until you pick (reported once at load — neither `local` nor `any` was manufactured); `409 server_busy` holds the row with the holder's name; queue page has the per-book picker; the Servers panel surfaces "N queued books are waiting for X, which is now disabled" with Change them to Any. 30-check keeper. Rulings owed: the GPU slot is still one global number (§2.4 per-server slots not built); only the render travels — RVC/align still run here; re-pointing an assigned book is refused.*

- [x] **2.6 Hosted Foundry's text acts through Crucible** (found 22:30 when Owen asked): translate / simplify / analysis in the hosted window and the narration clean-text pass still spawn the engine at BookForge's own text server. The engine gets `--endpoint <server>/v1/openai`, the header map (`FOUNDRY_ENDPOINT_HEADERS`) with the act named truthfully in `X-Crucible-Act`, and a per-act Crucible model id from Settings; residency checked by name before the spawn; text steps become travelling queue steps; the ONE legacy switch covers the local text engines too. *Built (`0088a296`, corrected `23700dc6`): `electron/crucible/text-acts.ts` (four acts, header map, base `<server>/openai`), `text-models.ts` (`<userData>/crucible-models.json`, one Crucible model id per act, picked in Settings → AI → Crucible), `text-venue.ts` (`decideWhereTextActRuns` + residency by name + `crucible_server_busy` → queue hold); BookForge's OWN driver path (narration clean-text, the CLI clean routes) runs a Crucible text act for real; the hosted job-queue path refuses `hosted_engine_takes_no_per_run_env` until Foundry's vendored `engine.ts` takes a per-run environment — Foundry's to build (they are building the reframe now). 31-check keeper. Ruling owed: what the hosted floor keys on once Foundry's next release exists.*

### Tier 3 — needs Owen, a free card, or days
- Owen's in-app pass on 2.2–2.4 with a free card; then delete the local narrator/llama/Ollama spawn layers.
- **After that pass — one copy per model per machine (Owen, 23:10).** Measured on the PC: Crucible holds the only copies that stay (`~/.crucible/{models 46G, voices 57G, envs 41G}`); to delete once the legacy paths are gone: WSL `~/higgs_v3_merged` (64G, the same merges Crucible mirrors), conda `higgs3`/`sglomni`/`orpheus_tts` (30G), the WSL HF cache (34G), the Ollama library (~120G: 9B ×3 quants, 27B ×2, cogito ×3, qwen3:32b), Foundry's own vLLM dots copy under `%LOCALAPPDATA%oundry` (its llama-server GGUFs stay for standalone). Formats cannot be shared across engines, so one copy = one engine per model per machine — the slot rule (a local Crucible replaces the local Ollama slot) is what makes it hold.
- [x] Streaming (Play tab, extension, Reader) through `stream-scheduler.ts` → Crucible streaming session — one seam, three consumers. *Built 00:45 (`f079b456`, `323803fb`): `electron/crucible/stream.ts` — a `StreamingEngine` whose worker is a Crucible session behind the same `getActiveEngine()` the scheduler drives; venue from `decideWhereGenerationRuns`; the idle sweep lifted into one `IdleWatch` owner; the streaming door carries NO guard verdict (Owen's ruling: streaming stays unguarded) so the ledger records `stream-unguarded`; 22-check keeper. Labelled stopgap: the SDK's session hides `ready`, so the first `say` waits out `stream_not_attached` — root fix owed in `sdk/ts/src/stream.ts`. Rulings owed: may a Listen make a voice resident (no BookForge door loads a VOICE yet); pack Listen rows to the venue's pace band; Orpheus Listen stays local until Orpheus manifests exist; where a Listen ledger persists. First Listen on this PC refuses `voice_not_resident` until a voice is loaded on the WSL Crucible.*
- [x] ASR and align → their Crucible job types *(01:00, `8c257cf1` the shared `runCrucibleJob` helper every later door uses, `838bf0f8` asr, `c1ba62ed` align, `48d10539` the rule that a run's later GPU steps follow the run's venue instead of deciding again — found live when an align step took the fine-tune's card). Align cannot FINISH remotely until narrator gains an items-in door (`narrator align --alignment <alignment.json>`): today a Crucible-venue alignment lands the model's items and then fails by name, and the legacy switch spawns the local narrator as before.*
- [x] Correct-sentences re-roll, RVC, denoise → their Crucible job types *(01:25: `9f24a629` the voice conversion, `ae22b855` the hiss pass, `6693c3ef` a sentence re-roll — each follows the book to the machine that rendered it, via `venueForRunStep`)*. VLM pages done *(01:55, `1e63e971`)*: the door that moved is the ENDPOINT, not a job submit — Crucible has no `pages` job type (PHASE3-VLM: pictures through the `llm` proxy; `pages` is a capability CLASS), and BookForge never holds a page image, so `vlm-convert.ts` now hands Foundry's engine a Crucible base + the header map with `X-Crucible-Act: pages`, and `vlm-page-server.ts`'s spawn half is labelled DATED. **Trap found:** Foundry's pages door does NOT normalise its endpoint while its text door does, so the bases differ by design — text `<server>/openai`, pages `<server>/openai/v1`; a keeper re-reads both rules from Foundry's source and they have been told. Refuses on this machine until a Foundry release carries the header support (the pinned 1.2.0 binary does not read it and reports the same version as one that does — the floor is conservative on purpose); the legacy switch keeps conversions working. Owed: `vlm:reader-status` still tells the renderer "this machine's GPU (WSL)" for a run about to go to a Crucible — a small `main.ts` + renderer follow-up. Deleting each spawn layer waits for Owen's in-app pass.
- [x] **The IPC collision** *(02:10, `bbac878a`)*: Foundry registered both `crucible:add` and `crucible:test`; two handlers of one name in one Electron process throw at startup, so the app would not have booted. Ours renamed — `crucible:add-server`, `crucible:test-server` — in `main.ts` and `preload.ts` only, since the renderer spells preload METHOD names and never a channel string (proved by grepping the built renderer). All sixteen of ours checked against their regenerated doc.
- The ~34 remaining log-line contracts (R4); next two: `isOomError` (`parallel-tts-bridge.ts:4773`) and the `.m4b` path extraction (`:6156`, `reassembly-bridge.ts:2100`).
- `orpheus-memory.ts` tier table deletion; narrator's verdict channel for the local driver.
- Crucible: `crucible service install` over an already-loaded launchd agent fails `launchctl bootstrap … exited 5` (already loaded) — install must bootout first (idempotency, PHASE11); found 23:50 on the Mac while re-recording the PATH.
- [x] Crucible SDK follow-ups *(01:40): `bcaa541` — `crucible.stream()` now RESOLVES ATTACHED (it reads the server's `ready` frame and checks voice/fingerprint/sample-rate/backend before returning), so BookForge's labelled ≤5 s `stream_not_attached` poll in `electron/crucible/stream.ts` can be deleted the day the client pin moves to a release carrying it; `f788faf` — `client.capability()` typed, with `capability_undecided` as a typed refusal; `3cd9d47` — `installed` on every capability row of `/v1/info` (align, asr, rvc, denoise, llm, tts) + the SDK type, and `describe_voices` stopped reading `residency._config` sideways. Crucible 1016 tests, SDK 182.*
- [x] `@crucible/bootstrap` (install Crucible from BookForge's setup wizard; client mints the token and passes `crucible init --token`). *Built 00:50 (crucible `3550763` `init --token`, `ab6d7bd` the package, `6b15c6d` release.sh fourth asset): `sdk/bootstrap/` — `detectHost / install / ensureRunning / readLocalConfig / health`, injectable runner, the WSL plumbing facts from Foundry carried with tests, 117 unit tests; live on this PC: WSL Ubuntu v2, the 3090 Ti, the conda interpreter, config read, service running with linger on. Rulings owed (PHASE12 §6): the Mac's conda root is `/opt/homebrew/Caskroom/miniconda/base` (not one of the three ruled roots — `condaRoots` override until ruled); may `install()` create the `crucible` env; prebuilt env archives need a catalog + downloader; what the app does with `enableLinger`. *Setup screens: BUILT in both apps (BookForge `eb7b4b73`, 02:30). BookForge's three doors live at
  **Settings → Crucible Servers → "Get a Crucible"** and as a first-run step, "Where the GPU work
  happens": connect to one elsewhere, use this machine's, or install one here — the third disabled
  with its reason stated, since no release carries the tarball. The install PLAN is drawn from facts
  BookForge can check itself (WSL2, the guest's card, an existing config) and lists the two commands
  the host must run. 41-check keeper. Also fixed there: the page-reader status cards said "this
  machine's GPU (WSL)" for a run about to go to a Crucible. Still owed: a Crucible release.*
- Foundry: its four items, Ollama retirement, and consuming `local`/registry the same way (Bun parses TOML natively).
- Foundry v1.3.0 (`83d7b66`) — tarballs built, `gh release create` needs Owen; the dev checkout's `dist/` already runs `83d7b66`, and until the release is cut the managed `foundry-cli` component stays on v1.2.0 (`eb69b7a`), whose engine still speaks `--server`. `foundry-cli-components.ts` pins nothing and takes the newest release, so cutting it is the whole fix.
- [x] Re-vendor `foundry-app/` *(02:10, `27da6a21`: Foundry `e6d5424`, 21 commits, 158/158 blobs hash-verified; app and engine shas agree for the first time)*. Three findings: **`--server` came back** as a wire dialect (`openai|ollama|anthropic`) after being retired as a server KIND, so a keeper guarding "nothing may write `--server`" was right to go red and now asserts the declared dialect instead; **`app/package.json` carries `"foundry": "file:.."`**, which made `npm ci` create a junction from the subtree to the BookForge repository root — a later recursive delete of `foundry-app/node_modules` would have deleted the repo (deleted non-recursively, repo verified intact, the step is now mandatory in VENDORED.md, and a ruling is owed on whether Foundry drops that devDependency for the snapshot); and the clean-text pin moved by exactly one docblock word (`both doors` → `every door`, the same nine characters), which is a port, not a rule move, so `NORMALIZER_VERSION` was right to stay.
- Merging `feat/narrator-guarded-serve` and `feat/phase6-remote-render` — after the pass.

## 2. Rulings taken tonight (defaults Owen can overturn in the morning)

1. **One Crucible per machine, and it is the shared pool.** Both apps are clients of the
   same server; job classes have independent envs under `~/.crucible/envs/`; weights live
   once under `~/.crucible/{models,voices,rvc}/<id>/<backend>/`. Two copies of the 9B only
   ever happen if two servers exist, so the rule is that none does. Existing weights on
   disk are **not adopted**: Ollama's GGUF blobs cannot be served by vLLM or MLX, and the
   PC and Mac need different formats of the same model anyway (AWQ safetensors vs
   `mlx-community` quantisations). BookForge's `runtime/higgs-models/` copies retire with
   the spawn layer that reads them.
2. **The local server has one owner: its own `config.toml`.** The registry holds remote
   servers only — the machines whose tokens Owen pasted, because no other source exists
   for them. This reconciles PHASE7 §7.1(A): the bootstrapper still mints the token, but
   it hands it to `crucible init --token` and keeps no copy.
3. **Listen never re-rolls.** The streaming verdict is recorded, not acted on; a listener
   cannot wait for a retake.
4. **Routing follows PHASE7 §4.2 as written** — per-row `waitFor`, rank by drag order,
   enable per server, "New jobs wait for" as the default's setting. (An earlier draft of
   tonight's questions said "one global choice"; the written contract wins.)
5. **Envs and weights are installed tonight; renders are not.** Installing is pip and
   disk. Rendering is the card, and the card is fine-tuning.
6. **The Mac restarts at 2 AM**, not during the stream.

## 2a. Rulings Owen made overnight (2026-09-14, 01:40)

1. **A model is unloaded the moment nothing holds it.** His words: *"Models should always be unloaded
   when we're done with them. Every time."* This OVERRULES PHASE5-APPS.md §7's proposed "no idle
   unload". "Done" is four facts, not a timer: no job on the lane, no lease open, no streaming
   session, no chat in flight. The lease built earlier tonight is what makes it safe — a client that
   intends a run of requests says so, and three books under one lease still load the model once.
   **Built 02:20 (crucible `5bf7d62`), and the measured consequence is worse than "reloads":** the chat
   door never loads, so a BookForge cleanup run finds the model gone the moment its previous
   completion returned and is answered `model_not_resident` until something submits a load. The lease
   is therefore what makes BookForge's Crucible provider FUNCTION, not merely what makes it fast.
   **BookForge leases as of 02:45** (`4267bf5e`, `1ac6fbd4`): `withCrucibleLease` takes one lease per
   RUN, heartbeats at a third of the ttl, and releases in a finally on success, throw, cancel and app
   quit; a 404 on release is a no-op and a 404 on a heartbeat re-leases (the server restarted and
   forgot). Leasing: the four text acts, the `crucible` AI provider, and page reading. NOT leasing, by
   name and pinned: every one-job door (render, asr, align, rvc, denoise, re-roll — they hold the lane,
   and `tts`/`align` would be refused by a lease they took themselves) and the streaming session (it
   holds the claim). A second agent is extending the server's lease to cover a voice and an aligner,
   because a book rendered or aligned chapter by chapter would otherwise pay a load per chapter.
   Owed: a queue row that cleans THEN simplifies takes two leases, so the model can go between them —
   one lease for the row needs a seam in `queue-engine.ts` and cannot carry one truthful act name.

2. **Cloud is how an underpowered machine lights translate and simplify.** His words: *"if a user
   can't run a 27b for translation, the only way the translate/simplify cards can light up is if we
   connect a cloud model."* So the local floor stays the 27B and is not lowered to the 9B; the tile's
   rule becomes *local floor OR an enabled cloud slot*. The key picks the models: the user enters an
   API key, the app calls the provider's own listing with it, and the dropdown is what came back —
   Briefcase's pattern (`backend/src/config/config.controller.ts`: OpenAI `/v1/models` with a bearer
   token filtered to chat models; Anthropic `/v1/models` with `x-api-key` + `anthropic-version`,
   labelled from `display_name`; no key is a stated empty list, not an error). Foundry owns the cloud
   slots and has been told. BookForge's own AI providers should take the same treatment.

## 3. Rulings owed (record here, do not guess)

- ~~**BIBLE BOOK ABBREVIATIONS BEFORE TTS (Owen, 2026-09-14 late, NOT BUILT)**~~ — **BUILT 2026-09-14**, `33ebaf27` (the table + expander), `9868d376` (Listen), `91088531` (the render door), `9df3d93a` (the prompt), `75db8497` + `f0267b22` (the keepers). Owen: "for ai cleanup, i want to deterministically expand bible book names. ex -> exodus, tim. -> timothy. or at least tell the ai cleanup model to expand them the rest of the way before going through TTS. it's a mess. you can pass this to foundry if they're better suited." Built as assessed, with one correction: the deterministic half does **not** carry its own "is this a reference" test. `scriptureSpans` (`electron/tts-number-rules.ts`) already answers that on four kinds of measured evidence, against Owen's must-NOT list, so `shared/listen-text/bible-books.ts` expands the book token only INSIDE a span that detector claimed — one definition of "this is a reference", and the negative corpus that guards it is the one already in the repository. Chapter and verse stay as DIGITS; the reading stays the model's. Runs in three places: `prepareNarrationCopy` for a book (through `writeNarrationEpub`'s verified rewrites; the cut is `.v3`), the `.txt` door, and Listen stage 1.5. Model half = class 2b of `electron/prompts/tts-narration-text.txt`. Doc: `docs/NARRATION_TEXT_PASS.md` §"The book NAME is printed in full deterministically". **Ordinal spelling was NOT guessed and is still Owen's** — taken as "First Corinthians" because the number prompt has asked the model for exactly that since 2026-09-05, and the two halves of one pass must not read one reference two ways; `BOOK_ORDINAL_WORDS` is the single edit if he wants otherwise. **STILL OWED: (1)** Owen's word on that ordinal; **(2)** whether Foundry marks citations as structure; **(3)** a RE-VENDOR of `tts-narration-text.txt` into Foundry `src/clean/prompts/`, because Foundry is what cleans a book and the model half does not reach one until then. Memory `bible-book-abbreviation-expansion`.
- **SERVICE DEFECT (found 2026-09-14 16:10 restarting the local server onto the page build):** the
  unit says `Restart=on-failure`, so a clean SIGTERM (an operator's, or a future self-reload) leaves
  the server DOWN, and the user manager's `/run/user/1000/systemd/private` had vanished so
  `systemctl --user` could not reach it ("Failed to connect to bus") — recovered with
  `systemctl restart user@1000` as root, after which the enabled unit came up on its own. Owed in
  crucible `service.py`: `Restart=always` (+ `RestartSec`), and `crucible service status` should
  detect an unreachable user manager and name the root command. Bootstrap's `ensureRunning()`
  should do the same rather than report "running" from a stale pid.

- **WHICH DEATHSTALKER IS THE VOICE? (found 2026-09-14 by Training pc's hash).** Crucible's
  `voices/deathstalker` (pulled from HF `owenmorgan/deathstalker-higgs-v3` rev d732c38, 09-13) and
  the local higgs gate's `ds_v7_930_prod` (merged 09-11) have identical safetensors headers and
  DIFFERENT tensor data — two merges of one family, each the "voice" to a different server. R1: one
  owner. Ruling owed: which merge is canonical; then the HF mirror is re-uploaded from it (or not),
  the voice manifest's revision pin follows, and the other copy is deleted. Same question applies to
  mistborn/sigma/owen/thirdreich — nobody has hashed those.

- **STOPGAP, 2026-09-14 — `@crucible/client` is pinned to a LOCAL TARBALL, and the pin is
  replaced by the v0.6.0 release URL the day Owen publishes.** `package.json` says
  `"@crucible/client": "file:vendor/crucible-client-0.6.0.tgz"`, with the whole of this
  paragraph restated at the top-level `"//crucible-client"` key so nobody has to find this
  file to learn why. The reason is C4's order read backwards: PHASE13 §5 is built entirely on
  the operator door — `setup`, `catalog`, `submitTask`, `task`, `tasks`, `taskEvents`,
  `cancelTask`, `parsePairing` and the `CrucibleCardHeld` error — and the only PUBLISHED asset
  is v0.5.0, which has none of them. So the doors could not be built against the published
  release, and waiting for the publish would have parked the whole of this phase behind one
  line Owen has to type. The tarball is `npm pack` of the crucible checkout's `sdk/ts` at
  branch `feat/phase6-remote-render`, commit `54fe7a0` — **the same commit the release will be
  cut from, so the bytes match**; the replacement is a one-line edit to `package.json` and an
  `npm install`, and `vendor/crucible-client-0.6.0.tgz` is deleted in the same commit. It is a
  tarball FILE and never a `file:` DIRECTORY, because a directory dependency makes
  `node_modules/@crucible/client` a junction into the crucible checkout, and a later recursive
  delete of `node_modules` would follow it and take the SDK's source with it (memory:
  `git worktree remove` FOLLOWS the node_modules junction).

- **CPU SLOTS STAY LOCAL — Owen, 2026-09-14: *"cpu slots stay local though."*** Confirms the
  scheduler's ruling 1 as HIS: a Crucible server has 0 CPU slots; assembly, muxing and every
  `resource: 'cpu'` step run on this machine in the `local-work` set. Not a default any more.
- **A ROW LEASE IS PER MODEL (Foundry, 2026-09-14, caught before it shipped):** A5 kept the lease
  across steps whenever the next step declared `leasesModel`; clean (9B) → simplify (27B) would
  keep the 9B lease and refuse the 27B load `leased` by our own name. Fix: keep only when the next
  step's resolved model id equals the lease's subject; `leasesModel` is necessary, not sufficient.
  Hosted, the lease is Foundry's dispatcher's — `foundry-job.ts` carries a comment, not the flag.

- **~~CLOUD KEYS HAVE ONE OWNER: FOUNDRY'S CLOUD CARD~~ — OVERRULED 2026-09-14 (evening), by
  Owen, through crucible `docs/PHASE15-HOST.md` §0.** It was taken that morning as a default
  *"Owen may overrule"*, and he did: *"they dont have ollama fallbacks or cloud anything at
  all."* **The keys live in the ENGINE** — `[upstreams.anthropic] key` in its `config.toml`,
  mode 0600 beside the token it already holds — and BOTH apps' settings sections became
  windows onto that one document (§0d). Neither stores a credential anywhere.
  `electron/cloud-credentials.ts` and everything that called it are deleted (`ec24f361`), and
  `tools/test-no-cloud-doors.js` pins it. The paragraph below is kept because the reasoning
  it records is still the reasoning — a BookForge user on a laptop must have SOME way to
  light translate and simplify — and the answer changed, not the question.

- **CLOUD KEYS HAVE ONE OWNER: FOUNDRY'S CLOUD CARD, HOSTED TOO (default ruling 2026-09-14, Owen may
  overrule — SEE ABOVE, THIS IS THE OVERRULED TEXT, KEPT FOR ITS REASONING).** Foundry's registry seam (990bd2e) suppressed cloud providers hosted, which would leave a
  BookForge user on a laptop with NO way to light translate/simplify — against the 01:40 ruling. And
  BookForge's own AI page ships hardcoded three-item Claude/OpenAI lists (audit finding 3). So: hosted,
  Foundry's cloud-card is editable and its record in app-settings.json owns keys + models both ways;
  BookForge DELETES its own Claude/OpenAI key rows and model lists and its OCR-cleanup AI provider reads
  Foundry's record (as the clean door already reads `cleanTextModel`); `servers?()` stays Crucible-only.
  Foundry asked to un-suppress cloud hosted and keep the engine-config form + "run first-run setup
  again" GUARDED hosted. **Foundry landed all four at 33bb187** (cloud drawn hosted; card editable; engine
  settings form hidden + `settings:write` refused hosted — it had been writing the engine's machine-global
  settings.json; "run setup again" hidden + refused). **The single re-vendor targets foundry ≥ 33bb187.**
  Note: `cloudProviders` entries hold the KEY in app-settings.json (our userData hosted).

- **OWED AT THE RE-VENDOR (foundry 990bd2e, 2026-09-14):** `foundry-job.ts` passes the row's
  `waitForResolved` (a server name or `any`) as `RunOptions.waitFor` on `runJob` — the field
  did not exist in the vendored e6d5424, so it lands in the same commit as the subtree move;
  the hosted refusal `hosted_placement_not_vendored` is deleted there too (the keeper's
  tripwire says how). Foundry's registry seam now answers `host_provides_no_registry` (seam
  missing) vs `host_registry_unavailable` (our `registry_snapshot_not_taken`, quoted) — two
  facts, two codes. Ruled by both sides: hosted, Foundry's capability record owns the per-class
  model and `<userData>/crucible-models.json` GOES.

- Extract `narrator` into its own repo? (PLAN.md owed 1 — the pin is currently a git sha
  into the private BookForge repo; WSL and the Mac both authenticate to it today.)
- The take ladder's steps, and whether a resident model ever unloads itself.
- What a Listen verdict is *for*, beyond the record.
- Where urvc's base assets live for `rvc`.
- Publish the promoted fine-tune merges so the pace bands match the measured arms.
- From 2.4 (`electron/crucible/render.ts`): zero-shot voices — Crucible's `zeroshot` takes clips from the request and the render door refuses it; upload the clip, or local-only? Does a *remote* render hold this machine's GPU lease? Resume across an app restart (the job id is not persisted yet; Continue submits a new job for the missing chunks).
- From 2.3: a queue row cannot yet name a Crucible server (2.5's `waitFor`), so the queue's `crucible` provider door refuses by name until then.
- From the Crucible agent: should `crucible install <type>` end by offering the service (assumed no)? Fold `service status` into `doctor` (assumed no)? Should `dots-ocr` state `temperature = 0` server-side? Does audio-separator reach the network with both model files present (first real install settles it)? `torchcrepe` fetches its own weights, so `f0_method: "crepe"` may reach the network inside a job (every BookForge recipe uses `rmvpe`).
- **Model lease (Q3 from Foundry):** on a shared Crucible a chat holds no lane, so a `load-voice` from BookForge evicts Foundry's translator at block 400 of 2000. Proposed: an explicit lease on the resident model (`POST /v1/models/{id}/lease`, heartbeat for liveness, `DELETE` at run end); loads and unloads refuse `409 leased` naming client, act, since. Not a timer. *Owen said go at 23:45; built 01:05 (crucible `c5eb431` server, `aa2a24f` SDK): one lease per server; blocks load-model / unload-model / load-voice / tts / align at the job door (each evicts the resident model), never a chat; `/v1/activity.lease`; in-memory, restart forgets; 22 + 11 tests. Foundry builds the client side in its dispatch (lease → spawn → heartbeat at ttl/3 → release in settle). BookForge's text acts and renders should take leases too — owed.*
- **`[local]` block on model manifests** (Q2): *built (crucible `4e17842`): `[local]` validated like every other table; `scripts/gen-foundry-lineup.py` → `foundry-lineup.json` (schema 1, CI `--check`); Foundry told.* Rulings owed: `qwen3.8:27b-24g` is Owen's local Modelfile, not a published tag (`ollama pull` fails elsewhere) — name the published parent `qwen3.8:27b` or publish the Modelfile; Foundry's page reader pins `ggml-org/dots.ocr-GGUF` with the Q8_0 projector while the lineup names anthonym21's F16 — one must move.
- **Owen's tile rule (recorded 2026-09-13 22:40):** translate/simplify tiles do not light unless the machine can run at least the 9B; a job that would take a week on CPU is disabled, not allowed.
- **For your eyes (Foundry, overnight):** Foundry-pc-1 merged a third engine door, `--server anthropic` (Messages API, for analyze's verdict, with rate-limit backoff), at foundry `12b065d`. An older note in my memory says Claude-API simplify/translate was "not authorized" — if that still stands, it is theirs to hear from you; nothing in BookForge uses it.
- **From D1 (the hosted Foundry seam, 2026-09-14): the per-row venue does not cross `runJob`.**
  BookForge resolves which Crucible a row waits for (`waitForResolved`, 2.5) and hands the job
  to the hosted window — but the vendored side chooses the slot itself with
  `waitForOfNewJob()`, which reads FOUNDRY's `newJobsWaitFor` and its derived slot list, and
  `RunOptions` carries nothing to say otherwise. So after the re-vendor a hosted text act will
  land on a machine the row did not name, and "one book, one GPU" (PHASE7-LANES §4.4) would be
  two answers again. **Foundry's, and small: one field on `RunOptions` (or on the request) the
  placement prefers over its own default.** Nothing here should paper over it — a second
  scheduler deciding the machine is exactly what centralising the queue removed.
- **From D1: who leases a hosted act.** Settled by reading, and written into
  `queue-steps/foundry-job.ts`: the vendored dispatcher takes a model lease between making the
  model resident and spawning the engine, Crucible allows one lease per server, so BookForge
  must NOT lease around a hosted run. BookForge leases where BookForge spawns. Recorded here
  because the file used to say the opposite and somebody would have built it.
- **From D1: the two per-act model ids.** BookForge picks a Crucible model per act (Settings →
  AI → Crucible, `<userData>/crucible-models.json`); the vendored dispatcher picks the SERVER's
  own `selected` model for the capability class and says a configured id would be one that
  server may have refused. Both are defensible and they are two owners of one fact (R1). Owen's
  to rule once a hosted act actually runs: does BookForge's per-act choice travel, or is the
  server's capability record the only picker, in which case BookForge's own text acts should
  read it too?
- **From A1 (`shared/queue/slot-sets.ts`), the per-server slot sets:** PHASE7-LANES §2.4's
  table gives every server `[gpu][cpu][cpu]` and says in the same row that the CPU number is
  *"Zero today — no CPU work is sent to a server yet, and the slots exist so the bench and
  the model do not change when it is."* Built as `SERVER_CPU_SLOTS = 0`, because a lane the
  scheduler will never fill is a maybe (ARCHITECTURE.md R3) — no step module declares
  travelling CPU work, so every `cpu` step goes to `local-work`. **Ruling: confirm 0, or say
  the bench should draw two idle lanes per server against Crucible's ancillary lane landing.**
  One constant, one line, either way.
- **From A1, the LEGACY set and this machine's one card.** The legacy narrator spawn has a GPU
  slot of its own so the stopgap keeps its old behaviour, and it is present for as long as that
  spawn layer is (§0b A2 deletes both). §2.4 says *"there is no local gpu row"* — under it, the
  legacy spawn would occupy the LOCAL server's slot instead. It does not, because a GPU step
  whose module has not been taught to travel spawns here whatever the switch says and would
  otherwise have no set to charge. `thisMachinesCardHeldBy` is what stops the two sets starting
  two jobs on one 3090 Ti, which the old global `gpu: 1` prevented by accident. **Ruling owed
  only if the layer outlives the in-app pass**; if it is deleted as planned the question goes
  with it.
- **From A3 (`electron/queue-steps/align.ts`): align refuses BEFORE it submits, not after.**
  `electron/crucible/align.ts` was built to land `alignment.json` and then refuse
  `crucible_align_narrator_door_owed` — a loud dated partial, with the GPU work banked (R6).
  A row assigned to a server now refuses at the STEP, before anything is submitted, because
  loading a 3 GB aligner on somebody's card to produce an artifact nothing can read is a worse
  answer than the honest one. The after-the-fact message is unchanged and still what the
  in-flight path says. **Ruling: is the banked `alignment.json` worth the card once narrator's
  items-in door (B5) exists? If it is, the early refusal comes out in the same commit that
  lands that door.**
- **From A3: the PASS steps travel — DONE 2026-09-14 (`2f154f0b`).** `simplify`,
  `translate-pass` and `narration-text` declare `machines()` through the shared
  `machinesForAiStep`, follow the run's venue (`runVenueOfRow` → `providerConfigOf`, and
  `cleanTextEpub`'s existing `crucibleServer` door for the `clean` act), and declare
  `leasesModel` + `leasedModel`; `footnote-refs` stays `local`, being a string replace over a
  zip. The hand-built fourth copy of the provider mapping is DELETED — `providerConfigOf` is
  asked first, before the book is resolved and before a stage directory is made, so a
  `crucible_server_not_named` refusal costs no work. Two things the audit had not seen: a pass
  config nests its provider under `simplify`/`translate`, so `resource`, `machines` and
  `leasesModel` were all reading a top-level `aiProvider` that is never there (every pass filed
  on the GPU pool, including one against Claude); and `callAI` — the transport a translate
  pass actually reaches the model through — knew four providers and no Crucible, so the
  declaration alone would have failed the row an hour in. It takes the same `AIProviderConfig`
  now, through the shared `crucibleChatOnce`. 16-check keeper
  `tools/test-queue-pass-travel.js`.
- **From A3 / Foundry: `foundry-job.ts` does NOT declare `leasesModel`, and the absence is the
  statement (settled 2026-09-14, `2f154f0b`’s sibling).** The one-line addition was offered and
  declined: a hosted act's lease belongs to the vendored dispatcher, which takes one between
  making the model resident and spawning the engine and releases it in its own settle
  (`crucible-dispatch.ts placeOnCrucible`). Crucible allows ONE lease per server, so
  `leasesModel: true` here would describe a lease that does not exist AND would refuse theirs —
  in this app's own name. It does not become right at the re-vendor either; a row-keyed lease on
  that path would be Foundry's design to make, if Owen asks for one. The reasoning is written
  beside `machines()` in that file so the next reader does not re-derive it.
- **From A5 / Foundry: a row's lease is kept only for an act that wants the SAME model —
  BUILT 2026-09-14 (`ad86d91f`).** `leasesModel` alone was not enough: a lease is per MODEL
  (`POST /v1/models/{id}/lease`) and a server holds one, while clean runs on `qwen3.5-9b` and
  simplify/translate on `qwen3.8-27b-4bit`. So the archetypal clean→simplify row carried the
  9B's lease into the step that must load the 27B, and that load is refused `leased` naming
  `bookforge`. `StepModule.leasedModel` named the id and `CrucibleLeaseHost.leaseSubject`
  reported what was held; since 2026-09-19 the pair is `StepModule.crucibleClass` and
  `CrucibleLeaseHost.leaseHeld` (the machine and the class), because no module could name an id
  after phase 15. The lease survives the seam only when they match. Scoped precisely, because
  half of it was already covered: `withRowLease` swaps on a model change, so BookForge's own
  chat acts never deadlocked — what the stale keep held was the GAP between steps, where
  `resolveCrucibleTextEngine`'s `loadFirst` door and an operator's CLI load both live. Also
  found: `settleStep` fires the release without awaiting it and pumps in the same tick, so a
  take could overtake it (`rowReleases` now waits). `tools/fake-crucible.js` enforces one lease
  per server, which is what makes any of this reproducible rather than assumed.
- **From A5 (`electron/crucible/lease.ts`): a row-wide lease carries the act that OPENED it,
  and Crucible has no name for "a row of acts".** `require_act_name` refuses anything outside
  its capability classes, and a lease carries one `act` with no route to re-state it. So a row
  that cleans and then simplifies holds one lease stamped `clean`. That is honest about what a
  lease answers (*why is this model being held*) and the other question — *what is running
  right now* — is answered per request by `X-Crucible-Act`, truthfully, and reported by
  `/v1/activity` as the in-flight entry's act. **Ruling: either Crucible gains a way to
  re-state a lease's act (an `act` on the heartbeat, or a PATCH), or a lease carries a LIST of
  acts, or this stays as the opening act's name.** The vocabulary is the server's, so it is not
  BookForge's to settle.
- Needs your hands, not a ruling: `sudo loginctl enable-linger <your-wsl-user>` in WSL (the service dies with your last shell otherwise); (Foundry v1.3.0 is VOID per Owen's 22:40 reframe — do not publish it); the first `wsl.exe` read against a cold VM returning −1 wants a reproduction.

## 4. State log

- **19:50** plan written; wake timers set for 02:03 and 06:03; questions with defaults sent
  to Owen before his stream.
- **19:45** Tier 1 installs started on both machines (nohup; logs `~/crucible-install-pc.log`
  in WSL, `~/crucible-install-mac.log` + `~/crucible-install-mac-tts.log` on the Mac). The
  Mac's first Higgs install failed on a zsh word-splitting quirk in my script (the argument
  arrived as one word); re-run under bash at 19:53.
- **20:05** 2.1 committed (`4787625a`). Three Opus agents running: 2.4 render seam, 2.2+2.3
  Servers row and provider, and the Crucible trio (service verb, manifest defaults, denoise).
  Foundry-pc-1 is taking its four engine items tonight and holds the registry work for
  Owen's word in its own session.
- **20:30** Foundry's engine phase landed (`646e8a1`, tag `engine-one-door`: one inference
  door, `--server`/`--ollama`/`--keep-model` retired, act-named log prefixes, `fitsWindow`).
  Fourth Opus agent started: re-vendor `foundry-app/` at 646e8a1 and retire the same flags
  on BookForge's side (`narration-clean-text.ts`, the clean CLI adapters, the `server`
  setting). Mac doctor found two real gaps — `ffmpeg` off the service PATH, RVC base
  assets with no source — both queued to the Crucible agent as root fixes.
- **21:30** 2.2, 2.3, 2.4 committed (`f6ef8d76`, `ea39bc58`, `3a98ff70`, `944d1f02`). Crucible:
  service verb landed (`a838101`), recipes resolved (`1ec936d`), branch pushed; the WSL
  server runs as a service with nothing resident. Foundry release prep done at `83d7b66`
  (publishing is Owen's). In flight: the render-routing follow-up, the re-vendor, and the
  Crucible agent's manifest defaults + denoise + rvc-base pull + service generator fix.
- **22:20** Re-vendor done: `foundry-app/` at Foundry `83d7b66` (`afe10842`, 145/145 blobs
  hash-verified), BookForge's own clean-text driver and CLI adapters stop speaking the
  retired dialect (`992a2fbf`: `--server`/`--ollama`/`--keep-model` gone from the spawn
  lines, refused by name in the CLI with the retiring commit; the `server` field of the
  clean-text settings is gone; the on-disk `llmServer` key is read as a selector and
  reported retired once). Found on the way: BookForge's mirror of Foundry's progress
  parser had already drifted — hosted Simplify progress stuck at block one — fixed,
  keeper 20/20. Two call-site pins moved for the Crucible seam (`8f3fd3d5`). Foundry's
  vendored `job-queue.ts` still spells `--keep-model` behind a flag nothing sets (dead,
  sealed subtree — reported to Foundry-pc-1). 2.5 (per-row `waitFor`) agent running.
- **23:40** 2.5 landed (`e91d2a41`); `foundry-app/` re-vendored at Foundry `81fdc30` after
  they removed the `--keep-model` residue (`a3ad3655`). Full keeper suite: 130 ran, 1 red —
  `test-bookshelf-stream-teardown`, a real race (the session said "released" before its
  snapshot directory was removed; lost under suite load, won alone). Fixed at the source
  (`cd6ab8b3`: the release is awaitable). Crucible: manifest defaults (`835628e`) and the
  service generator fix (`b40de27`) landed and are pushed; the WSL service was re-installed
  from the fixed generator and answers `/v1/ping`.
- **22:10** Owen finished streaming and gave me the card. The Mac is on v0.5.0 as a launchd
  service, reachable from the PC. **The first real Crucible `tts` render found the real
  gap**: narrator's vllm-omni arm needs `HIGGS_STACK`, `HIGGS_MAX_NUM_SEQS` and a launch
  script (`serve_higgs_v3.sh` + the certified frames-7500 deploy profile) that lived only in
  BookForge's `electron/scripts/higgs/`; Crucible set none of them. Fix in flight (Opus
  agent): the launcher becomes narrator's own package data, Crucible states the three
  variables (stack from the env spec, width from a `[serving]` table on each voice), the
  narrator pin moves, the tts envs are reinstalled, then the live keeper runs again. Also
  found: the regenerated rvc recipe had lost its PyTorch index line (`fa19e99`).
- **22:40** Owen reframed standalone Foundry: Crucible optional there (the speed tier and where
  WSL lives), Ollama through the one door as the beginner's default, the Ollama wizard stays,
  Foundry's own vLLM launcher is deleted for good; hosted Foundry inside BookForge stays
  Crucible-only. The 1.3.0 release prep is void. Foundry's three questions answered with
  evidence (dots.ocr runs under llama.cpp with a GGUF+mmproj pair, not under Ollama; one
  catalog of record = Crucible manifests with a `[local]` block; eviction mid-run = an
  explicit model lease, ruling owed).
- **22:57 THE FOUNDRY WIRE IS PROVEN LIVE.** `qwen3.5-9b` made resident on `local`; Foundry's
  `clean-text` ran against it — 734 blocks, 265 changed, 78.5 s, EPUB written. One Crucible
  fix on the way (`a97ef70`): the OpenAI door is also mounted at `/openai/v1/...`, where
  every OpenAI client composes it; the base is `<server>/openai`. 2.6 (`0088a296`) built the
  BookForge side with a version-floor refusal for that same gap — being corrected to the
  `/openai` base now; the one remaining hosted gap is Foundry's own: the vendored engine
  spawn takes no per-run environment, so the hosted job-queue path cannot carry a per-act
  header map until Foundry's `engine.ts` does. dots-ocr weights pulled on the PC (6.1 GB).
- **23:30** Owen handed the 3090 Ti to the fine-tuning agent (told "GPU free"; nothing of
  mine resident). **The live TTS keeper is deferred to the next free window** — the launcher
  fix lands without it and is verified only by narrator's and Crucible's own tests until
  then. Cleared on Owen's word: Ollama `cogito:8b/14b/32b`, the duplicate `qwen3.8:27b` tag
  (kept `27b-24g`), the 4-bit `qwen3.5:9b` (kept q8 and bf16); Foundry's ocr and footnote
  adapters and the orphan `foundry-4b-f16.gguf` base (~60 GB freed). Foundry landed its
  Package A (`374f18c`, engine `--server openai|ollama`) and B (`aa1c382`, local page reader
  = llama-server + dots GGUF); C (registry/slots/dispatch) in progress; one re-vendor after C.
- **23:45** Owen to bed: "power through until everything is done"; linger enabled; lease
  ruled go; the Mac's card offered for the TTS proof. Mac service re-recorded with
  Homebrew's PATH (ffmpeg refusal by name was the finding) and restarted; the live TTS
  keeper is running against the Mac in remote mode. Five Opus agents in flight: the
  narrator launcher (Crucible half), the model lease (server + SDK), the `[local]` block +
  `foundry-lineup.json` generator, `@crucible/bootstrap`, and BookForge streaming through a
  Crucible session. Foundry landed A and B; C and E in progress.
- **00:05** The launcher landed: narrator owns `serve_higgs_v3.sh` + the certified profile as
  package data (BookForge `0eeb0267`, narrator 1449 tests), Crucible states `HIGGS_STACK`
  (from the env spec), `HIGGS_ENV`, `HIGGS_MAX_NUM_SEQS` (from a `[voice.serving]` table)
  (`a7ab9af`, 894 tests); the tts envs were rebuilt on the new pin on both machines;
  both doctors healthy. Two more real defects behind it, agent assigned: Crucible's
  `load` sends `modelDir`, which narrator's served arm refuses by name, and narrator
  resolves the voice in a `NARRATOR_HIGGS_VOICES` document Crucible never wrote. Rulings
  owed: SGLang-Omni vs vllm-omni is named seven ways in this repo and the recipe installs
  vllm-omni; what Orpheus's narrator arm needs on cuda-linux (a whole `ORPHEUS_*` set).
- **00:20** Live keeper against the Mac (Owen offered its card): ping, voices, info, "nothing
  resident" all pass; the render fails at LOAD — narrator's MLX arm refuses the per-load
  `modelDir` exactly as the served arm does. Same defect, both arms; the load-contract agent
  has the live text. Two Mac service restarts tonight were for the same stale-process class:
  an editable install changed on disk under a running server (a doctor and a service verb
  that compare the running version to the checkout would name it — item for Tier 3).
- **01:20** The Higgs load contract landed (crucible `2d6b1f5`, 1009 tests): Crucible writes
  `~/.crucible/narrator-higgs-voices.json` from the voice manifest + pulled dir at every
  load and hands narrator `NARRATOR_HIGGS_VOICES`; the load message is the voice's name,
  no `modelDir`, on both arms; narrator unchanged. Both servers restarted on it (with the
  lease). Keeper rerunning on the Mac. Rulings owed: `higgs-default` (a token voice) on
  cuda-linux is refused until narrator can name the base dir for a `default` voice —
  Mac-only until then; zeroshot at the load door; take>0 needs a per-request sampling
  channel. Also landed: the lease (`c5eb431`/`aa2a24f`), `[local]` + lineup (`4e17842`,
  `9eb91bc`, `7e63905`), `@crucible/bootstrap` (`ab6d7bd` + `3550763` + `6b15c6d`);
  BookForge streaming via a Crucible session (`f079b456`, `323803fb`).
- **01:35 THE FIRST REAL HIGGS RENDER THROUGH CRUCIBLE PASSED — on the Mac, 8/8.** The live
  keeper in remote mode against `crucible@example-mac` (v0.5.0+ at `2d6b1f5`, narrator
  `0eeb0267`, deathstalker): voice loaded with `warming` lines, render ended `done`, a `chunk`
  event per row with its own seconds/chars/chars-per-sec (`capped` null, as PHASE3 says),
  both FLACs real and mono at 24 kHz naming the merge, a second render reused the resident
  voice without restarting narrator, `unload-voice` freed it. No memory measurement (remote
  mode cannot watch the card); the PC's measured peak still waits for a free 3090 Ti.
- **01:50 BookForge's own render seam reached the Mac too** (`bookforge-tts --tts --engine
  higgs --voice deathstalker --crucible-server mac`): the chunk rendered on the Mac's
  Crucible (its engine log shows the 00:49 start). Two defects behind it, both assigned:
  the new coverage-align step re-decided its venue (top-ranked = `local`) instead of
  following the run's, and loaded the qwen3-aligner on the fine-tune's card for ~1 min —
  unloaded, Finetuning-1 told; and a Crucible-bound run still preps its session inside
  WSL and then copies it to `Z:ookforge	mp`, a share WSL cannot mount (the known CLI
  defect) — a Crucible-venue run must live on a native path from the start.
- **02:00 Foundry drove the lease end to end on the Mac** (their hand-run of the app's
  dispatch against `2d6b1f5`): capability → load-model → lease (clean, ttl 120) → a second
  lease and a load-model both `409 leased` naming the holder → the built engine's
  `clean-text` through `<server>/openai` with the header map, 29 blocks in 99 s → heartbeat
  → release → a second release `404 unknown_lease (released)` → `activity.lease` null. The
  Mac's `qwen3.5-9b` is resident now (their run left it, by design). Foundry merged G (one
  queue lane per compute slot) at `f7ad5a9`; E still running.
