# Voices

How a voice is described, where its description comes from, and how narrator is
configured to serve it. Modules: `voices`, `voicefacts`, `voicerefs`, `voicerepo`,
`voicecatalog`, `voicecard`, `voicereference`, `narratorvoices`, `engines/narrator`,
`engines/higgs-v3/base.toml`, `voices/pins.toml`, `scripts/publish-voice.py`.

Crucible ships no voices. It downloads them, and a voice's facts come down with
its weights.

- The narrator engine names, their default sampling, which engines read the
  voices document and the declared per-engine footprints live in
  `narratorengines`, which imports only `backend`; `"higgs-v3"` is written once,
  as `narratorengines.HIGGS_V3`.
- A voice id is `tomltable.VOICE_ID_PATTERN`: lower-case letters, digits, dot,
  dash and underscore, starting with a letter or digit, at most 64 characters,
  because it becomes a file name. Voices, pins and home overlays use the same
  rule.

## Layers

- `voices` parses: the manifest schema, `parse_document`, `parse_voice`,
  `parse_engine_base`, overlay writes and the directory paths. It imports
  neither `voicerepo` nor `voicecatalog`.
- `voicerepo` reads and writes pins, fetches a pin's `crucible-voice.toml`
  and merges it with this machine's footprint (`merge`, `voice_for_pin`,
  `load_pinned`). It imports `voices` and nothing above it.
- `voicecatalog` sits above both and owns "which voices exist": the load
  order below, `load_voice`, `unserved_pins`, `resolve_weights_of`, and the
  build's **declared** voices for the module generator
  (`declared_voice_ids`, `declared_voice_backends`). `catalog` calls only
  these public functions.
- `load_all_voices`, `load_voice`, `unserved_pins` and `voice_aliases_of`
  are imported from `voicecatalog` only; `voices` does not forward them, so
  the import graph has no `voices` ↔ `voicecatalog` edge.

## Sources and precedence

`voicecatalog.load_all_voices` is the single owner of the order. Lowest precedence first:

1. **Pins**: `crucible/voices/pins.toml` (what this build offers) plus
   `<home>/voices/pins.toml` (what this machine chose). The home file wins per
   id. Each pin's `crucible-voice.toml` is read from the weights' own repo at
   the pinned revision.
2. **Engine base rows**: `crucible/engines/<engine>/base.toml`
   `[voices.<id>]` tables (`higgs-default`, `zeroshot`), each parsed by the same
   `parse_document` as any voice.
3. **Overlay**: `<home>/voices/*.toml`, which `PUT /v1/voices/{id}` with a
   `voice` body writes.

- `CRUCIBLE_VOICES_DIR` replaces **everything**: pins, engine rows and home.
- Shadowed pins are still loaded, so a broken pin surfaces before its
  overlay goes away.
- **One unreadable pin means one unserved voice, never zero voices**
  (2026-09-26). `load_pinned` keeps each pin's `VoiceError` beside its id, and
  `unserved_pins` reports it. Anything other than `VoiceError` is a defect and
  propagates.
- `pins.toml` is a reserved name in the voices directory.
- Voices are listed in id order by file **stem**, not by path, because `-` sorts
  before `.`: `zeroshot` and `zeroshot-deathstalker` would otherwise swap.
- `resolve_weights_of` checks every `weights_of` against the loaded set and
  hands each base the aliases it resolved (`VoiceManifest.weights_aliases`),
  so `aliases()` answers from the same load instead of loading again.
- **Declared voices** are what this build ships, whatever the machine chose:
  the packaged pins, the engine base rows and the packaged voices directory.
  A packaged pin's backends are its repo manifest's arms, read through the
  same fetch order and `<home>/voice-manifests` cache as a served pin.
- `higgs-default` and `zeroshot` are never pinned. They sit on Boson's
  `bosonai/higgs-tts-3-4b`, which cannot carry a manifest, and they are the
  engine's own base behaviour rather than trained voices.

## One source of truth: the repo, and its `crucible` tag

Owen, 2026-09-28: *"single source of truth. that source should be where the
models are served."* A voice IS its repo's `crucible-voice.toml` at the commit a
machine pulled: pace, safe band, caps, sampling, takes, chunk gap, edge fades,
reference cap, allowed controls. No client keeps a copy, and nothing in Crucible
has to change to deploy a voice.

**Deploying a voice is three steps:**

1. Push the weights and their `crucible-voice.toml` to the voice's HF repo, in one
   commit. Check it first: `crucible voices check <owner>/<name>@<sha>`.
2. Move the tag: `python scripts/publish-voice.py <voice-id> <sha>`. It reads the
   HF token from `$HF_TOKEN` or `huggingface-cli login` (never printed), refuses a
   sha the repo does not have and a manifest Crucible's own parser refuses, and
   only then moves `crucible`. `--create` makes the tag on a repo that has none, at
   the revision this build shipped before tags (`SEED_REVISIONS`).
3. Each machine runs `crucible voices pull <voice-id>` (or `--all`, or a pull task).
   `crucible voices check-updates` (`POST /v1/voices/updates`) says first which
   voices would move.

No BookForge JSON, no `pins.toml` edit and no Crucible release.

## Pins

A pins file maps each id to `hf_repo` plus exactly one of `ref` (a tag the
machine follows) or `revision` (a 40-character sha it serves and nothing else):

```toml
[mistborn]
hf_repo = "owenmorgan/mistborn-higgs-v3"
ref     = "crucible"
```

- The packaged rows all say `ref = "crucible"`.
- A pin with an exact `revision` (the home file, which the ladder's `PUT` writes)
  wins over the packaged ref and means exactly that sha.
- **Which commit a ref pin serves** (`voicerepo.settled`): the sha recorded in
  `<home>/voices/<id>/<arm>/crucible-pull.json`, else the last looked-up sha in
  `<home>/voice-refs.json`, else the voice is unserved with
  `voice_ref_unresolved` and the command that fixes it. The row's `revision` is
  that sha.
- **Reads never touch the network.** `GET /v1/voices` reads the pull record and
  `voice-refs.json`. A tag is looked up (`voicerefs.check`, `HfApi.model_info` at
  the ref) only by `crucible voices check-updates` / `POST /v1/voices/updates`,
  by a pull (`crucible voices pull`, a pull task, and a pull submitted for a voice
  never looked up), and once at server start for ref pins with no sha at all.
- **A pull moves the box** to the tag's current sha (`voicecatalog.pull_target`),
  fetches that commit's manifest and weights, and records the sha. A pull task
  for an installed voice is admitted only when the cached look-up says it would
  move. A voice that is resident is not swapped: the task refuses with
  `subject_in_use` (the same `held_on_card` guard `DELETE /v1/catalog` uses), and
  the CLI refuses while the server here has it loaded.
- **An unreachable Hub** leaves the box on what it has. `voice-refs.json` keeps
  the last sha it knew and records the error, and the row's `update_error` says so.
- A corrupt `voice-refs.json` is moved aside as `voice-refs.json.bad-<timestamp>`
  and read as empty.

- `PUT /v1/voices/{id}` with `{"pin": {"hf_repo", "revision"}}` (or `crucible
  voices pin <id> <repo>@<sha>`) writes the **home** pins file only. The
  packaged file is part of the install, and editing it would let an upgrade
  undo a deliberate repin.
- The door calls `voice_for_pin` **before** writing, so a pin that cannot load
  is never stored.
- Writes are validated by the same `parse_pins` and made atomically. A
  half-written pins file breaks every voice.
- Removing a home row restores the packaged pin, which is why trying a new
  checkpoint is safe.

## The repo manifest (`crucible-voice.toml`, `voicerepo`)

It lives at the repo root and is committed together with the weights it
describes, so a value and its description cannot drift apart. The card
(`README.md`) is generated from it (`crucible voices card`). The card is never
parsed.

- **It carries no machine facts.** `id`, `hf_repo`, `revision`,
  `memory_bytes_estimate`, `estimate_basis`, `estimate_note`,
  `[voice.serving]` and `backends` are each refused by name. The pin supplies
  the id, and the machine's `config.toml` `[tts.<engine>]` supplies the
  footprint and serving levers (`voicerepo.merge`). `merge` takes them from
  `EngineFootprint.to_dict()`: the `voices.SERVING_KEYS` go to
  `[voice.serving]` (Higgs v3 only), the rest to every arm.
- `RepoManifest.voice` is a `VoiceDocument`: exactly `display`, `kind`,
  `narrator_engine`, `language` and `sample_rate`.
- The repo schema says `[voice.arms.<backend>]` where the internal schema says
  `[voice.backends.<kind>]`. The name difference is deliberate, so neither
  file can be mistaken for the other.
- `merge` translates into the internal **document** and runs the same
  `voices.parse_document`, so every pace, sampling, clips, takes and cap rule
  applies unchanged.
- An unknown `schema` version is refused. Crucible never reads only the keys
  it recognises.
- **Pace basis.** A present `[voice.pace]` needs `basis`. `measured` requires
  `measured_from` and refuses `inherited_from`. `inherited` requires
  `inherited_from` and refuses `measured_from`. Each basis owes exactly its
  own sentence (2026-09-19). An inherited pace from a sibling checkpoint is
  close (mistborn 13.29/13.33/13.76). One from another corpus is not:
  deathstalker's 16.64 sat on weights that measured 15.91, 4.4% fast.
- **Cap basis.** `max_chars` is optional, since a cap only exists after a
  sweep. A stated cap requires `max_chars_basis` (`measured` or
  `placeholder`), and a basis without a cap is refused as a leftover.
- Fetch order: the pulled weights at this pin (the stamp is checked, so bytes
  from an older pin do not count), then the content-addressed cache
  `(repo, sha)`, then the Hub (`hf_hub_download` fetches that one file, so
  uninstalled voices still list real facts). "This revision has no manifest"
  and "the Hub did not answer" are different refusals.

## Facts clients read off the row (`voicefacts`)

Every per-voice fact a client used to keep a copy of is on the `/v1/voices` row,
so BookForge can delete its copies. All are optional in the manifest. Absent
means `null` on the row, and Crucible never invents one. Wire names are
snake_case.

| row field | manifest | meaning |
|---|---|---|
| `sampling` | `[voice.arms.<arm>] sampling` | `{temperature, top_p, top_k}` the row's arm renders take 0 with. This is exactly what Crucible writes into narrator's voice document (`narratorvoices`), which is what the engine applies. Crucible never reads `generation_config.json`; a repo whose `generation_config.json` disagrees with its manifest renders at the manifest's numbers. |
| `edge_fade_ms` | `[voice.arms.<arm>] edge_fade_ms = { in, out }` | raised-cosine fades at each chunk edge, in milliseconds (BookForge `edgeFadeMs`). Higgs is `pads=false`, so a chunk ends on a hard sample and an unfaded join clicks. |
| `reference_seconds_cap` | `[voice.arms.<arm>] reference_seconds_cap` | the most reference-clip audio the arm takes, in seconds, above 0 and at most `voicereference.MAX_REFERENCE_SECONDS` (BookForge `referenceSecondsCap`). |
| `allowed_controls` | `[voice.arms.<arm>] allowed_controls = [...]` | inline control tokens shaped `<\|group:name\|>` the arm allows; `[]` allows none on purpose (BookForge `allowedControls`). |
| `chunk_gap` | `[voice.chunk_gap]` | the silence a client adds after each chunk (BookForge `chunkGap`, which it passes to narrator as `NARRATOR_SENTENCE_GAP`). |

`[voice.chunk_gap]` keys, all seconds >= 0 unless prose:

- `inject_s` (required): the silence added after each chunk, **net of the tail
  the model already emits**.
- `target_join_s` (required): the join it should come to. It must equal
  `inject_s + model_self_tail_s` within 0.011 s, or the manifest is refused.
- `model_self_tail_s` (required): the trailing silence the model emits itself.
- `reader_sentence_gap_s`, `model_internal_gap_s` (optional): the reader's pause
  in the training clips, and the model's own sentence gap inside one render.
- `rule`, `method`, `source`, `measured_on` (required prose): which target was
  chosen, how it was counted, from which run, and when.

Each table has one validator in `voicefacts`, built on `tomltable.check_table`,
and the repo schema and the internal schema call the same one. `voice_document`
and `crucible voices export` write them back when present.

## The voice schema (`voices`)

- **`max_chars` counts characters, not tokens.** narrator derives the frame
  cap per chunk from the text. It is per backend, because each arm samples
  differently. It is optional. Absent means *not measured*: `null` on
  `/v1/voices`, omitted from the narrator document, and never borrowed from
  another arm. The render door does not refuse chunks by length.
- **`estimate_basis`**: `measured` or `declared` (for example SGLang's
  configured reservation). `declared` requires `estimate_note`, and `measured`
  refuses one. The basis is shown on the `/v1/voices` row.
- **Sampling.** The boson default is 0.8 / 0.95 / 50 for every Higgs voice on
  both arms (Owen, 2026-09-12: *"Let's set temp to 0.8 across the board"*). A
  deviation requires `sampling_reason`. `sampling` **replaces** the engine
  table and does not merge into it: every key must be present, because on
  SGLang an unfilled `top_k` samples the untruncated codebook tail (measured:
  80 s of silence). `repetition_penalty` is not allowed for Higgs.
- **Pace triple** (`pace_chars_per_sec`, `max_...`, `min_...`): all three or
  none, and `min < pace < max`. The band must be symmetric in ratio (every
  ladder writes `pace × 1.3` and `pace / 1.3`), within the rounding of
  two-decimal figures (`_PACE_HALF_ULP`). A band that really is lopsided
  states `edges = "percentile"`, which never reaches the wire. The whole
  `[voice.pace]` table may be absent, which marks an uncertified voice.
- **Why the base voices state no pace.** The old 15.0/20.0/14.5 triple was
  narrator's defaults copied back into it. 15.0 is `cap_frames()`'s divisor,
  not a speaking rate. narrator keeps a band's *ratios* and re-centres them on
  the book's median, so the 1.333/1.034 lopsidedness re-rolled healthy chunks
  to MAX_DEPTH. With no band, narrator uses its own default band centred on
  the geometric mean of the edges, and Crucible never computes a centre.
- **Packing shape**: either a `target_chars` (zero-shot) or a
  `safe_min_chars`/`safe_max_chars` band (fine-tunes: the corpus IQR), or
  neither, which means packing to `max_chars`.
- **`sample_rate`** is required per voice (24000 for every voice today, which
  is exactly why it must not become a constant).
- **Source shape**: exactly one of `hf_repo + revision` (a pin: fetched,
  stamped, `identity_basis = "verified"`) or `path + identity` (a local
  directory Crucible never fetches, stamps or deletes:
  `identity_basis = "asserted"`). Screening checkpoints use the local shape. A
  path must be absolute in **either** POSIX or Windows form, because the
  loader may run on another OS than the server. An empty string counts as
  absent.
- `fingerprint` is `<id>@<identity>` in both cases. `weights_identity` is the
  single reader of "revision, else identity".
- `weights_of` makes a voice share its base's download (Owen, 2026-09-24:
  *"lets reduce it to a single copy of everything"*). The base must be served,
  must not itself be an alias, and must pin the same repo and revision.

### Serving (`[voice.serving]`)

This section sizes the server narrator starts. It is **required for
`higgs-v3`** and refused on any engine that reads no `HIGGS_*` variable. The
`/v1/voices` row reports it as its `serving` block. Each lever requires a
`_note`. A note without a number is refused.

- `max_num_seqs` → `HIGGS_MAX_NUM_SEQS`: stage 0's admission width **and**
  narrator's batch width. narrator refuses to render without it. 16 is
  disputed (the deathstalker cap was measured at 64), which is why the note is
  required.
- `mem_fraction` → `HIGGS_SGL_MEM_FRACTION` (SGLang `--mem-fraction-static`).
  Absent means the launcher's 0.60. It is preallocated as KV on top of
  7.7 GiB of weights **whatever the width**, so narrowing the batch does not
  lower it. Measured 2026-09-19: 0.55 sat at 24.0 GB on a 24 GB card, where
  WDDM pages to host RAM 4–10× slower with no error. 0.48 at width 4 is ~20 GB.
- `context_length` → `HIGGS_CONTEXT_LENGTH`. Absent means SGLang-Omni's
  hard-coded 4096. That holds about 2,000 characters, so long rungs truncate on
  context and a screen records the truncation as the voice's length wall. A
  screening voice states 8192. **No narrator pin reads this variable yet.**
- `stall_guard` → `HIGGS_STALL_GUARD`, **on both arms, always stated**
  (`crucible/stallguard.py`). Absent means the default guard,
  `{ frames = 37, rate = 1.0, max = 20, window = 16 }` (`"37,1,20,16"`, accepted
  2026-10-02 from the PC A/B; see `stallguard.DEFAULT_NOTE`); a
  table states other numbers; `false` turns it off (`"off"`). `true` is
  refused (absent already is the default), and so is a partial table, an
  unknown key, or a number outside frames 1–10000, rate 0.001–100, max
  0.001–1000, window 1–64. A stated `stall_guard` needs `stall_guard_note`. The
  row's `serving.stall_guard` is the EFFECTIVE setting: `enabled`, the four
  numbers (null when off), `env` (the exact variable value), `basis`
  (`default` or `manifest`) and `note`. A document written back
  (`voice_document`) carries the key only when the manifest stated it. Repo
  voices take serving from `[tts.<engine>]`, which has no stall guard, so they
  get the default; an override manifest is how one differs.

  What the variable does (the contract both arms implement: the sglang-omni
  env patch on cuda-linux, narrator's own MLX loop on mlx-darwin):

  - `HIGGS_STALL_GUARD = "off" | "<frames>,<rate>,<max>,<window>"`, read once
    at startup. Unset is off. Anything else is a startup error naming the
    variable.
  - Per row: a ring R of the last `window` sampled cb0 codes and a counter
    `run`. A row is **counted** on a frame when it is active, past the delay
    window, not in EOC wind-down and not finishing on that frame.
  - Before temperature/top-k/top-p (so greedy is covered), a counted row with
    `run > frames` has `min(max, rate × (run − frames))` subtracted from the cb0
    logit of every distinct code in R. Nothing is forced.
  - After sampling cb0 code c on a counted frame: `run += 1` if c is in R, else
    `run = 0`; then c is pushed into R. **On a frame a row is not counted, run is
    zeroed and R emptied** (the prototype's rule), so a wind-down never carries
    a penalty and a new row starts clean.
  - The ring makes a loop hopping among a few silence codes one run. The
    prototype (`"37,0.5,20"`, campaign `2026-10-02-stall-guard`) reset on any
    code change, which let such a loop run 4.7 s.

- `mem_fraction` and `context_length` go to **both arms** (Owen, 2026-09-19:
  *"we're going to want to configure darwin to work the same way"*). narrator
  refuses a knob its backend lacks. A voice from a repo gets all three
  levers from `[tts.<engine>]`.

### Takes

A job sends `take: N`. The server decides what take N means.

- Take 0 may not deviate from the voice's sampling, and a deviating rung
  requires its reason.
- Rungs with no numbers are allowed (2026-09-19). narrator seeds
  `base + index + REROLL_SEED_STRIDE × (TAKE_REROLL_LANES × take + attempt)`,
  so a rung without numbers is a different draw at identical sampling.
  Screening sweeps need pure seed lanes.
- **At or past the end of the declared ladder**, a take uses the voice's own
  (take 0) sampling in its own seed lane. It is not clamped to the last rung.
  A negative take is refused.
- `applied_sampling` records the **full** triple a render used. The wire sends
  only the rung's declared keys.

### Overlay writes

`write_home_voice` parses the document with `parse_document` at its target path,
round-trips it through `tomli_w`, and writes it atomically. `voice_document` is
`parse_document`'s inverse and names what an override cannot carry (`pace_basis`,
`measured_from`, `inherited_from`, per-arm `max_chars_basis`). `voicerepo.merge` keeps
`measured_from` on the manifest for exactly this: it is a provenance fact, and a repo voice
saved back as a home voice once lost it with no line in `not_carried`. Voice ids must match
`^[a-z0-9][a-z0-9._-]{0,63}$`, so a request id can never become a path.

## Engine base rows (`engines/higgs-v3/base.toml`)

- `higgs-default` is `kind = "token"` (narrator calls it `default`). It is the
  base model's built-in speaker, used as a smoke voice to prove a new machine
  can start narrator. It is **not a production voice**: Owen ruled on
  2026-09-04 that production narration uses fine-tuned voices only.
- `zeroshot` is `clips = "from-request"`. One id covers every client clip, and
  the job carries the clip. `weights_of` points it at `higgs-default`'s
  download.
- Neither states a pace (see above). The zero-shot packing is one target:
  600 characters, `placeholder`. 900 drops the tail reproducibly. A zero-shot
  cap and a fine-tune cap are never inherited from each other: a fine-tune's
  stop length tracks its training clip length.
- SGLang-Omni **does** serve reference clones. The clip rides in the request
  body as base64 (`references[].data`), so the server needs no media path. A
  30 s clip is ~760 of the 4,096 positions.
- The base weights are licensed by Boson AI for research and non-commercial
  use, and so is every fine-tune of them.

## The narrator engine (`engines/narrator`)

- The wire is newline-delimited JSON on stdin/stdout. Readiness is narrator's
  `ready` line. There is no `base_url`. stdout carries protocol only, and a
  non-JSON line is a refusal, never skipped. stderr goes to the engine log.
  The pipes use UTF-8 with `PYTHONUNBUFFERED`.
- Rows retire **out of order**. `converse` yields in arrival order, and
  consumers key on the echoed `i`.
- The environment and its required variables are resolved at **construction**,
  with no defaults, so nothing starts, reads 8.5 GB and then exits 3.
  - Served arm (a recipe with a serving stack): `HIGGS_STACK` (from the env
    recipe), `HIGGS_SGL_ENV`, `HIGGS_MAX_NUM_SEQS`. `HIGGS_SGL_ENV` is the tts
    env itself, derived from the interpreter **without resolving the venv
    symlink** and confirmed by `bin/sgl-omni`, `pyvenv.cfg` or `conda-meta/`.
    Leaving it unset sends the launcher to a hard-coded conda path.
  - MLX arm: `NARRATOR_HIGGS3_MLX_BATCH`, `..._MEM_BUDGET_GB` and
    `HIGGS_MLX_CACHE_LIMIT_GB`, all from one `MLX_TIERS` row chosen by total
    memory. narrator computes headroom as `budget − weights − cache`, so all
    three must come from the same row. The width is a ceiling that narrator
    narrows. The rows are transcribed from BookForge:

    | total memory | tier | width | budget | cache |
    |---|---|---|---|---|
    | ≥ 60 GiB | extreme | 64 | 42 GB | 8 GB |
    | ≥ 44 GiB | fast | 72 | 34 GB | 8 GB |
    | ≥ 28 GiB | moderate | 48 | 22 GB | 6 GB |
    | < 28 GiB | light | 24 | 13 GB | 3 GB |

    `fast` really is wider than `extreme` (extreme was cut 96 → 64 and never
    re-measured). At width 1 the Mac rendered 7× slower. `NARRATOR_HIGGS3_MLX_BATCH`
    in the environment overrides the width.
  - Both arms: `NARRATOR_HIGGS_VOICES` (the document path) and
    `HIGGS_STALL_GUARD` (from `serving.stall_guard`; a higgs-v3 engine without
    it, or with a malformed one, is refused at construction). Variables an arm
    does not read are refused rather than silently set.
  - narrator does not export `HIGGS_STALL_GUARD` itself on the served arm: the
    sglang server it launches inherits narrator's environment
    (`sgl_served.start` → `subprocess.Popen` with no `env=`, then
    `setsid bash serve_higgs_sgl.sh` → `exec python sgl_omni_entry.py`), and
    the patched sampler reads it at import inside the server.
- **Start, served arm**: before narrator starts, `NarratorEngine.start`
  applies the tts env's sglang-omni stall-guard patch if it is missing or stale
  (`envpatches.ensure_applied`) and refuses `tts_env_unpatched` if it cannot.
  An unpatched server would accept the variable and render with no guard.
- **Cancel** sends `{"action": "cancel"}` and keeps reading until the terminal
  message. `CANCEL_GRACE_SECONDS` (120) starts at the cancel and is **not**
  reset by output. A narrator that keeps rendering past it raises
  `EngineWouldNotStop` (measured: 11 minutes of rows after a cancel).
- **Stop** sends `quit` on stdin and waits `QUIT_GRACE_SECONDS` (210 s) before
  SIGTERM, never SIGKILL.
  `detach` closes stdin first and closes stdout only after the reader thread
  finishes: closing a pipe another thread is reading deadlocks.
- `load` sends only the voice id and `warm` (`modelDir` is refused by
  narrator). The directory must agree with the document's. No `caps` are
  sent, because that channel raises on keys it does not know. Sampling
  travels in the document.
- `itemTake: true` on `ready` means this narrator build reads per-item
  `sampling` and `take`. Without it a rung is silently dropped (measured
  2026-09-15: take 0 and take 1 were byte-identical).

## The narrator voices document (`narratorvoices`)

Its refusals are `NarratorVoicesError`, an `errors.EngineError`, so the tts
job reports them as `engine_failed` without `narratorvoices` importing the
engines package. `engines.base` re-exports `EngineError` for its callers.

A Higgs v3 voice is a **name** in the JSON file `NARRATOR_HIGGS_VOICES`
points to, never a directory on the `load` message. The document is written
at every load and holds one voice, the one being loaded. The keys are
narrator's camelCase:

| key | from |
|---|---|
| `kind` | `checkpoint`, `default` (token) or `clips` (zeroshot) |
| `checkpointDir` | the pulled directory. For `clips` this is the base weights; writing `checkpoint` there would make narrator ask for a `generation_config.json` the base lacks |
| `clips` | the one reference `[{path, transcript, seconds}]` (vllm-omni takes exactly one) |
| `maxChars` | only when the manifest measured one |
| `targetChars`, `safeMinChars`, `safeMaxChars` | when declared |
| `paceCharsPerSec`, `maxCharsPerSec`, `minCharsPerSec` | all three or none |
| `sampling` | `temperature`, `topP`, `topK` (`topK` an int; narrator refuses `50.0`) |

- Not written: `maxCharsSource`, `scene`, `allowedControls`,
  `maxReferenceSeconds` and `_overrideNote`.
- **`token` on `cuda-linux` is refused.** The served arm then serves "the base
  snapshot out of the HF cache", which is not the pinned bytes. So
  `higgs-default` loads only on `mlx-darwin` (`NARRATOR_HIGGS3_MLX_MODEL` is
  set to the pulled base). Ruling owed. A lead not yet acted on: write the
  base directory as a `default` voice's `checkpointDir`.
- `take_sampling` returns **only the keys a rung declares**, and `None` (not
  `{}`, which narrator refuses) at take 0 or past the ladder's end.

## Reference clips (`voicereference`)

- Wire shape: `{"data": <base64 RIFF/WAVE>, "transcript", "name"}`. Crucible
  writes the file (narrator requires a path on the server's disk) next to the
  document at every load.
- `seconds` is **measured from the WAV header**, never taken from the client.
- The transcript is **required** and must be the book-exact text. A clone
  conditioned on a wrong transcript renders a whole book in a subtly wrong
  voice and reports success.
- Limits: 30 s total (narrator's `MAX_REFERENCE_SECONDS`) and 32 MiB decoded,
  with the encoded length checked first. base64 is decoded with
  `validate=True`.
- `sha256` is computed over the decoded audio, and residency reports its first
  12 hex digits plus the name. The clip itself is never published.

## Cards and export (`voicecard`)

- The card writer owns a fixed list of frontmatter keys. All other lines
  survive in their original order. An arm or pace with nothing to say writes
  **no line** (audit scripts read `null` as a value).
- `## Measured limits` is rendered only from manifest fields, with
  `measured_from` reproduced verbatim. It is inserted before the first other
  section when it is missing.
- `export` writes `[voice.chunk_gap]` and each arm's `edge_fade_ms`,
  `reference_seconds_cap` and `allowed_controls` when the voice states them.
- `export` converts a packaged manifest to `crucible-voice.toml` and returns
  the machine rows it drops. It **asks** for `pace` and `max_chars` bases and
  never guesses them. A voice with `edges = "percentile"` cannot be exported,
  because the repo schema cannot state it.
