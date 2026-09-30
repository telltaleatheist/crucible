# Jobs runtime internals

How the job lane, its records, the card's holders and the worker job types behave, and why.
Covers `crucible/jobtypes.py` (the job-type catalog), `crucible/jobs/` (registry,
`registry_table`, `binding`, `template`, `unload`, `leaseonload`, `base`, `queue`, `worker_type`,
`workerio`, and the `tts`, `rvc`, `denoise`, `llm`, `echo` types), `journal`, `workers`,
`installonsubmit`, `inflight`,
`leases`, `settle`, `ttsstream`, `rvcbase`, `rvcmodels`, `denoisemodels`, and the catalog files
`crucible/rvc/*.toml`, `crucible/rvcbase/*.toml`, `crucible/denoise/*.toml`.

The resume journal's wire contract is [docs/RESUMABLE-JOBS.md](../RESUMABLE-JOBS.md). Only the
implementation rules are repeated here. The `image` type (the fifth resident kind, `generator`)
has its own page: [image.md](image.md).

## 1. Standing rulings

- **Admission, not queueing.** Owen, 2026-09-13: *"i think all queuing logic should exist in the
  clients, not the server. if the server is busy, it cant receive a new job. if its not busy, it
  receives the next job requested."* Owen, 2026-09-27: *"Crucible isn't responsible for queuing.
  The apps that use it are. It grants and releases leases. That's it."* Crucible grants and
  releases leases. It never queues, never retries, and never resumes anything by itself.
- **Unload when done.** Owen, 2026-09-14: *"Models should always be unloaded when we're done
  with them. Every time."* This overrules the earlier "no idle unload" design. The card is
  not storage (see section 6).
- **Idiot proof.** A missing env or model is installed on submit (section 8). A 12-hour master
  can go to `rvc` whole, and input names need no extension (section 10.3).
- **Keep work on disk.** Owen, 2026-09-27: *"we should definitely be writing work to disk, so
  if something fails, we dont lose everything."* See the journal (section 4).
- **Retention.** Owen, 2026-09-18 and 2026-09-25: a job whose artifacts have all been fetched
  is deleted. *"keep all working files on the crucible side until the chain is complete. then
  remove them"*, and *"a garbage collector clean up files older than 7 days"*.
- **A cleanup failure is not an operation failure.** A reaper, a settlement or a record write
  that fails is logged loudly. It never turns finished work into an error the client sees, and
  it never kills the lane.
- **Null means "did not say".** `Job.client`, a session holder, an act, `capped`, `tokens`,
  `guard` and `gap_sec` are `null` when nobody stated them. The server never invents them, and a
  reader must never take `null` to mean `false`.

## 2. The registry (`crucible/jobs/__init__.py`)

### One catalog of job types (`crucible/jobtypes.py`, `crucible/jobs/registry_table.py`)

Every job type is one frozen `JobTypeSpec` in `crucible/jobtypes.py`: its public name, its
`Family`, its `CardEffect` (what it does to the card), whether it leaves its subject resident
(`leaves_it_resident`, the loads), and whether it keeps a resume journal (`journal_identity`).
A `Family` is the `[jobs] enable_<name>` flag and `CapabilityClass.job_type`: its `Env` (the
installer; `worker` says whether the env runs a `crucible/jobs/*/worker.py`), the capability
class names it covers (checked against `crucible/classnames.py` when the module loads, and
against `capability.CLASSES` when `crucible.jobs` loads), the refusal codes a pull fixes, the
base subjects it needs, and whether the catalog lists every model it can serve. Whether a type
is an unload, and of what, is `spec.unloads` (the card effect's `takes_off`).

Each job package exports `JOB_TYPES`: a `JobTypeBinding(spec, build)` per type it implements,
where `build(Wiring)` makes the type from the config, backend, the shared `Residency` and the
lease register. `crucible/jobs/registry_table.py` lists every package's bindings in one tuple
and refuses to load unless they cover `JOB_TYPE_SPECS` exactly, in order. `build_registry` is
that table filtered by the family flags; it also refuses a type whose `name` is not its spec's
or whose `journal_identity` disagrees with its spec.

Every former per-type table is a derivation, under its old name:

| Old name | Derived from |
| --- | --- |
| `jobs.ALL_JOB_TYPES`, `jobs.CAPABILITIES` | spec name to family name |
| `build_registry`'s `enable_*` mapping | `family.flag` |
| `leases.CARD_EFFECTS` | `spec.card` |
| `settle.LEAVES_IT_RESIDENT` | `spec.leaves_it_resident` |
| `installonsubmit.BASE_SUBJECTS` | `family.base_subject_kinds` (ids from `catalog`) |
| `installonsubmit.PULLABLE_REFUSALS`, `INSTALLABLE_REFUSALS` | `family.pullable_refusals` |
| `installonsubmit.CATALOG_IS_COMPLETE` | `family.catalog_is_complete` |
| `InstallOnSubmit.installable` (was `startswith("unload-")`) | `spec.installable` |
| `jobenv.WORKER_JOB_TYPES`, `JOB_TYPES_SERVED_BY_ENV`, `INSTALLABLE_JOB_TYPES`, `INSTALLER_FOR` | `ENVS` and each family's `env` |

`WORKER_HEADLINE_PACKAGE` and `SMOKE_IMPORT` stay in `jobenv`: they are facts about envs and
backends, not job types. `INSTALLER_FOR["pages"]` stays a literal there, because `pages` is a
capability class, not a job family.

The catalog is a leaf outside `crucible/jobs/` on purpose. `crucible.leases` must not import
`crucible.residency` (`tests/test_layering.py`), and `crucible.jobenv` is imported by
`residency`. Importing anything under `crucible/jobs/` runs `crucible/jobs/__init__.py`,
which imports `residency` and every job package. A table there could be read by neither.

### The job template (`crucible/jobs/template.py`, `unload.py`, `leaseonload.py`)

- `ManifestCatalog[T]` is a manifest loader plus the four questions every type asked of it:
  `all()` (the loader's error becomes `500 <kind>_manifests_unreadable`), `known(id)` (`400
  unknown_model`, or `unknown_voice`), `memory_estimate`, `provenance` (`{id, revision,
  fingerprint}`) and `descriptors`. Each type passes a lambda, not the loader itself, so a test
  that patches the loader on the type's module still reaches it.
- `parse_params(model, params, job_type, lead=None)` is the one `400 invalid_params`
  sentence: `"<job_type> params are not valid: <loc>: <msg>; ..."`, or `"<lead>: ..."` where a
  type has more to say (denoise explains why it takes no params).
- Each type has a `requirements(model, params)` step: everything preflight refuses, ending in
  the accelerator guard (`card_guard`). `preflight` calls it directly; `run` calls it through
  `as_job_error(...)`, which turns the same `ApiError` into a `JobError` with the same code and
  message. So each type calls the guard once in its source, not once per door. Types whose
  run-time guard happens later than the preflight check (the tts render, align and denoise,
  which load inside a claim or a session) keep the guard in one `_guard` method both sides call.
- `ResidentWorker` is the `_session`/`_forget` pair align and denoise shared: reuse the resident
  session when its process is alive, forget a dead one, guard, load, and name the job module
  as the bug if its occupant carried no session.
- `leaseonload.LeaseOnLoad` and `open_lease_for_load` are the `params.lease` of `load-model`
  and `load-voice`; the job packages import them from there.
- `UnloadJobType(spec, residency, describe=..., provenance=...)` is all four unload types.
  The kind comes from `spec.unloads`, the noun from `cardkinds.KIND_NOUNS`, and the codes stay
  `<noun>_not_resident` (`model`, `voice`, `aligner`, `separator`, since 2026-09-28
  `generator` for `unload-image`, since 2026-09-29 `audio_generator` for `unload-audio`: a
  space in the noun becomes an underscore in the code, and `segmenter` for `unload-segment`). The four copies had drifted;
  the rulings (2026-09-27):
  - progress is reported only after `await_clearance` returns (llm reported `unloading` before
    the wait, so a wait that then failed left a 0 % progress line on a job that never started);
  - `invalid_params` has the one `parse_params` format (unload-denoiser said
    `"<name> takes no params: <pydantic text>"`);
  - every type catches a `KeyError` from `Residency.unload` (the subject went between the check
    and the unload) as `<noun>_not_resident`, and names a failed stop by what failed:
    `engine_failed` for an engine, `worker_failed` for a worker session (llm only knew the
    first, align and denoise only the second);
  - `check()` names the resident of its own kind (`resident: <id>` or `no <noun> is resident`);
    unload-model used to name whatever held the card.

- `resolve` tells "that type does not exist" apart from "it exists but is off".
  `disabled_error` builds the one `400 job_type_disabled` sentence for all four doors that
  refuse (`resolve`, `/v1/models`, `/v1/voices`, the streaming door). It takes a job type or a
  capability name. `details.reason` is one of:
  - `undecided`: no `[capability]` record, so the client is sent to the probe.
  - `cannot_hold`: every class behind the flag is recorded disabled. Flipping the flag is named
    as the thing NOT to do, because the flag's next effect would be an OOM mid-book.
  - `not_installed`: the card can hold it. `details.install` is the `POST /v1/tasks` body that
    installs it. It names the INSTALLER, which is not always the type: `denoise` is built by
    installing `rvc`. No CLI command is ever shown.
  - `not_taken_up`: the flag is on but the running server did not take the type up
    (`api.take_up_enabled_types`).
- `build_registry` hands every card-touching type the SAME `Residency` holder. That shared holder
  makes "loading a voice unloads a model" true across the four resident kinds: model, voice,
  aligner and separator. `crucible doctor` passes none and gets an empty holder. It cannot see
  another process's resident engine.
  - `asr` and `rvc` get only the owned-pid set. Nothing of theirs is ever resident.
  - `align-longform` gets no holder. It drives the asr and align workers directly and starts
    and stops its own aligner session. Nested jobs would deadlock, because a job waiting on a
    job waits on the lane it holds.
  - `denoise` shares `rvc`'s env but not its flag. A host can have the env and no separator.
  - `align-longform` shares `align`'s flag and needs `asr`'s env too. Its `check()` names which
    half is missing.
- `_assert_every_type_implements_the_protocol` refuses, at build time, any type missing a
  `JobType` member. Members come from the Protocol's class body, because
  `__protocol_attrs__` only exists from 3.12 and 3.11 is supported. Background: a type written
  against a six-method `JobType` once met a seventh added in another branch. The resulting
  `AttributeError` happened inside `_finish`, so no terminal event was sent and client streams
  hung forever.
  - `OPTIONAL_JOB_TYPE_MEMBERS` (`journal_identity`) are declared on the Protocol with a
    default of `None` and left out of that check. `journal_identity` is `None` on every type
    that keeps no resume journal and a method on the one that does (`asr`). `crucible/inputs.py`
    reads it with `getattr(..., None)`, so a type without it refuses `resume` with
    `resume_unsupported`.
- Every job type must map to a capability class, or a refusal would have nothing to name.
- The registry's installer facts (`INSTALLABLE_JOB_TYPES`, `INSTALLER_FOR`, `SMOKE_IMPORT`,
  `no_installer`) live in `crucible/jobenv.py`, derived from `WORKER_JOB_TYPES`,
  `JOB_TYPES_SERVED_BY_ENV` and the headline packages. The server side (`jobs`, `tasks`,
  `installonsubmit`, `api/routes/capability.py`) imports them from there and never imports
  `crucible.cli`; `tests/test_layering.py` holds that line.

### Layering leaves

These modules import nothing heavier than `backend`, `config` or `errors`, so the lane,
leases, the journal and the job types can use them without pulling in residency, the
ladder or the CLI:

- `crucible/clock.py`: `now()` (an aware UTC `datetime`), `utcnow()` (its ISO string) and
  `utcnow_to_the_second()` (the ladder's record stamps). The lane, the journal and the leases
  call `clock.now()` through the module, so a test that moves time patches `clock.now`.
- `crucible/cardkinds.py`: the four resident kinds (`KIND_*`) and `KIND_NOUNS`. `leases`
  and `residency` read them from here, and so does every caller that names a kind.
- `crucible/cardfacts.py`: the ladder record reader (`record_path`, `load_record`,
  `stale_reason`, `card_for`, the rung and outcome names, `RungResult`). The job types read
  `card_for` from here, as do the CLI and the routes; `ladder` keeps the measuring.
- `crucible/capabilitystore.py`: `decide_for(config, backend)` (decisions on the card's
  total bytes, the config's reserve and local models, and `card_for`),
  `decide_on(...)` for callers deciding on recorded numbers (`settings`), `record_of(...)`
  and `write_capability(config, backend, decisions, flags)`. The CLI, `api/app.py`'s
  first-request decision, `installonsubmit.live_decisions` and `settings` all go through it.

## 3. The lane and the job record (`crucible/jobs/queue.py`, `base.py`)

### Admission

- The lane is a single slot, not a queue: `LaneSlot` (`JobStore._admitted`) holds at most the
  one admitted job id, and `admit` refuses a second. `queue_depth` is 0 or 1, `position()` is
  0 (running), 1 (admitted) or null, and `queued()` is empty or that one job; the names and the
  `queued` event's `position` field stay because clients read them. `JobStore.admitted` is
  that slot, for code and tests that must see or hold it.
- `refuse_if_busy` answers "is there room now" and nothing else. It reads the admitted slot as
  well as `_running_id`. `enqueue` admits and wakes the lane, but the lane task cannot run until
  the current handler yields. A check of `_running_id` alone would therefore admit a second job
  in that window. The check and the admit run in one synchronous stretch on the event loop, and
  `enqueue` repeats the check because it is where the admit happens.
- The refusal is `409 server_busy` with facts (`busy_details`): holder, job, state, `since`
  (`started` for a running job, `created` for an admitted one) and the latest progress line.
  A bare "busy" makes clients poll, and polling rewards luck rather than who asked first.
  `Job.busy()` builds them as a `BusyHolder` and `Job.busy_details()` is its `to_dict()`;
  it is also what
  `POST /v1/tasks` reads through `Settlement.holder()`, which is why `settle` needs no import of
  the queue.
- `position`, `queue_depth` and cancel are unchanged. Under the admission rule they only take
  the values 0, 1 or null.
- `discard` forgets a created job that was never admitted. A leftover record would sit at
  `queued` forever, with no position.
- `_lane_lock` guards the admitted slot and `_running_id` together, because the settlement reads the
  lane from a worker thread. The move from queued to running happens in one step under that
  lock. Without it, there was an instant (while the record was being persisted) when a job
  admitted before a clearance was on neither side, so the card could be cleared under it.
- Uploads are MOVED into the job that names them. A blob is consumed once. A second job naming
  it is told which job took it (`refuse_if_blob_consumed`), not `unknown_blob`. The record lasts
  for the life of the process, including for discarded jobs.

### Running

- `run()` executes on a worker thread. `JobContext` marshals every mutation back to the loop with
  `call_soon_threadsafe`, which keeps events in emission order.
- The lane is one coroutine. Nothing the queue's own bookkeeping raises may escape it: an escape
  kills the lane, and every later job would sit at `queued` while the server kept answering 202.
  `_fail_out_of_band` marks the job failed without going back through `_finish`, and survives
  even failing to append the event. It still persists the record, so a restart reads it back as
  `failed` and not `interrupted`.
- `_execute` settles the card (section 6) BEFORE the terminal event, while the job still holds
  the lane. That way the settlement's `note` lands on a stream the client is still reading, and
  `refuse_if_busy` still refuses anyone who would race the unload.
- `JobContext.progress(extra=...)` carries a type's own measurements (asr sends processed and
  total seconds). `warming` is for loads, where no fraction is honest. `note` is a fact about
  the machine, not the work. `cue` is one unit of the answer as it lands, so a killed run keeps
  what it sent. `chunk` carries Crucible's measurements plus `guard` verbatim (section 10.1).
- Artifacts are published with a provenance sidecar written immediately. `index=` marks an
  artifact as chunk N, so a resume is a set difference on `chunks_done` rather than filename
  parsing. `expect_chunks` sets the denominator `chunks_total`, and `last_chunk_at` gives a
  pace from two reads of the record.

### Shapes

- `Job.failure` is a `JobFailure(code, message)`; `Job.error` is its `to_dict()`, the
  `{"code", "message"}` the record, the `failed` event and `GET /v1/jobs/{id}` carry.
- `hold`/`hold_record` build a `HoldRecord` and answer its `to_dict()`.
- `Reaped.why` is a `ReapReason` (`fetched`, `aged`, `released`), a `str` enum whose values are
  the strings `job_reaped` has always carried.
- The resume note reads the journal manifest through `JournalProgress.of(manifest)`;
  `crucible/journal.py` still hands out the manifest as a dict.
- `SECONDS_PER_DAY` names the retention arithmetic.

### Durability and restart

- `_persist` writes `job.json` beside the artifacts, whole-file-and-replace, at the moments that
  change what a recovery would say: created (before the 202 answers), started, each artifact,
  and the terminal end. It does not write on progress. It never raises.
- The record deliberately omits `params`. A render's params hold a chapter of somebody's book,
  and the client keeps its own text. This was agreed with BookForge on 2026-09-20.
- `restore` runs once at startup, before the API answers. A job found `running` or `queued` comes
  back as **`interrupted`**, which is distinct from `failed`. `failed` is the server judging the
  work. `interrupted` is weather: the client should collect what landed and re-ask for the rest.
  A recovered job is never re-queued or resumed. Continuing it is a NEW job from the client
  (`chunks_done`, or `params.resume` for journaled types). Events are not restored.
- An interrupted job's `finished` is its `interrupted_at` (when the restart noticed it), and the
  coerced record is written back at once. Without a `finished` stamp the reaper cannot age the
  job, and without the write-back every restart would restart its retention clock.
- A clean stop ends the running job the same way, before anything else stops.
  `JobStore.stop()` shuts the lane to new work, sets the running job's cancel flag (the path
  a `DELETE /v1/jobs/{id}` takes, so the plugin asks its worker to stop cooperatively), and
  waits for the job to finish for up to `procgroup.stop_budget_seconds(STOP_TIMEOUT_SECONDS)`.
  Only then is the lane task cancelled, and the lifespan stops tasks, the job store, streams and
  residency in that order. The job ends **`interrupted`** (the client did not cancel it), with
  a note saying so. Before this, cancelling the lane task left the job's thread running: after
  `residency.shutdown()` it could start a worker nothing stopped, a CUDA process outliving the
  server. A job past the budget is named on stderr and the server stops anyway; never SIGKILL.
- `client_ref` is the client's own name for the work, echoed back and never read. BookForge puts
  its queue step id there so it can match interrupted jobs after both sides restart.

### Provenance sidecars

- The `model` block comes from the job type (`model_provenance`), because the queue only knows
  an id. It is `{id, revision, fingerprint}`, or null for a model-less type.
- `_params_for_artifact`: for an INDEXED artifact, each list of `{index, ...}` items is cut down
  to this artifact's item. Job-wide keys stay, and `params_sha256` names the full request.
  Before this, sidecars grew O(chunks²): 2.4 GB of sidecars per 1.2 GB of audio. Non-chunk
  artifacts keep the whole params.
- A sidecar carries `{"artifact": {job_id, name}}` so it can be cited from itself.

## 4. Retention and the reaper

- The reaper ticks inside the lane coroutine while the lane is idle: once on the way into the
  first wait, then every `REAP_INTERVAL_SECONDS` (60 s). It never competes with a render for the
  disk. The same tick settles a lapsed lease (section 5).
- A job is reaped for one of two reasons, and the reason is recorded:
  - **fetched**: `Job.collected()`, meaning every artifact AND every sidecar has been GET.
    Sidecars count because the SDK fetches the artifact and its sidecar in parallel. A job with
    an EMPTY artifact list is never "collected": that case is left to the age rule, otherwise a
    load job would be reaped the instant it ended.
  - **aged**: finished more than `retention_days` ago.
- Queued and running jobs are never touched, however old.
- **Holds.** `hold` keeps a job for a chain at any status, except a job that has ended with
  nothing published. A held job is never reaped for being fetched. It is still aged out.
  `release` reaps at once. Holds are persisted and survive a restart.
- **Orphan directories** under `jobs/` that no live record owns belong to dead processes and are
  reaped by age only, never immediately. A second Crucible sharing the home (on another port)
  may be writing into them.
- A reaped job leaves a `Reaped` tombstone so its id answers `job_reaped` (with when and why)
  instead of `unknown_job`. Tombstones are not capped: each one is smaller than the `Job` it
  replaces. If a directory cannot be deleted (on Windows, typically a file still being
  streamed), the record is kept and the next tick retries.
- `_age_seconds` raises for a terminal job with no `finished`. Guessing either way would keep the
  directory forever or delete it at once.
- Each job is reaped in isolation: a record the reaper cannot judge is logged ("the reaper
  skipped job ...") and left, and the tick goes on to the other jobs, the orphan directories and
  the journals. One bad record once stopped all reaping for good.
- `rvc` stages its pieces inside the job directory, so a killed worker's scratch is the reaper's
  to collect and never lands in the system temp.

## 5. The journal (`crucible/journal.py`)

Contract: [RESUMABLE-JOBS.md](../RESUMABLE-JOBS.md). Implementation rules:

- Journals live at `<home>/journals/<resume_id>/`, not under `jobs/`, because the reaper deletes
  job directories and the journal must outlive every job that writes it. Layout:
  `manifest.json`, `units/<key>.json` (plus `<key>.bin` for byte units), and
  `journals/_gone/<id>.json` tombstones so an expired id is refused as expired, not unknown.
- **Atomic writes, always** (`write_atomically`): write to a dot-named temporary file in the
  same directory, flush and fsync, `os.replace`, then fsync the directory where possible
  (Windows cannot, and NTFS journals its metadata anyway). Readers only open `<key>.json` names,
  so a crash leaves nothing that reads as finished. On Windows, a replace that hits a
  reader-held target is retried (`REPLACE_ATTEMPTS`).
- `put_file` writes and replaces the bytes (`<key>.bin`) first, and only then the `<key>.json`
  that marks the unit finished.
- An unreadable unit is logged and treated as not done: the work is redone, never stitched from
  a bad file.
- The manifest has one writer at a time. Snapshot and replace happen under a lock, because both
  the job thread and the lane write it. Progress-only manifest rewrites are throttled to one per
  `PROGRESS_WRITE_SECONDS`. Units are written the moment they land.
- `verify` refuses a resume, naming the first difference, checked in this order: job type,
  model/revision, format version, each input's sha256, each output-affecting param.
- Writer state: a writer recorded `queued` or `running` whose job the lane is not running is
  reported `interrupted`.
- Removal writes the tombstone FIRST. A crash in between leaves a journal that is still read
  and gets removed again next tick.
- `Journals.reap` runs on the same tick and the same `retention_days` as job directories. It
  measures from the last save, and never takes a journal whose writer is live.
- `_journal_started` puts `resumed: N of M ... done` on a resumed job's stream before the type
  skips anything.

## 6. Holders of the card: leases, chats, settlement

### Four facts, no timer (`crucible/settle.py`)

The card is cleared the moment the last of these goes false:

1. **the lane**: no job running or admitted;
2. **the lease**: no open lease;
3. **the claim**: no streaming session holds narrator's wire;
4. **the chats**: no chat completion in flight.

There is no keep-warm window and no config key, because an idle interval is a guess.
`Residency.warming` is not a fifth fact, since it is a sub-state of the lane. A client that
does not lease pays a reload between chats or chapters. That is the price of not stating an
intention, and the fix is a lease on the client side, never an exception here.

- `Settlement` is the ONE place that decides. Five moments call it: a job ends, a lease is
  released, a lease lapses, a streaming session closes, the last chat returns.
- **Loads are exempt only when they ended `done`** (`LEAVES_IT_RESIDENT` plus outcome). A
  cancelled or failed load settles like any other job. The outcome is passed in because the
  lane settles before `_finish`, while `job.status` still reads `running`. `tts` and `align`
  are NOT exempt, because a render that left its voice resident would strand it. What holds a
  voice across chapters is a lease.
- A load that succeeded leaves the card resident and held by nothing. `unheld_since()` makes
  that visible without deciding anything about it.
- Settlement never runs on the event loop: `SubprocessEngine.stop()` can wait 180 s. It takes a
  claim with `claim_to_clear`, which reads the four facts and claims in ONE step under the lock
  every client door records its hold under. Either a door's hold is seen and nothing is
  claimed, or the door sees the clearance and waits for it.
- **Every door waits out a clearance** within the resident engine's own stop budget plus
  `CLEARANCE_MARGIN_SECONDS` (`Residency.clearance_timeout`: `stop_budget_seconds` when the
  engine states one, else `CLEARANCE_TIMEOUT_SECONDS`), then answers from the settled card. A
  narrator stop can legitimately take far longer than a plain SIGTERM, so one fixed budget
  called a normal narrator stop a wedge. A clearance past the budget is a wedge and is refused
  as `engine_in_use`.
  An `unload-*` of the very subject being cleared is admitted and ends `done` (`clears=True`,
  `being_cleared`).
- The id to unload is read under the claim. `Residency.unload` raises `KeyError` for a stale id
  rather than unloading the wrong engine.
- Every clearance is logged, because a chat-triggered or lease-triggered unload has no job to
  carry an event.
- `held_by` and `unheld_since` do not take the settlement lock. A bench read must never queue
  behind a 180 s unload.
- Streaming door gap: nothing holds the card between `load-voice` finishing and
  `POST /v1/tts/stream` opening. `params.lease` on the load closes that gap.

### Leases (`crucible/leases.py`)

- A lease is a refusal, not a reservation. It admits nothing and reserves no lane. It says only
  "nothing may take the leased thing off the card". It does not gate chats, because chats are
  what it protects. `accepts_work` is untouched.
- There is one lease per server, because there is one card. A second lease is refused
  `409 leased`, the same code and details a loader gets.
- A lease names the RESIDENT THING, of any kind. The server reads the kind off
  `Residency.resident` at open time. The client never sends a kind, because there is only ever
  one candidate.
- `CARD_EFFECTS` is what each job type does to the card (`makes_resident`,
  `reuses_what_it_names`, `takes_off`), derived from each `JobTypeSpec.card` in
  `crucible/jobtypes.py` (section 2). `Lease.evicted_by` derives every (lease, job) answer from it.
  Under a voice lease, a `tts` render of the leased voice is ADMITTED (it reuses the voice). A
  render under a model lease is refused. A type with no row, asked for while a lease is open,
  is refused `lease_scope_unknown`; the tests require every type to have a row.
- **Expiry is read, never swept.** A lease past `expires_at` is simply not open. Heartbeats push
  it out by the ttl it was opened with. The ttl range is `MIN_TTL_SECONDS`–`MAX_TTL_SECONDS`
  (30 s to 1 h). Below that range a lease expires between heartbeats. Above it, a lease would
  outlive its holder's crash.
- A lapse fires nothing by itself. `Settlement` arms a one-shot timer at the lease's own
  `expires_at`, re-armed on open, heartbeat and release. The idle reaper tick also checks. Each
  lapse is evaluated ONCE (`forget_lapse`, called in a `finally`). If a lapse stayed on offer, a
  later unleased load would read it as a holder letting go and be unloaded.
- Leases live in memory only, and a restart forgets them. A restarted server holds nothing to
  protect.
- `params.lease` on `load-model` / `load-voice` holds what the load made resident from the
  instant it exists. It goes through the same validators as the lease door. It is opened AFTER
  the load's last cancel check, because a lease opened by a job that then raises
  `JobCancelled` would strand the card behind its own hold. `done.lease` is present and `null`
  when none was asked for. An absent key would mean "this server does not speak leases here".

### Chats in flight (`crucible/inflight.py`)

- A chat takes no lane and reserves nothing. vLLM batches, and serialising chats to fix a display
  would defeat it. `InFlight` is a RECORD, so `/v1/activity` stops showing a busy server as idle,
  and it is one of the settlement's four facts.
- The streamed door opens the record in the handler and closes it where the relay really ends
  (`_RelayResponse`'s `finally`). Unloading under a live token stream is the eviction this server
  refuses to do.
- The one bound: `chat_queue_full` (503) past a SERIAL engine's own measured concurrency plus
  one. mlx-lm accepts every connection and generates on one thread. vLLM states no
  `chat_concurrency` and is never refused.
- `X-Crucible-Act` must be a capability class name, or it is refused by name. A typo would put a
  false act on a bench. An absent header records `null`.
- `retry_after()` is the median of the last 20 completions, rounded up, with a floor of 1 s. It
  is `None` (header absent) until one completion has finished. A Retry-After is never invented.

### A stop that did not finish, and engines a crash left behind (`crucible/residency.py`)

- Each job package loads through `Residency.occupy`, which owns the shared preamble (refuse,
  evict, warming) and publishes what the package's `start()` returns. `residency.py` imports no
  job package and no engine module, so `residency -> jobs -> residency` is not a cycle.
  `jobs/tts/render.py` loads a voice with `occupy_voice(residency, ...)` from
  `jobs/tts/common.py`; `Residency` has no per-kind load method.

- `unload` records the stopping engine as `DyingResident` before it asks it to stop. If the stop
  raises (SIGTERM deadline passed), the record stays and every load or claim goes through
  `refuse_if_stopping`. That check first asks the process table whether any of the dying pids
  are still alive: if none are, the card is let go and the load proceeds with no restart. If
  some are, the stop is asked once more (never SIGKILL), and only then is the load refused as
  `engine_still_stopping`, naming the live pids, `kill <pids>` (never -9) and the engine's log.
- Every time the card gains or loses a resident, `Residency` writes the pids it owns (engine,
  worker session, anything still dying) with each one's command and start time to
  `<home>/run/resident.json`, whole-file-and-replace. It is deleted when nothing is owned.
- Engine children run in their own session so a Crucible crash does not take them down, which
  also means a restarted Crucible does not know them. At startup (app lifespan,
  `Residency.start_reclaiming`, on a daemon thread so the API answers at once) the record is
  read. A pid whose command and start time still match is sent SIGTERM (its own process group
  when it leads one) and waited for up to the engine's stop budget. A pid now belonging to
  another process is never signalled. A record written by a Crucible that is still running is
  left alone. A record that will not parse is moved to `resident.json.bad-<timestamp>`.
- A survivor is logged and kept as a leftover (`accelerator.note_leftover`). While it lives,
  `accelerator.guard`'s `accelerator_busy` message says which pid and command a previous Crucible
  left, when it was asked to stop, and to run `kill <pid>` (never -9), with the rows under
  `details.left_by_previous_run`. The survivor stays in `resident.json`, so the next start asks
  again.

## 7. Worker processes (`crucible/workers.py`, `crucible/jobs/workerio.py`)

Phase 4 types run a library in their own env: `<env python> crucible/jobs/<type>/worker.py`. The
request goes in on stdin and newline-delimited JSON comes back on fd 1.

- **fd 1 is results and nothing else.** A worker calls `workerio.claim_stdout()` before importing
  any library. It dups fd 1 aside and points fd 1 at stderr. On the server side, any fd 1 line
  that is not a known message kind (`ready`, `progress`, `result`, `failed`, `done`) is refused,
  quoting the line. Background: on 2026-09-05 narrator's aligner lost a 401-chunk book because a
  library printed to stdout (`json.loads` failed with "Extra data"). `workerio.send` is locked,
  because denoise sends from a heartbeat thread.
- **Results match work by position.** Workers never echo an index.
  `require_positional_results` refuses a count mismatch.
- **Never SIGKILL a worker.** Killing a CUDA holder wedges WSL2 until Windows reboots. On POSIX
  the sequence is close stdin, wait up to `STOP_ON_EOF_SECONDS`, SIGTERM the process group, and
  wait up to `STOP_TIMEOUT_SECONDS`. If the worker still will not go, the server says so. On
  win32 it is CTRL_BREAK_EVENT, then the tree is terminated. There is no console for the break
  in some cases, so the tree is ended at once. `os.killpg` does not exist on win32; before
  2026-09-23 this bug meant a cancel raised and the worker kept running.
- Workers spawn in their own process group, so a stop reaches ffmpeg or urvc children.
- `run_worker` is `start`/`send`/`stop` in a row. A one-shot worker's stdin is closed after its
  request, because a process blocked on a read it will never get is a silent hang.
  `WorkerSession` (align, denoise, image) keeps stdin open for the next request. It is not a pool, and
  the residency holder stops it. `rvc` deliberately does NOT use a session (section 10.3).
- A session request may carry a `cancel_request`: on a cancel the server writes that line to the
  worker's stdin instead of stopping it, and the worker (`workerio.serve(..., interrupts)`) stops
  between steps and stays up; after `CANCEL_GRACE_SECONDS` it is stopped the old way. Only
  `image` sends one (image.md, "Cancel between steps"). `_Reader.lines` hands control back at
  least every `POLL_SECONDS`, so a worker that reports progress faster than that is still
  checked for a cancel.
- `ready_silence_timeout` is a silence timeout, reset by every message. After `ready` there is no
  timeout at all, because an 18-hour transcription is legitimately quiet for long stretches.
  Cancel is polled every `POLL_SECONDS`.
- `WorkerSession.start` has no cancel hook. A half-loaded model is VRAM nothing tracks. Loaders
  check `ctx.cancelled` before and after the load instead.
- If a request fails to write with EPIPE, the worker usually died at import. The report gives
  the death and log tail, not the pipe error.
- Worker logs are opened for APPEND with a per-run header. The quoted log tail
  (`LOG_TAIL_LINES`) comes from the latest run only, so a previous run's failure is never shown
  as this one's.
- `worker_environment`:
  - Crucible's own tools directory goes first on PATH, so urvc and audio-separator find the
    pinned ffmpeg.
  - `cuda_library_path` PREPENDS every `nvidia/*/lib` and bundled `*.libs` directory the env
    actually has, derived at spawn time and never hard-coded. ctranslate2 resolves cuBLAS lazily
    at the first matrix multiply, so a missing loader path passes import, load and doctor, and
    fails only on the first real transcription (measured 2026-09-15).
  - `torch_allocator_environment` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for
    plain-torch workers on CUDA (the Qwen3 aligner and audio-separator), unless the operator has
    already set it. `torch_memory_cap` sends the manifest's admitted `memory_bytes_estimate`,
    which `workerio.cap_memory` applies with `set_per_process_memory_fraction`. Measured
    2026-09-26 on a 3090 Ti: with varying input lengths, reserved memory climbed from 7.7 GB to
    over 21 GB and spilled into Windows shared memory. With the setting and the cap it held at
    5.6–5.9 GB. vLLM, faster-whisper and rvc are excluded.
- The stdin loop is written once. A one-shot worker (asr, mlx asr, rvc) calls
  `workerio.read_request(label)`, which answers `failed` for no request, a line that is not
  JSON, or JSON that is not an object, and returns None so `main` exits 1. A session worker
  (align, denoise, qwen asr) calls `workerio.serve(label, OPS)`: blank lines are skipped, and the
  first unreadable request, unknown op, `KeyError` from `require` or other exception from a
  handler is answered `failed` and ends the worker with exit 1; end of input exits 0.
- A one-shot worker's `main` is `read_request`, `parse_request`, then the work. A step
  that cannot go on raises the worker's own failure (`WorkerFailed` in the asr workers,
  `RvcFailed` in rvc) and `main` answers it with `fail`, so the order of `failed`
  and cleanup stays in one place. rvc's batch loop is `Conversion` (`announce`, `run`,
  `deliver`, `finish`); asr's windows are `transcribe_window` under `transcribe`.
- `workerio` imports only the standard library at load (numpy only inside `decode`), because no
  worker env has `crucible`. Workers load it by path (`load_sibling`) and never leave
  `crucible/jobs/` on `sys.path`, where `queue.py` would shadow the stdlib `queue`.
- `workerio.decode` uses ffmpeg (`f32le`, 16 kHz mono, streamed with stderr drained on a thread),
  never PyAV, which silently truncates some assembled m4b files. `probe_duration` is only a
  progress denominator.

`worker_type.py` holds the checks and refusals shared by `align`, `asr`, `denoise` and `rvc`.
The **refusal order** for every type (llm's order) is: what no install can fix
(`backend_unsupported`, a card too small), then what an install or pull fixes (`env_missing`,
`model_not_installed`), then the live card. So a model too big for the card is refused for its
size, not for a 55 GB download it would need first.

## 8. Install on submit (`crucible/installonsubmit.py`)

Rulings (Owen, 2026-09-26 and 2026-09-27): install a missing env on submit, pull the missing
model, decide the card automatically if it was never decided, and do not hold the job.

- `POST /v1/jobs` for a type this card can run but has not installed, or for a declared model,
  voice or base asset set that has not been pulled, starts ONE `module` task and answers
  **`409 installing`**. The message reads like: *"installing the rvc environment (about 3.3 GB),
  then pulling its base assets (about 900 MB) and the RVC voice 'sigma' (about 55 MB); submit
  this job again after it. Task … is doing it: GET /v1/tasks/…"*. `details` carries the task id,
  steps, the current step, byte progress and pace, the last line, and `plan` (the install
  modal's sentences from `GET /v1/capability/plan`).
- A type that is ON whose env has gone or drifted (`preflight` refuses `env_missing`, after an
  upgrade changed a recipe) starts the same install. The installer is the env the refusal names
  (`details.env`, plus `details.narrator_engine` for tts), so a denoise job installs rvc and word
  timestamps install align. Nothing about `env_missing` is left to a hand step.
- The job is **refused, not held**. Holding it would make the lane a queue of two, with Crucible
  owning the order. The client retries the way it retries `server_busy`.
- One install per env: there is one task lane. A submit during a related install is pointed at
  that task, and one during an unrelated task is told to come back after it. A failed install is
  reported ONCE, with its one-line reason, to the next submit. The submit after that retries.
- Refused at once, with no install: `cannot_hold`, a model the card cannot run, a type with no
  installer (`echo`), `unload-*` of an uninstalled type, an env already on disk with its flag
  off, and an undeclared model for a type whose catalog is complete
  (`CATALOG_IS_COMPLETE`: rvc, denoise, asr, align). llm and tts are not in that list, because
  their models may be upstream routes or local directories.
- Installs and pulls are considered only when `preflight` refused with one of the
  `INSTALLABLE_REFUSALS` (`env_missing` and the `PULLABLE_REFUSALS`), so a job whose env and
  weights are present costs nothing extra.
- `[jobs] install_on_submit = false` restores the plain refusal.

### Operator tasks (`crucible/tasks/`)

`POST /v1/tasks` runs one operator task at a time: `pull`, `install`, `module`, `engine`,
`engine-restart`. The package is split by what each part owns:

| module | owns |
|---|---|
| `states` | `Task`, `TASK_TYPES`, `HISTORY`, the terminal set. The state words are `jobs/base.py`'s; a task is never `interrupted`, so its terminal set is the job one without it |
| `validate` | one validator per task type (`VALIDATORS`, checked before the task exists) and one per module table (`job_type_entry`, `need_entry`, `subject_entry`); `validate_module` walks `MODULE_TABLES` and reports every problem, in table order, in one 400 |
| `hostdoor` | finding the orchestrator's door (`$CRUCIBLE_HOST_DOOR`, owned by `platform/paths.py`) and relaying its NDJSON stream line by line into the task's events |
| `runner` | the `crucible install … --verbose` child: argv, environment, the `crucible: ` reason line, throttled byte progress |
| `store` | `TaskStore`: history, subscribers, the busy and card-held refusals, and one `_run_*` per type, with a module's steps split into `_need_step`, `_install_entry_step` and `_subject_step` |

The package answers `TaskStore`, `Task`, `TASK_TYPES`, `PROGRESS_INTERVAL_SECONDS` and the
four patch points; everything else is imported from its submodule. Tests patch
`tasks.install_command`, `tasks.env_installed`, `tasks.which` and `tasks.searched_note`
on the package, so the submodules look those names up through `crucible.tasks` at call
time rather than binding their own copies.

## 9. Per-type notes: llm and echo

- **llm** (`load-model`, `unload-model`). Chat never touches the lane (it is proxied). The lane
  serialises the lifecycle only.
  - `_reclaimable` counts the resident model itself: `Residency.occupy` always evicts, so reloading
    the resident model is a full restart and its memory is free for the guard. Types that REUSE
    what they name (tts, align, denoise) exclude it instead.
  - The guard runs at submit and again in the lane, because the card moves. The KV-pool check
    ("can one request fit at the served context") is a second question after "does it fit".
  - `params.context` is refused above the host ceiling (`capability.check_load_context`, the
    same function `GET /v1/capability` uses) before anything is evicted.
  - Model rows: `loadable` is a disk fact and never runs nvidia-smi. `max_model_len` for the
    resident model is what the engine was STARTED with. `max_context` is the ceiling with its
    terms and basis (`measured`, `computed` which is a floor, or `declared`). The Ollama-store
    copy is reported on llama-windows only, and nothing loads from it yet.
  - On llama-windows there is no env: the engine is `llama-server.exe`. `env_missing` is still
    the refusal name for "cannot start anything".
- **echo**: a test type, registered only with `[jobs] enable_echo = true`. It sleeps in 20 ms
  slices so a cancel lands promptly.

## 10. Per-type notes: audio workers

### 10.1 tts render (`crucible/jobs/tts/render.py`)

- A render MAY load its own voice. This is the one asymmetry with llm, which never loads from
  the proxy. The accelerator guard applies only to that load: rendering the already-resident
  voice runs no guard. Measured 2026-09-15: the guard refused a resident voice for its own
  18.1 GiB, because under WSL2 our own engine is unattributed.
- A zero-shot voice renders only if it is already resident. The render door has no `reference`
  field.
- **Arms.** `retake: true` uses narrator's guarded `render_many` against the caller's `band`
  (and `retake_without_band` refuses one with no band). `retake: false` (the default) renders
  each chunk once. The bare arm is the default because screening measures the failures a guard
  hides. A band is never read off the voice: two incidents on 2026-09-18 came from inferred
  bands. A band is validated even when unused (`band_malformed`, a whole-request refusal).
- **Width** has one owner: whatever the engine was started with. An absent `width` is ABSENT on
  the batch and never substituted. Substituting `max_num_seqs` cost the Mac a measured 2.3x
  (12.9x→5.5x realtime). A stated width above the engine's is refused, never clamped. On the
  served arm (cuda-linux) Crucible refuses it (`width_over_serving`). On mlx-darwin narrator
  does, because only narrator knows its MLX tier width. The deciding question is
  `jobenv.tts_env(...).serving_stack`.
- **Guard verdicts** are forwarded VERBATIM and never read inside. The only check is "is an
  object", and a non-object fails its row. `null` means nobody judged the chunk.
  `capped`/`tokens` are `null` when narrator did not say. `_optional_bool` passes a real bool
  through, with no branch to add.
- **Takes.** `take` rides on every item, including 0. `sampling` is ABSENT at take 0, because
  `{}` is refused by narrator. Above take 0, the live narrator is asked whether it reads rungs,
  and `sampling_not_wired` is refused otherwise. A stale pinned narrator once returned
  byte-identical takes. A take past the ladder is a seed lane at take-0 numbers, not a clamp.
  Take 0's sampling reaches narrator through the `NARRATOR_HIGGS_VOICES` document written at
  load. `done` reports the full applied sampling triple and the weights identity with its basis.
- The whole job is ONE `generate_batch`. Rows retire out of order and are keyed by the client's
  `index` (narrator's `i`). One artifact per requested index: a split chunk is joined by
  narrator. A failed row is reported and the run continues. `done` lists the failed indices. A
  batch that ends short is a protocol failure.
- `_render` builds the request (`_batch_request`) and hands each engine message to a
  `_BatchTally`: `claim` enforces the protocol (unknown row, row answered twice, a message
  with no type; `batch_done` and `stopped` are passed over), `record` counts the row,
  `require_complete` refuses a short batch. `_require_band` is `_require_band_keys`,
  `_band_rate` per rate and `_require_band_order`.
- FLAC is encoded by ffmpeg (`s16le` at the rate narrator reported on `loaded`, which was already
  checked against the manifest), not soundfile, because the server process takes no compiled
  audio dependency. A row at another sample rate is a failed row, never resampled. Reported
  duration must match the PCM within `DURATION_TOLERANCE_SECONDS`.
- `stopped` (narrator's cancel acknowledgment) is ignored wherever it arrives. It can arrive
  after `batch_done`. A cancelled render whose engine does not stop within the grace period has
  its voice unloaded through `Residency.unload`, even under a lease.
- `RENDER_SILENCE_TIMEOUT_SECONDS` is 600 s: Owen's standing ceiling for a single Higgs chunk.
- There is no pace round-trip: narrator's wire has no state for it at the pinned sha.

### 10.2 tts lifecycle and streaming (`jobs/tts/__init__.py`, `common.py`, `crucible/ttsstream.py`)

- `load-voice` validates `params.reference` for shape here and for content in
  `voicereference.parse_reference`. It is required for `zeroshot` and refused for other kinds.
  `done.reference` carries the digest, so two clients loading one zero-shot id can tell whose
  clip won. `/v1/voices` rows report `serving`, `takes`, `reference_required`, `identity` and
  `identity_basis` (`verified` or `asserted`), `pace_basis` and `orphan`. They never report
  sampling or rung numbers, which are engine tuning.
- The streaming door is SSE plus three POSTs, not a WebSocket. Electron 33 bundles Node 20, which
  has no global `WebSocket`. `Last-Event-ID` gives reattach for free.
- A session owns one worker thread and holds the residency claim (`may_mutate=False`) for its
  whole life. The session is built BEFORE the claim is taken, because a claim has no expiry and
  a refusal after claiming used to hold the card for the life of the process. `open` checks
  `engine_still_stopping` before `voice_not_resident`. The door never loads a voice.
- Frames: `ready`, `audio`, `restart`, `done`, `error`, `closed`. `gap_sec` is on `done` only,
  relayed verbatim from narrator's `gapSec`. The client inserts the silence. A row that retires
  with audio and no `gapSec` is failed by name, and `null` means cancelled.
- **Unguarded by ruling** (Owen, 2026-09-13: *"streaming can stay unguarded. it needs speed over
  all else"*). Every row is sent with `stream: true`, which routes the whole batch past
  narrator's guarded arm. Never add a `guard` field to this door.
- `STREAM_BATCH_WIDTH` is a table keyed by engine with no default: `higgs-v3` is 1 (measured
  worthless above 1). The 25 ms coalescing window is skipped at width 1.
- Per-row cancel: a row not yet dispatched is dropped for free. A row in flight is stopped by
  aborting its batch, and the survivors are resubmitted with a `restart {id, from_seq}` frame.
  `seq` never restarts. At width 1 there are no survivors.
- `GRACE_SECONDS` (15 s): a dropped reader can reattach with `Last-Event-ID` and be replayed.
  After that the session closes and cancels in-flight rows. The event log is pruned by time AND
  by the slowest reader's cursor, never by count. A `Last-Event-ID` below the oldest frame is
  refused, never skipped.
- `progress_report` has no percentage: a session has no denominator. `progress` is `null`, and
  the counts are reported instead.

### 10.3 rvc (`crucible/jobs/rvc/`)

**Every input must produce an output.** A run missing one output FAILS. Finished inputs are
still published as each one completes, so a cancel or failure keeps them.

Client parameters: `index_rate`, `protect_rate`, `n_semitones`, optional `f0_method` and
`hop_length`, plus `piece_s`, `overlap_s`, `crossfade_s`, `output_rate` and `output_channels`.
Everything else belongs to the server.

- **`protect_rate` is inverted.** urvc's pipeline gates protection on `protect < 0.5`, so LOWER
  protects MORE and 0.5 turns protection off. The bound is [0, 0.5]. It is also a no-op at
  index rate 0. Do not "fix" the bound.
- **An absent `f0_method` or `hop_length` omits the flag**, so urvc keeps its tuned default. This
  is the one place on the wire where absence is meaningful. `hop_length` is tested with
  `is not None`. `n_semitones` 0 omits the flag.
- `has_index` is declared in the manifest. A non-zero `index_rate` against a model without an
  index is refused, not clamped.
- urvc runs as `sys.executable -m ultimate_rvc.cli.main`, never through the `urvc` console
  script. The console script bakes in an interpreter path that goes stale when an env is
  relocated, and then exits 1 with no output.
- `_stage_models` builds a per-job `URVC_MODELS_DIR` of symlinks: the shared base assets plus
  ONE model, so the name urvc is given cannot resolve to another voice.
- Engine environment: `URVC_SKIP_INIT` (no first-run downloads), `HF_HUB_OFFLINE` and
  `TRANSFORMERS_OFFLINE` (no network), `KMP_DUPLICATE_LIB_OK` (three bundled OpenMP runtimes),
  `OMP_NUM_THREADS=1`, and `PYTHONUNBUFFERED`. With more than one thread, the cross-runtime
  barrier SIGSEGVs on the first sentence, on mps and cpu alike. The worker adds
  `URVC_MODELS_DIR` and a PATH of the checked ffmpeg/ffprobe, then the env's `bin`. urvc
  otherwise reaches for `static_ffmpeg`, which the recipe no longer carries.
- ffmpeg and ffprobe are refused at submit (`ffmpeg_missing`) like asr, align and tts.

**Cut, convert, stitch** (`worker.py`). Any length goes in; memory is bounded by a piece.

1. **Plan** (`plan_input`, one streaming read). Each cut lands at the CENTRE of the quietest
   100 ms within the 10 s before the nominal cut. This is asr's rule, ported rather than
   imported, using a running sum rather than a convolution. No final piece is shorter than
   `MIN_TAIL_SECONDS` (1 s): the last cut moves earlier instead. Frames are COUNTED, not read
   from the header, because VBR MP3 headers can be wrong. Unreadable inputs fail the job before
   the engine starts.
2. **Cut** (`read_pieces`). Each piece carries `overlap_s` of REAL neighbouring audio on both
   sides. Pieces are written as float WAV at the input's rate and channels, one batch at a time,
   into staging under the job directory.
3. **Convert.** One urvc `convert-dir` process per batch (see recycling below).
4. **Stitch** (`Stitcher`). Each piece is resampled to the output rate with a rational polyphase
   filter, then trimmed or zero-padded AT THE END to its exact span. The overlap is dropped, and
   neighbours join with a raised-cosine crossfade centred on the seam. The writer checks the
   frame count. A piece whose length is off by more than 100 ms + 0.5% is refused.

- Defaults: `piece_s` 60 s (range 10–600), `overlap_s` 0.5 s (maximum 5), `crossfade_s` 20 ms
  (maximum 1, and never more than `2 * overlap_s`). A caller who sets `overlap_s: 0` without a
  fade gets hard seams, not a refusal.
- The overlap is **context**. rmvpe ends in a bidirectional GRU and contentvec is a transformer,
  so a piece's edge frames need real neighbours, not reflect padding. The overlap also absorbs
  urvc's length shortfall (about 20 ms per 60 s). The crossfade is **click removal**, not
  blending. NSF vocoder phase is integrated from each piece's start, so two conversions agree in
  pitch but not in phase, and a long fade through voiced sound partly cancels.
- **End padding is accepted.** The input's own end has no overlap past it, so up to ~20 ms per
  minute of the last piece is zero-padded. Owen, 2026-09-26: *"i guess we can accept 20 ms of
  padding"*. Do not add an end pad to "fix" it.
- **Output format.** The container and sample format are the input's, read from the BYTES, never
  the name. RF64 is used past WAV's 4 GiB. The converted signal is urvc's 16-bit output.
  - **Rate** (Owen, 2026-09-26: *"i would like to keep 48 khz but if we cant then we cant"*):
    `native` (the default) is `max(input rate, rate urvc wrote)`. The urvc rate is read from its
    first converted piece and is 48 kHz for the published models. `input` keeps the input rate.
    Length is exact in time: `out_frame = round(in * out_rate / in_rate)`, half up, integers
    only. At non-integer ratios each piece sits within half an output frame of a whole-file
    resample.
  - **Channels** (Owen, 2026-09-26): `input` (the default) keeps the input's count and writes
    the ONE converted voice identically to every channel. `mono` writes one channel. A
    per-channel conversion is not offered: it doubles the cost and smears the voice through
    phase differences.
- **Worker recycling** is a memory bound, not a throughput choice. urvc leaks and never gives
  memory back, and the per-file `torch.mps.empty_cache` patch is not enough. Only process exit
  reclaims it. A batch closes at whichever comes first:
  - `BATCH_SIZE` = 96 PIECES (never input files; 96 ten-minute files in one process is the
    2026-09-26 incident).
  - A seconds bound: budget / `LEAK_BYTES_PER_AUDIO_SECOND` (3.4 GB per 600 s, measured on the
    Mac 2026-09-26), capped at `MAX_BATCH_AUDIO_SECONDS` (1800 s). This is the only bound that
    holds on macOS, where `ps` cannot see Metal memory.
  - A live check after each piece: the process's RSS + swap against the budget. Swap counts,
    because kylies-pc's leak went there. A process over budget is stopped after the piece it is
    on, and unreached pieces carry to the next process. At least one piece per process is always
    finished.
  - The budget is `MEMORY_FRACTION` (0.5) of memory available at job start (never more than that
    fraction of the total), or `FALLBACK_MEMORY_BUDGET_BYTES` (3 GiB) when the host will not
    say. Available memory comes from `/proc/meminfo` MemAvailable (inside WSL, the VM's), or from
    macOS `vm_stat` free + inactive + speculative + purgeable.
- urvc's stderr goes to a file, not a pipe that would fill during a long batch. Its stdout is
  parsed for `[RVC] n/total` and never forwarded.
- rvc runs the accelerator guard with no `reclaimable_bytes`: it never unloads somebody else's
  resident engine. Nothing of rvc's is ever resident.

### 10.4 denoise (`crucible/jobs/denoise/`)

- Shares the `rvc` env (`ENV_JOB_TYPE = "rvc"`; `jobenv.JOB_TYPES_SERVED_BY_ENV`).
  `envs/rvc/*.txt` pins `audio-separator`, and `crucible install rvc` decides both flags.
- **The separator is RESIDENT** (the fourth resident kind, Owen's ruling 2026-09-15), but the
  wire is still one block per job. The client does the blocking: it sends about 44 blocks of
  roughly 22 minutes for a 15-hour book and slices stems back at recorded offsets. One load
  serves the whole book, where a per-block load was measured at about a third of the pass.
  `done.extra.load_seconds` is non-zero only on the block that loaded. A pass where every block
  reports a load has lost residency. There is no `load-denoiser`, because the load always
  precedes its block; `unload-denoiser` exists.
- `params` is empty with `extra="forbid"`. Every separation parameter is an audio-separator
  default, byte-identical to BookForge's.
- `use_autocast` is on for CUDA only, decided from the backend and sent on the LOAD. It was
  measured at 20.1x→29.2x realtime, with the output delta below the 16-bit noise floor.
- **Invariants**, all refusals:
  - The input must already be at the model's native rate (44.1 kHz). The librosa front-end
    crashes on other rates, and the server never resamples.
  - The primary stem comes back the same length, sample for sample. This is checked on the
    primary stem only, the one it was measured on.
  - Exactly one output names the primary stem.
- Only the primary stem is PUBLISHED. The others are measured and reported. Publishing them used
  to cost about 10 GB of downloaded noise per book. Output is WAV, because the client slices at
  sample offsets.
- Worker (`denoise/worker.py`), a `WorkerSession` with ops `load` and `separate`:
  - `output_format` and `sample_rate` ride on `separate`, not `load`.
  - The per-request output directory is set by re-pointing `output_dir` AND `output_format` on
    both the `Separator` and its model instance. The model instance copies both at load and
    writes stems with its own copy. Setting only the separator's copy once returned FLAC where
    WAV was asked (1.0.38).
  - The attribute's existence is asserted at load time. The output directory's contents are
    the answer, not `separate()`'s return value, and the returned container is checked.
  - `_Heartbeat` sends progress every 30 s during `separate()`, which otherwise reports nothing
    for hours on long inputs.
  - A failed request ends the session.
- If the worker dies between blocks, the resident row is removed (`_forget`). A stop failure is
  logged and put on the job's stream, never raised over the real error.
- Weights come only from `crucible denoise pull` (HF mirror, pinned, per-file digest). The
  library's own GitHub downloader is never used. Doctor checks presence, not digest.

## 11. Catalog files and loaders (`rvcmodels`, `rvcbase`, `denoisemodels`)

- `rvc/<id>.toml`: every published RVC model is a `.tar.gz` under `rvc/` in ONE repo
  (`owenmorgan/owen-morgan-bookforge`). A manifest therefore names the repo, revision, archive
  path, `archive_sha256` and `archive_bytes`, and `weights.pull_archive` fetches that one file.
  - Each archive unpacks to `rvc/voice_models/<model_name>/`, a whole `URVC_MODELS_DIR` root.
    This was verified against the tarballs on 2026-09-13.
  - `model_name` is the trained folder name. It cannot be derived from the id.
  - `files` is empty on purpose: the tarball is deleted after unpacking, and `pull_archive`'s
    stamp proves completeness.
  - The archive path must be anchored and traversal-free.
- Both backends name the same engine and archive for rvc and denoise. torch has MPS, and
  checkpoints are not quantised per platform. The only difference is denoise's `use_autocast`,
  read off the backend.
- Every `memory_bytes_estimate` in these catalogs is **computed, not measured**:
  - rvc: about 540 MB of base weights, plus the model, plus a declared 1.5 GiB for the CUDA
    context and activations.
  - denoise: the checkpoint plus that 1.5 GiB.
  - On a Mac the number is compared against free unified memory, which makes it conservative.
  - The first real measured pass should replace them.
- Model revisions are `main` as the HF API served it on the recorded date (2026-09-13, or
  2026-09-23 for the vocals separator). HF's LFS oid is the sha256 for large files. Small
  configs were hashed from the served bytes.
- `rvcbase/ultimate-rvc.toml`: the engine's shared base assets (contentvec
  `pytorch_model.bin` + `config.json`, `rmvpe.pt`, `fcpe.pt`). They come from the repo urvc's own
  first-run downloader uses (`JackismyShephard/ultimate-rvc`, `Resources/`), pinned, not from the
  ancestral upstreams. `why` on each file is required.
  - `fcpe.pt` is shipped because `f0_method` is a client parameter.
  - `config.json` is required, or transformers refuses the directory. An earlier job-side list
    missed it, which is why `rvcbase.targets` is now the one owner of the file list.
  - `crucible rvc pull-base` places them under `~/.crucible/rvc-base`.
- `denoise/<id>.toml` declares both halves of each file:
  - `model_path`/`config_path`: where the bytes live on the HF mirror (`Politrees/UVR_resources`).
  - `model_filename`/`config_filename`: the names audio-separator resolves inside
    `model_file_dir`.
  - These differ, and nothing derives one from the other. A wrong filename would make
    audio-separator try to download over it.
  - Filenames are validated as bare names.
  - `primary_stem` is `dry` for the denoiser and `vocals` for the vocals separator.
    `sample_rate` is 44100 for both.
  - The vocals checkpoint is byte-identical to the author's own repo (KimberleyJSN/melbandroformer).
  - `denoisemodels.denoise_models_root` and `model_files` are the one owner of that layout and
    file list. Pull stamps are per model (`stamp_name`), because the directory is flat.
- `missing()` in rvcbase and denoisemodels checks presence, not digests. Files Crucible placed
  were verified at placement, and re-hashing hundreds of MB on every doctor run is not worth it.
- The manifest loaders (`manifests`, `asrmodels`, `alignmodels`, `rvcmodels`,
  `denoisemodels`) are one loader in five copies. Merging them is an open follow-up.
