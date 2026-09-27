# Proposal: a job whose env is missing gets it installed (fresh-install #2)

> **BUILT (option 2, server side), 2026-09-26.** Owen's ruling: *"yes, we need to install a missing environment when a job is submitted"*, under the standing rule that Crucible is idiot proof and the caller need not know an environment exists. `crucible/installonsubmit.py` has the whole of it; in short:
>
> - `POST /v1/jobs` for a type whose refusal would be `job_type_disabled` / `not_installed` (the record says this card fits it, and the live install plan agrees) answers **202**. The job is `queued` with `position: null` and a `waiting_for` object: `message` in words ("installing the rvc environment (about 3.3 GB), then pulling its base assets (about 900 MB) and pulling the RVC voice 'sigma' (about 55 MB); your job starts after it"), `plan` (the install modal's sentences, `capability.install_plan`, since an API client has no modal), `task_id`, `steps`, `step`, `progress`, `line`. The job's stream carries it as `waiting` events.
> - The server runs one `module` task (the operator page's install, plus `rvc`'s base assets and the named voice/model when declared and missing), started with `on_submit`: not gated on the four facts, ending in the additive take-up (#42) rather than the registry swap. One driver, one task lane: jobs arriving together or mid-install share it; an operator's own running install is waited out, not raced.
> - When it lands the job is admitted as the door would (model, lease, clearance, `preflight`) and queued behind the lane. If the install fails, every job waiting on that step fails with the install's own one-line reason (`jobenv.failure_message`'s head, now carried on the task as `Task.reason`).
> - Still refused at once: `cannot_hold` (or a live plan that says nothing runs here), `undecided` (no record: nothing measured the card), types with no installer, an env already on disk with its flag off, an undeclared model for a catalog-complete type.
> - The switch: `[jobs] install_on_submit = true` (default; `false` restores the refusal with `details.install`).
>
> Not built, still open: a job whose env IS installed but whose model/voice is not pulled is refused `model_not_installed` as before (item 1's second half); `undecided` servers are not installed for; the waiting message states sizes and a download's remaining time, not a pip time estimate.

2026-09-26. Snag #2 in `FRESH-INSTALL-KYLIES-2026-09-26.md`: the Windows install ends at a running engine with no job envs, and kylies-pc's rvc env, base assets and model were a command typed inside the guest. PHASE19's rule is that nobody is ever shown a command. This note says what already exists, what package D built, and what is left for Owen to decide.

## What already exists

- **`POST /v1/tasks`** runs `install`, `pull` and `module` on the server itself (PHASE13 §3.3, `crucible/tasks.py`). An install ends with the registry reload (§3.4), so the type is served without a restart.
- **The operator page** (`GET /`, `crucible/ui/app.js`) draws an Install button per job type and a Pull button per subject, and both send that request.
- **A module** (`modules/*.module.json`, `crucible/modules.py`) is an app's whole need in one POST: job types plus subjects. BookForge and Foundry post theirs. This is PHASE19 2.12's "coordinate step on first connect".
- **The gap:** a client with no module (training-pc-1's rvc run, a script, the SDK used directly) submits a job and gets `job_type_disabled`, whose sentence used to end "Install it with `crucible install rvc`": a shell command, on the server, typed by a person.

## What package D built (the smallest honest piece)

- `job_type_disabled` now carries `details.reason`: `undecided`, `cannot_hold`, `not_installed`, or `not_taken_up` (`crucible/jobs/__init__.py`, `disabled_error`).
- On `not_installed` it carries `details.install`, the exact body to POST to `/v1/tasks` (`{"type": "install", "job_type": "rvc"}`; `denoise` names `rvc`, its installer). The sentence names that request and the operator page's Install button, not a command.
- `not_taken_up` no longer happens after a CLI install. The server takes up a type turned on in config.toml on the next request (`crucible/api.py`, `take_up_enabled_types`, #42).

So any client can now fix the refusal by itself in one request, and a person can do it with one click.

## What is left, for Owen to decide

1. **Client side (small, recommended first).** The TS SDK gains `ensureJobType(type)`: on `job_type_disabled` with `reason == "not_installed"`, it POSTs `details.install`, follows the task's events, and retries the submit. The weights (rvc base assets, a voice model) are a second refusal of their own. `rvc_base_models_missing` and the model refusals would need the same `details.pull` body to be followed the same way (package E's files).
2. **Server side (a policy question).** Should `POST /v1/jobs` start the install itself and hold the job `queued` until it lands? It is 3.3 GB and minutes of pip for rvc, started by whoever can submit a job. It would need a config switch (`[jobs] install_on_demand`), the task lane's one-at-a-time rule, and a job state that says "waiting for its install". Not built: a job POST that quietly starts a multi-gigabyte download is a decision, not a fix.
3. **The Windows installer (package A2/B).** The installer could post a default module at the end (for example "rvc for this card"), so a fresh box serves something without an app. That needs a ruling on which types a bare install should carry.
