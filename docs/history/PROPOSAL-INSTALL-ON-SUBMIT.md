# Proposal: a job whose env is missing gets it installed (fresh-install #2)

> **BUILT, server side: 2026-09-26, reworked 2026-09-27.** Owen's rulings: *"yes, we need to install a missing environment when a job is submitted"* (idiot proof: the caller need not know an environment exists); then, 2026-09-27, *"Yes, it should try to pull the model"*, *"Yes, it should automatically be checked"* (a server with no capability record), and *"Crucible isn't responsible for queuing. The apps that use it are. It grants and releases leases. That's it"*. `crucible/installonsubmit.py` has the whole of it; in short:
>
> - `POST /v1/jobs` for a type this card can run and has not installed, or for a declared model / voice / `rvc` base assets this card can run and has not pulled, starts the operator page's install (one `module` task, `on_submit`) and is **refused `409 installing`**: "installing the rvc environment (about 3.3 GB), then pulling its base assets (about 900 MB) and pulling the RVC voice 'sigma' (about 55 MB); submit this job again after it. Task … is doing it: GET /v1/tasks/…". `details` carry `task_id`, `steps`, `step`, `progress` (bytes), `line`, and `plan`, the install modal's sentences for this card (an API client has no modal). The task's own record carries the sentence as `message`.
> - **No job is created or held.** The first build (2026-09-26) accepted the job, held it through the install and then put it on the lane behind whatever was there, which made the lane a queue. The ruling of 2026-09-27 rules that out, and so does the gentler version (hold it, then one normal admission): the app owns the queue, so the server names the work under way and when to come back, the `server_busy` shape apps already retry on.
> - One install per env: a submit while the task it needs runs is pointed at that task; one while another task has the lane is told to come back after it (`reason: task_busy`). A failed install is reported once, with the install's own one-line reason (`install_failed`, `Task.reason`), to the next submit that needed it; the submit after that tries again.
> - No capability record (`undecided`): the job door decides this card (`cli._decide_here`, with `ladder.card_for`) and records it (`cli._write_capability`, no flag changed), then answers as above.
> - Still refused at once: `cannot_hold` (or a live plan that says nothing runs here), a model the card cannot run (`model_cannot_run`), no installer (`echo`), `unload-*`, an env on disk with its flag off, a missing or undeclared model for rvc/denoise/asr/align.
> - The switch: `[jobs] install_on_submit = true` (default; `false` restores the plain refusals, with `details.install`).
>
> Still open: a type that is ON whose env has gone (`env_missing`) is refused as before; the sentence states sizes and a download's remaining time, not a pip time estimate.

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
