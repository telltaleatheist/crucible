# Phase 13 — Crucible has its own UI: the operator door

**Owen, 2026-09-14:** *"not microservices. but crucible has its own ui. and it provides the
token or whatever else we need to set it up on foundry or bookforge."*

This document is the CONTRACT. Three builds read it — the server (section 3), the page
(section 4), and each app's connect door (section 5) — and it is the single owner of every
name on the wire between them (ARCHITECTURE.md R1). A build that needs a name this file does
not have adds it HERE first, in its own commit, and says so.

## 0. What is decided, and what it deletes

**One server process.** Not microservices: the lease, the four facts and the unload ruling
(`crucible/settle.py`) work because ONE process arbitrates the card. Job types stay what they
are — isolated envs the server spawns — which is a modular server, not a distributed one.

**Crucible serves its own operator page** at `http://<server>:<port>/`. Everything a person
does to a server after it exists happens there: see what is resident, install a job type,
pull weights, watch the progress, read the token off the screen and hand it to an app.

**What that deletes in the apps.** BookForge and Foundry each carry an install story — a host
probe, a step list, a printed list of `crucible … pull` lines, a disabled "run it" button.
Two copies of one fact, kept in step by hand, which is exactly R1's shape. Both shrink to
two doors:

1. **Connect** — name, address, token; or ONE pasted line the page hands out (5.1).
2. **Get one on this machine** — `@crucible/bootstrap`'s `install()`: the chicken-and-egg
   minute a page cannot do for itself, after which the door is "Open Crucible".

Nothing else about a server is an app's to draw. The printed pull lists go. The wizard's
"install the engine here" steps become "which server" once the local spawn layers are
deleted (BookForge `docs/CRUCIBLE_ROLLOUT_PLAN.md`), which is a BookForge phase, not this one.

**What stays exactly as it is.** The registry in each app (a server list is per app because
each app talks to different servers). `local` reading `config.toml`. Every job route. The
lease. The bootstrap package — its one job gets smaller, not different.

## 1. Auth: how a page on a server with a bearer token gets in

- `GET /` and `GET /ui/*` are **public static** — HTML, CSS, JS, no secret in any of them.
- Every `/v1/*` route is private, unchanged.
- The page keeps the token in `localStorage` for its origin. First visit: it asks for the
  token. It also accepts `#token=<t>` in the URL fragment, stores it, and REPLACES the URL
  without the fragment (`history.replaceState`) so the token is not in the address bar or
  a bookmark. A fragment never reaches the server, which is why it is the fragment and not
  a query string.
- Where the first token comes from: the person who installed the server. `crucible init`
  already prints it; from this phase `init`, `service install` and the new
  `crucible token --url` print the **pairing line** (5.1), which opens the page already
  signed in. An app that installed the server through bootstrap holds the config and opens
  the page the same way (5.3). Nobody types a token twice.

Showing the token on the page to somebody who authenticated with that token reveals nothing.

## 2. The vocabulary

| word | meaning |
|---|---|
| **subject** | one pullable thing: a model, a voice, an RVC voice, the RVC base assets, a denoise checkpoint. `{kind, id}`. `kind ∈ model, voice, rvc, rvc-base, denoise`. `rvc-base` has exactly one id, `base`. |
| **catalog** | every subject this BACKEND can hold, installed or not, with what the manifests already say about it. |
| **task** | one operator operation the server runs on itself: `pull` a subject, `install` a job type, or a `module` (an ordered list of both). One at a time per server (R5: admission is the server's). |
| **module** | an app's statement of what it needs: job types + subjects. The APP owns its module and posts it; Crucible never learns what the app is for. |
| **pairing line** | `crucible://<name>@<host>:<port>/#<token>` — everything an app's connect door needs, in one string. |

## 3. The server (Crucible, Python)

### 3.1 `GET /v1/setup` — "whatever else we need to set it up"

```json
{
  "name": "crucible@mac-studio",
  "version": "0.6.0",
  "backend": "mlx-darwin",
  "bind": "http://0.0.0.0:7100",
  "urls": ["http://192.168.68.20:7100", "http://100.64.0.3:7100"],
  "token": "…",
  "pairing": ["crucible://crucible@mac-studio@192.168.68.20:7100/#…", "…one per url…"],
  "job_types": ["llm", "tts"],
  "config_path": "/Users/telltale/.crucible/config.toml"
}
```

`urls` = the bind address made reachable: `0.0.0.0` becomes one entry per non-loopback
IPv4 interface, in the order the OS lists them; a concrete bind host becomes exactly one
entry. Never a guess, never a hostname lookup — an interface the host has is a fact; a name
somebody else's DNS may resolve is not. `job_types` is what `/v1/info` says; it is repeated
here so the page draws from one read.

### 3.2 `GET /v1/catalog`

```json
{ "rows": [
  { "kind": "model", "id": "qwen3.5-9b", "name": "Qwen3.5 9B", "job_type": "llm",
    "installed": true, "installed_bytes": 20718362624, "expected_bytes": 20700000000,
    "floors": ["clean"], "license": "Apache-2.0", "source": "hf:Qwen/Qwen3.5-9B",
    "resident": false },
  …
]}
```

- One row per subject the manifests declare FOR THIS BACKEND (`crucible/lineup.py`'s
  `floors` table is the source of `floors`; the `installed` fact is `weights.installed()`;
  `resident` is `/v1/activity`'s `resident.id`). `expected_bytes` is what the manifest
  declares or `null` — never estimated. Every field is derived from something the server
  already owns (R1); no new table.
- A `job_type` whose env is NOT installed still lists its subjects (you may pull weights
  before the env); the page says so on the row.

### 3.3 Tasks — `POST /v1/tasks`, `GET /v1/tasks`, `GET /v1/tasks/{id}`, `GET /v1/tasks/{id}/events`, `DELETE /v1/tasks/{id}`

Request bodies, one `type` each:

```json
{ "type": "pull",    "kind": "model", "id": "qwen3.5-9b" }
{ "type": "install", "job_type": "tts", "narrator_engine": "higgs" }
{ "type": "module",  "module": { "name": "bookforge", "version": "1.0.0",
                                 "job_types": [ {"type":"llm"}, {"type":"tts","narrator_engine":"higgs"}, … ],
                                 "subjects":  [ {"kind":"model","id":"qwen3.5-9b"}, … ] } }
```

- `202 {"task_id": …}` — the task runs in the server (a subprocess of the console script
  for `install`; the `weights.pull*` functions for `pull`, run off the event loop).
- **One task at a time.** A second POST answers `409 task_busy` naming the running task —
  the same shape `POST /v1/jobs` uses for `server_busy` (ARCHITECTURE.md section 3).
- **Tasks and jobs do not exclude each other** EXCEPT: an `install` task may not START while
  a job is on the lane, a lease is open, a streaming session holds the card, or a chat is
  in flight — the four facts (`settle.py`), because 3.4 restarts the server. `pull` may run
  beside a job (it is disk and network). Refusals name the holder: `409 server_busy`.
- Refusals by name, at POST time: `unknown_subject`, `already_installed` (a pull of an
  installed subject is refused, not skipped — the page greys the button instead),
  `unknown_job_type`, `job_type_installed` (for `install`), `narrator_engine_required`,
  `narrator_engine_refused` (given for a non-tts type), `task_busy`, `server_busy`,
  `invalid_module`. A `module` is validated WHOLE before anything starts; installed job
  types and installed subjects inside a module are SKIPPED with a `skipped` event each
  (a module is idempotent; a single pull is not — the difference is written here on purpose).
- **Events** stream on the same SSE envelope jobs use (`api.py`'s `text/event-stream`
  routes; same `{"type": …}` discipline). Task event types: `started`, `step` (`{name,
  index, total}` — for a module, one per entry), `progress` (`{bytes_done, bytes_total|null,
  file}` for pulls; `{line}` for install — pip's line, and it is NOT load-bearing, R4),
  `skipped` (`{reason}`), `done`, `failed` (`{code, message}`), `cancelled`.
- `GET /v1/tasks/{id}` is the same status shape jobs have: `state ∈ running, done, failed,
  cancelled`, the request body echoed, timestamps, and `error` on failure. `GET /v1/tasks`
  lists the last N (in-memory, N = 50; a task is not a record anybody keeps).
- `DELETE` cancels: a pull is stopped and its partial directory is REMOVED (R6 does not
  apply — half a safetensors file is not partial work anybody can use; `weights.pull`
  already writes into a temp dir and moves on success, keep that); an install is SIGTERMed
  and the env is left for `--force` to rebuild. Cancel is refused `not_running` after the
  fact.
- R6 for a module: a failed step stops the module; every completed step STAYS (envs and
  weights are on disk; the task's `failed` event names the step index).

### 3.4 An installed job type is live before the task says `done`

The running server built its registry from `[jobs]` at startup (`create_app` →
`build_registry`). `crucible install <type>` rewrites `config.toml` with the new flag
(`_write_capability`, which MERGES — every other flag survives, verified `cli.py:413`).

**Requirement:** a client that posted `{"type":"install"}` sees the new job type in
`/v1/info` before the task's `done` event, with nothing to run by hand. The builder decides
HOW with a written reason in this section — either re-read `[jobs]` and rebuild the
registry in place, or (if the app object cannot safely grow a mounted route set) re-exec
under the service supervisor when the four facts are settled — and writes the reason here.
Whichever it is, the events say what happened (`step {name:"reload"}` or
`step {name:"restart"}`), and the page reconnects. "Restart it yourself" is not an option.

### 3.5 The CLI grows one verb and two lines

- `crucible token --url` prints the pairing lines, one per `urls` entry (section 3.1). No
  `--show` needed; the flag name says it prints the secret.
- `crucible init` and `crucible service install` end by printing the same lines.

### 3.6 `@crucible/client` grows the same doors

`setup()`, `catalog()`, `submitTask(request)`, `task(id)`, `taskEvents(id)`, `cancelTask(id)`,
plus a PURE `parsePairing(line): {name, url, token}` that refuses malformed lines by name
(`invalid_pairing`). Typed against section 3 verbatim; tested against a fake server like
every other method in `sdk/ts/test`.

### 3.7 Tests

pytest for every route and every named refusal; a task test that pulls from a FAKE source
into a temp `CRUCIBLE_HOME` (no network, no real weights — C: has 5 GB free today); the
install test runs a fake console script that writes a config flag; the reload/restart path
under 3.4 is tested with the facts settled and with each of the four facts unsettled. The
static mount is tested (`GET /` is HTML, `GET /ui/app.js` is JS, no `/v1` under `/ui`).

## 4. The page (Crucible, `crucible/ui/`, shipped as package data)

- **No build step, no CDN, no framework.** One `index.html`, one `app.css`, one `app.js`,
  vanilla, served by FastAPI's static mount. The Mac may be on a LAN with no internet; a
  page that needs a CDN is a page that is sometimes blank. Package data is declared in
  `pyproject.toml` so the wheel carries it; a test asserts the files are in the built wheel.
- Both themes via `prefers-color-scheme`; keyboard-operable; nothing animates in from
  `opacity: 0`.
- **Sections, in this order,** each drawn from exactly one read:
  1. **Status** — `/v1/setup` + `/v1/activity`: name, version, backend, card, what is
     resident and who holds it (the lease holder, the job, the session), refreshed on an
     interval and on every task event.
  2. **Tasks** — the running task with its live progress (bytes, or pip's lines in a
     scrolling pane), the last few finished. Cancel button on the running one.
  3. **Job types** — every type this backend could hold: installed → enabled/disabled with
     `capability`'s reason; not installed → an Install button (tts asks which narrator
     engine, from `/v1/capability`'s rows, never a hard-coded list).
  4. **Catalog** — `/v1/catalog` grouped by kind: name, id, size (installed or expected),
     floors as chips ("minimum for translate"), license, Pull / Installed / Resident,
     progress inline when its pull is the running task.
  5. **Connect an app** — the server name, each URL, the token (masked, reveal, copy), and
     the pairing line per URL with a copy button. One sentence: "Paste this into
     BookForge or Foundry → Settings → Crucible Servers → Add." Also a **Module** box: paste
     a module JSON (or drop the file) → `POST /v1/tasks {type:module}`.
  6. **Service** — how the server is run (from `/v1/info`'s service facts if present; else
     the `crucible service …` lines), config path, and the pairing lines again for a
     terminal person.
- Every refusal the API names is shown with its code and message, verbatim, next to the
  control that caused it. Never a toast that says "something went wrong".
- Sign-in state: token missing → one field and a sentence about where the token comes from
  (`crucible token --url` on the server). 401 from any call → back to that field with the
  reason.

## 5. The apps

### 5.1 The connect door takes a pasted line

Beside Name / Address / Token: **"Paste from Crucible"** — one field. A `crucible://` line
fills the three (through the SDK's `parsePairing`; a malformed line shows `invalid_pairing`
and fills nothing). Test and Add are unchanged. Foundry's door does the same, through the
same SDK function.

### 5.2 The install door becomes "Open Crucible"

- A local server exists (`local` resolves): the door is one button, **Open Crucible**, and
  one sentence. The printed step list and pull list are DELETED from this case.
- No local server: the bootstrap document stays (it is the pre-server minute), with its last
  step now "Open Crucible" instead of the pull list, and the driven button still gated on
  the published release.

### 5.3 Opening the page

`crucible:open-ui` (BookForge; Foundry names its own) opens a `BrowserWindow` at
`<url>/#token=<token>` for a NAMED registry entry — the token from the registry (or from
`config.toml` for `local`), never typed. An external browser is not used: the fragment would
land in its history. Every server row in Settings → Crucible Servers gets **Open**.

### 5.4 The module

`shared/crucible/bookforge.module.json` in BookForge (Foundry ships its own beside its
package): the section-3.3 shape, versioned. A **"Set up for BookForge"** button on a server
row posts it and shows the task's progress in the row. It is the ONLY place BookForge says
what it needs from a server, replacing `BOOKFORGE_JOB_TYPES` and the pull list in
`electron/crucible/install.ts`, which are deleted (one fact, one owner: the module file).

## 6. Not in this phase, written so it is not forgotten

- **Packed envs on the release** (conda-pack per backend per job type, split at 2 GiB like
  BookForge's own) — the same tasks, faster. After the door exists.
- **Service control from the page** (start/stop): the page runs inside the server, so
  "stop" would take the page with it. `restart` is 3.4's mechanism and is enough.
- **A module registry** (a list of known apps inside Crucible) — refused: Crucible would
  then know what BookForge is. Apps post; the page pastes.
