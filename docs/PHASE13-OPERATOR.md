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

### 2.1 How a pairing line is spelled, exactly

A server name **contains an `@`** (`crucible@mac-studio` is the default, and
`config.default_server_name()` builds it from the hostname), so the naive line
`crucible://crucible@mac-studio@192.168.68.20:7100/#…` has two `@` in its
authority. Some parsers split at the last one and some at the first; a format
whose meaning depends on which parser read it is not a format.

**The rule: `<name>` and `<token>` are RFC 3986 percent-encoded.** Only the
unreserved set passes through literally — `A-Z`, `a-z`, `0-9`, `-`, `.`, `_`,
`~`. Every other byte of the UTF-8 encoding is written `%XX` with **uppercase**
hex. So the Mac's line is:

```
crucible://crucible%40mac-studio@192.168.68.20:7100/#bXktdG9rZW4
```

and nothing else is. Three consequences, all of them wanted:

- The line is a genuine URI. `new URL(line)` and Python's `urlsplit` both read
  it, and both agree about where the name ends.
- **A line with two literal `@` in its authority is refused `invalid_pairing`**,
  not repaired by guessing which `@` was meant. Crucible is the only producer of
  these lines; a line it did not write is a line whose meaning nobody knows
  (R3 — nothing is ever told "maybe").
- The token survives whatever it is made of. `secrets.token_urlsafe` emits only
  unreserved characters today, so today's tokens are unchanged by the encoding —
  which is exactly why the rule has to be written down now rather than the first
  time a token contains something else.

The producer is `crucible/pairing.py` (`pairing_line`); the inverse is the SDK's
`parsePairing` (3.6), and the two are tested against each other in both
directions. The trailing `/` before the `#` is part of the format and a line
without it is refused: it is what keeps the fragment out of the path.

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
  "pairing": ["crucible://crucible%40mac-studio@192.168.68.20:7100/#…", "…one per url…"],
  "job_types": ["echo", "load-model", "tts", "unload-model"],
  "config_path": "/Users/telltale/.crucible/config.toml"
}
```

`urls` = the bind address made reachable: `0.0.0.0` becomes one entry per non-loopback
IPv4 interface, in the order the OS lists them; a concrete bind host becomes exactly one
entry. Never a guess, never a hostname lookup — an interface the host has is a fact; a name
somebody else's DNS may resolve is not. `job_types` is what `/v1/info` says; it is repeated
here so the page draws from one read.

**`job_types` is the POSTABLE list, not the capability list, and this example
said otherwise until 2026-09-14** (it read `["llm", "tts"]`; the code has
always answered `sorted(store.registry)`). The two lists are the pair
`/v1/info` deliberately keeps apart: `job_types` is *what you may send to*
`POST /v1/jobs` — `load-model`, `unload-model`, `tts`, `echo` — and
`capabilities[].job_type` is *what this server serves* — `llm`, `tts`. `llm` is
never in the first list, because there is no job you POST called `llm`.
Anything asking "is this capability here" must read `capabilities`; the page
does, and section 3.2's sentence below was corrected for the same reason.

**Where the interfaces come from: `getifaddrs(3)`, through `ctypes`**
(`crucible/interfaces.py`). It is the question the OS answers, on both backends,
and it is stdlib. The three alternatives were each rejected for saying something
else: `socket.gethostbyname(socket.gethostname())` is the hostname lookup this
route exists to avoid; a UDP socket `connect()`ed to a routable address and asked
its `getsockname()` reports **one** interface chosen by the routing table, which
is a guess about where a client will come from rather than a list of what the
host has; and `psutil` is a dependency for a fact the C library already states.
A host on which `getifaddrs` cannot be called at all is a **named refusal**,
`interfaces_unreadable` (503), never an empty list — an empty `urls` reads as
"this server is reachable from nowhere", which is a different and false claim.
Loopback (`127.0.0.0/8`) and link-local (`169.254.0.0/16`) addresses are
excluded: neither is an address another machine can use, so neither can carry a
pairing line.

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
  before the env); the page says so on the row, by comparing `job_type` against
  **`/v1/info`'s `capabilities[].job_type`**. There is no `env_installed` field on
  the row: that is a fact about the job type, not about the subject, and one copy
  per subject is how it would come to disagree with itself (R1).
  (**Corrected 2026-09-14**: this said `/v1/setup`'s `job_types`, which is the
  POSTable list — see 3.1. A `model` row's `job_type` is `llm`, a word that list
  never contains, so the comparison said "llm not installed" on a server that was
  serving it. Nothing was wrong with the route; the doc named the wrong list.)
- **Subjects with no block for THIS backend are absent, not listed as
  unsupported.** The route's sentence is "every subject this backend can hold";
  a row for a `mlx-darwin`-only voice on the PC would be a row with nothing
  truthful to put in `installed`, `expected_bytes` or `source`.
- **Every `license` is `null` in this build, and that is the honest value.** No
  manifest schema in the repo — model, voice, rvc, rvc-base or denoise — carries
  a licence key, so there is nothing to derive one from. Reading "Apache-2.0"
  off a HuggingFace repo name would be Crucible making a licence claim on
  somebody else's weights, which it has no basis for. The field stays on the
  wire so the page and the SDK are built against the shape that will carry it
  the day a manifest declares one; the day it does, the manifest is where it is
  declared.
- **`expected_bytes` is `null` for models and voices, and a real number for
  `rvc`, `rvc-base` and `denoise`.** The second three declare the bytes they
  fetch (`archive_bytes`, the base assets' summed `bytes`, the separator's
  `total_bytes`), because each is a named file with a pinned digest. A model or
  a voice is a `snapshot_download` of a whole repo and no manifest states its
  size; `[local] download_bytes` is the OLLAMA/GGUF artifact's size, a different
  file from a different repo, and lending it to this field would be the
  two-owners bug wearing a plausible number.
- **`floors` comes from `crucible/lineup.py` and nothing else.** A catalog that
  could not read the lineup answers `503 catalog_unreadable` naming the manifest
  that broke it, rather than dropping the `floors` key or emitting `[]` — an
  empty floors list means "this model floors nothing", which is a claim.

### 3.2a `GET /v1/capability` grows `job_types`

**Added 2026-09-14, while building the page (section 4).** Section 4 says the Job
types section is drawn from `/v1/capability`, and it could not be: the route
answered the stored `[capability]` record, whose rows carry a capability CLASS
(`clean`, `translate`, `tts`, …) and no job type. A page cannot turn `clean` into
an `{"type": "install", "job_type": "llm"}` without a table of its own, and
section 4 forbids exactly that ("never a hard-coded list").

So the route now answers the record **plus** one derived key:

```json
"job_types": [
  { "job_type": "llm", "classes": ["clean","translate","simplify","analysis","generate","decide","pages"],
    "installer": "llm", "narrator_engines": [] },
  { "job_type": "tts", "classes": ["tts"], "installer": "tts",
    "narrator_engines": ["higgs-v3"] },
  { "job_type": "denoise", "classes": ["denoise"], "installer": "rvc",
    "narrator_engines": [] },
  { "job_type": "echo", "classes": ["echo"], "installer": null,
    "narrator_engines": [] }
]
```

- One row per job type named by `crucible/capability.py`'s `CLASSES`, in that
  table's report order; `classes` partitions the same read's `classes` list
  exactly, so the two halves of the section cannot disagree.
- `installer` is the job type `POST /v1/tasks {"type": "install"}` must be given
  to build this one's env — `crucible/cli.py`'s `INSTALLER_FOR`, which is almost
  always the type itself. `denoise` is `rvc`, because it shares that env; `echo`
  is `null`, because it is compiled in. A page that did not know this would draw
  an Install button the task door refuses `unknown_job_type`.
- `narrator_engines` is the whole of what `narrator_engine` may be, from the
  table the task door validates against (`crucible/voices.py`'s
  `NARRATOR_ENGINE_SAMPLING`), and `[]` for every type the field means nothing
  for. A list and not a default: on cuda-linux the two engines cannot share a
  venv and there is no default.
- **`job_types` is built live and is NOT part of the record**, because all three
  facts are about this BUILD rather than about the card. Written into
  `[capability]` they would be a second copy that goes stale the day an engine is
  added, which is R1's shape; held in the page they would be the same defect one
  layer further out.
- **Whether a type is OFFERED here is deliberately not in this row.**
  `/v1/info`'s `capabilities[].job_type` is that list — the capability spelling,
  which is what a row here and a catalog row both speak. (`/v1/setup`'s
  `job_types` is the other list, the POSTable one; see 3.1.)
- A server that has decided nothing still answers `503 capability_undecided` and
  the whole section goes with it, including the list. That is the honest state:
  the page shows the refusal with its code and the command that fixes it.

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
- Five more refusals this build needed, and where they come from:
  `unknown_task` (404, `GET`/`DELETE` of an id this server does not hold — the
  shape `unknown_job` already has), `not_running` (409, cancelling a task that
  has finished), `install_command_missing` (503, from an `install` task: the
  `crucible` console script is not beside this interpreter and not on `PATH`,
  and the refusal names both places searched, exactly as every "tool missing"
  refusal does since `crucible/hosttools.py`), and the two a RUNNING task fails
  with rather than falling into a generic bucket `crucible/errors.py` says this
  server does not have: `pull_failed` (the weights module's own sentence — a
  gated repo, a revision the manifest names and the repo does not, a digest
  that did not match) and `install_failed` (the console script's exit code,
  with its output already on the event stream). Both arrive as the task's
  `failed` event, never as a status on the POST, because by then the task has
  been admitted.
- `GET /v1/tasks` answers `{"tasks": [ …the section's status shape… ]}`,
  newest first — the same `{"rows": …}` / `{"tasks": …}` envelope
  `GET /v1/catalog` uses, so a listing can grow a cursor one day without
  becoming a different document.
- **`409 server_busy` on a task carries the holder in the shape the API already
  names it**, which is not one shape but four, because four different things can
  hold the card (`crucible/settle.py`). `details.fact` says which — `a job`,
  `a lease`, `the claim`, `a chat` — and beside it are that fact's own fields:
  a job's are `POST /v1/jobs`' verbatim (`holder`, `job_id`, `type`, `model`,
  `status`, `since`, `progress`, `message`); a lease's are `409 leased`'s
  verbatim plus the `subject` a receipt carries (`lease_id`, `kind`, `client`,
  `act`, `subject`, `since`, `expires_at`); the claim's is `held_by`; a chat's
  is `in_flight`. One producer, `Settlement.holder()`, so the task door and the
  job door cannot come to disagree about who is in the way.
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
`/v1/info` before the task's `done` event, with nothing to run by hand.

**DECIDED, 2026-09-14: an in-place reload. `step {name: "reload"}`.** Re-exec was
not a close second — it cannot satisfy the requirement as written. A task's
record and its event log live in this process's memory (3.3: *"a task is not a
record anybody keeps"*), so a server that re-exec'd would take the task with it
and the `done` event would never be written. "The client sees the type before
`done`" and "the process that owes it a `done` is replaced" cannot both be true.
Re-exec also presumes a supervisor: a `crucible serve` in a terminal has none,
and the one door where an operator is most likely to type `crucible install` is
exactly that one.

**What the reload actually swaps, and why it is two things.** The routes read
capability off `Config` (`config.enable_llm`, `config.enable_tts`,
`config.capability`) and read what is POSTable off `JobStore.registry`, which
`build_registry` built from the same flags. `crucible install` rewrites BOTH —
the `[jobs]` flag and the `[capability]` record — so a reload that rebuilt only
the registry would leave `/v1/models` refusing `llm` while `POST /v1/jobs`
accepted `load-model`. The reload therefore:

1. re-reads `config.toml` with `load_config`, and **adopts the result into the
   Config object this process already holds** (`Config.adopt`). One `Config` per
   server process is not a convenience here, it is R1: every route, the
   residency, the store and every plugin hold a reference to that one object,
   and handing half of them a second one is the two-owners bug built on purpose.
   `adopt` refuses a config from a different path or home by name, because those
   are identity rather than capability;
2. rebuilds the registry with `build_registry(config, backend, residency)` —
   the SAME residency, so a model that was resident stays resident and the new
   plugin instances hold the live engine — and swaps the contents of the dict
   the store already has, so nothing holds a stale mapping;
3. writes the new `job_types` into the task's `step` event, so the client is
   told what became reachable rather than having to diff two `/v1/info` reads.

**The four facts are read TWICE, and the second read can fail the task.** They
gate the task at `POST` (an install may not start while the card is held), and
they are read again immediately before the swap, on the event loop, in one
synchronous stretch with it. A job admitted during the minutes the pip install
took would otherwise have the registry replaced underneath it — and worse, a
capability step that turned a flag OFF (an env that built on a card too small)
would remove the very type that job is running. So a holder found at the second
read is `failed {code: "reload_refused"}` naming it. The env stays on disk
(R6), and re-running `install` finds it built and reaches the reload in seconds.
That is a loud wrong answer instead of a quiet one (R3): the alternative — swap
anyway and hope — is the shape of every defect in ARCHITECTURE.md's table.

### 3.5 The CLI grows one verb and two lines

- `crucible token --url` prints the pairing lines, one per `urls` entry (section 3.1). No
  `--show` needed; the flag name says it prints the secret.
- `crucible init` and `crucible service install` end by printing the same lines.

### 3.6 `@crucible/client` grows the same doors

`setup()`, `catalog()`, `submitTask(request)`, `task(id)`, `taskEvents(id)`, `cancelTask(id)`,
plus a PURE `parsePairing(line): {name, url, token}` that refuses malformed lines by name
(`invalid_pairing`). Typed against section 3 verbatim; tested against a fake server like
every other method in `sdk/ts/test`.

**BUILT, 2026-09-14, with two additions that are named here because they are
now part of the contract:**

- **`tasks()`** — `GET /v1/tasks`. The doc listed the single read and not the
  listing, and the page's Tasks section draws "the last few finished" from it;
  a client that could not ask for them would have had to keep its own list of
  ids across a reload, which is a second record of something the server
  already keeps.
- **`CrucibleCardHeld`**, for the four-shaped `409 server_busy` above. The
  SDK's existing `CrucibleBusy` reads a JOB's eight fields, so a lease-shaped
  body read as one comes back a *protocol error* — a page told its server sent
  nonsense when it sent exactly what this document specifies. The client
  discriminates on `details.fact` and produces the job type or the card type
  accordingly. `details.who` is the server's own sentence and an app's row
  shows it verbatim (5.4).

The pairing line's two implementations are held together by a LITERAL: the same
line appears in `tests/test_setup_route.py` and `sdk/ts/test/unit-pairing.test.ts`,
one asserting the producer emits it and the other asserting the parser reads it.
Neither is written against the other's code, so a change to 2.1 fails on both
sides with the old and the new spelling visible.

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

### 4.1 BUILT, 2026-09-14 — and the decisions the building forced

`crucible/ui/index.html`, `app.css`, `app.js`. Tests: `tests/test_ui_mount.py`.

**`GET /` is a 307 to `/ui/`, and the page has one home.** `index.html` asks for
`app.css` and `app.js` by RELATIVE name, which is what lets the same three bytes
be served from any mount — and from `/` those names resolve to `/app.css` and
`/app.js`, which nothing serves, so the page arrived unstyled and inert. Three
ways out: register two more routes for the assets (one file at two URLs, and a
third place to remember when a fourth file is added); write `/ui/` into the HTML
(an absolute path, which pins the page to today's mount); or make `/ui/` the
page's one home and have `/` say so. Only the last leaves a single owner of
where the page lives. The pairing line survives it: a redirect whose target
carries no fragment of its own keeps the request's, so
`http://host:7100/#token=…` lands on `/ui/#token=…` signed in. The mount gained
`html=True` so `/ui/` is the page; a miss under `/ui` is still a 404, so
`/ui/v1/info` never answers with HTML.

**Events are read with `fetch`, not `EventSource`.** Every `/v1` route needs a
bearer token and an API version header and `EventSource` can send neither. The
frame parsing in `app.js` is the SSE envelope the job stream already uses.

**Status reads `/v1/setup` + `/v1/activity`, and takes the card from
`/v1/info`'s `host.gpu`.** The page never asks for `?accelerator_probe=true`:
that spawns `nvidia-smi` per read and Status is on a 4 s interval, which is the
exact waste the route's own opt-in ruling exists to prevent. The card's
identity — vendor, model, VRAM — is a static host fact and costs nothing.
Live totals remain `GET /v1/accelerator`, for somebody who asks.

**Job types read `/v1/capability` (3.2a) and `/v1/info`'s
`capabilities[].job_type`.** Not `/v1/setup`'s `job_types` — see the correction
in 3.1. A class that is off is drawn with the number that turned it off.

**The `tts` engine picker shows one engine, because the server lists one.** When
this section was written the list held two and the page drew both, correctly —
applying a deprecation was never this page's job. Owen's ruling landed later the
same day and `voices.NARRATOR_ENGINE_SAMPLING` dropped `orpheus`, so the control
dropped it in the same tick with no page change. That is the property worth
keeping: a page that filtered would be a second opinion about what this build
ships, and the day a second engine returns this picker grows it for free.

**Service draws the `crucible service …` lines, because `/v1/info` carries no
service facts in this build.** The page tests the READ (`info.service`), never a
version number, so a build that grows them is drawn without a page change.

**Invalid module JSON is named by the page, as `invalid_json`, before anything
is sent** — it is the browser's finding and not the server's, and dressing it as
`invalid_module` would attribute a refusal to a server that was never asked. The
server's own `invalid_module` (with its collected `details.problems`) is shown
verbatim when it comes.

**Pull and Install are DISABLED while a task runs**, because the door answers
`409 task_busy` and a control that can only be refused teaches its operator that
the page is broken. A pull of an installed subject is greyed with the reason on
its `title`, for the same reason and 3.3's (`already_installed` is a refusal,
not a skip). Every other refusal is shown where it was earned, code first; a
`409 server_busy` additionally shows `details.who`, the server's own sentence
about the holder.

**The token lives in `localStorage` under `crucible.token`**, per origin. A
`#token=` fragment is stored and then removed with `history.replaceState`; a
401 from any call clears it and returns to the field with the refusal.

## 5. The apps

### 5.1 The connect door takes a pasted line

Beside Name / Address / Token: **"Paste from Crucible"** — one field. A `crucible://` line
fills the three (through the SDK's `parsePairing`; a malformed line shows `invalid_pairing`
and fills nothing). Test and Add are unchanged. Foundry's door does the same, through the
same SDK function. The spelling `parsePairing` accepts is 2.1's, exactly: it
percent-decodes the name, refuses a second literal `@` in the authority, and
refuses a line with no `/` before the fragment.

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

**The operator window does NOT inherit the app's bridge, and this is a contract
for both apps, not a preference.** A Crucible's page is code the app does not
own — the server may be the Mac Studio, a friend's box, or a machine whose
address somebody pasted — and `<url>/#token=` hands that page a window inside
an Electron process that has a preload, an IPC bridge to the filesystem, and a
session holding the user's cookies. So the window is built with **no preload at
all**, `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`, and
**its own `session.fromPartition()`** so nothing it stores can reach the app's
storage or be reached from it. Navigation is pinned to the server's origin:
`will-navigate` is denied for any URL whose origin differs, and
`setWindowOpenHandler` returns `{action: 'deny'}` for every `window.open`. A
page that wants to send the user somewhere says so in text they can copy.

Nothing about this is a response to distrusting Crucible; it is that "the page
is served by the thing it administers" stops being a safe sentence the moment
the thing is on another machine, and a window with no bridge costs nothing
because the page needs none — it talks to its own server over HTTP.

### 5.4 The module is GENERATED, never hand-written

`bookforge.module.json` and `foundry.module.json` are **written by a generator
in this repo, from these manifests**, and each app vendors its file byte for
byte. `scripts/gen-modules.py` is that generator and `--check` is the guard CI
runs, exactly as `scripts/gen-foundry-lineup.py --check` already guards the
lineup (which this leaves byte-identical).

**Why not a file typed by hand beside each app.** Foundry already vendors
`foundry-lineup.json` from this repo's manifests, for the reason
ARCHITECTURE.md R1 gives: "what can this machine run" has one owner. A module
file typed beside it would restate the same model ids — `qwen3.8-27b-4bit`
appears in both — with nothing comparing them, so the day a manifest is renamed
the lineup is regenerated and the module is not, and a "Set up for Foundry"
button asks a server for a subject that does not exist. That is the same defect
as every row in ARCHITECTURE.md's table, introduced on purpose, in a file whose
whole job is to be correct about ids.

**What an app declares, and what the generator derives.** In `modules/<app>.toml`
an app states only what Crucible cannot know: the job types it needs (with the
narrator engine, for `tts`), the capability CLASSES it uses, and any subject it
names outright — a specific voice, the RVC base assets, a separator. The
generator resolves every one of those against the manifests and refuses by name
if it cannot:

- a class with a **floor** in `crucible/lineup.py` resolves to the floor — the
  floor is by definition the smallest model the class may run on at all, which
  is precisely the one an app must have pulled;
- a class with **exactly one** candidate model resolves to it;
- a class with several candidates and no floor **must be named** in the
  declaration (`model = "qwen3.8-27b-4bit"`), and the generator checks that the
  named model really serves that class. `analysis` is such a class today, and a
  generator that picked for it would be inventing a policy nobody wrote down;
- every explicitly named subject is checked to exist, in the right kind's
  catalog, before it can be written.

**`version` is derived, not typed.** It is `<crucible version>+<12 hex of the
sha-256 of the module's own content, version excluded>`, so two apps at the same
crucible version whose needs differ have different versions, and a regenerated
file that says the same thing keeps the same one. A hand-typed semver on a
generated file is a number somebody forgets to bump.

A **"Set up for BookForge"** button on a server row posts the vendored file and
shows the task's progress in the row. It is the ONLY place BookForge says what
it needs from a server, replacing `BOOKFORGE_JOB_TYPES` and the pull list in
`electron/crucible/install.ts`, which are deleted.

**When that button is refused `server_busy` because a LEASE is open, the row
shows the holder verbatim** — "held by foundry — translate, qwen3.8-27b-4bit,
until 03:12" — out of the `details` 3.3 specifies. It must not render as a
generic failure: a lease means another app on the same machine is mid-run, which
is the system working, and an operator shown a dead button with no name will
conclude the button is broken and press it until it is.

### 5.5 The setup step PROBES, and shows ONE face

Written 2026-09-14, from BookForge's `docs/CRUCIBLE_ROLLOUT_PLAN.md` §0b C2 and
`docs/SETUP-AND-SETTINGS-AROUND-CRUCIBLE.md` §6. The doc specified the doors and
not the moment a person first meets them, and the two are different problems.

**In SETTINGS, the three doors stay closed until one is opened.** That is
correct there, and it is a measurement decision rather than a layout one:
composing the install picture spawns `wsl.exe -l -v` and an `nvidia-smi` query,
and a settings page that did that every time it was opened would cost a second
of somebody's time to answer a question they did not ask.

**In the WIZARD, the step probes on entry and shows exactly one of three
faces.** A person setting the app up for the first time is being asked *which
server*, not *read these three options and work out which applies to you* —
three collapsed doors is the app handing back the question it exists to answer.

1. **Connected** — a local server resolves. Its name, its address, where the
   config was read from, and two buttons: **Open Crucible** (5.3) and **Set up
   for BookForge** (5.4), with the task's events drawn in place.
2. **Install here** — no local server, and this machine could hold one. The
   measured machine, the pre-server sequence, the commands the app cannot run
   for anybody (`wsl --install -d Ubuntu`, `sudo loginctl enable-linger
   "$USER"`), and the driven **Install** button gated on the published release.
   No printed pull list and no printed env installs: 5.4's module is what
   stocks a server, and the page is where it is watched.
3. **Connect only** — this machine cannot hold one, said by name. One
   "Paste from Crucible" field, with Name / Address / Token still fillable by
   hand, then Test and Add.

**THE VERDICT IS THE MAIN PROCESS'S, AND IT HAS THREE VALUES, NOT TWO.** The
renderer draws a decision rather than making a second one out of the same
nulls (R1). And the decision is `yes` / `no` / **`unknown`**, because on Windows
the honest answer is sometimes the third one: whether a Crucible can run here is
whether the GUEST sees a card, and an app must not answer that from the
Windows-side `nvidia-smi` — a Windows driver that answers says nothing about
whether the passthrough works. So a machine with no WSL2 guest, or one where
nobody has named the guest, is `unknown`, and `unknown` draws the **install**
face: its first step is the thing that would settle it. Telling somebody with a
4090 "this machine cannot host one" because they have not installed Ubuntu yet
would be a wrong answer stated confidently, which is worse than a maybe (R3
forbids the maybe; it does not license the confident error). The verdict always
carries its own sentence, whichever way it went.

**After a DRIVEN install completes, the step posts the module by itself.** The
install ends with a server that answers and holds nothing — no job
environments, no weights — and leaving somebody there with a second button to
find would be handing the install story back in two halves. A person who ran
the sequence by hand presses **Set up for BookForge** instead; it is the same
task either way, and it is idempotent, so pressing it on a stocked server is
how you find out that it is one.

## 6. Not in this phase, written so it is not forgotten

- **Packed envs on the release** (conda-pack per backend per job type, split at 2 GiB like
  BookForge's own) — the same tasks, faster. After the door exists.
- **Service control from the page** (start/stop): the page runs inside the server, so
  "stop" would take the page with it. `restart` is 3.4's mechanism and is enough.
- **A module registry** (a list of known apps inside Crucible) — refused: Crucible would
  then know what BookForge is. Apps post; the page pastes.
