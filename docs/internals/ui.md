# Operator console internals (`crucible/ui`)

The operator console is `crucible/ui/index.html`, `app.css` and `app.js`: vanilla,
one file each, no build step, no framework, no CDN and no external font. The Mac
may sit on a LAN with no route out, and a page that needs a network to be legible
is sometimes blank. The original design is `docs/history/PHASE13-OPERATOR.md` section 4
(history, not maintained).

On the machine Crucible runs on, the primary UI is now the desktop window (`crucible app`,
`crucible/desktop_app`, described in `host-and-platform.md`); the tray's "Open Crucible"
opens that window. This console stays the way to reach a server from another computer.

## Sections and the read that owns each

| Section   | Reads |
|-----------|-------|
| Status    | `GET /v1/setup`, `GET /v1/activity`, the card from `/v1/info` |
| Tasks     | `GET /v1/tasks` and the running task's event stream |
| Job types | `GET /v1/capability` and the capabilities `/v1/info` reports |
| Settings  | `GET/PUT /v1/settings` |
| Voices    | `GET/PUT/DELETE /v1/voices/{id}` (`voices.md`) |
| Catalog   | `GET /v1/catalog` |
| Connect   | `GET /v1/setup` |
| Service   | `GET /v1/info`, `GET /v1/setup` |

## Rules the whole page is written to

- **Refusals are shown with their code and message, verbatim, beside the control
  that caused them.** No toast, no "something went wrong", nothing told "maybe"
  (ARCHITECTURE.md R3). A 409 `server_busy` names its holder by `details.door`:
  `job` (the lane; `details.holder` with the job's id, type and status) or
  `operator` (the card; `details.fact` and `details.who`, the server's own
  sentence about the holder). Any other refusal carrying `fact` and `who`
  (`subject_in_use`) shows them the same way. An operator told only "busy"
  concludes the button is broken. A settings refusal is placed by `details.field` (the dotted path
  refused) on the control that earned it. A non-Crucible error (proxy, gateway)
  is named as what it is, not dressed up as a Crucible refusal.
- **No second copy of a server-side table.** Job types, narrator engines, subject
  kinds, catalog order, what installs what and the upstreams (their names, order,
  which field each takes, and `upstream_labels` for display) all arrive on the wire;
  the page has no list of them and must never grow one. The same goes for the task
  stream's terminal events (`info.terminal_states.tasks`), the label and tone of a voice's
  `manifest` source (`info.voice_sources`) and the service commands shown in Service
  (`info.service_commands`): all three come from `/v1/info`, and until it is read the
  page shows the info refusal or "reading…" instead of a guess. An upstream's field is
  whichever of `url` or `key_hint` its settings entry carries. Catalog rows are grouped by kind in
  the order the route gives.
- **Which read says a capability is served.** `setup.job_types` and
  `info.job_types` are POSTable job types (`load-model`, `tts`, `echo`, ...). The
  Job types section and catalog rows speak in capabilities (`llm`, `tts`, `asr`),
  which is `info.capabilities[].job_type`. Asking `job_types` whether `llm` is
  here answers no on a server that serves it. The served-capabilities set is
  `null` (not empty) when `/v1/info` has not been read or was refused. The
  catalog deliberately has no per-subject `env_installed`; a row compares its job
  type against the served capabilities.
- **No local state about the server.** Every write re-reads what it changed
  (settings redraw from the document the PUT returns; a catalog Remove re-reads
  the catalog; a route change re-reads capability and catalog) rather than
  patching in place. When a task stream reaches its terminal event, everything is
  re-read.

## Redraws and operator input

Status refreshes on a four-second interval and on every task event (throttled),
and the whole console is redrawn from `state`. Therefore:

- Anything the operator has typed or chosen but not yet sent (module text,
  engine choice, upstream keys, the voice draft and its JSON, a half-typed pin)
  lives in `state`, never only in a DOM element, or the redraw throws it away.
- Every focusable control has an id so `render` can restore focus and keyboard
  position after a redraw; keyboard operability is required.
- An upstream key field is cleared on success only: a refused key is still the
  one the operator has in hand.
- The voice editor stays open on a refusal so the named field can be fixed.
- Upstream `test` results (the model ids an upstream offers) are kept in memory
  for the route pickers, never cached across reloads or written to disk.

## Transport and token

- Events are read with `fetch`, not `EventSource`: every `/v1` route needs a
  bearer token and the API version header, which `EventSource` cannot send. The
  SSE framing is the same envelope the job stream uses.
- Every path the page calls is written as a whole template literal, never built
  from fragments, so `tests/test_ui_mount.py` can extract each path and check it
  against the app's route table.
- The token lives in this origin's `localStorage`. A pairing line's `#token=` is
  stored and then removed from the address (history, bookmarks); a fragment never
  reaches the server, which is why it travels in one. A 401 from any call returns
  to the sign-in gate with the reason. If storage is refused the page still runs
  for this visit and asks again on reload.
- Status does not pass `?accelerator_probe=true`: the probe spawns `nvidia-smi`
  per read and Status is on a timer. The card's identity comes from `/v1/info`.

## Specific controls

- **Engine (Windows only; `host-and-platform.md`, "The Windows to WSL move").** On Linux and macOS the backend
  is the engine and there is nothing to move. Whether a `cuda-linux` server is a
  WSL guest cannot be told by the server (`host.platform` is `linux` either way),
  so the page asks the browser, which runs on the Windows side. The server does
  not perform the move: it hands it to the host's loopback door and relays the
  host's events under the task id, so the Tasks panel watches it like a pull. A
  hand-started server refuses `engine_move_needs_host`.
- **Plan before install or pull** (Owen, 2026-09-26: *"that can be in a modal or
  something that pops up when the user tries to install a pakcage from the
  crucible ui"*). The words come from `GET /v1/capability/plan` and are shown in
  `window.confirm`. A 404 (a subject no capability class runs) means nothing to
  say and the act proceeds; any other refusal is shown and nothing is submitted.
- **Pull** is greyed on installed subjects (the API would refuse
  `already_installed`). **Remove** is offered on installed rows and refused by
  name when something holds the subject (resident model, task). It is the
  only irreversible act on the page and gets one `window.confirm` naming the
  bytes; the browser's own dialog, not a custom modal.
- **Voices** (Owen, 2026-09-26: *"it should be possible to do directly by the
  user"*). A voice's settings travel with its weights (`crucible-voice.toml` at
  the pinned revision). Add = `PUT /v1/voices/{id}` `{pin: {hf_repo, revision}}`;
  Edit = `PUT` `{voice: {...}}`, a whole local override that wins over the pin;
  Revert = `DELETE`, removing the override or the pin. Weights are pulled in the
  Catalog. Edit is disabled for a voice that is on the card, since the server
  refuses to rewrite it.
- **Settings** (Owen, 2026-09-14: *"Settings live in the engine and nowhere
  else."*). This panel, BookForge's settings and Foundry's card are three windows
  onto one store. Route choices always include `local`, labelled with what local
  means on this machine; "nothing fits" is an answer, not an absence. The three
  upstream `test` refusals are answers to "does this work" and are shown on the
  upstream's card verbatim.
- **Numbers.** Sizes are decimal (as model cards and downloads quote them), and a
  `null` size stays unknown, never 0. A new task step resets byte progress, since
  bytes from the previous file are not progress on this one.
- **Module validation** in Connect is the browser's own finding and is named
  before the request; the server's `invalid_module` is different.

## Styling (`app.css`)

- The complete palette is defined on bare `:root`; dark mode redefines only
  tokens, from `prefers-color-scheme: dark` or an explicit `data-theme` on the
  root, which wins in both directions.
- Nothing fades in from `opacity: 0`; the only motion is a progress bar's width,
  and `prefers-reduced-motion` removes it.
- `.setting` rows have their own shape rather than reusing `.row` (a four-column
  catalog grid); the Voices editor's field row wraps with a 240px basis per field.
