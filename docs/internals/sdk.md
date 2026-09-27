# TypeScript SDK internals (`sdk/ts`)

What a maintainer of `@crucible/client` needs that the code does not say. The
public API's own JSDoc is one sentence per member; the reasons live here.

## Runtime rules

- **Zero runtime dependencies.** The package runs in Node 20+, bun and the
  Electron main process. Nothing is installed to use it.
- **Base64 is hand-written** (`base64.ts`). `Buffer` is Node/Electron only, and
  `btoa(String.fromCharCode(...bytes))` overflows the call stack on a large
  input. `decodeBase64` refuses a character outside the alphabet instead of
  skipping it: a skipped byte shifts the rest of a PCM chunk, which is heard as a
  click mid-sentence.
- **`node:fs` is never imported statically.** A static import puts fs into the
  module graph of `import {CrucibleClient}` and breaks browser-targeted bundles.
  `loadNodeFileApis` (client.ts) and the loader in `pairing-file.ts` build the
  specifier at run time (`'node:' + 'fs/promises'`) and import it inside the one
  method that needs it. Keep the `/* webpackIgnore: true */` markers on those
  `import()` calls: they are bundler directives, not comments. The loaded module
  is `any`, so its shape is checked at the seam and a runtime without it gets a
  `CrucibleError` naming what is missing.
- `pairing-file.ts` is re-exported from its own module so a browser bundle that
  imports `parsePairing` does not pull `node:fs` in behind it.
- `CrucibleError` assigns `cause` explicitly so the CJS build carries it.
- The streaming door is `fetch` + SSE, not a WebSocket: Node 20 (and Electron 33's
  bundled Node 20.18, where the SDK runs in the main process) has no global
  `WebSocket`, and `ws` would be a dependency.
- `SDK_VERSION` (version.ts) is a literal so ESM, CJS and bundlers agree without
  reading `package.json`. It is a build label for `User-Agent`, never a
  compatibility statement: compatibility is `API_VERSION` (`X-Crucible-Api`).
  `test/unit.test.ts` pins it to `package.json`; `scripts/release.sh` pins both to
  `crucible/__init__.py`.

## Reading the wire (`shape.ts`)

The SDK reads the current server's wire and nothing older (docs/INTENT.md,
2026-09-27).

- A key the server always sends is read with a strict reader (`str`, `num`,
  `bool`, `objectField`, `strArray`); a value it may honestly send as `null` uses
  the `nullable*` readers. A missing key is a `CrucibleProtocolError` naming it.
- The `opt*` readers are only for keys the server sends in some cases and not
  others: a job's `lease_id` (loaders only), a model row's `reason` (only on a
  refusal), and the relayed chat body's `id`, `model` and `usage`.
- `oneOf` is for vocabularies the client or caller branches on (job and task
  state, a decision's type, subject kind, a stream cancel outcome). Words only a
  display reads (a voice's `kind`, `estimate_basis`, lane health, `residentKind`,
  an orchestrator's owner word, `pages_engine.engine`, `finishReason`) are carried
  as the server's own string, so a new word from the server is news, not a
  protocol error.
- **Null is never zero and never false.** `AcceleratorHolder.bytes`,
  `unattributedBytes`, `ChunkData.tokens`/`capped`/`guard`, `cachedTokens`,
  `maxInFlight` and a voice's backend-block fields keep `null` all the way to the
  caller. `capped: null` read as `false` reports every runaway as a long
  sentence; `bytes: null` read as 0 makes a busy card look free; `guard: null`
  says nothing about whether a take was clean.
- A pinned voice this host cannot read is a `/v1/voices` row with every key
  present and `null` where nothing is known (`crucible/jobs/tts/common.py`):
  `kind`, `sample_rate`, `pace`, `serving` and the backend-block fields are
  nullable, and `takes` is 0.
- **Unknown event kinds are carried** as `UnknownEvent` / `StreamUnknown`, never
  terminal. The job event vocabulary grows without moving `api_version`. The
  switch in `readEvent` stays exhaustive over `EVENT_NAMES`, so adding a known
  kind without a case is a compile error.
- **`info()` contains bad rows.** `llm` and `tts` rows are read with the
  `/v1/models` and `/v1/voices` readers; a row that throws a
  `CrucibleProtocolError` goes to `unreadableRows` with its raw data. Any other
  capability is tried as a descriptor and carried raw (`RawCapability`) when it
  does not fit. `models()` and `voices()` stay strict. Only a
  `CrucibleProtocolError` is contained; anything else propagates.
- `progress` and `done` frames keep every unmodelled key in `extra`, verbatim
  (asr's `{stage, processed_s, total_s, cues}`, tts's `failed`, load-voice's
  `fingerprint`). `done` must carry `artifacts` or `resident`.
- Voice pace: the three rates (`pace_chars_per_sec`, `max_chars_per_sec`,
  `min_chars_per_sec`) are all three or none; a half-stated triple is refused as
  the wire disagreeing with itself. Which of `target_chars` / `safe_min_chars` /
  `safe_max_chars` are null tells a band from a target from neither. Nothing
  derives a centre from a null: narrator owns that derivation. `min < pace < max`
  and "never both a band and a target" are the server loader's rules, not
  re-checked here.
- The sampling block on a `done` is read as "every value is a number", not a fixed
  set of keys: the engine's levers can grow.
- `readRenderResult` requires `artifacts` and `failed`; a `failed: []` invented
  for an absent key would read as a clean render.
- An open streaming session's `progress` is read and required to be `null`: a
  session has no denominator.
- `readAlignment` requires exactly one of `items` and `error` per window.

## Request rules

- **Optional request fields are omitted when absent, never sent as `null`.** The
  server's params models forbid unknown keys and treat `null` as a value. This
  covers `clientRef`, `hold`, `retake`, `band`, `width`, a load's lease, a
  zero-shot clip's `name`, asr's `initial_prompt`, `context`, speech-only knobs
  and `resume`, decide's `missing`, and a task's `narrator_engine`.
- **The client keeps no second copy of the server's rules**: no voice `maxChars`
  check, no asr language list, no decide option/level limits, no act vocabulary,
  no ttl range. The server refuses by name. What the client does check are facts
  about the request itself: duplicate render chunk indices (an index is a file
  name), empty chunk text (narrator fails a whole batch on one empty row),
  question shape and unknown keys (stripping one would send a different
  question), and `responseFormat`'s `type`/`name`/`schema` presence.
- Render chunks are sent as `index`, the key the server's `TtsChunk` model
  declares; translating to narrator's `i` is the server's job.
- `X-Crucible-Act` is a header, not a body field, because the chat body is
  proxied to the engine verbatim. It is sent only when the caller names an act.
- `thinking` sends `chat_template_kwargs: {enable_thinking}`, read per request by
  mlx-lm and vLLM. `contextTokens` reaches Ollama as `options.num_ctx`; omitted,
  the server sends the tag's own context. A reasoning model that runs out of
  budget answers with `reasoning` and no `content`; that is a protocol error whose
  message names the remedy (raise `maxTokens` or send `thinking: false`).
- A load's lease is sent snake_case (`ttl_seconds`) as the server's `LeaseOnLoad`
  declares.
- A task request carries only its own type's fields; the server refuses a `pull`
  carrying `job_type`. A module document is posted byte for byte as the app
  vendors it from `scripts/gen-modules.py`. `engine` tasks accept only `wsl`.

## Transport (`#fetch` in client.ts)

- `X-Crucible-Client` carries the bare `clientName` beside `User-Agent`, because a
  browser silently drops `User-Agent`. The server validates 1-80 characters with
  no control characters and falls back to `User-Agent` otherwise; the header name
  must match `crucible/__init__.py`'s `CLIENT_NAME_HEADER`. It is sent on the
  unauthenticated pairing doors too.
- **Stale pooled sockets.** undici keeps an idle connection about 4 s and
  uvicorn's default keep-alive is 5 s, so a request a few seconds after the last
  one could be written onto a closing socket (`ECONNRESET`, seen 2026-09-18..20).
  The server now sets `KEEP_ALIVE_SECONDS = 75`, and the client retries **once**,
  only for GET (`isSafeMethod`), only when `fetch` itself rejected with
  `ECONNRESET` or `UND_ERR_SOCKET` (on the error or its `cause`), and never after
  the caller's signal aborted. A POST is never retried: a reset gives no way to
  know whether the server read it, and a repeated `POST /v1/jobs` is a second
  render.
- The timeout signal is built per attempt, since `AbortSignal.timeout` starts
  counting when created. A caller's own `signal` replaces the constructor's
  `timeoutMs` rather than composing with it; a probe's `timeoutMs` does compose
  with a probe's `signal`. A caller's abort rejects with their own reason, so
  `TimeoutError` and `AbortError` stay distinguishable.
- The client never retries `server_busy`, `leased`, a 5xx, or an upstream rate
  limit. Queues belong to clients (ARCHITECTURE.md R5); a retry loop in the SDK
  would be an invisible queue with a policy nobody chose.

## Errors (`errors.ts`, `#failure`)

- Every non-2xx maps to one type. Three 5xx/4xx codes get subclasses because one
  conclusion must never be drawn from them: `accelerator_unreadable` is not an
  idle card, `capability_undecided` is not "nothing fits", and `server_busy` /
  `leased` carry bodies a bench needs (holder, job, progress, `busyLine`).
- `server_busy` has two shapes, discriminated on `details.fact`: the job door's
  (the holder is always a job) and the operator door's (`CrucibleCardHeld`: a
  job, a lease, the streaming claim or a chat).
- `holder`/`client` is `null` when the busy job or lease came without a name; show
  "an unnamed client", never a guess.
- `testUpstream` decides its three result refusals (`upstream_unreachable`,
  `upstream_rejected`, `upstream_unconfigured`) on the **code**, never the status
  or class: `upstream_rejected` arrives as a 401 and maps to `CrucibleAuthError`,
  and keying on that would tell the user their Crucible token is wrong. It returns
  the server's own sentence (`serverMessage`), not `error.message`, which names
  Crucible as the refuser.
- `isServerSpecificRefusal` lists codes that may differ on another server
  (`server_busy`, `engine_in_use`, `job_type_disabled`, `model_not_resident`,
  `not_resident`, `unknown_model`, `stream_session_open`, `leased`,
  `env_missing`, `task_busy`, `already_installed`, `job_type_installed`). An
  unknown code answers `false`, so a new refusal is surfaced once rather than
  swallowed by a walk across machines.
- `connect.ts` distinguishes a Crucible on another API major (named, both
  versions) from "not a Crucible", and prefixes the code onto the message because
  a connect door shows only `err.message`.

## Events and streams

- `events()` never reconnects: a job keeps its event log for its life and a
  caller resumes with `lastEventId`. A stream that ends without a terminal event
  throws `CrucibleUnreachable`; ids must increase.
- `sse.ts` handles `\n`, `\r\n` and a lone `\r`; a trailing `\r` waits for the
  next byte so CRLF is not read as two breaks. A frame with no `data:` line is not
  dispatched.
- `chatStream` ends on `data: [DONE]`; a body that ends without it throws.
- **TTS sessions (`stream.ts`) reattach on their own**, unlike `events()`. The
  server's grace window is 15 s (`crucible/ttsstream.py` `GRACE_SECONDS`), so the
  client reattaches with `Last-Event-ID` for `REATTACH_BUDGET_MS` (16 s, one
  second of margin) and then gives up. A refusal (`unknown_session`,
  `replay_unavailable`) is never retried. A frame the client cannot read or a
  session-wide `error` travels back; only a dropped socket is reattached.
- `openTtsStream` attaches the event stream and reads `ready` before it resolves,
  so a caller's first `say` can never be refused `stream_not_attached`. The
  session holds one generator shared by `attach` and the public iterator, so audio
  for a row said before iteration waits in the stream and a second `for await`
  resumes where the first stopped. If attaching fails the client closes the
  server-side session, since the server allows only one.
- `ready` repeats the session's identity; it is compared with the open reply and a
  mismatch is a protocol error.
- PCM16 is little-endian; `pcm16` reads it with a `DataView`, which also
  handles an odd byte offset.
- `StreamRestart`: narrator has no per-row cancel, so cancelling one row in a
  batch aborts its neighbours, which the server resubmits. Audio with
  `seq < fromSeq` for that row is void. A `higgs-v3` voice's batch width is 1, so
  it cannot happen there today.
- `gapSec` is narrator's own gap for the row; the player inserts it. `null` means
  the row was cancelled.

## The batch writer (`writeArtifactsTo`)

- There is no shared mount between a client and a Crucible (WSL cannot mount the
  `Z:` NAS share), so artifacts are fetched over HTTP even from localhost.
- Each artifact is fetched as its `artifact` event lands, at most
  `DEFAULT_ARTIFACT_CONCURRENCY` (4) at once. It is a ceiling, not a throughput
  knob: a replayed history of 1,400 chunks would otherwise open 2,800 sockets.
- Each file is written to a sibling `<name>.<stamp>-<counter>.part` and renamed. BookForge's
  resume counts any `<index>.flac` over 1024 bytes as done, so a half-written file
  must never carry that name. The temporary must be in the same directory (a
  cross-filesystem rename is a copy). Node's `rename` replaces an existing file on
  Windows too, so no unlink window.
- The sidecar is written first, so `<index>.flac` existing implies
  `<index>.flac.provenance.json` exists. The sidecar is parsed to prove its shape,
  but the server's own bytes are what is written.
- Artifact names are checked as single path members before becoming a path, even
  though the server validates them too.
- Without `lastEventId`, the terminal `done` list is reconciled so an artifact
  whose event was missed is still written. With it, only what arrives after the id
  is written.
- A failed write throws out of the iterator; the first failure is kept, and
  in-flight writes are settled before the error leaves.

## Pairing

- A `crucible://` line percent-encodes the server name as RFC 3986 userinfo
  because the default name (`crucible@host`) contains `@`. A second literal `@`
  is refused, not guessed. A port is required. `parsePairing` is pure and never
  puts the token (the fragment) in an error.
- `cruciblePairingPath` mirrors `crucible/config.py`'s `crucible_home()`:
  `$CRUCIBLE_HOME`, else `%LOCALAPPDATA%\Crucible\pairing` on win32 (unset
  `LOCALAPPDATA` is refused, never assembled from a username), else
  `~/.crucible/pairing`.
- `readPairingFile` answers `null` only for ENOENT/ENOTDIR. An empty file, a file
  with more than one line (the writer, `crucible/pairing.py`, writes exactly one)
  or any other read error throws: answering "no server here" would send a user to
  install a second Crucible over a running one.

## Orchestrators

`engineOf(info)` returns `null` for an engine, the `EngineRef` for an
orchestrator (follow `engine.url` once, with the same token, and refuse a second
hop), or throws `orchestrator_has_no_engine`. An orchestrator's `engine: null` is
read strictly.
