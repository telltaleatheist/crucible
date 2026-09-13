# Phase 3b: the `tts` job type

Contract for the largest capability. Extends DESIGN.md; where the two disagree this file
wins for `tts`. Written 2026-09-13 from `docs/CLIENT-SURFACES.md` sections 3 and 10 (tier 3)
and Owen's two rulings — the division of knowledge (DESIGN.md section 3.1) and *voices come
from the server*.

Owen, 2026-09-12, on the second one, verbatim:

> we're going ot have to pass in the actual voice. maybe we retrieve available voices from
> crucible and we send in the order for deathstalker, the voice that was returned by the
> server. and the server will have to have a streaming or a rendering option for higgs. the
> bookforge browser extension uses streaming a lot. i use it every sunday

Those are the two doors, and the last sentence is why both are mandatory rather than one
being a later addition.

## 1. What moves, and what does not

**Moves to Crucible.** The engine process and its lifetime. The accelerator lease. Which
voice is resident. Sampling per (voice, backend). Cap certificates per (voice, backend). The
token-budget formula that turns characters into `max_new_tokens`. Orpheus's EOS levers. The
reference clips a zero-shot voice is conditioned on. The WSL spawn, the path rewriting and
the per-engine VRAM arithmetic — all of it deleted rather than ported.

**Stays in BookForge.** Chunking (`listen-chunks.ts`, the paragraph packer). Text
normalisation (`listen-text.ts`, `tts-punctuation.ts`, the number rules, the CAPS fold, the
glyph strip). The pace guard's *judgment* — which takes are good, which rows need a retake,
when a hole is a hole. The retake decision. Assembly. The session layout and the resume
rule. The ledger.

The line between those two paragraphs is the division of knowledge: **the server measures,
the client judges.** Crucible reports what a chunk actually did — its duration, its
characters per second, whether it hit the cap — and never decides what to do about it.

## 2. A voice is a manifest, and the server advertises it

`voices/<id>.toml` in this repo, one file per voice id, mirroring `models/<id>.toml`
exactly. The id is Crucible's and stable across backends; the weights and the numbers differ
per backend.

```toml
[voice]
id = "deathstalker"
display = "Deathstalker"
kind = "checkpoint"           # checkpoint | zeroshot | token
narrator_engine = "higgs-v3"  # which of narrator's engines serves it
language = "en"

# The band the chunk packer works to. Advertised so a client can pack to it; the
# client does the packing, the server states the shape. These are the numbers that
# live in BookForge's higgs-models.json voice document today.
[voice.pace]
target_chars = 300
max_chars = 800
safe_min_chars = 120
safe_max_chars = 700
pace_chars_per_sec = 15.0
max_chars_per_sec = 19.0
min_chars_per_sec = 9.0

[voice.backends.cuda-linux]
hf_repo = "owenmorgan/deathstalker-higgs-v3"
revision = "<40 hex>"
memory_bytes_estimate = 0        # MEASURED on the card it names, never computed
cap_tokens = 800                 # THE cap certificate for (voice, cuda-linux)
sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }

[voice.backends.mlx-darwin]
hf_repo = "owenmorgan/deathstalker-higgs-v3"
revision = "<40 hex>"
memory_bytes_estimate = 0
cap_tokens = 800
sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }
```

Four things in that file are load-bearing.

**`cap_tokens` is per backend and must stay per backend.** Every voice's two blocks carry
identical numbers today — 600/600, 800/800, 1000/1000, 1100/1100 (CLIENT-SURFACES.md section
3.1). That is a coincidence of the current catalog, not a property of the world, and
DESIGN.md section 3 already promises a cap certificate per (model, backend). One number
shared between two accelerators is how a cap measured on one card silently governs the
other.

**`sampling` is the engine-level default, and a deviation owes a reason.** Owen's rule, in
memory as `higgs-sampling-default-no-deviation`: 0.8 / 0.95 / 50 is *the boson default*, one
engine-level number, and a per-voice deviation requires a written reason. So a backend block
whose `sampling` differs from its engine's default must also carry `sampling_reason`, and
the loader refuses it if it does not. On SGLang, sampling is **mandatory per request** —
omitting `top_k` samples the untruncated 1026-way codebook tail, measured as one chunk
running to the cap with 80 s of silence — so there is no "leave it to the engine" option
here, and the manifest is where the numbers have to be.

**`memory_bytes_estimate` is measured or the block does not exist.** The same rule the model
manifests learned the hard way: a computed estimate came out 34% light on `mlx-darwin` and
6% light on `cuda-linux`. A voice that has not been measured on a backend carries no block
for that backend, and the server refuses to load it there by name rather than guessing.

**Zero-shot clips belong to the voice, not to the request.** A `kind = "zeroshot"` voice
carries its reference clips the way a checkpoint carries its weights:

```toml
[voice.backends.mlx-darwin]
hf_repo = "owenmorgan/higgs-zeroshot-refs"
revision = "<40 hex>"
clips = [
  { file = "stranger-01.wav", transcript = "...", seconds = 8.4 },
]
```

The transcript is the book-exact text the clip was cut from and is **never an ASR guess**
(`narrator/engine/protocol.py` refuses an empty one; the training-text doctrine is the same
law). There is one exception and it is named rather than general: a voice may declare
`clips = "from-request"`, which means this voice id exists precisely so an operator can hand
over a clip that is not published yet, and a job naming it must carry the clips in its
`inputs`. That is the only way audio ever crosses the wire in the render direction, and a
voice that does not declare it **refuses** clips in the request rather than ignoring them.

### `GET /v1/voices`

The rows are what `/v1/info`'s `tts` capability carries **verbatim**, produced by the same
function — the same rule, and for the same reason, as `llm`'s models (PHASE2-LLM.md section
5). One voice, one description; a client never reconciles two.

```json
{
  "id": "deathstalker",
  "display": "Deathstalker",
  "kind": "checkpoint",
  "language": "en",
  "backend_supported": true,
  "installed": true,
  "resident": false,
  "loadable": true,
  "reason": null,
  "revision": "<40 hex>",
  "fingerprint": "deathstalker@<40 hex>",
  "memory_bytes_estimate": 0,
  "cap_tokens": 800,
  "sample_rate": 24000,
  "takes": 3,
  "pace": { "target_chars": 300 }
}
```

`revision`, `fingerprint`, `memory_bytes_estimate` and `cap_tokens` are `null` when
`backend_supported` is false, because they live in the backend block this host does not
have — and `0` would read as "needs nothing".

**`sampling` is deliberately not on that row.** It is engine tuning, it is the server's, and
publishing it invites a client to send it back. The same goes for the EOS levers, the
`max_new_tokens` formula and the engine flags. What a client gets is the shape it must pack
to (`pace`, `cap_tokens`) and the identity it must record (`fingerprint`).

## 3. The take ladder is the server's steps and the client's judgment

`docs/PLAN.md`: *the ladder's steps are server config; the client asks for take N.*

```toml
[[voice.takes]]
# take 0 — the boson default. No deviation, so no reason is owed.

[[voice.takes]]
temperature = 0.7
reason = "measured 2026-09-11 over the same 88 chunks: 0.8 gave 4 guard fires / 3 holes / 0 drops, 0.7 gave 8 / 7 / 0. A second take at 0.7 is a different draw, not a better setting."
```

A `tts` job carries `take: N`, an index into that list, and nothing else about sampling. The
client decides *that* a row needs another take and *which* take it keeps; the server decides
what take 1 means for this voice on this backend. A `take` past the end of the list is
refused by name (`unknown_take`) rather than clamped — a silent clamp is a retake ladder
that stops climbing without telling anyone.

This is the one place where the division of knowledge had a genuinely arguable alternative
(move the whole ladder, judgment included). It is written here so that changing it later is
a decision rather than a drift. **Owen has this open as a question.**

## 4. Engines: narrator is the managed subprocess

Crucible does not reimplement Orpheus's EOS surgery. It runs the code that already has it.

`python/narrator` in the BookForge repo is a proper installable Python package with exactly
the shape Crucible's engine layer wants: an engine registry keyed by id (`orpheus`,
`higgs-v3`), a per-engine extras matrix in its `pyproject.toml` that mirrors Crucible's
`envs/<type>/<backend>.txt` one for one, a resident server (`python -m narrator.serve`)
speaking newline-delimited JSON over stdin and stdout, and the EOS levers, caps, guards and
codec arithmetic that CLIENT-SURFACES.md row 15 calls "the single hardest thing in the `tts`
contract" already implemented and already measured.

So: **`narrator` is to `tts` what `vllm` is to `llm`.** A pinned dependency in the env
recipe, started and stopped by `crucible/engines/narrator.py` behind the same `Engine`
interface, SIGTERM only, its stdout and stderr in `~/.crucible/logs/engine-<voice>.log`.
narrator itself starts SGLang-Omni on `cuda-linux` and loads MLX in-process on
`mlx-darwin`, and tears its engine down when it exits — which is exactly the lifetime
Crucible's lease wants (CLIENT-SURFACES.md section 3.4).

Two consequences, both named rather than hidden:

- **The readiness probe is not HTTP.** Every other engine answers `/v1/models`;
  `narrator.serve` answers a `ready{device,backend}` line on stdout. `SubprocessEngine` has
  to grow a seam for that, rather than the narrator engine faking an HTTP server to fit the
  one that exists.
- **Crucible ends up depending on a package that lives in an app's repo.** narrator's
  `engine/` and `serve/` know nothing about audiobooks, but `compat/` and `assemble/` do,
  and the whole thing is versioned with BookForge. The env recipe pins it by git sha so a
  server is never surprised. **Extracting `narrator` into its own repo is an owed ruling for
  Owen**; until he makes it, this pin is the honest arrangement rather than a workaround.

`envs/tts/` holds one recipe per (narrator engine, backend) rather than one per backend, and
the reason is in narrator's dependency matrix rather than in Crucible's design: Orpheus
needs `vllm==0.7.3` (the last version whose V0 engine takes per-request logits processors,
which is what the EOS boost *is*), Higgs v3 needs `vllm-omni==0.28.0` against torch 2.13,
and installing both into one env resolves torch twice and breaks whichever loses. So
`cuda-linux` gets `~/.crucible/envs/tts-higgs-v3/` and `~/.crucible/envs/tts-orpheus/`, and
the voice manifest's `narrator_engine` picks which one a load uses. On `mlx-darwin` both
engines share one env, because on the Mac they genuinely do.

Two site-packages patches must be re-applied after any upgrade of the `higgs-v3-server`
group — `work/patch_vllm.py`, without which every voice-clone request is HTTP 400, and
`work/patch_tail_trim.py`, without which every chunk ends in about 240 ms of audible
garbage. pip cannot express that. `crucible doctor` checks for both and reports them by
name.

## 5. Residency holds one thing, whatever kind it is

`crucible/jobs/llm/residency.py` today holds at most one resident *model*. The accelerator
does not care what kind of thing is on it, and a card holding a Higgs checkpoint has no room
for a 9B. So `Residency` generalises: **at most one resident engine, of either kind**, and
loading a voice unloads a model exactly as loading a model unloads a voice.

`GET /v1/health`'s `resident_models` keeps its name and its shape (a list of ids) and gains
`resident_kind: "llm" | "tts" | null`, so a client can tell which door to knock on.

New job types, mirroring the model pair exactly:

| Type | Refusals, all before queuing |
|---|---|
| `load-voice` | `unknown_voice`, `voice_not_installed`, `backend_unsupported`, `env_missing`, `accelerator_busy`, `insufficient_memory` |
| `unload-voice` | `voice_not_resident` |

## 6. The render door — job type `tts`

A normal job on the exclusive lane.

```json
{
  "type": "tts",
  "model": "deathstalker",
  "params": {
    "language": "en",
    "take": 0,
    "chunks": [
      { "index": 41, "text": "He had been walking for some time." },
      { "index": 42, "text": "The road did not appear to end." }
    ]
  }
}
```

**`model` is the voice id.** The wire's word for "the thing that produces the bytes" is
`model`, and for `tts` that thing is the voice — which for Higgs is not a pun but the
literal truth: a v3 voice *is* the merged checkpoint the engine was started on
(CLIENT-SURFACES.md section 3.3, "a Higgs voice change is a full worker restart"). So
`describe_models()` for this type returns the voices, `/info`'s `tts` capability rows are
voices, and provenance records `model: {id: "deathstalker", revision: "<checkpoint sha>"}`
with no new vocabulary invented for it.

**Artifacts: `<index>.flac`, one per chunk**, mono 24 kHz PCM_16 — byte for byte the format
BookForge's assembly and resume already expect, so nothing downstream changes. Each carries
its provenance sidecar, as every artifact does.

**A render job may load its voice; a stream may not.** This is the one asymmetry with `llm`,
and it is deliberate. A chat request is fine-grained and unattended — two clients alternating
would thrash the card, so the proxy never loads (PHASE2-LLM.md section 5). A render job is an
operator's explicit order, it owns the exclusive lane for its whole duration, and it is the
thing the queue was built to serialise. So if the wrong voice (or none) is resident when a
`tts` job reaches the front of the lane, the job loads it, emitting `warming` events exactly
as `load-voice` does. The streaming door, being a connection rather than a job, behaves like
chat: it refuses with `voice_not_resident` and names what is resident instead.

**Progress and measurement.** Alongside the standard `progress {fraction, message}` and
`artifact {name}`, a `tts` job emits one new event per chunk:

```
chunk {index, seconds, chars, chars_per_sec, tokens, capped, take}
```

That is the whole guard interface. `capped` is true when generation stopped because it hit
`cap_tokens` rather than because the model finished — the difference between "a long
sentence" and "a runaway", which BookForge's PaceTracker needs and cannot infer from a
duration. The server measures and reports; **it decides nothing**, and it never retakes on
its own.

`chunk` is an addition to DESIGN.md section 4's event vocabulary. It is additive and
`api_version` does not move: a client that does not know the kind still sees every
`progress`, `artifact` and `done` it saw before. The SDK yields unknown event kinds through
rather than dropping them, so that stays true for the next one too.

**Resume stays client-side.** BookForge already knows which `<index>.flac` files exist and
exceed 1024 bytes; it sends the chunks it still needs. The server has no session, no project
and no memory between jobs — DESIGN.md section 10, "no library, no project files, no
per-user state" — and resume is exactly the kind of state that would break it.

**The batch writer is the SDK's.** No shared mount, ever: `Z:` is invisible to WSL, which is
the whole reason whole-m4b alignment cannot run on this PC today. The SDK fetches each
artifact as its `artifact` event lands and writes `<index>.flac` where assembly and resume
look, overlapped with the next chunk's generation.

## 7. The streaming door — `GET /v1/tts/stream` (WebSocket)

The Listen path, the in-app Play button, and the browser extension Owen uses every Sunday.
Its requirements are not the render door's with a smaller buffer; they are different in kind
(CLIENT-SURFACES.md row 18): sub-sentence audio emitted *while a row is still generating*,
rows retired out of order within a batch, and a cancel that aborts work in flight.

WebSocket rather than SSE, because every one of those needs the client to speak mid-stream.
The bearer token travels in the `Authorization` header like every other route. Browsers
cannot set headers on a WebSocket, but no browser talks to Crucible: BookForge's own TTS
WebSocket on 8766 stays exactly where it is and becomes a **relay**, which is also what
keeps the extension working unchanged.

**Control frames are JSON text.** Client to server:

```
{"op": "hello",  "voice": "deathstalker", "language": "en"}
{"op": "say",    "id": "r12", "text": "...", "take": 0}
{"op": "cancel", "id": "r12"}
{"op": "cancel_all"}
{"op": "close"}
```

Server to client:

```
{"op": "ready",  "voice": "...", "fingerprint": "...", "sample_rate": 24000, "backend": "..."}
{"op": "done",   "id": "r12", "seconds": 3.41, "chars": 98, "chars_per_sec": 28.7, "capped": false}
{"op": "error",  "id": "r12", "code": "...", "message": "..."}
{"op": "closed", "reason": "..."}
```

**Audio frames are binary and self-describing**, so PCM never pays for base64 and a frame
never depends on the JSON frame before it:

```
b"CRU1" | uint32 seq | uint16 id_len | id (utf-8) | pcm16le mono at sample_rate
```

Out-of-order retirement falls out of that: ids are the client's, `seq` counts within an id,
and `done` for one id may arrive while another is still emitting. There is no batching
parameter on the wire — how many rows the engine runs at once is engine tuning and belongs
to the server (Higgs measured worthless above width 1; Orpheus runs 16).

**A keepalive, because the thing being replaced has none.** `narrator.serve` has no
heartbeat of any kind, which is why BookForge carries a 12-minute no-heartbeat watchdog and
a 30-second poll. Crucible sends a WebSocket ping every 15 s — the interval the SSE streams
already use — and a client that stops answering is disconnected and its rows cancelled. A
dropped connection cancels everything it had in flight: the same rule as the `llm` proxy,
for the same reason. Work nobody is waiting for is time stolen from the next job.

## 8. API additions

| Route | Auth | Returns |
|---|---|---|
| `GET /v1/voices` | yes | the rows in section 2 |
| `POST /v1/jobs {type: "load-voice", model}` | yes | a job; `warming` while narrator starts, `done {resident: id}` |
| `POST /v1/jobs {type: "unload-voice", model}` | yes | a job; `done {resident: null}` |
| `POST /v1/jobs {type: "tts", model, params}` | yes | a job; `chunk` per chunk, `artifact` per FLAC |
| `GET /v1/tts/stream` | yes | WebSocket, section 7 |

`GET /v1/info` gains a `tts` capability whose rows are `/v1/voices`' rows verbatim.
`GET /v1/health` gains `resident_kind`.

## 9. SDK additions (`@crucible/client`)

- `voices()` → `VoiceInfo[]`, `loadVoice(id)` / `unloadVoice(id)` → job ids.
- `render({voice, language, take, chunks, signal})` → a job handle whose events are typed,
  `chunk` included, plus `writeArtifactsTo(dir)` — the batch writer of section 6.
- `stream({voice, language})` → a session: `say(id, text, take?)`, `cancel(id)`, `close()`,
  and an `AsyncIterable` of `{id, seq, pcm: Int16Array}` interleaved with `{id, done}`.
  Zero runtime dependencies still: Node 20's `WebSocket` is global, and Electron's renderer
  has one too.
- Typed refusals for every named code in this document.

## 10. Verification

No new number is written into a manifest without the accelerator it was measured on, and
both of Owen's cards were busy the night this was written. So the order is: build against a
**fake narrator** — the same shape as `tests/fake_engine.py`, a script that speaks the
JSON-lines protocol and emits a sine wave — prove every wire behaviour in pytest, and leave
exactly one owed item.

**Owed, and it must be done before BookForge renders a book through this:** on the PC, load
`deathstalker` on `cuda-linux`, render a chapter, measure the real VRAM and write it into
the manifest, and confirm the audio is identical in format to what
`narrator.compat.worker` produces today. On the Mac, the same through MLX. Then the
streaming door with the browser extension pointed at BookForge's relay — the Sunday test,
and the only one that matters to Owen personally.

## 11. What this deliberately does not do

- **No assembly.** Crucible returns chunks; the m4b is BookForge's.
- **No chunking and no text normalisation.** Section 1.
- **No retake decision.** Section 3.
- **No session.** Section 6.
- **No voice creation.** Training a checkpoint and publishing it to HuggingFace is
  `orpheus-finetune`'s job and stays there. Crucible pulls a published voice at a pinned
  revision, exactly as it pulls a model.
