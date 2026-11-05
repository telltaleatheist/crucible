# narrator — the BookForge-owned replacement for ebook2audiobook

Status: PROPOSAL (2026-09-04). Nothing here is built. Name is a placeholder.

## Why

ebook2audiobook was useful when it managed XTTS for us. Today Orpheus is the whole
pipeline, and the fork is 155 commits / +11,118 lines past upstream: the engine
(`orpheus.py`, 5,507 lines vs 706 upstream), the parallel worker layer
(`bookforge_ext/parallel`, 3,015 lines) and 35 of `core.py`'s functions are ours.
What upstream still provides is three things — stanza sentence segmentation under our
chunk packer, the assembly recipe, and the VTT — and it costs us three checkouts
synced by git, a fork that can never merge upstream, and gradio/stanza/calibre
weight in every env. Calibre is never used: by the time text reaches the engine it is
an EPUB from Foundry, every time.

## Non-negotiable contracts (what other pieces rely on)

These are the seams the new module must honour byte-for-byte on day one, because the
app, the CLI, Studio, the training tools and the streaming server all read them:

1. **The headless door.** `parallel-tts-bridge.ts`, `reassembly-bridge.ts`,
   `correct-sentences-bridge.ts`, `book-render-service.ts` and `cli/bookforge-tts.py`
   spawn `app.py --headless` with ~40 flags (`--prep_only`, `--worker_mode`,
   `--sentence_start/end`, `--assemble_only`, `--sentences_dir`, `--session`,
   `--resume_session`, `--list_sessions`, `--sentence_indices`,
   `--sentence_overrides`, `--num_takes`, `--take_temperatures`,
   `--orpheus_model_dir/--orpheus_base_dir/--orpheus_adapter_dir`,
   `--post_render_filter`, …) plus `ORPHEUS_*` env caps.
2. **The session layout.** `sessions/<lang>/ebook-<uuid>/<hash>/chapters/sentences/*.flac`
   and its JSON sidecars. Resume, the project sentence cache, Studio's per-sentence
   retakes, clip/cut/training tools and the reassembly bridge all walk it.
3. **The worker progress lines** (`worker_core.py` format) — the bridges' regexes and
   the watchdog read them; the MLX heartbeat line is additive-compatible by design.
4. **The streaming worker protocol** — `orpheus_stream.py`'s stdin/stdout JSON lines,
   including `batch_chunk` / `batch_item{streamed}` for fast start.
5. **The VTT.** Cue text = the exact chunk text as rendered, cues in sentence order,
   sidecar binding beside the m4b. (Measured 2026-09-04: e2a writes NO cue identifiers
   and NO `NOTE` blocks; an earlier draft of this line said otherwise.) Training and
   debugging read it as the contract between sentence index, rendered file, text and time.
6. **Guard events** — `[ORPHEUS][ORPHEUS_GUARD_EVENT] {json}` lines the bridge parses
   into the reject dir.
7. **Voice caps** — `orpheusVoiceCapsForModel` → env / `register_voice_caps`
   (`VOICE_CAP_SOURCES` keys) unchanged.

## Shape

One Python package in the BookForge repo — `python/narrator/` — installed editable
into the two Orpheus envs (WSL `orpheus_tts`, Mac `ebook2audiobook-orpheus`). ONE
checkout per machine: WSL imports the package over `/mnt/c/.../bookforge/python`
(code only — sessions and scratch stay on ext4 as today). No more e2a checkouts, no
more `cp`/CRLF trap, no more "which checkout is stale".

```
python/narrator/
  manifest.py    THE render manifest — the one source of truth (see below)
  text/          EPUB → chapters → sentences → chunks (the packer, ported from core.py)
  engine/        OrpheusEngine: vLLM + MLX backends, caps registry, LoRA adapters,
                 EOS boost/floor, truncation guards, rate ratchet, batched decode,
                 generate_batch_stream + the windowed SNAC decoder, fastpath
  render/        session store (layout v1 = today's), workers, resume, retakes,
                 progress protocol   (port of bookforge_ext/parallel + worker_core)
  assemble/      per-chapter parallel AAC, stream-copy concat, remux with chapter
                 atoms + cover, transcript embed, VTT from sample counts, gap
                 realization, duration guards
  serve/         the streaming server (orpheus_stream.py moves here, same protocol)
  compat/app.py  accepts today's --headless flags and routes them; retired flag by
                 flag once each bridge is modernized
  cli.py         python -m narrator prep|render|assemble|serve|retake|sessions
  tests/         the e2a tools/test_*.py move here + golden-book parity fixtures
```

**The render manifest** is the design's centre. Prep writes it; render, assembly, the
VTT, resume, retakes and the training exporters all READ it and never re-derive it:

```
{ version, book: {epubPath, sha}, voice, caps: {maxChars, ...},
  chapters: [{index, title, chunks: [{index, text, kind: prose|heading|item,
              gapBefore, gapAfter, file: "chapters/sentences/<index>.flac" | null,
              samples: <int|null>, take: <n>}]}] }
```

Cue times are a running sum of `samples` (from each FLAC's STREAMINFO — exact, no
ffprobe) plus the realized gaps. The concat list for the encoder is generated from
the same rows. That is what makes the VTT and the audio unable to disagree, which is
a stronger guarantee than today.

**The engine** keeps every behaviour it has now; it loses `TTSUtils`/`TTSRegistry`,
the e2a session dict and `_save_audio`, and gains a small `SentenceSink` interface
(write a rendered chunk + its sample count into the manifest). `generate_batch_stream`
and the caps registry are already shaped for this.

**Assembly** encodes each chapter as its own AAC in parallel (16 wide here; `aac_at`
on the Mac), stream-copies the chapters together, and remuxes once with chapter
atoms and cover. Later: encode a chapter the moment its last chunk lands, so the
end-of-job assembly becomes a concat and a remux. Chapter boundaries carry ≥0.5 s of
silence, so AAC's ~21 ms priming at a join is inaudible; the duration guards (post
Witnesses / Nuremberg) move to per-chapter checks.

## What is deleted

gradio, calibre, stanza (after step 4 decides), XTTS/F5/Voxtral/bark/vits/tacotron/
fairseq engines, TTSManager/registry, device_installer (the component system owns
envs), argos translation, voice_extractor, the 63 upstream `core.py` functions not
on the headless path, `lib/conf.py`'s UI-era settings (the CUDA/WSL env block moves
into `engine/`).

## Migration — each step shippable, each parity-tested

0. **Golden set.** Three books rendered with today's e2a (short; long; headings-heavy):
   chunk lists, session dirs, VTTs, m4b durations, guard events → fixtures.
1. **assemble/** replaces `--assemble_only` (reassembly bridge + CLI) reading the SAME
   session dir. Parity: identical cue text and count, cue times within 1 ms (then
   exact), equal m4b duration and chapter count. Biggest visible payoff (Mutineer
   8 min → <1 min), lowest risk.
2. **engine/ + serve/** move together (they are one import). Parity: vLLM is
   deterministically seeded per batch composition, so a fixed batch must produce
   byte-identical FLACs; streaming server smoke on both machines.
3. **render/** ports the worker layer, keeping session layout v1. Parity: resume an
   existing e2a session dir untouched; progress lines identical.
4. **text/** ports the packer (stanza stays until parity is shown on the golden set;
   diffs are ear-checked, not just diffed — chunk boundaries change the read).
5. **Cut over**: `getDefaultE2aPath()` → the package; delete the WSL and Mac e2a
   checkouts; `pip install -e` in both envs. Then retire compat flags one by one.

## Risks

- The session layout has many implicit readers (`grep chapters/sentences`); step 3
  must enumerate them before changing a byte.
- Two Orpheus envs are hand-built (memory: torch pins, vLLM 0.7.3, mlx); the package's
  `pyproject` must pin what those envs already have, not what pip prefers.
- The packer is the one piece whose "parity" is partly a judgment (2026-08-27
  headings, 2026-08-29 list items, min-chars floor); keep it last.

## Managed WSL engine (Windows) — BookForge provisions the GPU environment itself

Orpheus on Windows needs WSL2 for vLLM with CUDA graphs (native Windows is eager-only,
~6x slower). Today that means a hand-built Ubuntu, a hand-built conda env and a git
checkout the user maintains. With narrator as our own package, the WSL side becomes
an ARTIFACT BookForge installs, versions, updates and removes — like the conda-pack
components in Settings → Add-ons, but for a whole distro.

**A dedicated distro, not the user's.** `wsl --import bookforge-orpheus <dir> image.tar`
from an image we build. Isolated from any Ubuntu the user has; `wsl --terminate
bookforge-orpheus` kills only ours (today a wedge means `wsl --shutdown` for
everything — see wsl-wedge-proofing); `wsl --unregister` is a clean uninstall.
The image carries a plain venv at a fixed path (no conda, no `bash -ic`, no
`conda run`): `wsl.exe -d bookforge-orpheus --exec /opt/orpheus/bin/python -m narrator …`.
The narrator package itself is imported from the Windows checkout over `/mnt/c`, so
app updates never need a new image — only torch/vLLM pin changes do.

**Building the image** is a script, not a ritual: a Dockerfile (Ubuntu base + the
pinned wheels: torch 2.5.1+cu121, vLLM 0.7.3, numpy 1.26.4, snac, soundfile…) →
`docker export` → tar → zstd, versioned `orpheus-wsl-vN`. Same hosting as the Windows
env (GitHub Releases, split under the 2 GiB cap, sha-verified, resumable). ~3-4 GB
compressed, ~10-15 GB on disk; say so in the UI. The self-test that gates "ready":
`nvidia-smi` inside the distro, `import vllm`, and a one-sentence render that must
log CUDA graph capture (the exact line in CLAUDE.md's troubleshooting section).

**What BookForge drives, in order** (Settings → Add-ons → "Orpheus GPU engine (WSL)"):
1. Preflight, loud on failure: Windows 11 / 10 21H2+, an NVIDIA Windows driver
   (WSL gets libcuda from it — no CUDA toolkit in the distro), virtualization enabled
   (detectable via `wsl --status`; a BIOS toggle we can only name, not flip), disk.
2. Enable WSL if absent: `wsl --install --no-distribution` under a UAC prompt; this
   one step can need a REBOOT, and the installer must say "reboot, then press
   Install again" rather than pretend. On most Windows 11 machines WSL is already
   present and this step is a no-op.
3. Download + import the image; write nothing to the user's `.wslconfig` unless they
   opt in (it is global to all distros — the memory cap story).
4. Self-test; register the component with its version; the existing
   `spawnWithWslSupport` / path-rewriting machinery targets the new distro name.
5. Updates: compare the installed image version to the catalog; re-import on bump
   (sessions and models live outside the distro — models stay in the Windows
   orpheus-models dir, sessions on the distro's ext4 scratch that a re-import
   recreates).

**Honest limits.** NVIDIA only (vLLM on ROCm-in-WSL is not a real target); the
first-ever WSL enable needs admin and possibly a reboot; corporate policy can block
WSL; a machine with virtualization off in firmware cannot be fixed from software.
Each of those becomes a named preflight failure, never a silent fallback to the slow
native path — that path stays available as an EXPLICIT choice ("no GPU engine").

**The alternative worth keeping on the table: no WSL at all.** NOT vLLM — its
Windows wheel is a broken community build and CUDA graphs do not work through it
(Owen, 2026-09-04; that is the whole reason for WSL). llama.cpp is a different
engine: its own C++/CUDA backend with its own graph capture, no PyTorch, no vLLM,
and it runs Orpheus as GGUF natively on Windows and loads GGUF LoRA adapters at
runtime. UNMEASURED here — treat every throughput claim as a hypothesis until one
chapter has been rendered with it on this machine. If it works: zero-setup for
users, good for streaming/single listener. The cost is batch throughput for whole-book renders (no continuous
batching like vLLM) and a second engine backend to keep at parity with vLLM/MLX
(caps, EOS boost/floor, guards, SNAC decode). Decide by measuring one book's render
time on llama.cpp vs vLLM before choosing; the managed-WSL design above is what
keeps vLLM's throughput for the render box, and the two are not mutually exclusive
(WSL when available, llama.cpp otherwise).

## Multi-engine direction (Owen, 2026-09-04, relayed by the orpheus-training session)

narrator must be able to run other LLM-codec TTS engines, not only Orpheus. Named
targets: **Maya1** (Maya Research; 3B Llama-style + SNAC 24 kHz; Apache-2.0; emotion
tags + voice-description prompt) and **Llasa-8B** (HKUST; Llama-3.1-8B + X-codec2:
ONE codebook of 65,536, 50 tokens/s, 16 kHz output; 1B/3B siblings).

What varies per engine and therefore sits behind an interface in `engine/`:

1. **Codec** — SNAC (3 RVQ levels, 7 tokens per ~12 Hz frame, 24 kHz) vs X-codec2
   (1 stream, 50 Hz, 16 kHz): token->audio decode, frame arithmetic, sample rate. The
   windowed streaming decoder is codec-specific.
2. **Prompt format** — Orpheus/Maya voice-token framing (`[SOH] "voice: text" [EOT]
   [EOH][SOAI][SOS] audio [EOS][EOAI]`) vs Llasa's text + optional reference-audio-
   token prompt for zero-shot cloning.
3. **Stop handling** — audio-token cap (Orpheus 3700 = 44 s; X-codec2 ~4000 = 80 s),
   EOS boost/floor levers, repetition-penalty scope: engine-specific numbers, never
   constants in the scheduler or the bridges.
4. **Chunk sizing** — maxChars / maxCharsPerSec are per-model already; the
   seconds-per-token conversion is per-codec.
5. **Voice identity** — fine-tuned voice token (Orpheus) vs reference clip (Llasa
   zero-shot) vs description prompt (Maya1): three different "voice" objects.
6. **Serving** — vLLM serves all three Llama-style backbones, but the logits
   processors and the decode step differ per engine.

The per-voice tuning catalog (`orpheus-models.json`) and the ladder/caps machinery
key on **(engine, voice)**, not voice alone.

Sequencing: the Orpheus port in `engine/` lands first as a faithful port (step 2).
The interface is extracted AFTER the orpheus-training session reports its Llasa-8B
zero-shot render (prompt format, token rates, decode path measured on the
deathstalker reference), so the seams are shaped by two real engines, not one.
Nothing in the manifest, session layout, VTT or assembly contracts depends on the
engine; `sampleRate` is already a manifest field for this reason.

### Llasa-8B measurements (orpheus-training session, 2026-09-04; full notes in
`E:\training\_campaigns\2026-09-01-cod-full-rebuild\llasa\LLASA_NOTES.md`)

- Codec X-codec2: single codebook 65,536, 16 kHz mono float out, measured 49.07 audio
  tokens/s (nominal 50); decode = X-codec2 decoder over the token list, NO trim
  convention (unlike SNAC's +75 samples per window).
- Prompt: Llama-3.1 chat framing; speech tokens are ordinary vocabulary entries
  `<|s_N|>` plus a few special tokens (ids/order in the notes). Training rows start with
  ONE BOS 128000 - mlx-lm's loader adds its own (trainer trap).
- Zero-shot cloning: the reference clip is embedded as its X-codec2 tokens after the
  reference transcript (12 s clip = 552 tokens). So "voice" = reference clip + transcript.
- Stop invariant is TOTAL SEQUENCE LENGTH (trained window 2048): chunks over ~2200
  total tokens stop early with a CLEAN EOS and silently drop the remaining text (never
  a loop). Ladder: 1500 chars 3/3 failed, 750 chars 5/6 failed, 375 chars 12/12 clean.
  => the chunk-sizing cap for this engine is tokens-in-window (prompt + reference +
  audio), not chars/sec; the manifest's caps object must be per (engine, voice).
- EOS fires reliably unaided: no boost/floor needed (Orpheus-specific levers stay
  behind the Orpheus backend).
- Serving: plain Llama causal LM with an extended vocab; TokensPrompt in, ids out,
  decode outside; transformers bf16 single-stream 20 tok/s (0.38x realtime, 19 GB).
- License CC BY-NC 4.0 (personal use here; Owen: not a blocker).
- Fine-tune prepared on the Mac (MLX bf16, LoRA r32 attn+MLP, 20.6 GB peak, 3.1 s/it)
  but NOT run: Owen wants Higgs auditioned first; Higgs notes follow, and the engine
  interface is extracted after BOTH are in hand.

### Ruling 2026-09-04 ~16:25 (Owen, relayed): Llasa is OUT; second engine = Higgs v3

Llasa is counted out (16 kHz codec, 2048-token stop wall). Likely engine set:
**Orpheus + Higgs v3 (4B)** - Owen's ear on a v2 sample: "not bad, about Orpheus
quality"; the v3 render is in progress and the eight-field notes follow. Card facts:
backbone Qwen3-4B, context 8,192; tokenizer = 8 codebooks x 1,026 entries with a delay
pattern and a fused multi-codebook embedding, 25 frames/s, 24 kHz mono (NOT SNAC), so
~5 min of audio per window vs Orpheus's 44 s; voice object = reference clip in the
conversation history (or a scene description; multi-speaker assignment is native);
serving via vLLM-Omni / SGLang-Omni (a separate serving stack from our vLLM 0.7.3; a
WSL env `higgs3` exists with vLLM-Omni); license research/non-commercial (accepted:
personal use); no training code - a LoRA fine-tune is build-it-ourselves. Higgs v2
(3B Llama-style, transformers HiggsAudioV2, LoRA-able) is the comparison only.

Interface consequences: the codec seam must cover multi-codebook-with-delay-pattern
decode (Higgs) as well as SNAC's 7-token frames; the serving seam must allow a
different server stack per engine (vLLM 0.7.3 vs vLLM-Omni) behind the same
JSON-lines worker protocol; the voice seam carries a reference clip + transcript in a
conversation history. Extraction starts when the Higgs v3 measured fields arrive.

### Higgs measurements (orpheus-training session, 2026-09-04; full notes in
`E:\training\_campaigns\2026-09-01-cod-full-rebuild\higgs\HIGGS_NOTES.md`)

Both v2 and v3: 24 kHz mono; tokenizer 8 codebooks x 1024 + stream bos/eos 1024/1025;
25 LM steps per second of audio (delay pattern); decode = codec over the frame matrix
with a `frames - 7` delay-pattern trim; NO pads or fades at either end (assembly must
supply gaps); EOS fires reliably unaided; no cap hits.
- **v2** (`higgs-audio-v2-generation-3B-base`, transformers `HiggsAudioV2ForConditionalGeneration`,
  `use_text_head=True` for PEFT/SFT): deathstalker clone from two reference clips in the
  chat history; nine Pokemon paragraphs (132-898 chars) at length ratios 0.83-1.26, EOS
  9/9, zero runaways; gen RTF 1.12 single-stream, decode RTF 0.004, 16.3 GB peak, 15.6 s
  load. Licence: Boson Higgs Audio 2 Community (attribution; "Higgs Audio 2" in any
  fine-tune's name).
- **v3** (`higgs-audio-v3-tts-4b` via vllm-omni 0.28.0, `--attention-backend FLASH_ATTN`;
  FlashInfer fails on sm_86 with torch 2.13+cu130): serves text-only (default voice) but
  every reference-audio shape returns HTTP 400 "Token id -100 is out of vocabulary"
  (config `audio_token_id -100`, tokenizer has no audio placeholder) - cloning BROKEN in
  that stack; stops early on long paragraphs un-cloned (ratio 0.49-0.64); RTF 2.46;
  cold start 55 s; preallocates ~24 GB. Licence research/non-commercial. A v3 port waits
  for a serving stack whose cloning works (SGLang-Omni docker is the card's path).
- Bandwidth vs the real narrator (>4 kHz energy): Orpheus 0.346 % vs 0.358 % real;
  Higgs v2 duller (0.229 %); v3 brighter but a different speaker.

## Engine interface (EXTRACTED 2026-09-04 - see "What is real" at the end)

Shaped by two measured engines (Orpheus/SNAC and Higgs/8-codebook delay pattern) and
one rejected (Llasa). Everything below the manifest is engine-specific; nothing above
it is. `engine/` gains a small protocol and the Orpheus mixins become its first
implementation without changing a line of their bodies.

```
engine/
  protocol.py     Engine, Codec, VoiceRef, Budget, Backend  (typing.Protocol + dataclasses)
  orpheus/        today's engine/ modules moved unchanged (OrpheusEngine implements Engine)
  higgs/          later: v2 via transformers first (the one whose cloning works)
  registry.py     engine id -> factory; catalog rows carry {engine, voice}
```

- **`Codec`**: `sample_rate`, `frames_per_second`, `tokens_per_frame` (SNAC 7 @ ~12 Hz;
  Higgs 8 codebooks @ 25 Hz with delay pattern), `decode(tokens) -> float32 mono`,
  `trim` convention (SNAC: +75 samples per window; Higgs: `frames - 7`), and the
  windowed streaming decoder (`WindowedFrameEmitter` is SNAC's; Higgs needs a
  delay-pattern-aware one or a per-frame decoder - measured later).
- **`VoiceRef`**: a tagged union - `{kind: 'token', name}` (Orpheus fine-tuned voice),
  `{kind: 'clips', clips: [(path, transcript)], history: ...}` (Higgs chat-history
  clones), `{kind: 'description', text}` (Maya1). The catalog (`orpheus-models.json`
  becomes `engines/*.json`) keys tuning on `(engine, voice)`.
- **`Budget`**: `max_chars(voice)`, `max_chars_per_sec(voice)`, and
  `max_total_tokens(prompt_tokens, voice)` - Orpheus's audio-token cap (3700) and
  chars/sec guard; Higgs's 8,192-token window at 25 tok/s. The chunk packer (step 4)
  asks the Budget, never a constant. Llasa's 2048 wall is what proved this seam.
- **Stop levers** (EOS boost/floor, rate ratchet, truncation ladder, short-chunk
  backstop) are Orpheus-private; the protocol exposes only `stop_policy(voice)` so the
  worker and the streaming server stay engine-agnostic. Higgs: none needed (measured).
- **`Backend`**: vLLM 0.7.3 (Orpheus), MLX (Orpheus on Mac), transformers (Higgs v2),
  vLLM-Omni/SGLang-Omni (Higgs v3, later). The serve worker's JSON-lines protocol and
  `generate_batch_stream(texts, voices, stream_rows, on_chunk, on_row, should_stop)`
  are the engine-agnostic surface; each backend implements them.
- **Gaps**: Orpheus bakes lead/trail silence into each chunk FLAC; Higgs emits none,
  so gap realization moves to the SentenceSink (engine reports `pads: bool`) and the
  manifest's `gapBefore/gapAfter` become live for Higgs. The assembler already reads
  them.
- Guard events, progress lines and the VTT are unchanged by engine choice.

### Ruling 2026-09-04 (evening): the second engine is Higgs v3; v2 is DROPPED

Owen, relayed: "basically just Orpheus and we know Orpheus better". Higgs **v2**
is not shipped. The v2 code built earlier that day is KEPT as interface
SCAFFOLDING - registry id `higgs-v2-scaffold` - because it is a complete, tested
reference implementation of the Protocol for an engine that is not SNAC, emits no
pads and carries its voice as clips in a chat history, and it proves the seams
fit something that is not Orpheus without a server in the way. No GPU smoke is
owed for it and nothing renders a book with it.

The v3 cloning failure reported in the earlier audition was WRONG about the
cause: it was a one-line vLLM 0.28 bug rejecting vllm-omni's -100 audio
placeholder, fixed by `work/patch_vllm.py`. Cloned v3 measures ECAPA cosine
0.704 against a 0.766 narrator self-ceiling (92 %).

### What is REAL vs what is still DESIGNED (2026-09-04, end of the engine task)

The sections above are the design. This is the state of the code, so nobody
plans against a paragraph that has not been built. Full detail:
`python/narrator/engine/PORT_NOTES.md` section 12.

**Real, tested, on both interpreters (Windows 3.12 and WSL `orpheus_tts` 3.11):**

- `engine/protocol.py` - `Codec`, `Budget`, `Engine`, `ServedBackend` as
  runtime-checkable Protocols; `StopPolicy` (with `coverage_check`),
  `BackendSpec` (`inprocess` | `served`), `SpeechRequest`, `ReferenceClip` and
  the `VoiceRef` union as frozen dataclasses. No torch, no backend.
- `engine/registry.py` - `orpheus`, `higgs-v3`, `higgs-v2-scaffold` as lazy
  factories; an unknown id raises rather than defaulting.
- `engine/orpheus/**` - every Orpheus module, MOVED, bodies unchanged, plus
  `interface.py` (`OrpheusCodec`, `OrpheusBudget`, the `stop_policy` view). The
  engine test suite is 147 tests, unchanged, OK on both interpreters.
- `engine/higgs/v3_served.py` + `v3_engine.py` - the SERVED second engine:
  launch (invoking their `serve_v3.sh`), `/health` polling, stop, the request
  builder (sampling in `extra_params` and nowhere else), the response decoder,
  the 45-token control-token allowlist, the one-reference / 30 s rules, the two
  the checkpoint rules, `pads=False`, `edge_fade` 10/25, `max_chars=600`,
  `coverage_check='asr'`. **Rendered a sentence end to end on the GPU** - see
  PORT_NOTES 12.7.
- `engine/higgs/**` - 163 tests across the v3 backend and the v2 scaffold, none
  of which needs a model or a GPU (the v3 client is driven against a real
  stdlib HTTP server standing in for vllm-omni).
- `serve/worker.py` selects its engine from `NARRATOR_ENGINE` (default
  `orpheus`) and gained ONE branch for a served backend
  (`engine.render_audio(text)`). `--fake-engine` has a fake per engine.
- `pyproject.toml` gains three mutually exclusive extras, pinned to what the
  three WSL envs actually have.

**Hardened after adversarial review + the Mac and app-side runs (2026-09-04):**

- voice validation is the ENGINE's (`resolve_load_voice` on the protocol) - the
  worker was checking Higgs loads against Orpheus's stock allowlist;
- a failed load STOPS the server it started (no orphaned 14 GB vllm-omni), and
  the ready timeout is 900 s against measured cold starts of 55 / 146 / 297 s;
- server adoption checks `/v1/models` for `higgs-v3` and for the configured
  adapter, instead of trusting whatever answers the port;
- a tail-trim readiness probe (one fixed-seed word, trailing 300 ms RMS below
  -45 dBFS) catches an unpatched server, which otherwise answers 200 and returns
  audible garbage on every chunk;
- `maxChars` is per VOICE, and a FINE-TUNE without one is refused by name;
- a non-WAV 200 and the `-100` HTTP 400 are refused with what to do about them;
- `edge_fade` is an `EdgeFade(in_ms, out_ms)` - asymmetric 10/25 - and a test
  asserts it equals `assemble/engine_profiles.py`;
- a worker that cannot render NEVER handshakes (exit 3, reason on stderr);
- ONE chunk writer, `soundfile` PCM_16, on every backend (torchaudio's FLAC
  writing is wheel-dependent and produced PCM_24 on the Mac against PCM_16 in
  WSL - a bit-depth mix the concat demuxer drops frames on);
- `DefaultVoice` - the model's own voice, and a fine-tune's text-only prompt -
  as its own member of the union, with `ClipsVoice`'s >= 1 clip rule intact.

**Designed, not built:**

- **The ASR coverage gate.** `StopPolicy.coverage_check == 'asr'` is a hook with
  a name. It matters more on v3 than anywhere: a chunk measured a duration ratio
  of 0.99 while dropping 22 % of its text and inserting filler.
- **Serving a v3 fine-tune end to end.** The SHAPE is settled - a merged
  checkpoint the server runs on, `kind: 'checkpoint'` - but narrator cannot
  launch a chosen one: `serve_v3.sh` `exec`s a hard-coded snapshot path and
  takes no arguments, so the operator starts the server and states which
  checkpoint through `NARRATOR_HIGGS3_CHECKPOINT`. A checkpoint voice with
  nothing stated is refused by name. Making narrator own that launch is the
  next served-backend step.
- **v3 streaming and batching.** The buffered POST endpoint is what is wired;
  `/v1/audio/speech/stream` and `/v1/audio/speech/batch` are unmeasured.
- **`SentenceSink` reading `pads` / `edge_fade`.** There is no SentenceSink
  yet; the engines report both and the assembler must consume them - for v3 that
  means realizing the manifest's gaps AND fading 10 ms in / 25 ms out.
- **The catalog keyed on (engine, voice).** `orpheus-models.json` is still
  Orpheus-only, so `OrpheusBudget.max_chars` reads the payload the engine was
  constructed with and REFUSES anything else, and v3's 600 is a placeholder.
- **The rest of `serve/worker.py`.** Its load path is still Orpheus-shaped
  (VALID_VOICES, merged/adapter/base modes, `set_voice`, per-request LoRA); the
  Higgs engines refuse the parts they cannot honour by name.
- **A managed `higgs3` env.** The two site-packages patches are hand-applied
  today and must be re-applied after any pip upgrade there.

**Budget rule for adapter voices (measured by the training session, 2026-09-04 night):** a
fine-tuned Higgs adapter's stop length tracks its TRAINING CLIP LENGTH, not the text (a
30-minute adapter trained on 8-22 s clips stops after ~6-10 s of audio on any prompt over
~150 characters; 900 chars -> 9.6 s). So `max_chars` for an `{kind:'adapter'}` voice MUST
come from that model's own length sweep, carried per voice in the catalog with its
provenance, and an adapter voice without a measured cap is REFUSED by name. The 600-char
zero-shot placeholder applies only to `{kind:'clips'}` voices. Also measured: one reference
clip per request under the 30 s cap (multi-clip = one pre-joined wav + one transcript);
cold start 297 s to /health (55 s only on a warm restart).

### Higgs v3 facts from the trainer work (orpheus-training session, 2026-09-05, HIGH confidence)

1. vllm-omni cannot load a LoRA at runtime (no adapter flags; the talker class does not
   implement SupportsLoRA). Every Higgs voice ships as a MERGED checkpoint directory
   (~8.5 GB) and the server is started on that directory; switching voice = restarting
   the server on a different path (~55 s warm, ~300 s cold). The LoRA (~200 MB) is the
   archival artifact; merging is a CPU step outside narrator. Catalog and VoiceRef
   kind: `checkpoint` (checkpointDir); the `adapter` strategy switch is gone.
2. Owen: production is fine-tuned voices ONLY. Zero-shot cloning (reference audio in
   the request) is diagnostic, not a product feature; the text-only request to a merged
   model is the primary path.
3. The chunk-tail sentinel trim is a band-aid on the server side; a token-level fix in
   vllm-omni's decode is queued there. narrator never trims client-side.

### Higgs on the Mac (BUILT + SMOKED 2026-09-05; full notes in engine/PORT_NOTES.md 13)

Owen: *"make sure the Mac has Higgs built in for streaming the model via the browser
extension. I use that constantly on the Mac."* Done. `higgs-v3` is now ONE engine with
TWO backends chosen by `sys.platform`: **darwin -> in-process through mlx-audio**
(`engine/higgs/mlx_backend.py`, `BackendSpec.kind == 'inprocess'`), everything else ->
the vllm-omni server. Every number the plan states about v3 - 8 codebooks at 25 fps,
24 kHz, `pads = False`, `EdgeFade(10, 25)`, EOS reliable, `coverage_check = 'asr'`,
the voice document, the 45-token control allowlist - is unchanged on both arms; only
where the weights run differs.

- **No conversion step.** mlx-audio 0.4.8 (already pinned for Orpheus) loads the
  official `bosonai/higgs-audio-v3-tts-4b` safetensors directly, codec included, from
  the same shards. No mlx-community repo, no second download.
- **Measured, Mac, 2026-09-05:** 8.7 GiB download; 2.8 s cold load; 8.89 GB peak;
  **RTF 0.571** (5.90 s of audio in 3.37 s) at 15.9 chars/s. `ready`/`loaded` carry
  `engine: higgs-v3`, `backend: mlx`, `sampleRate 24000`, `pads false`,
  `edgeFadeMs {in:10, out:25}`. NOT ear-checked.
- **Point 3 below ("narrator never trims client-side") STANDS, and now has a
  measurement behind it on this path.** At the token level, mlx-audio's delay-pattern
  revert already ends one frame BEFORE the EOC diagonal, so on a clean ending it is
  exact and there is nothing to trim - 0 out-of-range codes reached the codec in 2/2
  fixed-seed renders, and the trailing-300 ms RMS was identical treated and untreated.
  What narrator ships is a token-identity GATE (keep a frame iff all 8 codebooks are
  in [0, 1023]), which is a defect sensor for ragged endings, not a trim.
- **Streaming is per ROW at retirement.** A delay-pattern codec has no sound windowed
  decode. `should_stop` is checked every generation step, so a cancel still lands in
  milliseconds.

## Higgs v3 path design points (Owen, approved 2026-09-05 night, relayed by the orpheus-training session)

Orpheus keeps its 44 s sentence split. For Higgs v3:
1. **Chunk = whole paragraphs packed** until the next would exceed a MEASURED per-voice cap
   (placeholder ~2,000 chars ~ one page ~ 140 s; the served sweep of the 100 s-clip model gives
   the real number); never split a paragraph unless it alone exceeds the cap. Evidence: the
   matched-checkpoint sweep - training clip length sets the stop wall (98 s clips read 1,500
   chars at 84-98 %, 8-22 s clips at 9-25 %), so the cap is a property of the voice's training
   corpus and must come from its sweep (the catalog's `maxChars` + `maxCharsSource:
   'length-sweep'` per checkpoint voice).
2. **Headings, list items and table-like fragments render as their own chunks**, never packed
   into prose (the paragraph packer's walls; already the rule for `[heading]`/`[item]`).
3. **Sentence-level VTT comes from POST-RENDER FORCED ALIGNMENT** of each chunk's audio
   against its known text (a CTC aligner: wav2vec2 via torchaudio, or WhisperX align mode;
   CPU, seconds per chunk) - the model exposes no text-to-time mapping. narrator's VTT
   contract (cue text = chunk text, times from sample counts) stays the CHUNK-level truth;
   the sentence-level cues are derived from the alignment inside each chunk's span.
4. **The same alignment is the coverage guard**: text with no aligned audio = dropped /
   truncated; audio with no text = insertion. It replaces the duration-ratio guard, which
   cannot see a measured 22 % text loss. This is the `coverage_check: 'asr'` hook already on
   the Higgs StopPolicy, made concrete: an ALIGNMENT check, not a transcription diff.
Design-note level; no code yet. The existing `electron/scripts/align_audiobook.py` (WhisperX,
paragraph-aware, silence-snapped) is the natural home for the aligner.

### BUILT 2026-09-05 - `python/narrator/align/` (README there carries the numbers)

Points 1 and 2 were already `text/paragraph_packer.py` except for tables, which are now a
sixth block kind: a `<table>` becomes one block per data row (e2a's own cell recipe,
core.py:1461-1481), a text-shape detector catches the same thing in a PDF-derived block
list, and both are walls carrying `[item]` - a new `[table]` token is in neither `TTS_SML`
nor `SML_UNSPOKEN_PATTERN`, so the engine would read it aloud.

Points 3 and 4 are `narrator align --session-dir <hash dir> [--out sentences.vtt]
[--report coverage.json]`. `<stem>.sentences.vtt` is ADDITIVE - the chunk-level VTT is
untouched and both are generated from one `assemble/vtt.chunk_spans`, so a sentence cue
cannot leave its chunk's cue. The thresholds are DATA in `assemble/engine_profiles.py`
(`CoveragePolicy` per engine, `audited` for `higgs-v3` and not for `orpheus`) and the
reading is `assemble/coverage_gate.py`, pure stdlib because assembly runs on a CPU env
with no torch.

#### THE AUDIT REPORTS. IT DOES NOT BLOCK. (Owen, 2026-09-05)

> there will always be truncations or errors of some sort. thats the nature of tts.
> nothing is going to come out perfect. we try our best to detect and reduce the number of
> errors but assembly will never function, ever, if we expect it to come out the other side
> flawless. we need to base assembly on the expected text and the actual real length of the
> audio. with orpheus, for truncations, we split at sentence boundaries and re-rendered.
> but the goal is to have zero truncations.

Point 4 as BUILT, therefore:

* `narrator align` audits the WHOLE book and always writes both outputs. A chunk it cannot
  place is recorded in the report's `errors` by index and message; the run exits 0 whenever
  it happened. `--continue-on-error` is accepted and ignored.
* EVERY chunk gets sentence cues. Measured ones where the aligner placed the words;
  otherwise the expected text spread over the chunk's REAL audio span by character share,
  each run marked `NOTE estimated chunk <i>` in the VTT. The geometry, the marker and the
  writer live in `assemble/sentence_vtt.py`, so assembly can lay the same cues when there
  is no report at all.
* `coverage_gate.check()` LOGS what the report found - every failed index, the dropped text,
  the `narrator retake --indices ...` line - and assembles. A missing report is logged and
  assembles too. The only refusals left are integrity ones: a report about another engine,
  another session or an older render.
* BookForge's Align queue row succeeds whenever the run happened and carries the counts and
  the retake list on its card; the assembly row repeats that list once on the finished book.

The backend is **WhisperX**, measured against torchaudio `forced_align` on ten kershaw
chunks: identical word times (median delta 0.000 s, p95 0.020 s) and identical cost, but
only WhisperX localizes text inside longer audio, which is what makes "audio with no text"
detectable at all; torchaudio's API is also removed in 2.9. No automatic switching - a
backend that fails records the chunk by name and the pass moves on. Deviation from the relayed note on
point 1: a Higgs CHECKPOINT voice with no measured cap stays REFUSED rather than taking a
2,000-char placeholder; the 600 is a MEASURED zero-shot number and belongs only to the
served default. OWED: a Higgs v3 render to align (every threshold is calibrated on Orpheus
output plus hand-built failures).
