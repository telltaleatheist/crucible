# Crucible changelog

What changed in each release, newest first. Every change adds a line under
**Unreleased** as it lands; `scripts/ship.sh` refuses to cut a release with nothing
there, renames the section to the version it cuts, and puts it in the GitHub release
notes.

## Unreleased

- docs/RETRIEVAL.md says which reranker suits which job, measured: the dedicated reranker for query-to-passage retrieval; a decide model (qwen3.5-9b) for short items such as tags judged against a long description, where the dedicated reranker could not separate good from junk (Content Studio, 2026-10-10).
## 1.0.136 — 2026-10-10

- An instrumental whose score ended but which the instrument transfer refuses (the skill's checks: a bar longer than its meter, the first planning-lyrics song on the PC) now keeps the plan in `failed-plan/` too and says the skill's own error; `scripts/check-instrumental-planning.py` prints a failed song instead of stopping.

## 1.0.135 — 2026-10-10

- `scripts/check-embed-rerank-live.py`: the batch check holds the measured tolerance (a short text padded beside a long one reads cosine 0.99989 of itself alone on the Mac), and docs/RETRIEVAL.md states the measured figure.
- A YuE2 instrumental is now planned from planning lyrics: words its score is planned from and never sung, so the melody has a sung song's bounded phrases before it moves to the instrument (YuE2 then re-plans from the fixed score with section tags only, as before). Planned from empty sections, nothing bounded the score: on Victoria's 8 GiB laptop 4 of 13 instrumentals ran it to its 4096-token cap and failed, and the rest scored 1023 to 3911 tokens where sung songs score 1800 to 2600. A client may send its own as the new param `planning_lyrics` (only with `instrumental: true` and never beside `lyrics`, both `audio_param_conflict`; section tags from YuE2's vocabulary, never the same tag twice in a row, at least one line of words, at most 36 lines and 2000 characters, else `invalid_params` saying why; `audio_param_unsupported` on Stable Audio). Without them, and without section-tag `lyrics`, the server picks one of ten original sets in `crucible/audio/planning/yue2.toml` from the seed (set `seed mod 10`), each a different structure sized like B-Sides' sung songs, so the same params and seed plan the same song. `audio.planning_lyrics` in the `done` event and `settled.planning_lyrics` in a kept request say which (`{source: pool|request, id, lyrics}`). The SDK's `AudioOptions` takes `planningLyrics` and `AudioResult` reads `planningLyrics`. `scripts/check-instrumental-planning.py` renders N pool instrumentals on a card and reports score tokens, how each stage ended and the audio's length (docs/AUDIO.md "Planning lyrics").

## 1.0.134 — 2026-10-10

- Rerank's token counts are honest on the Mac. `tokens.total` is every document's prompt with its query, as llama-server is sent it, and `tokens.cached` is what no pass read again, so `total - cached` is what the engine read. On the Mac (1.0.133) a rerank reported its query once per document and 0 cached, although it read the query once (a ~2,500-token query: 1 document ran 3.8 s, 20 documents 6.9 s). The Mac's items route says per question what its passes read (`read_tokens`, `ITEMS_VERSION` 7; the engine re-patches its env at start). A likelihood decision on the Mac now counts each candidate's prompt to its boundary, as on llama-server, and its `cached_tokens` the same way.
- `scripts/check-embed-rerank-live.py`: check 4 compares each encoding against a float call of the same batch (a text embedded beside another reads up to ~1.4e-3 off itself alone on the Mac, which is the batch, not the encoding) and checks the two are the same vector to cosine search; check 6 checks `total - cached` is less than the documents' prompts (the query read once), and is skipped on vLLM, which re-reads it.
- `docs/RETRIEVAL.md`: how to use embed and rerank, for app authors. Covers installing the package, curl, TypeScript and OpenAI-client examples, the fingerprint, base64 decoding, cutoffs and reranking with a decide model. It also gives the measured costs: a model nothing holds is unloaded after each call, so an unheld call reloads it first, inside `timing_ms.queued` (~3 s on the Mac), and a run of calls should hold a queue session. The same text in another batch differs slightly on the Mac under the same fingerprint. The reference sections, the SDK README and the route docs link to it.

## 1.0.133 — 2026-10-10

- An instrumental whose score comes back unusable now says which way it failed (`truncated`: it ran to the score's token cap without ending, or `empty`: the model ended it with no score, with the token count) and keeps the plan YuE2 wrote (tokens, timing, how it ended) in the job's `failed-plan/` beside its kept request. Victoria's two-album run had 2 of ~11 instrumentals fail there with nothing saying why. Nothing re-runs it.
- Two new verbs, `embed` and `rerank`, in an optional package: `crucible install retrieval` pulls Qwen3-Embedding-8B and Qwen3-Reranker-8B at bf16 (about 16 GB each; the PC runs both as GGUFs on llama-server, the Mac as Qwen's safetensors on mlx-lm; on the PC's card they swap, they do not stay loaded together) and records `[packages] retrieval = true`; a server without it refuses both by name (`409 package_not_installed`) and never pulls them, `load-model` and install-on-submit included. `POST /v1/embed` turns up to 256 texts into unit-length vectors: `input_type` `query` (the model's instruction prefix, with an optional `instruction`) or `document`, an optional Matryoshka `dimensions` (32 to 4096), `encoding_format` `float`, `base64` or `base64_float16`. Every answer names what wrote the vectors in `model.fingerprint` (model, weights revision and file, engine build, Crucible's scheme); an app stores it with the vectors and sends it back as `fingerprint`, and a server that would write anything else refuses `409 fingerprint_mismatch` instead of mixing incomparable vectors. `POST /v1/rerank` scores one query against up to 256 documents: P(yes) against P(no) under the model's own prompt (Crucible's, never the app's), in document order and sorted, on the decide door's likelihood machinery, so llama-server, mlx-lm and vLLM all serve it; the query is read once where the engine keeps a cache. Any decide model reranks too when named (`model: "qwen3.5-9b"`, Crucible's general template), with no package. Compatible routes: `POST /v1/openai/embeddings` (OpenAI's shape) and `POST /v1/openai/rerank` (Cohere's and Jina's, `top_n`, `return_documents`), each also under `/openai/v1/`. Both verbs follow the sizing ruling: `model`, then `max_params_b`, then Settings, then the automatic pick; a named model that does not fit is refused `model_does_not_fit`, never swapped. `GET /v1/models` rows carry `verbs`, `package`, `embed` and `rerank` (dimensions, templates, limits), `GET /v1/info` carries `verbs`; features `embed`, `embed.openai`, `rerank`, `rerank.compat`, `packages`. An embedding model is refused at the chat and decide doors (`model_embeds_only`). The Mac's items route is `ITEMS_VERSION = 6` and a new patch (`mlx-lm-qwen3-bare-checkpoint`) lets mlx-lm read Qwen3-Embedding's bare-transformer checkpoint; both apply themselves at the next engine start (docs/internals/api.md "Embed", "Rerank"; docs/VERB-SIZING.md section 9). The SDK has `embed()` and `rerank()` (`EmbedRequest`, `RerankRequest`, their responses, `decodeEmbedding` for the base64 encodings), and reads `ModelInfo.verbs`, `package`, `packageInstalled`, `embed`, `rerank` and `ServerInfo.verbs` (null from an older server).
- An audio job (a song) that fails, is cancelled or is interrupted can now be run again: from the moment it starts it keeps its request on disk (`request.json` in the job's directory, `request` on `GET /v1/jobs/{id}`): the params as sent with the seed it uses written in, the seed the server chose when the client sent none, the settled values, `low_vram` and a `reproduce` sentence. It clears once the job finishes successfully: a song that ends `done` drops it (`done_extra.audio` is the record); one that ends otherwise keeps it until the job is reaped. Victoria's job 1f3da14c (CUDA OOM in synthesizing) left no params and no seed. No other job type keeps anything of its request; a narration's params stay off disk (docs/AUDIO.md "A sound that did not finish").
- YuE2 under `[audio] low_vram` no longer runs out of memory synthesizing a long song (Victoria's 8 GiB laptop: a song composed to 8,960 tokens failed 4 s into `synthesizing` at the 6.8 GiB cap). yue2-infer synthesizes a song of up to about 10,000 frames as one chunk whose prefill runs the AR half over the whole prefix and every codec token and keeps each layer's keys and values for the solve (112 KiB a token, 1.44 GiB for that song); it ran with the whole AR half, embeddings and lm_head on the card (4.03 GiB), so the keys grew on top of them. The synthesis prefill now brings the AR half to the card one layer at a time and the solve holds the NAR half alone beside the keys, so the stage's weights on the card no longer depend on the song and synthesizing stays below the composing stage at any length; the arithmetic is unchanged (docs/internals/audio.md "Synthesizing under low_vram").

## 1.0.132 — 2026-10-10

- Chat on llama-server on cuda-linux (`qwen3.5-4b-bside`): every JSON schema is now enforced by llguidance, the compiler vLLM and the Mac's mlx-lm use, so a schema means the same grammar on every engine, and `"json_whitespace": "compact"` is kept there (it was refused). Crucible's llama-server build is now `b10970-llg1.7.6-cuda13.0` (a new asset on the `tools` release; a host replaces its binary at its next `crucible install llm` or GGUF load): `LLAMA_LLGUIDANCE=ON` with llguidance 1.7.6 (the llm env's) in place of b10970's 1.0.1, and patched so a grammar llguidance will not compile is llama-server's own 400 instead of an answer sampled without it (`scripts/llama-server-linux.patch`). The door sends `response_format` (json_schema, json_object) and llama-server's `json_schema` field as the grammar `%llguidance {}
start: %json <schema>`, held from the first generated token as on the other engines; such a chat must state thinking off (`400 structured_output_with_thinking`; B-Sides' manifest does). A schema stated twice or beside a `grammar` is `invalid_request`. A `%llguidance` grammar sent to llama-server on llama-windows (ggml-org's build, no llguidance, which aborts on one) is refused `structured_output_not_served`; compact stays refused there (docs/internals/engines-and-capability.md "Structured output").

## 1.0.131 — 2026-10-10

- `POST /v1/decide` answers carry `timing_ms.queued`: the ms a decision waited in the server's line before it ran. `timing_ms.total` was always the run alone, so a client that sent several decisions at once (Briefcase: three chapter calls) read the later ones' waits as a slowdown.

- Chat: `"json_whitespace": "compact"` beside a JSON schema or json_object makes the answer compact JSON (no whitespace between tokens, whitespace only inside strings), for clients whose models are trained on compact JSON (B-Sides); `"flexible"`, the default, is unchanged. vLLM and mlx-lm keep it: the door writes llguidance's own `"x-guidance": {"whitespace_flexible": false}` into the schema, which llguidance takes over the engine's flexible default (a json_object goes as the schema `{"type": "object"}`); the mlx-lm patch is unchanged. llama-server (b10970's schema converter has a fixed whitespace rule and no option for it) and mlx-vlm are refused `json_whitespace_not_served` before the chat waits or loads anything, so on the PC `qwen3.5-4b-bside` (llama-server) is refused it; an upstream model is refused it too. Without a JSON constraint it is `json_whitespace_without_json`; a schema that states `x-guidance` whitespace itself is `json_whitespace_conflict`. `structured_outputs.disable_any_whitespace`, `disable_additional_properties` and `whitespace_pattern`, which vLLM 0.29.0 reads only from its server config and dropped from a request without a word, are now refused `structured_output_not_served`. The SDK's `ChatOptions` takes `jsonWhitespace` (docs/internals/engines-and-capability.md "Structured output").
- Mac: the desktop no longer freezes while the 27B reads a prompt. mlx-lm read prompts 2,048 tokens per GPU evaluation, which is 12-13 s on qwen3.8-27b-8bit, and macOS can make the window server wait for all of it (measured: the window server unresponsive 12.6 s at a time, a 60 Hz Metal client blocked 13.2 s; the 9B's 3.7 s steps never froze it). Crucible now starts mlx-lm with `--prefill-step-size` derived from the model's size, its attention layers (the weights' config.json) and the context it is loaded at, so a step is at most about 1.6 s of GPU even at the end of the longest prompt the context allows (27B 234 at 12288 and 217 at 24576, 9B 686, 4B 1523, smaller models 2048); prefill is as fast near the start of a prompt and about 10% slower on a 24k-token one; the decide items and likelihood route (ITEMS_VERSION 5) reads its state and its rows in the same steps. A manifest may not state the flag itself (`prefill_step_stated`).

## 1.0.130 — 2026-10-10

- YuE2 no longer grows in host memory song after song. Every move of its backbone between the card and host memory (yue2-infer's decode parks the whole 7.26 GB backbone while the VAE runs; `[audio] low_vram` swaps the halves around every synthesis chunk) allocated fresh host copies, and glibc's heap kept the freed ones: since 1.0.125 kept YuE2 loaded between queued songs, a worker on an 8 GiB laptop in a 15.8 GB WSL guest grew to 15.5 GB and was OOM-killed on track 12 of an album. Each parameter and buffer now keeps the host tensor its load made for the life of the worker (`yue2_worker.HostHomes`): a move to the card copies from it, a move back points at it, nothing is allocated in host memory after the load. Measured on the 3090 Ti with low_vram on: anonymous memory 1.95 GB after song 1 and 1.98 GB after song 6 (it was 9.76 GB after song 1 and 12.24 GB after song 2). Each audio job's `done_extra.audio.host_memory` now reports the worker's `before`, `after` and `peak_rss_bytes` and the `host_homes_bytes` it keeps; the YuE2 manifest declares `host_memory_bytes_estimate`, and `crucible doctor` names the model and both figures (`audio_host_memory`) when this machine has less memory than that (docs/internals/audio.md "Host memory").

## 1.0.129 — 2026-10-10

- Decide: a `likelihood` question takes up to 256 candidates (was 26), so one question carries a whole tag set, and it is now served on llama-server too (it was refused `likelihood_unsupported_on_engine`). On llama-server each candidate is generated as a forced continuation: its exact token ids under a token-id grammar, each token's log-probability read from the raw logits before the grammar and the sampler (`post_sampling_probs: false`), the shared context reused from the prompt cache; qwen3.5-4b-bside on the PC (llama-server, BF16) and the Mac (mlx-lm, bf16) agree on 256 tags to 0.06 nat on average, same winner and top 10. A question takes `normalize`: `softmax` (the default, today's numbers) or `none`, where each candidate's `probability` is exp(`logprob`) on its own; the answer echoes it. For "which of these apply", use the items form with one yes/no item per option (docs/internals/api.md "Likelihood questions"). On the Mac every question's context is read once and every candidate's first token comes off its last position, so a k-token candidate is a row of k-1 tokens and the head runs once per forward (`mlx-lm-decide-items-helper` is now `ITEMS_VERSION = 4` and applies itself at the next engine start): 256 tags in 2.4 s instead of 8.7 s on the M1 Ultra. The SDK's `DecideLikelihoodQuestion` takes `normalize` and `DecideLikelihoodAnswer` reads it.

- A worker the kernel's out-of-memory killer ends no longer takes the server down with it. The systemd unit now sets `OOMPolicy=continue`: systemd's default stops the whole service when any process in it is OOM-killed, and the workers run inside it (Victoria's laptop: one OOM-killed YuE2 worker stopped the server and ended the song queued behind it). An existing unit is rewritten, reloaded and restarted onto by the update itself: every update path (scripts/deploy.sh, the Windows tray carrying its WSL guest, the install one-liner) runs install.sh, which runs `crucible service install`. The failed job now says how its worker ended instead of "exited -9": "was killed by the out-of-memory killer (SIGKILL, signal 9)" when the kernel's count of out-of-memory kills (the service's cgroup `memory.events`, else /proc/vmstat) moved while it ran, "was killed (SIGKILL, signal 9)" naming someone else when it did not, and "out of memory?" when there is no count to read. When the event loop stalls, the server log now also says where every other thread is and the machine's memory pressure, RAM and swap left, so a stall can be told apart as a held lock or a machine out of memory.

## 1.0.128 — 2026-10-10

- A model may come in more than one FORM (precision) of the same weights, and each server serves the best form its card holds: the first, best first, whose estimate fits the card less the desktop allowance; with none fitting, the smallest is tried. `qwen3.5-4b-bside`'s PC block now has two, `bf16` (`qwen3.5-4b-bside-BF16.gguf`, 9.27 GiB) and `q8_0` (5.49 GiB), both at `owenmorgan/qwen3.5-4b-bside-gguf` @ 2a9bc567: the 3090 Ti takes bf16 and an 8 GiB card q8_0; the Mac stays bf16. Same id, same served name. `GET /v1/models` rows carry `form`, `form_reason` and `forms`, `/v1/catalog` rows `form` and `form_reason`; `crucible models list` and `crucible doctor` (`model_form_not_installed`) say when the installed form is not the one this card takes, and a load is refused naming the form and the pull rather than serving a smaller one. `crucible models pull <id>` pulls the form this card takes, `--form <name>` another; a second form is pulled beside the first, which stays installed. A chat, a decision and `load-model` (`params.form`) may name a form: another form on the card is a reload, a call naming none takes the resident form, an unknown name is `400 unknown_form`. The SDK reads `ModelInfo.form`, `formReason` and `forms` and `CatalogRow.form` and `formReason`, and sends `form` from `ChatOptions`, `DecideRequest`, `DecideItemsRequest` and `LoadModelOptions`. Feature `models.forms` (docs/FITS-AND-THE-CARD.md section 8).
- Chat: `response_format` is now enforced on the Mac. Stock mlx-lm read none and answered every `json_schema` unconstrained; Crucible's `mlx-lm-structured-output` patch (applied by the engine at its next start) compiles `response_format` `json_schema` / `json_object` and `structured_outputs` with llguidance as vLLM does, one matcher per sequence, so the answer is the schema and stops at its closing brace (qwen3.5-4b-bside under B-Sides' describe schema: every answer valid, about 4% slower per sequence). It also fixes two mlx-lm batch defects that killed the generation thread, or dropped a constrained chat's grammar, when it shared a batch with a plain one. Every engine now states which constraints it enforces, and the chat door refuses the rest `400 structured_output_not_served` instead of sending them to be ignored: mlx-vlm enforces none, vLLM no `guided_*` or `grammar`, llama-server no `structured_outputs`. A constrained chat is answered from its first token, as on vLLM, whether thinking is on or off.

## 1.0.127 — 2026-10-10

- Chat: a request may carry `"prefill": "<text>"` to start the model's answer with that text; the model writes on from it and the reply's content is what it wrote after it. Served on vLLM and llama-server, with thinking off and no `response_format`; refused by name elsewhere (`prefill_not_served` on mlx-lm, mlx-vlm and upstream models, `prefill_with_thinking`, `prefill_with_grammar`, `prefill_conflict`). The SDK's `chat()` takes `prefill`, and its `ChatUsage` now carries `cachedTokens`, the prompt tokens the engine read from its prefix cache.

## 1.0.126 — 2026-10-10

- `qwen3.5-4b-bside` is v2: one model for every B-Sides text call (describe, album, tracks, lyrics, album title, artist, track titles, cover), the task named by the first line of the user message. Same id; the Mac pins `owenmorgan/qwen3.5-4b-bside` @ 3e499e62, a PC the Q8_0 GGUF @ 5e326c86. On a Mac, mlx-lm does not yet enforce `json_schema`, so the strict schema each call sends applies on a PC only.
- `POST /v1/decide` scores free-text candidates: a question of `type: "likelihood"` (`instructions`, `candidates` name → reply text, `rank_by` `total` or `mean`) answers each candidate's summed log-probability, token count and mean as the start of the model's reply, a softmax over the totals and the `winner`, with nothing generated (so no format failures and no runaways). It sits beside the label questions in the questions form. vLLM reads prompt log-probabilities (`/tokenize` first, so every refusal comes before a forward pass); mlx-lm and mlx-vlm score every candidate as a row of the items route over the state read once (`mlx-lm-decide-items-helper` is now `ITEMS_VERSION = 3` and applies itself at the next engine start); a llama-server model is refused `400 likelihood_unsupported_on_engine` (b10970 returns no prompt-token log-probability). The SDK's `decide()` takes `DecideLikelihoodQuestion` and reads `DecideLikelihoodAnswer`; a `DecideAnswer` must now be narrowed by `type` before reading `labelMass` (docs/internals/api.md "Likelihood questions").

## 1.0.125 — 2026-10-10

- Fix: weights left in the store by a model that is no longer in the catalog (`qwen3.5-4b-bside-4bit`, 3.6 GB on an 8 GiB laptop after 1.0.124) can be removed: `crucible remove model <id>` and `DELETE /v1/catalog/model/{id}` find them in the weights store when the catalog has no such model, instead of refusing `subject_unknown`. Nothing is deleted on an update by itself; `crucible doctor` lists such weights and now names that command for this machine's own.
- Fix: a job's done event says what is on the card even when the settlement after it took the model off and then failed stopping its process; before, `resident` kept the id the job read before the settlement.
- `crucible install llm` on a PC says the llama-server download (107 MB, xz) and the binary it unpacks (140 MB) apart, so the two figures no longer look like a mismatch.
- Docs: the README says a Windows install moves to the WSL2 engine by itself (nothing to click), and how another device reaches a server that only its own machine can after an install: `crucible lan enable` on Windows, Private networks only, a Public network kept shut until it is marked Private.
- CLI: a refused `crucible api` request (a `job submit --follow` included) now leads with `crucible: HTTP <status> <code>: <message>` and exits 1; the server's whole error document still follows on the next lines.
- Fix: `crucible uninstall` left behind what Crucible itself had placed in its home and reported it as "Crucible did not put this here": `tools` (ffmpeg, silero-vad, zig), `run`, `ladder`, `voice-manifests`, `servers`, `interpreters`, and the host's `host.log`, `host.lock` and `tray.*` files. They are now removed. `journals` and `playground-presets.json` are kept with the other user data, `voice-refs.json` is kept with the voice weights (removed by `--purge-weights`), the Windows home's `wsl` folder (the distro's disk) is kept and named, and a LAN door (`landoor.json`) is withdrawn before the engine stops instead of being left on the machine.
- Fix: after an update, `install.ps1` said "Crucible is ready" while the tray was still carrying the Linux engine to the new release. It now says ready only when the engine answers `/v1/info` on the release just installed, waiting up to 20 minutes and saying every 30 s which release answers; if that runs out it says so and names the controller's log. The setup `.exe` waits the same way.
- `install.ps1 -FromApp`: an app that launches the installer says so (the setup `.exe` and `@crucible/bootstrap`'s `install()` pass it). It replaces the guess from the script's file name and a redirected output. Apps on an older `@crucible/bootstrap` must update it: without the switch the script follows the whole Linux setup as a terminal would.
- A model whose manifest says `keep_calls_together = true` (now `yue2-3b`) keeps its queued calls together: while it is resident, its queued songs run back to back ahead of a queued job that would take it off the card (a CLI `load-model` sent mid-batch on Victoria's laptop), and the card is not unloaded between them. The job kept waiting says why (`waiting_for` / a `waiting` event with code `keeping_calls_together`, "waiting: yue2-3b has N queued call(s) ahead …"), and waits only for the calls queued when it would have run, never for later ones (docs/QUEUE.md, "A model that keeps its calls together").
- A finished `audio` song now says how each of YuE2's token stages ended, so a song that ran long (a 6:00 song on an RTX 3070, 2026-10-09) is read off the job instead of found by re-running the seed: `audio.decode_stages` gives `scoring` and `composing` their tokens, `cap`, `ended` (`eos`, or `cap` when the model never ended the stage), `execution` (`cuda_graph` or `eager`), `low_vram`, wall time and tokens per second, and `audio.stages_at_cap` names the stages that ran to their cap. The job still succeeds and nothing re-runs it. The SDK's `readAudioResult` reads both, and `lowVram`.
- Fix: under WSL2 every load stopped the server for over a second ("the event loop has not run for 1.0 s"), because a job's preflight read the card with `nvidia-smi` on the event loop. A preflight now runs off the loop, and the lane checks are made again after it; `GET /v1/info`, `/v1/models` and `/v1/voices` (which read each env, a `pip list` the first time) and `/v1/setup` (which spawns PowerShell on Windows) are read off it too.

- The engine log no longer shows `'_POSIX_C_SOURCE' macro redefined` on every Triton compile (zig's glibc headers say POSIX.1-2024, Python's pyconfig.h says 2008). Crucible's `cc` passes `-Wno-macro-redefined`, and only that; a host's existing `cc` is rewritten the next time an engine or worker starts.

- A vLLM engine (and the vLLM ASR worker) no longer logs a `deep_gemm` `AssertionError` traceback as a WARNING on every start: Crucible runs vLLM with `VLLM_USE_DEEP_GEMM=0`. DeepGEMM needs a Hopper or Blackwell card and a CUDA toolkit, which Crucible never places, so it could never run here; with the switch off vLLM stops before trial-importing it.

- A failed load (or any engine or worker that stops) now leads its refusal with the first error its log reports, then the last 40 lines as before. A vLLM start that died on its KV cache showed only the API server's "Engine core initialization failed. See root cause above" traceback, with the engine core's `ValueError: ... KV cache ...` cut off above the tail; it now reads `First error in its log: ValueError: ...`. A traceback logged as a warning (vLLM's deep_gemm import) is not taken for the cause.

- A vLLM load now says what it is doing while its engine starts, and for how long: reading its weights, compiling the model with torch.compile, compiling Qwen's linear-attention Triton kernels (slow only when its cache is cold, as on the first load after an install), capturing CUDA graphs, starting its HTTP server, read from the phase vLLM last logged. A job's `warming` messages are now its `message` and its `job.progress` (with the fraction unchanged), and `GET /v1/jobs/{id}` carries `message`, so a client reading the job sees the load move instead of "loading <model>" for the two minutes a first load spends compiling.

## 1.0.124 — 2026-10-09

- `qwen3.5-4b-bside` on a PC now runs on llama.cpp from a Q8_0 GGUF (about 5.4 GB) instead of vLLM, so B-Sides' tag model loads in seconds instead of minutes and fits an 8 GiB card whole; `qwen3.5-4b-bside-4bit` is gone. `crucible install llm` on a PC also places Crucible's pinned Linux llama-server, which runs on the CUDA libraries the llm env already has.

## 1.0.123 — 2026-10-09

- Fix: a vLLM model run one request at a time at its full context (B-Sides' `qwen3.5-4b-bside-4bit` on an 8 GiB card) was refused at startup: "0.35 GiB KV cache is needed, which is larger than the available KV cache memory (0.29 GiB)". The memory plan now gives each in-flight request 1024 tokens beyond its context for vLLM's block padding and the linear-attention state it keeps in the pool, and the Qwen3.5 4B family uses the measured 46,581 B/token.

## 1.0.122 — 2026-10-09

- Fix: on a cuda-linux host with no C compiler (a fresh WSL distro), a model loaded its weights and then died with `RuntimeError: Failed to find C compiler`, because Triton compiles a C launcher on its first kernel. `crucible install` now places a pinned Zig 0.17.0 as `~/.crucible/tools/bin/cc` and Crucible's engines and workers run with `CC` set to it; a host updated by a deploy places it the first time an engine or worker needs it (57 MB, once); only a placement that fails is refused by name (`c_compiler_missing`), and `crucible doctor` names the fix.

- `qwen3.5-4b-8bit` carries its measurement from an RTX 3070 Laptop (8 GiB): 4.86 GiB of weights, about 0.65 GiB of warm-up above weights and KV, 46,581 B/token of KV. It holds 5.95 GiB resident; an 8 GiB card's text verbs need 5.9-6.2 GiB with it.

- Fix: updating Crucible on a Windows PC whose engine runs in WSL ended with a false "its engine did not start" (`local_config_unreadable: … no config at …\config.toml`) when the Windows side had never run an engine of its own. A Windows home with no `config.toml` runs no engine, so the guest's engine (carried to the new release by the tray) is no longer judged against it.

## 1.0.121 — 2026-10-09

- Fix: on a card where the desktop allowance leaves about as much as is free (an 8 GiB card), vLLM refused to start ("Free memory on device … is less than desired GPU memory utilization"). The memory plan now leaves vLLM's own CUDA context (0.94 GiB, measured on an RTX 3070 under WSL2) out of the budget it hands the engine.

## 1.0.120 — 2026-10-09

- New model `qwen3.5-4b-8bit` (cuda-linux): the 4B at 8 bits, text only (`owenmorgan/qwen3.5-4b-w8a16`). On an 8 GiB card every text verb now picks it instead of `qwen3.5-0.8b`; larger cards are unchanged.

## 1.0.119 — 2026-10-09

- Settings: every config.toml key can be changed from the operator page; a host or port change offers "Restart now?" inline (the address is written into the service definition, so the page names `crucible service install` when that has to run first). On a PC the port stays fixed at the Windows host's 7100 (`port_fixed_by_windows_host`).
- New routes: `PUT /v1/settings/jobs/{job_type}`, `PUT /v1/settings/tts/{engine}`, `POST /v1/settings/token/rotate`, `POST /v1/capability/record`, `POST /v1/server/restart`.
- `PUT /v1/settings` now also takes name, host, port, advertise, cors_origins, open_pairing, install_on_submit, retention_days, max_session_hold_s, hf_token (write-only) and video_desktop (Mac; out of range refused by name).
- Operator page: "Re-measure this card", "Replace the token" (the new token is shown once), and "Reload now?" for a voice started with old `[tts]` numbers.
- `[auth] open_pairing` now applies without a restart.
- `docs/CHANGELOG.md`: what changed in each release, back to 0.1.0. `ship.sh` refuses to cut a release with nothing under Unreleased and puts the version's section in the GitHub release notes (`scripts/changelog.py`).

## 1.0.118 — 2026-10-09

- New per-model `[llm.concurrency]` config table (`"<model>" = n`): how many requests a chat model runs at once on this server. It can only lower the manifest's width, and takes effect the next time the model loads.
- CLI: `crucible models concurrency [<model> [<n>|default]]`.
- API: `PUT /v1/settings/llm/concurrency`; `GET /v1/settings` carries `llm_concurrency` rows.
- Settings → Requests at once, with "Reload now" to apply it to a model already on the card.

## 1.0.117 — 2026-10-09

- Verb sizing, phase 1: each text verb has a goal (27B for generate, translate, simplify and analysis; 9B for decide and clean) and the automatic pick is the largest model at or below it that fits. Decide now runs on `qwen3.5-9b` on the PC and the Mac instead of the 27B.
- The 9B floor is gone: every text verb's lineup runs down to `qwen3.5-0.8b`, so a small card gets a smaller model instead of "off". The `minimum_for` manifest key is removed (now refused as unknown), and the Foundry lineup drops `floors` (schema 3); `GET /v1/catalog` still sends `floors: []`.
- `bits` is now a required fact on each manifest backend block, checked against the weights and refused under 4. `/v1/capability` and install-plan rows carry `goal`, and the record says why a model was picked.
- Decide switches to a vision model by itself when a request carries images: the capability row records `with_images` / `with_images_reason`, and `model` on `POST /v1/decide` is now optional (also in `crucible api decide` and the SDK's `decide`/`decideItems`). New refusals: `capability_undecided`, `capability_disabled`, `no_image_model_fits`.
- A text verb whose automatic pick is under 4B adds a sentence recommending an API key (Settings "Accounts this engine may spend", `[upstreams]`/`[routes]`); it never changes the pick.
- An unmeasured card's desktop reserve is now an eighth of the card (1-3 GiB) instead of a flat 3 GiB. `[audio] low_vram` provenance reads "set automatically" / "set manually". SDK: `activity()` reads the deploy hold `updating` (now required) and a running job's `cancelling`.

## 1.0.116 — 2026-10-09

- `[audio] low_vram` is now decided by Crucible: it turns it on where a model does not fit the card whole but fits at its low-VRAM figure, records `low_vram_auto` beside it, and says so in one sentence. A person's setting is never overridden.
- `crucible audio low-vram on|off|auto` (no argument shows the state), `PUT /v1/settings/audio/low-vram`, `crucible api low-vram`, and an On / Off / Let Crucible decide switch in the operator Settings panel. The settings document carries `audio_low_vram`.
- The server now says who can reach it and how to open it: `/v1/setup` carries `network`, `crucible serve` prints it, and the new `crucible lan offer [--ask]` prints it for a person. A fresh Windows install typed at the PC asks whether to share on the network. `lan enable` names a Public network before the admin prompt; inside the WSL guest `lan` refuses `lan_inside_wsl`.
- SDK pairing tells `connection_unreachable`, `connection_timed_out`, `connection_cancelled` and `not_crucible` apart, and a LAN address that does not answer names `crucible lan enable`. `setup()` reads `network` as `ServerNetwork`. The unused `ensureDistro` and related exports are removed from `@crucible/bootstrap`.
- `install.sh --token-env` reads the token from `$CRUCIBLE_INIT_TOKEN`, so the one-line install keeps it off the command line. `crucible uninstall` refuses a packaged (MSIX) shell like the installer does.
- WSL install: sets `crucible` as the distro's default user through `wsl --manage --set-default-user` where WSL supports it, and restarts the guest with `crucible service restart`, waiting for the server to answer.

## 1.0.115 — 2026-10-09

- Fix: `stable-audio-3-medium` requests over 120 s came back 120 s long. The worker now uses the model's own window (Medium up to 380 s) and refuses audio shorter than the request instead of returning it.
- Medium measured at 6.8 GB on cuda-linux (was 8 GB declared), so music now fits an 8 GiB card.

## 1.0.114 — 2026-10-09

- New CLI: `crucible jobs list|enable|disable <type>` and `crucible audio low-vram [on|off]` change config without minting a new token; a running server picks the change up on its next request.
- New `crucible service restart`: one restart through the service manager, then waits up to 120 s for the server to answer (refuses `restart_not_answering`). The tray no longer fights a person's `service stop`, and waits on a unit that is still starting instead of running a recovery.
- Capability now weighs audio models at the `[audio] low_vram` figure when it is on, so `crucible install audio`, `/v1/capability`, the doctor and the job agree; with it off, a model that would fit names the setting as the fix.
- `crucible init --token-env` reads the token from `$CRUCIBLE_INIT_TOKEN`; installers no longer put the token on argv.
- Install fixes: refuses to run from an MSIX-packaged shell (`packaged_shell`); the WSL distro is verified to enter as `crucible`; env builds print progress every 20 s; `crucible install audio` places the pinned ffmpeg; no voice-pin warnings on hosts that do not serve TTS.
- A job's done event `resident` now says what is on the card after the settlement (null when it was unloaded).

## 1.0.113 — 2026-10-09

- New model `qwen3.5-4b-bside-4bit` (W4A16, 3.79 GB): B-Side's describe model for an 8 GiB card, served by vLLM one request at a time. In the `bside` family, so no capability class offers it.

## 1.0.112 — 2026-10-08

- New `[audio] low_vram` config: holds only one half of YuE2 on the card at a time, so songs run on an 8 GiB card (measured ~6.5-6.8 GB peak, same audio). Off unless a host sets it.
- Audio manifests may declare `low_vram_memory_bytes_estimate` and `low_vram_memory_note`; `yue2-3b` on cuda-linux declares 7.3 GB. The done event reports `audio.low_vram`.

## 1.0.111 — 2026-10-06

- Reverted 1.0.110's separator change: the value is the seconds between window starts, not an overlap count, so 4 made voice isolation slower. The manifest field is renamed `hop_s` and both separators are back to 8 s.
- The Throughput docs say what the number means and that windows are not batched.

## 1.0.110 — 2026-10-06

- The served API docs gain a Throughput section: what runs together and how to send work for chats, `asr`, `rvc` and `denoise`. Those job docs point to it.

## 1.0.109 — 2026-10-06

- Separator overlap now comes from each separator's manifest (`[model] overlap`, required); `vocals-roformer` set to 4, `denoise-roformer` stays 8.
- Fix: a resident vLLM model's reclaimable memory is what its load measured on the card, not only its estimate, so Crucible evicts its own model instead of refusing work as `accelerator_busy`. `card.loaded` carries `measured_bytes`.

## 1.0.108 — 2026-10-05

- vLLM models now constrain JSON with llguidance (`--structured-outputs-config {"backend": "guidance"}`). xgrammar could not put a newline or quote in a length-capped string.

## 1.0.107 — 2026-10-05

- New model `qwen3.5-4b-bside`: B-Side's describe fine-tune, served on both the PC (vLLM) and the Mac (mlx-lm). Its own `bside` family, so it is only used when asked for by name.

## 1.0.106 — 2026-10-04

- The running server documents itself, without a token: `GET /docs` (HTML with a filter), `GET /v1/docs` (JSON index, job types marked enabled), `GET /v1/docs.md` and `GET /v1/openapi.json`. Includes every route and job type with its params, inputs and returns. The operator page links to it.

## 1.0.105 — 2026-10-04

- `denoise` takes `stems: "all"` to return every separated stem, the primary first. Default stays `"primary"`.
- Deploy hold: `POST /v1/server/updating` now takes the hold before checking for work, so nothing admitted in the gap is killed. While held, new work is refused with a retryable 503 `server_updating`; `GET /v1/activity` shows `updating`. SDK: `CrucibleUpdating`, and requests wait out the update and retry.

## 1.0.104 — 2026-10-04

- SDK: a job or chat stream that breaks mid-read now raises `CrucibleUnreachable` (naming where to resume) instead of the platform's raw error.

## 1.0.103 — 2026-10-04

- Fix: a job recovered after a restart gets its ending back (`done`, `failed`, `cancelled`, `removed`), and its event stream closes instead of hanging forever.

## 1.0.102 — 2026-10-04

- Audio models take `format: "mp3"` (192 kbps CBR) beside flac and wav; flac stays the default. Artifacts carry their real media type.
- CORS also allows `User-Agent` and `X-Crucible-Session`, which the SDK sends.

## 1.0.101 — 2026-10-04

- New `[server] cors_origins`: exact origins whose web pages may call this server. Empty by default; `*` and paths are refused. Edits take effect without a restart.

## 1.0.100 — 2026-10-04

- SDK: `playground()`, `playgroundPresets` / `savePlaygroundPreset` / `deletePlaygroundPreset`, and `instrumental` on `audio()` and its result.

## 1.0.99 — 2026-10-03

- Playground presets stored on the server: `GET/PUT/DELETE /v1/playground/presets/{model}[/{name}]`. A preset holds the form's params, never the seed.
- Playground: a Copy button for style tags, and contradictory tags shown in red (from `[[conflict]]` tables in `song.toml`).

## 1.0.98 — 2026-10-03

- Playground audio pages: a queue panel on the right and a player bar at the bottom (previous, play/pause, next, scrubber); finished songs play in order.

## 1.0.97 — 2026-10-03

- Playground: every generation is its own card, so you can listen to one while others queue, and a count sends 1-20 in a row (a fixed seed becomes seed, seed+1, ...).

## 1.0.96 — 2026-10-03

- Playground: tag suggestions laid out in a two-column grid, one row per group.

## 1.0.95 — 2026-10-03

- Fix: the playground's Instrumental switch now actually locks the lyrics box.

## 1.0.94 — 2026-10-03

- Playground: style tags as chips with grouped one-click suggestions (`crucible/audio/tags/song.toml`).
- An instrumental song with sung lyrics is refused as `audio_param_conflict`, naming the first sung line; the playground disables lyrics while Instrumental is on.

## 1.0.93 — 2026-10-03

- New audio param `instrumental: true` for `yue2-3b`: YuE2's own instrumental workflow, so nothing is sung and lyrics are optional. The playground gains an "Instrumental (no vocals)" switch.
- Every audio render now reports `effective_params.notes`.

## 1.0.92 — 2026-10-03

- `yue2-3b` songs now run on the Mac (MPS, torch 2.14), with a causal-attention self-check on every load that refuses to generate if the kernel leaks future tokens.
- Recipes may mark a line `# crucible: no-deps` to install it without its own dependency pins.

## 1.0.91 — 2026-10-03

- Stopping an engine re-sends SIGTERM every 10 s within its budget (never SIGKILL), and a process that outlives its stop is reaped when it exits.
- `/v1/activity` running rows carry `cancelling`; the desktop app and web console show "stopping a cancelled ..." for them.

## 1.0.90 — 2026-10-02

- Fix: `GET /v1/info` ran `pip list` on the event loop, stalling the whole server for seconds at a time during renders. Package lists are now cached until the env changes.

## 1.0.89 — 2026-10-02

- The proxy retries a request the wire lost five times over ~7.5 s instead of twice.
- `crucible local shutdown` waits 30 s for its reads, so it no longer leaves a second orchestrator running beside the old one.
- The server logs the stack of whatever blocks its event loop for more than 1 s.

## 1.0.88 — 2026-10-02

- Higgs stall guard v2: an engaged row stays penalised until it has been off the stuck code for 10 frames, so it no longer snaps straight back. Narrator pinned to match on the Mac.

## 1.0.87 — 2026-10-02

- A cancelled load stops at the next readiness poll instead of holding the card until the load ends.
- Desktop app: a slow status answer shows "slow to answer" instead of flipping to "Crucible is not running".
- Narrator pin: unloading a voice no longer waits 60 s.

## 1.0.86 — 2026-10-02

- Higgs stall guard default changed to `37,1,20,16` after an A/B on the PC (longest generated pause 35.3 s to 4.5 s).

## 1.0.85 — 2026-10-02

- Narrator pin: a hyphen that can only be a dash reaches the engine as an em dash.

## 1.0.84 — 2026-10-02

- Narrator pin: every printed ellipsis reaches the engine as "...".

## 1.0.83 — 2026-10-02

- A card wait whose holder has not changed is repeated every 60 s with a fresh `next_check_at`, so client watchdogs see it is alive.
- SDK: the job `waiting` event is typed (`CardWaitData`); `session()`, `stream()` and `fleetSession()` take `onWaiting`.

## 1.0.82 — 2026-10-02

- A card held by a process Crucible does not own (`accelerator_busy`) is now waited on instead of failing queued jobs, sessions and chats. Queue and activity rows carry `waiting_for`. `"queue": false` still refuses.
- TTS streams: with `X-Crucible-Queue-Ticket: 1` a stream-open that must wait answers 202 with a queue session to follow instead of holding the request open. SDK `stream({onQueue})` reports its place in line.
- Narrator pin: the MLX stall-guard parser reads the PC's grammar.

## 1.0.81 — 2026-10-02

- Higgs stall guard on cuda-linux: stops runaway silences. Default `37,0.5,20,8`, overridable per voice with `[voice.serving].stall_guard` (or `false`); shown on the `/v1/voices` row.
- The guard is an sglang-omni env patch applied by `crucible install tts`, `crucible env patch tts` and at engine start; `crucible doctor` reports `tts_patches`.

## 1.0.80 — 2026-10-02

- Narrator pin caps interior pauses at 1.5 s; the render job's chunk events carry `pause_cuts` (SDK `ChunkData.pauseCuts`).

## 1.0.79 — 2026-10-01

- Fix: `"queue": {}` waits with the default again instead of being refused (1.0.78 broke every older SDK).

## 1.0.78 — 2026-10-01

- Waiting in line is now the default: a job, chat, decision or TTS stream that finds the server busy queues (up to an hour) unless it sends `"queue": false`. `{"max_wait_s": N}` still sets the wait. CLI: `--max-wait N` / `--no-queue` replace `--queue`.
- SDK: `fleetSession()` asks several servers for a queue session at once and takes the first to open; `closeSession(id)`.
- Desktop app and web console follow `GET /v1/events` instead of polling, with a Queue panel showing the open session and every waiting row.
- Decide on mlx-lm is much faster (items batched in one forward, recent states cached); a new mlx-lm patch removes ~150 ms per chat request.
- Fixes: a job type turned off in `config.toml` is refused while the server runs; a removed queue session cancels its own model load; a token voice on cuda-linux is listed unloadable and refused `voice_not_served_here`.

## 1.0.77 — 2026-10-01

- SDK: `closeSession(id)` so a restarted app can close a session it recorded. Docs: one client name per install.

## 1.0.76 — 2026-10-01

- Queue sessions replace leases: `POST /v1/queue/sessions` waits in line and, once open, runs one client's requests back to back while others wait. Routes `GET .../{id}`, `GET .../{id}/events`, `POST .../{id}/touch`, `DELETE .../{id}`; CLI `crucible api session`.
- Leases are removed entirely: the lease routes, `params.lease`, `lease_id` and the lease CLI verbs.
- New `GET /v1/events`: one SSE stream of every change (jobs, queue, card, tasks, settings), with `?topics=` and resume by Last-Event-ID; CLI `crucible api events`. `GET /v1/info` lists `features`.
- TTS streams run inside a queue session. SDK: `session()` and `events()` replace `lease()` and friends.

## 1.0.75 — 2026-10-01

- Decide logs each decision's wait and answer time.

## 1.0.74 — 2026-09-30

- A lease request (`POST /v1/models/{id}/lease`) can wait in the queue with `"queue": {}`, loading the model with the lease when needed. The SDK's `lease()` queues by default.

## 1.0.73 — 2026-09-30

- Video on the Mac renders clips up to 21 s at 1280x704, tiled by default to keep the desktop responsive (`[video_desktop] enabled = false` turns it off). The playground shows the machine's limits.

## 1.0.72 — 2026-09-30

- New playground: `GET /v1/playground` and a page per image, video and audio model with a prompt, its settings and Generate; opened from the tray or `crucible local playground`. A model not yet on the server downloads on first Generate.
- Chats and decisions can wait in the queue with `"queue": {}` (loading the model if needed); they appear in `GET/DELETE /v1/queue` and the SDK queues them by default.

## 1.0.71 — 2026-09-30

- Server-side queue: `POST /v1/jobs` with `"queue": {"max_wait_s": N}` waits in a FIFO line instead of being refused busy. New routes `GET /v1/queue`, `DELETE /v1/queue/{id}`, `POST /v1/queue/{id}/heartbeat`, `GET /v1/queue/events`; new terminal state `removed`; `job submit --queue`. The desktop app lists the queue.
- New `[video_desktop]` table splits the Mac's video GPU work so the desktop stays responsive; video done events report GPU busy figures.

## 1.0.70 — 2026-09-30

- New optional `[video_trial]` config table to lift one machine's video clip limits for measuring.

## 1.0.69 — 2026-09-30

- `ltx-2.5-distilled` video now runs on the Mac (ltx-2-mlx, int8), with the same params and outputs as the PC, on Macs with 48 GB or more.

## 1.0.68 — 2026-09-30

- Fix: config rewrites (every `crucible install`) dropped `[hf]` and other tables they do not own, losing the HF token; they are now kept.
- Fix: the video job lost the finished clip at the mux step on Linux.
- Masked image jobs on the Mac report `mask_blend_steps` and `mask_outside_drift`.

## 1.0.67 — 2026-09-29

- New `video` job family (`video`, `load-video`, `unload-video`) with `ltx-2.5-distilled`: clips with sound on the PC (cuda-linux only), text-to-video and image-to-video, output `video.mp4`. SDK: `video()`, `loadVideo()`, `readVideoResult()`.

## 1.0.66 — 2026-09-29

- Fix: inpainting on the Mac now carries the blend between steps, so masked regions no longer come back as unrelated scenes.

## 1.0.65 — 2026-09-29

- New `segment` job family (`segment`, `load-segment`, `unload-segment`) on PC and Mac: `birefnet` for subject cutouts and `sam2.1-hiera-large` for point/box selections. Returns `mask.png` and `cutout.png`. SDK: `segment()`, `loadSegment()`, `readSegmentResult()`.

## 1.0.64 — 2026-09-29

- Image inpainting and outpainting: `params.mask` names a mask input (white regenerates), with `mask_blur` (default 8). New refusals `mask_size_mismatch`, `mask_empty`, `inpaint_unsupported`.

## 1.0.63 — 2026-09-29

- Fix: image-to-image on the PC failed on a 3-channel start image.

## 1.0.62 — 2026-09-29

- Image-to-image (`image_strength`) now works on the PC as well as the Mac.

## 1.0.61 — 2026-09-29

- Fix: Stable Audio 3 Medium ran out of memory while loading on the PC; it is now halved to float16 on the CPU before moving to the card.

## 1.0.60 — 2026-09-29

- Fix: `crucible install audio` failed on the PC because of a stray line in the stable-audio-3 recipe.

## 1.0.59 — 2026-09-29

- New `audio` job family (`audio`, `load-audio`, `unload-audio`) for sound effects, music and songs: `stable-audio-3-small-sfx`, `stable-audio-3-medium` (PC and Mac, gated, need an HF token) and `yue2-3b` songs (PC only). Output `audio.flac` or `.wav`. SDK: `audio()`, `loadAudio()`, `readAudioResult()`.

## 1.0.58 — 2026-09-28

- `asr`: a piece that loops at every retry is left empty and listed under `decode_loop` instead of failing the whole job; the done event counts `decode_loop_pieces`.

## 1.0.57 — 2026-09-28

- Voices follow their Hugging Face repo's `crucible` tag: deploying a voice no longer needs a Crucible release. `crucible voices check-updates`, `POST /v1/voices/updates` and `crucible voices pull <id>|--all`; voice rows carry update status and serving facts (sampling, `edge_fade_ms`, `chunk_gap`, ...).
- Image jobs take `params.lease` and a new `load-image` job warms the model; repeated prompts reuse cached text-encoder output.
- Desktop tray: clicking opens the window, the window's X hides it, and tray Quit closes both.

## 1.0.56 — 2026-09-28

- New `image` job family (`image`, `unload-image`) with Qwen-Image 2.1 on the Mac (mflux) and the PC (diffusers), with cancel between steps. SDK: `image()`, `readImageResult()`.
- New native desktop window: `crucible app` (Home, Models, Voices, Packages, Activity, Settings), installed to the Start Menu / `~/Applications/Crucible.app`, plus a Windows setup exe `crucible-setup-<version>.exe`.
- Mistborn voice pinned back to v16.

## 1.0.55 — 2026-09-28

- `POST /v1/decide` takes an items form (`items: [{text, options?}]`) that asks many choice questions about one state in one call, much faster on the Mac. SDK: `decideItems()`.
- Mistborn voice pinned to v17.

## 1.0.54 — 2026-09-28

- Decide can answer image questions on the Mac: `qwen3.5-9b-vl` gains an mlx-darwin build served by mlx-vlm with logprobs. `model_text_only` refusals name the image models (`image_models`).

## 1.0.53 — 2026-09-28

- The Mac uses a pinned ffmpeg instead of Homebrew's.
- Fix: every Mac deploy was refused as busy-state unknown because the probe broke under zsh.

## 1.0.52 — 2026-09-28

- ASR: `speech_only` is now on by default unless the caller sets `vad_filter: true`; `speech_only: false` transcribes everything. A Qwen3-ASR piece that just recites the job's `context` is left empty (listed under `context_echo`) instead of failing the job or shipping the instruction as text.
- Crash recovery: Crucible records its engine processes in `<home>/run/resident.json` and asks survivors of a crashed run to stop at startup. Interrupted jobs are reaped correctly again, and a server shutdown interrupts the running job instead of leaving a worker behind. No CUDA process is ever SIGKILLed; refusals name the pid and a `kill <pid>` step.
- Install-on-submit now also covers `env_missing`, answering `409 installing`. `409 server_busy` details carry `door` (`"job"` or `"operator"`). New `crucible api job hold` / `job release`. `/v1/info` gains `terminal_states`, `voice_sources` and `service_commands`.
- `crucible remove` and `crucible voices pin` go through the running server when there is one, so weights and voices in use are protected. Error messages across the CLI and the Windows host now name the next command or file to fix the problem; corrupt host records are set aside automatically.
- Windows uninstall stops the engine and controller cooperatively instead of force-killing them. `deploy.sh` refuses a machine it cannot probe (unless `--interrupt`) and refuses a downgrade without `--force`. `promote_release.py --publish` requires every fleet machine to report the release.
- Large internal restructuring with no wire changes; `GET /v1/capability?class=generate` is about 9x faster, and a failed llama.cpp re-pull leaves the old engine serving.

## 1.0.51 — 2026-09-27

- Resumable jobs: `POST /v1/jobs` returns a `resume_id`, and `params.resume` continues a job from where it stopped. New `GET` / `DELETE /v1/resumable[/{id}]`, `crucible api resumable list|get|discard`, `job submit --resume`, and SDK `resumable()`, `resumableEntry()`, `discardResumable()`. Qwen ASR is the first job type that keeps a journal.
- The TypeScript SDK now reads the current server's wire strictly again, reversing the 1.0.25 tolerance for older servers. Voice rows may have null `pace` and `sample_rate`.
- Legacy paths removed: the `host` alias of `crucible orchestrator`, WSL user-scope units, the renamed/retired ASR id tables, the vllm-omni TTS patches and the packaged voice source. Config keys the writer emits are now required.
- Narrator launches SGLang-Omni on a free loopback port, and a voice counts as unloaded only once that server has exited too.
- Denoise sends a progress event every 30 s while separating. Voice pin updated for sigma v2.

## 1.0.50 — 2026-09-27

- Install on submit: a job that needs a missing env, model, voice or rvc base assets starts the install and gets `409 installing` with what is being installed and its progress; resubmit after it finishes. `[jobs] install_on_submit` controls this.
- Quantize to fit: capability considers precision (never below 4-bit) and narrows batch width before quantizing, for Higgs TTS and Qwen3-ASR. New `GET /v1/capability/plan` and `crucible ladder`. Cards without bf16 run vLLM in fp16; jobs a card cannot start are refused with `card_lacks_feature`.
- ASR: new `speech_only` option (Silero VAD) with `speech_threshold`, `speech_pad_s` and `speech_min_gap_s`, off by default. Qwen alignment runs in batches of 16 with progress, and `transcript.text.json` is published before alignment.
- RVC: long inputs are cut at quiet points and stitched frame-exact, output keeps the input's rate and channels (`output_rate`, `output_channels`), finished inputs are published as they land, and urvc is recycled by memory use. Refuses `ffmpeg_missing` up front.
- Crucible ships its own pinned ffmpeg on Linux. `crucible init` measures the desktop GPU reserve on NVIDIA cards (`crucible capability --measure-desktop`); new `crucible pair <address>` and `crucible guest`.
- Many fresh-install fixes for Windows: WSL restart handling with "Update and restart" wording and Try again (`crucible orchestrator --try-again`), the installer console follows the move, per-network LAN checks, and upgrades that stop the old server with the new release's code.

## 1.0.49 — 2026-09-26

- The WSL guest gets root through passwordless sudo when available, fixing `Exec format error: 'wsl.exe'` during service install.
- Host error messages keep the end of the output, so the actual error is no longer cut off.

## 1.0.48 — 2026-09-26

- Every command in Crucible's WSL distro now runs as the `crucible` user, fixing a fresh install that put the engine in `/root/.crucible` and later minted a new token.

## 1.0.47 — 2026-09-26

- The rvc env on cuda-linux gets `diffq` from Crucible's own `wheels` release, so it installs on a fresh WSL guest with no C compiler.

## 1.0.46 — 2026-09-26

- Fixed the WSL move on fresh machines: the host reads the server's catalog correctly, waits up to 180 s for the guest to boot, and reports the real error. Cloud-init is disabled in the guest to speed up boot.
- `crucible api` accepts `--pairing-file` and `$CRUCIBLE_PAIRING`, so the token no longer has to appear on the command line.

## 1.0.45 — 2026-09-26

- Fixed fresh Windows installs on a machine with WSL but no Linux distribution yet.

## 1.0.44 — 2026-09-26

- One unreadable voice pin no longer makes `GET /v1/voices` fail for every voice; it is listed as not loadable with its reason.
- Env installs that compile from source use the host's compiler (gcc) when the interpreter's recorded one is missing.

## 1.0.43 — 2026-09-26

- The five fine-tuned voices now come from their own HuggingFace repos via pins; Crucible ships only the engine's own voices.
- Fixed Qwen ASR failing whole jobs on words like "life-changing" or "U.S." when trimming overlap.
- The rvc env on cuda-linux pins `onnxruntime-gpu` 1.26.0, so denoise runs on the PC. `crucible init --help` no longer crashes.

## 1.0.42 — 2026-09-26

- The Qwen3 aligner's CUDA memory estimate is now measured (5.5 GiB), fixing false out-of-memory errors on 300 s align windows.

## 1.0.41 — 2026-09-26

- Qwen ASR cuts audio into 30 s pieces at pauses with real-audio overlap; callers can set `piece_s` and `overlap_s`. This fixes dropped sentence openings.
- New Voices panel in the operator console for pinning a repo, editing a voice's settings and reverting; new `GET /v1/voices/{id}/manifest`.
- Plain-torch workers on CUDA (aligner, denoise) use expandable allocator segments and are capped at their admitted memory share.
- Mistborn v16 is the packaged voice.

## 1.0.40 — 2026-09-26

- Fixed the Mac's 27B crashing with `metal::malloc` resource limit on long thinking runs (new mlx-lm env patch).
- A crashed mlx-lm generation thread now ends the engine; chat and decide refuse a dead engine with `502 engine_exited`. `/v1/activity` reports `resident.engine_exit_code`.

## 1.0.39 — 2026-09-25

- Denoise stems are always WAV as requested (they could come back as FLAC).
- The SDK README documents per-call thinking and how a token run-out looks on each backend.

## 1.0.38 — 2026-09-25

- Each render chunk's provenance sidecar now holds only its own chunk's params, plus `job_id` and `params_sha256`, cutting sidecar size dramatically.

## 1.0.37 — 2026-09-25

- `POST /v1/jobs` accepts `hold: true` to hold a job's artifacts from submission; the SDK's `submit()` and `render()` take `hold`.
- `POST /v1/jobs/{id}/hold` works at any job status; one align can cite several render jobs.

## 1.0.36 — 2026-09-25

- New `POST` / `DELETE /v1/jobs/{id}/hold` keeps a finished job's artifacts on the server for a chain of jobs, surviving restarts.
- Job inputs can reference another job's artifact (`{"artifact": {job_id, name}}`) instead of re-uploading it; an expired one gets `409 artifact_expired`. SDK: `artifactRef()`, `holdArtifacts()`, `releaseArtifacts()`.

## 1.0.35 — 2026-09-25

- Align jobs send each window's result as it finishes instead of all at the end.

## 1.0.34 — 2026-09-24

- Voices can share downloads: `[voice] weights_of`. `zeroshot` now reuses `higgs-default`'s weights instead of pulling its own 9.3 GB copy, and the base cannot be removed while an alias uses it.
- The TTS "not installed" message names the folder the weights are actually in.
- `crucible doctor` reports a folder left under an alias's own id as stranded weights.

## 1.0.33 — 2026-09-24

- ASR manifests support `[model] weights_of`: `qwen3-asr-1.7b-mlx` and `qwen3-asr-0.6b-mlx` now share their official sibling's download instead of pulling a second copy.
- A base folder shared by an ASR alias cannot be removed while the alias uses it.

## 1.0.32 — 2026-09-24

- New ASR models: `qwen3-asr-0.6b` on every backend (vLLM on the PC, Qwen's own package on the Mac) and `qwen3-asr-0.6b-mlx` (mlx-audio port, Mac only).

## 1.0.31 — 2026-09-24

- New align verb: SDK `client.align()` and `readAlignment()`, and CLI `crucible api align --model --language --window INDEX TEXT AUDIO`. Each window (up to 300 s) returns word timings relative to its own start, or its own error.

## 1.0.30 — 2026-09-24

- On the Mac, `qwen3-asr-1.7b` now runs Qwen's official `qwen_asr` package on torch MPS (new engine `qwen-asr`).
- New Mac-only id `qwen3-asr-1.7b-mlx` keeps the faster mlx-audio port (about 2.5x faster) for callers who want speed.

## 1.0.29 — 2026-09-24

- New ASR model `qwen3-asr-1.7b` (vLLM on the PC, mlx-audio on the Mac), with Qwen3-ForcedAligner word times and a loop guard that re-cuts looping pieces and otherwise fails with `asr_decode_loop`.
- ASR gains an optional `context` param (Qwen only); Qwen refuses `initial_prompt`, `vad_filter: true` and `language: auto` by name. SDK: `AsrOptions.context`.
- The ASR lineup is now exactly `qwen3-asr-1.7b`, `whisper-large-v3-turbo` and `whisper-tiny`; other Whisper sizes are removed. Old ids get an `unknown_model` refusal that names the offered models.
- `crucible serve` moves weights pulled under a renamed id into the new id's folder; `crucible doctor` reports model folders no manifest owns.

## 1.0.28 — 2026-09-24

- On the Mac, mlx-lm now runs batched: every mlx-darwin block states `--decode-concurrency`, `--prompt-concurrency` and `--prompt-cache-size`, and the chat door admits that width plus one (previously capped at 2).
- vLLM's chat admission now reads `--max-num-seqs` from the running engine, so `/v1/activity` reports `chat.max_in_flight` (17) instead of null.
- New mlx-lm env patch returns float32 logprobs, fixing `/v1/decide` `label_mass` values above 1 on the Mac. Generation is unchanged.

## 1.0.27 — 2026-09-24

- When a caller disconnects, the chat and decide doors now stop sending its work to the engine and release in-flight slots (broken since 1.0.18 by the config-reload middleware).
- New model `qwen3.5-2b` (bf16 on every backend), filling the `decide` ladder between 0.8B and 4B.

## 1.0.26 — 2026-09-24

- Requests that arrive while the card is being cleared (job submit, lease, chat, decide, stream open) now wait for the clearance to finish instead of failing; only a clearance that outlives its budget returns `409 engine_in_use`.
- SDK: a job read without `chunks_done` returns null instead of a protocol error, so older servers still work. The SDK README documents which fields each client path relies on.

## 1.0.25 — 2026-09-23

- SDK: "any Crucible that answers works". Informational fields are now tolerant (absent reads as null); load-bearing fields stay strict, and a field of the wrong type is still an error. Unknown TTS session events and unknown host-door events are passed through instead of failing.
- `/v1/decide` puts the state in the system message, so mlx-lm on the Mac reuses the cached prefix instead of re-reading the whole transcript per question.

## 1.0.24 — 2026-09-23

- New `POST /v1/decide`: send a state (text, JSON or images) and questions with fixed answer sets, get a probability distribution per question from one forward pass of the resident model. SDK `decide()` and CLI `crucible api decide`; options `logprobs` and `missing: "refuse" | "report"`.
- New capability classes `decide` (no size floor) and `generate` (client-sized context). `GET /v1/capability` accepts `class`, `context_tokens` and `concurrency`. New models `qwen3.5-4b` and `qwen3.5-0.8b`, and vision aliases `qwen3.5-9b-vl` and `qwen3.8-27b-4bit-vl` that share their base's download via `[model] weights_of`.
- `load-model` accepts `context` (refused above the backend's `max_context` with `400 context_over_limit`); SDK `loadModel(model, {context})`. The 8-bit 27B is now Mac only.
- Ollama upstreams are called through native `/api/chat`, so `context_tokens` (`ChatOptions.contextTokens`) and the tag's own context are honoured instead of a silent 4096.
- ASR: optional `initial_prompt`, new `faster-whisper-large-v3-turbo` on cuda-linux; denoise: new `vocals-roformer`.
- mlx-lm `top_logprobs` raised to 40 via an env patch (`crucible env patch llm`); stopping engine processes now works on Windows.

## 1.0.23 — 2026-09-22

- On the Mac, the accelerator guard now checks what the unified memory pool can hold rather than a free-memory sample, so an open browser no longer blocks loading a model the capability walk selected.

## 1.0.22 — 2026-09-21

- Job reads (and the SDK's `Job`) gain `chunks_total` and `chunk_at`, so clients can show progress and pace for chunked renders.
- `DELETE /v1/voices/{id}` returns 204 for a voice that is already gone; shipped voices still refuse with `voice_not_custom`.
- Voice rows gain `orphan`: true for a local-path voice that nothing is using.

## 1.0.21 — 2026-09-21

- The chat proxy retries a request once on a fresh connection when the socket is lost before a reply, and keeps pooled connections shorter than the engine's keep-alive. This fixes spurious `502 engine_unreachable` errors. Timeouts are not retried.

## 1.0.20 — 2026-09-21

- Fixed every Higgs TTS batch failing with a NameError (narrator pin updated).

## 1.0.19 — 2026-09-21

- Mac page reading batches scanned pages of slightly different sizes again, by grouping pages on the image processor's grid instead of exact pixel size. This fixes timeouts on whole books.

## 1.0.18 — 2026-09-21

- A running server now picks up changes to its own `config.toml` on the next request, so `GET /v1/capability` no longer needs a restart after `crucible install`.

## 1.0.17 — 2026-09-21

- The Mac can now read pages: `dots-ocr` gains an mlx-darwin block, served by Crucible's own in-process server on mlx-vlm 0.7.1, 12 pages wide.

## 1.0.16 — 2026-09-21

- The plain-language capability `summary` now actually appears in `GET /v1/capability` (it was missing in 1.0.15).
- Configs written by older versions still load after the capability record gains new fields.

## 1.0.15 — 2026-09-21

- Jobs survive a server restart: job records are saved next to their artifacts. A job that was running comes back as `interrupted` (not `failed`) with `chunks_done`, so a client can resubmit only the missing chunks. New optional `client_ref` on jobs; SDK adds `interrupted`, `clientRef`, `interruptedAt` and `chunksDone`.
- Each capability decision gains a plain-language `summary` for end users, beside the operator-facing `reason`.
- The SDK sends `X-Crucible-Client`, so browser clients show their own name instead of the browser's User-Agent.
- `deploy.sh` refuses to restart a server that is running jobs, streams, chats or a lease; `--interrupt` overrides.
- Unjudged TTS batches (`retake: false`) now render batched instead of one at a time, and rows are written as they finish (narrator pin updated).

## 1.0.14 — 2026-09-20

- A `load-model` / `load-voice` job now always reports `lease_id`, as `null` when no lease was asked for, both in the `done` frame and in `GET /v1/jobs/{id}`.
- The TypeScript SDK's `JobStatus` gains `leaseId`, read strictly for the two loader job types.
- Note for clients: servers older than 1.0.13 refuse the `lease` param on a load with `400 invalid_params`.

## 1.0.13 — 2026-09-20

- `load-model` and `load-voice` accept `params.lease {act, ttl_seconds}` and return a `lease_id`, so a load can hold the card it made resident. Without it, behaviour is unchanged.
- A lease that expires now unloads the card on the next idle tick, so a client that dies mid-run no longer strands a model on the GPU.
- `GET /v1/jobs/{id}` now includes a job's terminal facts (such as `lease_id` and `resident`), so a client that lost the event stream can recover its lease.
- SDK: new `LeaseOnLoad` and `LoadModelOptions` types, `lease` on both load calls, and `resident.heldBy` / `unclaimedSince` and `chat.maxInFlight` / `maxInFlightBasis` on activity.

## 1.0.12 — 2026-09-20

- A model or voice load that fails in any way (including a cancel mid-load or a malformed engine reply) now always shuts its engine down, instead of leaving an untracked GPU process behind.
- New tests pin the refusal paths: a refused chat request never takes an in-flight slot and never counts toward the `Retry-After` median.

## 1.0.11 — 2026-09-20

- `/v1/activity`'s `resident` block gains `held_by` (what is currently holding the loaded model) and `unclaimed_since` (when it stopped being held by anything).
- These fields are reporting only; nothing is unloaded automatically in this release.

## 1.0.10 — 2026-09-20

- The chat proxy now limits in-flight requests on serial engines (mlx-lm on the Mac) to the engine's measured concurrency plus one, instead of silently queuing them until clients time out.
- Excess requests are refused with `503 chat_queue_full` (nothing reaches the engine), with a `Retry-After` based on the median of recent completions.
- `/v1/activity` publishes `chat.max_in_flight` and `chat.max_in_flight_basis` so clients can size their request pool. Batching engines (vLLM, mlx-vlm) remain unbounded.

## 1.0.9 — 2026-09-20

- Engine and worker logs are now appended across restarts instead of overwritten, so reloading a hung model no longer erases the log of the hang. Each run starts with a visible header.
- Log tails and fatal-error checks read only the current run, so an old out-of-memory line no longer blocks the next start.

## 1.0.8 — 2026-09-20

- The server's keep-alive is now 75 seconds on both its HTTP entry points, fixing intermittent `ECONNRESET` on the first request after an idle gap.
- The TypeScript SDK retries a GET or HEAD once if the connection resets before any response arrives (never a POST).
- A render no longer substitutes the voice manifest's `max_num_seqs` when the client sends no `width`; the engine keeps its own width. This restores full batch width on the Mac (about 12.9x realtime vs 5.5x). The job's `done` reports the stated width, or `null`.

## 1.0.7 — 2026-09-19

- The TTS render request gains `retake`, `band` and `width`; new refusals `retake_without_band`, `band_malformed` and `width_over_serving`. `chunk_too_long` and `unknown_take` are retired, so renders are no longer refused by chunk length.
- Render results now report the full sampling settings applied and the weights used.
- Voice manifests may omit `max_chars` and `[voice.pace]`. `[voice.serving]` gains optional `mem_fraction` and `context_length`, and `serving` now appears on the `/v1/voices` row.
- The narrator pin moves to a build that reads the new retake, band and width fields.

## 1.0.6 — 2026-09-19

- Voices can now be published with their weights: a `crucible-voice.toml` at the root of the HuggingFace repo, plus per-machine `pins.toml`. `PUT /v1/voices/{id}` accepts `{"pin": {...}}` (repin) or `{"voice": {...}}` (local manifest), and the pin is validated before anything is written.
- New CLI: `crucible voices card`, `crucible voices check`, `crucible voices pin` and `crucible voices export`. The `/v1/voices` row gains `manifest`, `pace_basis`, `max_chars_basis` and `inherited_from`; `basis = "inherited"` requires `inherited_from` in the manifest.
- `crucible init` writes a `[tts.<engine>]` table to `config.toml` holding the box's TTS memory footprint. A server without one refuses pinned voices with `engine_footprint_unset`.
- Automatic WSL on Windows: the tray decides and starts the WSL move at every start. The host door gains `GET /install` and `GET /install/events`, the outcome is recorded in `wsl-outcome.json`, and the bootstrap SDK gains `installStatus()` and `watchInstall()`.
- Installs now show download progress for the Ubuntu image and the interpreter. The network probe checks every package index the install needs, and `crucible install` refuses with `env_disk` before downloading when the disk is too small.
- `crucible lan enable` refuses when the engine on the port is the native Windows engine (the forward would loop back on itself).

## 1.0.5 — 2026-09-19

- Fixed the Windows host carrying its WSL guest forward in the wrong distro name; it now uses the distro it manages.
- `wsl.exe` error messages (UTF-16) are now decoded correctly instead of appearing as NUL-interleaved text.
- `deploy.sh` waits up to 300 s for the PC's guest record.

## 1.0.4 — 2026-09-19

- The Windows host now waits for presence to be measured before carrying the WSL guest to the new release, and logs why when it does not.
- Fixed `engine_version_stale` wrongly refusing a Windows install over a WSL guest that had not yet upgraded; an unreadable config is now refused as `local_config_unreadable`.
- `ship.sh` prints its timing table even when a step fails.

## 1.0.3 — 2026-09-19

- Releases now carry only code: the Python interpreter is downloaded from python-build-standalone, the WSL image from Canonical, and environments are installed from their publishers. Environment packs, `crucible envpack` and `crucible install --build` are removed. Installers verify the wheel against a published `.sha256`.
- Installers install the latest promoted release and never downgrade: new refusals `crucible_already_latest`, `install_older_than_running`, `install_would_downgrade` and `release_channel_unreadable`; explicit rollback via `--rollback-to` / `-RollbackTo`. An API version mismatch is now reported as `api_version_mismatch`.
- New `crucible api` client command covering every server route (jobs, streams, voices with `voice-write` / `voice-remove`, settings), printing JSON.
- Finished jobs are now cleaned up: fetched or older than `[jobs] retention_days` (default 7). A reaped job answers `404 job_reaped`; uploads are moved into their job and a reused one gets `409 blob_consumed`.
- `/v1/activity` and `/v1/health` gain `stopping`; while an engine process is still shutting down, every load, claim and stream refuses with `engine_still_stopping`. A cancelled load no longer leaves the model resident.
- Voices may name local weights by `path` + `identity`; stream `done` frames carry `gap_sec` (the player inserts the pause); voice pace may be unmeasured or must be symmetric unless `edges = "percentile"`; `ServerInfo.pagesEngine` in the SDK; page concurrency is now 12.

## 1.0.2 — 2026-09-18

- On WSL (cuda-linux), vLLM's KV cache is now sized in bytes against the card (`--kv-cache-memory-bytes`) instead of a fraction of it, fixing models that failed to load with "No available memory for the cache blocks".
- A load that cannot fit enough KV cache is refused up front with `insufficient_kv_cache`, naming every term.
- `qwen3.5-9b`'s memory figures on cuda-linux are re-measured.
- `API_VERSION` is unchanged; 1.0.0 and 1.0.1 clients still work.

## 1.0.1 — 2026-09-17

- New `PUT /v1/voices/{id}` and `DELETE /v1/voices/{id}` for installing and removing custom voice manifests on a remote server; `revision` may be omitted and is resolved to the repo's head.
- The bf16 27B model is replaced by `qwen3.8-27b-8bit` (FP8 on cuda-linux, MLX 8-bit on the Mac). Display names now show precision on both 27B variants.
- SDK: `PairingRequest` gains `approvalRequired`, so clients can tell open pairing from approval-required.

## 1.0.0 — 2026-09-17

- New `crucible lan {enable,disable,status,reconcile,explain}` opens the WSL engine to the local network (port proxy plus firewall rule, one UAC prompt).
- Pairing is now open by default: requests are approved immediately and the first poll returns the token. Set `[auth] open_pairing = false` to require approval.
- `/v1/info` probes get a 15 s timeout, so a healthy engine with many models no longer reports "timed out".

## 0.6.12 — 2026-09-17

- An engine that is answering always has an owner, fixing the case where every authenticated route returned 503 (including `/quit`). Empty-token refusals now say which cause applies.
- Installers download the script to a file before running it, so a truncated download can no longer half-install; `--force` reinstalls a machine already on that release.

## 0.6.11 — 2026-09-17

- Recovery can now start a WSL guest installed as a system unit; previously the tray's Start could not bring a stopped guest back.

## 0.6.10 — 2026-09-17

- Windows can stop a WSL guest that runs as a system unit, fixing upgrades stuck with `HTTP Error 409: Conflict`.
- Error messages from the host now include the server's error code and message instead of just the HTTP status.

## 0.6.9 — 2026-09-17

- Stopping a WSL guest's user unit is now confirmed correctly, fixing `local_stop_failed` during upgrade.
- `install.ps1` uses Windows' own `tar` instead of whichever one is on PATH (fixes zstd failures from Git Bash).

## 0.6.8 — 2026-09-17

- Custom voices without a release: TOML manifests in `<CRUCIBLE_HOME>/voices/` are served alongside (and override) the shipped set after an engine restart.
- Stopping a user unit inside a WSL guest now works (via the system manager), fixing upgrades that could never activate.
- Releases rebuild only environment packs whose recipe changed; unchanged packs are carried from the previous release by reference. New release tooling: `scripts/bump.py`, `scripts/ship.sh`, `scripts/deploy.sh`, `scripts/tests.sh`.

## 0.6.7 — 2026-09-16

- Installers no longer embed a version: with no `--release` or `CRUCIBLE_RELEASE` they ask GitHub for the newest release, and fail with `release_lookup_failed` if they cannot.
- Can read models from the local Ollama store on `llama-windows`; llm rows report `ollama_copy` when the machine already has a matching Ollama model.
- The translate/simplify floor drops to `qwen3.5-9b`; a refused class now mentions using API keys when it can be routed.
- Memory estimates split into weights, overhead and KV-per-token (`[backends.<kind>.memory]`); llm rows publish `max_context`, and model manifests require `trained_context`.
- New generated `docs/API.md` reference. Voice pins for deathstalker, mistborn and owen updated to the weights in service.
- Fixed upgrading a WSL guest that still has an older user unit.

## 0.6.6 — 2026-09-16

- The WSL install now checks the guest (systemd, network, root access) after importing the distro; `linger_unreadable` is renamed `guest_root_unreachable`. The install can no longer loop forever on a repair that changes nothing.
- The guest service is installed with root via `wsl.exe -u root`, and a guest's old user unit is retired on upgrade.
- Fixed upgrade refusals: the stale-engine check now only judges the engine this installation runs, and a controller that closes its socket on quit counts as gone.
- SDK: both local-model settings fields are now required.

## 0.6.5 — 2026-09-16

- Fixed upgrades blocked by orphaned `wsl.exe` processes holding the install directory; child processes now start in `CRUCIBLE_HOME`.
- An existing launcher for this Crucible is replaced on upgrade; someone else's file is refused with instructions to move it.
- Removed three upgrade refusals that fired on healthy machines (engine owned by WSL unit, guest outliving its controller, `/quit` missing on 0.6.0).

## 0.6.4 — 2026-09-16

- Inside WSL, the server is now installed as a system unit (`/etc/systemd/system/crucible.service`), because WSLg hides the user D-Bus on a stock install. Outside WSL nothing changes.
- An upgrade now restarts a running service whose definition changed, and starting refuses with `engine_version_stale` if the old engine is still answering.

## 0.6.3 — 2026-09-16

- Apps can choose a local model per class through a `[local_models]` table. `GET /v1/settings` gains `local_models` and `local_model_choices` (with `fits` and `installed`); a model that will not fit is refused with `local_model_does_not_fit`.
- A failed WSL move is rolled back as a whole.
- POSIX installs wait for authenticated readiness before reporting success; the installer's error message no longer runs npm by mistake.

## 0.6.2 — 2026-09-16

- Fixed fresh POSIX installer parsing and Mac core initialization.
- Module installation checks recognize the native llama.cpp runtime.
- WSL activation is verified before native model weights are retired.
- Release core artifacts are built from an explicit source commit, and release publication is serialized.

## 0.6.1 — 2026-09-16

- New `align-longform` job type for aligning a whole audiobook. The client sends the book's sentences and the m4b. The server transcribes, coarse-aligns, runs the Qwen3 aligner and writes the VTT. Bad input is refused before any work starts: sentences out of reading order, blank or duplicate sentences, an unsupported language, or a `chunk_s` over 300 s.
- On cuda-linux, `tts` now renders Higgs on SGLang-Omni instead of vllm-omni, using a Python 3.12 env. `crucible install tts` now creates the two CUDA toolkit symlinks that flashinfer needs. On mlx-darwin, renders now use a measured batch width and run up to about 7x faster.
- Cancelling a `tts` render now stops it. The job ends `cancelled` instead of failing on narrator's `stopped` acknowledgement. If the engine does not stop within 120 s, the voice is unloaded and the job still ends `cancelled`.
- The `denoise` separator now stays loaded across jobs under a lease, so a book costs one model load instead of one per block. `unload-denoiser` was added. Only the primary stem is published, and `done.extra.load_seconds` reports the load time.
- `llm` models now start with `--language-model-only`, so the vision tower is never loaded (about 0.85 GiB freed). `qwen3.5-9b` now serves a 16384 context on every backend.
- New config key `advertise` (defaults to empty) declares addresses that something else forwards to this server. They are added to the URLs and pairing lines the server hands out. The Windows orchestrator has a new `POST /quit` route for a clean shutdown. Fixes: `asr` on cuda-linux reported ready but could not find its CUDA libraries at the first transcribe. A drifted env stamp can now be corrected without rebuilding the env.

## 0.6.0 — 2026-09-15

- The GPU card is cleared as soon as nothing holds it. Clients can take a lease with `POST /v1/models/{id}/lease`, renew it with `POST /v1/leases/{id}/heartbeat` and release it. A lease can name a voice or an aligner, so a book rendered chapter by chapter loads its voice only once. Lease refusals are `leased` and `not_resident`.
- Installs no longer need conda. `crucible install <type>` downloads a prebuilt env pack from the release; `--build` builds it locally instead. `install.sh` and `install.ps1` install a bare server and print a pairing line. New CLI commands: `crucible uninstall` (keeps weights unless you pass `--purge-weights`; supports `--dry-run`) and `crucible service install|uninstall|start|stop|status`. `@crucible/bootstrap` is published as a release asset.
- New operator web page at `/ui/` (`GET /` redirects to it), with status, tasks, job types, catalog, settings and app pairing. New routes: `GET /v1/setup`, `/v1/tasks` for installs and weight pulls over HTTP (pulls can be cancelled), and `DELETE /v1/catalog/{kind}/{id}`. New command `crucible token --url` prints the pairing line.
- New settings and routing: `GET /v1/settings` and `PUT /v1/settings`, the `[routes]` config table, `[upstreams.anthropic|openai|ollama]` and `POST /v1/settings/upstreams/{name}/test`, so a capability class can forward chat to a cloud or Ollama upstream. Upstream keys are write-only. The OpenAI-compatible API is now also served at `/openai/v1`. Model manifests can set sampling and thinking defaults in a `[defaults]` table. `ChatOptions.act` in the SDK sends `X-Crucible-Act`.
- Windows is now a backend, `llama-windows`, running llama.cpp `llama-server` with GGUF weights for `dots-ocr`, `qwen3.5-9b` and `qwen3.8-27b-4bit`. `crucible host`, now `crucible orchestrator`, is a Windows tray app that keeps the WSL engine running and can restart it. A distro you name in `[orchestrator] distro` can also be managed. On the Mac, `asr` now runs on mlx-whisper (`vad_filter` is refused there) and `align` runs on MPS.
- New `denoise` job type, which shares the rvc env. Zero-shot voices now load with a reference clip, and voice rows report `needs_reference`. Fine-tuned voices now declare a second retake rung at temperature 0.7, so a retake does not reuse the settings that caused the problem. App modules now name capability classes, and each server picks the model for its own machine.

## 0.5.0 — 2026-09-13

- New `GET /v1/activity` shows the whole server in one read: running and queued jobs with position, progress, last message and the calling client, the resident model, any open TTS streaming session (`claim`, with counts and a `progress` that is always `null`) and in-flight chat completions. The accelerator probe is opt-in with `?accelerator_probe=true`. The SDK gains `activity()`.
- `POST /v1/jobs` no longer queues behind a busy lane: it answers 409 `server_busy` naming the holder, job, type, model, status, since and progress. The SDK reads this as `CrucibleBusy` (with `busyLine`) and adds `isServerSpecificRefusal`; it does not retry on its own.
- Capability selection: `crucible install` now probes the card and records, per capability class, which model fits or the number that ruled it out. New `crucible capability` command re-decides without reinstalling (`--write` can only turn a type off), and new `GET /v1/capability` returns the ten `classes` (align, analysis, asr, clean, echo, pages, rvc, simplify, translate, tts), or 503 `capability_undecided`.
- Chat completions accept an optional `X-Crucible-Act` header naming the act (for example translate vs simplify); an unknown value is refused before the completion runs.
- TTS `chunk` events now carry narrator's `guard` verdict, forwarded unchanged (`null` when narrator says nothing), and the SDK exposes it. This SDK needs a server that sends the field. The tts recipes now pin a narrator commit that can emit it. The streaming door stays unguarded by design.
- Fixes: `desktop_allowance_bytes` is 3 GiB on cuda-linux and 25% of unified memory on mlx-darwin (`--desktop-allowance-bytes` still overrides). `crucible doctor` no longer calls a healthy Mac unhealthy over vLLM patches it does not use, and it checks the v3 sentinel patch (older envs now report stale). The `thirdreich` voice's safe band is corrected from 600-1000 to 500-700.

## 0.4.0 — 2026-09-13

- New TTS streaming door: `POST /v1/tts/stream` opens a session (one at a time), `GET /v1/tts/stream/{id}/events` streams its audio and events over SSE, `POST /v1/tts/stream/{id}` sends an op (`say` with a required `take`, or `cancel`), and `DELETE /v1/tts/stream/{id}` closes it.
- Per-row cancel works even though narrator can only cancel a whole batch: rows that are cancelled along with it are resubmitted, and a new `restart {id, from_seq, reason}` frame tells the client where the good audio starts again.
- A dropped stream can reattach with `Last-Event-ID` within a 15-second grace window. A request for frames that are no longer held is refused with `replay_unavailable`.
- A streaming session and the job lane can no longer use the engine at the same time. Jobs that touch the card are refused with `engine_in_use` while a session holds it.
- SDK: `stream({voice, language})` returns a session that you iterate for audio, plus `say()`, `cancel()`, `cancelAll()` and `close()`, and it reattaches by itself after a drop. `info()` no longer fails on a capability it cannot read: it returns that capability marked `unreadable`.
- Fix: older configs no longer break when new job types are added. A missing `enable_<type>` key now reads as off, and `crucible doctor` lists those keys (there is no need for `init --force`).

## 0.3.0 — 2026-09-13

- New `tts` job type, with narrator as a managed engine. Voices are manifests listed at `GET /v1/voices`. `load-voice` / `unload-voice` put a voice on the card. The render job turns text chunks into one `<index>.flac` per chunk and emits a `chunk` event per row. A voice's cap is `max_chars` (characters), and `capped`/`tokens` are `null` when narrator does not report them. New CLI: `crucible voices list|pull` and `crucible install tts --narrator-engine`. Turn it on with `[jobs] enable_tts`.
- New `asr` job type (faster-whisper, six pinned models, cuda-linux only). It has no default model, and both ffmpeg and an explicit model, language and options are required.
- New `align` job type (Qwen3-ForcedAligner) with per-chunk `cue` events and an `unload-aligner` job. New `rvc` job type, with seven pinned models and a `crucible rvc` command; it needs its base assets under `~/.crucible/rvc-base/`. Enable them with `crucible init --enable-align --enable-rvc`.
- New `GET /v1/accelerator` reports what holds the card, including `unattributed_bytes`, without evicting anything. It returns 503 `accelerator_unreadable` when it cannot see the card. `/v1/health` gains `resident_kind`. `/v1/info` lists each capability once and adds `job_types`.
- Models: the `dots-ocr` page reader (cuda-linux). Every manifest now requires `modalities`. `qwen3.5-27b` is renamed to `qwen3.8-27b`, and `qwen3.8-27b-4bit` is added for 24 GB cards. A backend can set its own `context_default`. `/v1/models` and `/v1/openai/models` report `max_model_len` and `fingerprint`, and provenance sidecars now record the revision and fingerprint.
- Proxy and fixes: a client that disconnects now cancels the engine's request (499 `client_disconnected`). Request bodies are forwarded byte for byte on vLLM, and an engine's own 400 is passed through. vLLM now loads under WSL2 (pinned memory). The 9B no longer fills the card. `crucible doctor` no longer crashes. The SDK passes unknown event kinds through as `unknown` and adds `voices()`, `loadVoice()`, `unloadVoice()`, `accelerator()`, `asr()`, `render()`, `writeArtifactsTo()`, and `responseFormat`/`seed` for chat.

## 0.2.0 — 2026-09-12

- New `llm` job type: `crucible install llm`, manifests for `qwen3.5-9b` and `qwen3.5-27b`, and `load-model` / `unload-model` jobs that keep one model resident (vLLM on cuda-linux, mlx-lm on mlx-darwin).
- New routes: `GET /v1/models`, `/v1/openai/chat/completions` (streaming and non-streaming) and `/v1/openai/models`. Completions always report Crucible's model id, and `/v1/health` reports `warming` for the whole of a load.
- Loads are refused by name, permanent reasons first: `backend_unsupported`, `insufficient_memory`, `env_missing`, `model_not_installed`, `accelerator_busy`. A model too big for the card is no longer sent off to download.
- SDK: `models()`, `loadModel()`, `unloadModel()`, `chat()` and `chatStream()`, with a `thinking` option for reasoning models. The server can also be run with `python -m crucible`.

## 0.1.0 — 2026-09-12

- First release: the server handshake with API v1, bearer-token auth, SSE job events and provenance sidecars, on two backends (`cuda-linux` and `mlx-darwin`). Windows is refused.
- CLI: `crucible init`, `crucible serve`, `crucible doctor` (`--json`) and `crucible token --show`, plus an `echo` test job type enabled with `[jobs] enable_echo`.
- `@crucible/client`, the TypeScript SDK with no runtime dependencies, ships in the same release as the server (sdist, wheel and `crucible-client-<ver>.tgz`).
