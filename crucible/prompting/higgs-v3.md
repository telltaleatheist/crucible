# Higgs TTS v3: how to feed it text

Model `higgs-default` / `zeroshot` (and every fine-tune merged from it), upstream `bosonai/higgs-tts-3-4b` @ `239f63fb7b02b1aa085f98d9efae5e35cc5523e8` (from `crucible/engines/higgs-v3/base.toml`), Crucible job type `tts` (also `load-voice` and the `/v1/tts/stream` session).

Note on the name: the repo was renamed. `bosonai/higgs-audio-v3-tts-4b` (the name narrator's code and higgs-tts-3-4b/AGENTS.md still use) now redirects to `bosonai/higgs-tts-3-4b`. Same weights.

## Sources

Read on 2026-10-15 unless stated.

Official (Boson AI and the serving stacks the model card names):

- [O1] Model card README, HF `bosonai/higgs-tts-3-4b` @ 239f63fb (last modified 2026-09-04). https://huggingface.co/bosonai/higgs-tts-3-4b/blob/239f63fb7b02b1aa085f98d9efae5e35cc5523e8/README.md
- [O2] `higgs-tts-3-4b/PROMPTING.md`, same repo and revision.
- [O3] `higgs-tts-3-4b/AGENTS.md`, same repo and revision.
- [O4] `tokenizer_config.json`, `tokenizer.json`, `config.json`, `chat_template.jinja`, same revision. No `generation_config.json` exists at this revision (HTTP "Entry not found").
- [O5] Boson hosted API docs, overview page: https://docs.boson.ai/models/higgs-tts/overview (no date on page).
- [O6] Boson hosted API docs, voices page: https://docs.boson.ai/models/higgs-tts/voices (no date on page).
- [O7] Boson blog, "Higgs TTS 3", 2026-06-04: https://www.boson.ai/blog/higgs-audio-v3-tts
- [O8] SGLang-Omni Higgs TTS cookbook: https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html (page dated 2026-10-15 at fetch).
- [O9] SGLang-Omni 0.1.4 source as installed in WSL (`/home/telltale/anaconda3/envs/sglomni/.../sglang_omni`): `models/higgs_tts/text_tokenizer.py`, `stages.py`, `payload_types.py`, `serve/speech_service.py`, `serve/protocol.py`. Caution: this env carries local patches (`.orig` files beside several modules, and a "night4 2026-09-08" edit in `speech_service.py`). It is the legacy BookForge env; Crucible's own env pins the same `sglang-omni==0.1.4` (`crucible/envs/tts/higgs-v3-cuda-linux.txt`).
- [O10] vLLM-Omni Higgs recipe: https://github.com/vllm-project/vllm-omni/blob/main/recipes/BosonAI/Higgs-Audio-V3-TTS.md (main, no date visible).
- [O11] Higgs Audio **v2** repo `boson-ai/higgs-audio` @ `05a145bb` (2026-06-05), `examples/generation.py`. v2 only. Used where noted.

Community:

- [C1] HF discussion #18 "Glitches and Silence" (2026-06-24 to 06-28).
- [C2] HF discussion #20 "Control tags shift the speaker's timbre" (2026-07-04 to 08-18).
- [C3] HF discussion #8 "Tupt sound at regular intervals" (2026-06-09 to 08-02).
- [C4] HF discussion #17 "Cross-linguistic Voice Clone Stability" (2026-06-22 to 06-25).
- [C5] HF discussion #13 "How to use Control Tokens?" (2026-06-13).
- [C6] `duct-tape2/higgs-voice-kit` README @ `764cad0f` (2026-09-16).

Ours (all read-only):

- [R1] `crucible/engines/higgs-v3/base.toml`, crucible @ `5a579e7a` (file last changed a067ebd, 2026-10-02).
- [R2] `crucible/docs/internals/voices.md`, same checkout.
- [R3] `crucible/docs/API.md` (generated), `tts` job section, same checkout.
- [R4] narrator at the commit Crucible pins, BookForge `94c1f49d` (2026-10-02): `python/narrator/engine/higgs/v3_served.py`, `sgl_served.py`, `prompt.py`, `python/narrator/serve/worker.py`.
- [R5] BookForge `bookforge/docs/HIGGS_ENGINE.md` (last changed 3605eec3, 2026-09-19).
- [R6] Owen's project notes (BookForge, not published): `higgs-sampling-default-no-deviation.md`, `higgs-temperature-control-2026-09-11.md`, `higgs-zero-shot-path.md`, `higgs-guard-book-pace.md`, `deathstalker-band-500-800-longest-pause.md`, `narrator-unspoken-glyphs-and-caps-headings.md`, `higgs-first-inapp-render-findings.md`, `voice-facts-live-on-crucible.md`, `em-dash-deterministic-where-trustworthy.md`, `narrator-build-state.md`.

## What the model was trained to see

**Architecture.** An autoregressive decoder over interleaved text and audio tokens. Audio is 8 codebooks at 25 frames per second, with a delay pattern. OFFICIAL [O1]. Output is 24 kHz. COMMUNITY/search summary and OURS [R1] (`sample_rate = 24000`).

**The prompt is not a chat.** On SGLang-Omni (the stack Crucible runs on cuda-linux) the prompt is a flat token string with three shapes. OFFICIAL [O9] `text_tokenizer.py` docstring:

- zero-shot: `<|tts|> <|text|> {text} <|audio|>`
- clone with transcript: `<|tts|> <|ref_text|> {ref transcript} <|ref_audio|> [audio placeholders] <|text|> {text} <|audio|>`
- clone without transcript: the same, minus the `<|ref_text|>` part.

The same file warns that leaving out `<|tts|>` "yields fluent-but-wrong output". The caller never writes these tokens. It sends `input` and `references` to `/v1/audio/speech`.

**No system prompt and no scene description.** v3 has none. The v2 model took a system message plus a `scene` role (`<|scene_desc_start|>...<|scene_desc_end|>`), defaulting in our port to "Audio is recorded from a quiet room." OFFICIAL v2 [O11]; OURS [R4] `prompt.py`, which states "Higgs v3 has NO scene-description token at all". The v3 tokenizer has no scene token. OFFICIAL [O4]. Sending `<|scene_desc_start|>` to v3 is read aloud. OURS [R4] `v3_served.py`.

higgs-tts-3-4b/AGENTS.md says to build prompts through `chat_template.jinja` and "do not hand-assemble the chat prompt". OFFICIAL [O3]. That template at this revision is a plain Qwen ChatML template with tool calls. It has no audio or reference handling. OFFICIAL [O4]. The serving stacks do not use it for `/v1/audio/speech` [O9]. See Conflicts.

**Control tags are vocabulary, not text.** The model knows exactly these inline tags, each one token. OFFICIAL [O1][O2][O4]:

- emotion (21, sentence-level): affection, amusement, anger, arousal, awe, bitterness, confusion, contemplation, contentment, determination, disgust, elation, enthusiasm, fear, helplessness, longing, pride, relief, sadness, shame, surprise.
- style (3, sentence-level): singing, shouting, whispering.
- prosody: sentence-level `speed_very_slow` (~0.65x), `speed_slow` (~0.85x), `speed_fast` (~1.2x), `speed_very_fast` (~1.4x), `pitch_low` (~-3 st), `pitch_high` (~+2.5 st), `expressive_high`, `expressive_low`; inline `pause` (~400-700 ms), `long_pause` (~700-1500 ms).
- sfx (9, inline): cough, laughter, crying, screaming, burping, humming, sigh, sniff, sneeze.

higgs-tts-3-4b/PROMPTING.md counts 43. The tokenizer also carries `<|env:music|>` and `<|env:noise|>`, which no official doc describes. OFFICIAL [O4]; narrator's allowlist includes both. OURS [R4].

Placement rules. OFFICIAL [O1][O2]: delivery tags (emotion, style, speed/pitch/expressive) go at the start of the sentence or turn. `pause`, `long_pause` and `sfx` go inline exactly where they fire. An sfx tag is followed immediately, with no space, by its onomatopoeia ("`<|sfx:laughter|>Haha`"). Tags stack. "Only the tags below are recognized — anything else degrades output or gets read literally" [O2].

**Reference audio (zero-shot cloning).** One reference: audio plus its transcript. "Supplying the reference transcript (`text`) materially improves cloning fidelity." OFFICIAL [O1]. The audio is encoded into codec frames and spliced into the prompt as placeholders, one per delayed frame row (T frames cost T+7 positions). OFFICIAL [O9]; OURS [R4] `sgl_served.py`. So reference length is paid for in context.

## Writing the text

**Length per call.** Boson's hosted API: keep `input` to "roughly 300 characters". Longer text "may come back truncated or with garbled words as a normal 200 response rather than an error". Split long scripts on sentence boundaries. OFFICIAL [O5]. The SGLang server refuses `input` over 4096 characters with HTTP 400 (configurable in our patched env via `HIGGS_MAX_INPUT_CHARS`). OFFICIAL [O9]. The SGLang Higgs builder serves a 4096-token context by default (`context_length = 4096`); prompt plus `max_new_tokens` must fit. OFFICIAL code via OURS [R4] `sgl_served.py`.

**Normalisation.** The hosted API runs text normalisation by default: it expands "numbers, dates, times, currency, units, URLs, email addresses, phone numbers, abbreviations, and symbols into spoken form"; `enable_tn: false` turns it off. OFFICIAL [O5]. Self-hosted SGLang-Omni and vLLM-Omni have no such pass in the code or docs we read [O8][O9][O10]. So a self-hosting caller owns this step. That the hosted API normalises suggests the model reads spoken-form text best; Boson does not say this directly. (Inference: see Open questions.)

v2's example script did its own light clean-up of the transcript: Chinese punctuation to English, parentheses removed, `°F`/`°C` spelled out, whitespace collapsed, and a period added if the text did not end in punctuation. OFFICIAL v2 only [O11]. Not documented for v3.

**Punctuation and pauses.** Officially, pauses come from `<|prosody:pause|>` and `<|prosody:long_pause|>`. `speed_very_slow` only slows so far; for slower delivery use `long_pause` between phrases. OFFICIAL [O2]. Unknown tags are read aloud. OFFICIAL [O2]. Ours measured the same: an out-of-vocabulary tag derailed a render into a loop with ASR coverage 0.000. OURS [R4] `v3_served.py`.

**Language.** Use only the listed languages. 85 are under 5 % WER/CER, 17 between 5 and 10 %. OFFICIAL [O1][O3].

## Voices

**Built-in voice.** With no reference, the server uses the base model's own speaker. OFFICIAL [O1] (zero-shot example). In Crucible this is `higgs-default`, a smoke voice, "not a production voice". OURS [R2].

**Zero-shot reference clips.** Official guidance:

- "5-30 seconds of clean speech, with no music or background voices". OFFICIAL [O6].
- A verbatim transcript "including filler words". OFFICIAL [O6].
- The transcript "materially improves" cloning. OFFICIAL [O1][O8].
- Formats wav, mp3, opus, pcm, flac (hosted API) [O6]; WAV/FLAC/MP3 (vLLM-Omni) [O10].
- You must own the right to clone the voice; cloning without consent is prohibited by the licence. OFFICIAL [O1][O6].

What Crucible enforces. OURS [R2] `voicereference`:

- one clip, 30 s total, 32 MiB decoded; seconds measured from the WAV header;
- transcript required and book-exact;
- WAV in base64 on the wire.

**Fine-tuned voices.** These are merged checkpoints, so the weights are the voice and no reference is sent. OURS [R2][R3]. Officially there is a fine-tune discussion (HF #5) but no Boson fine-tuning guide was found. A clip adds nothing to identity on our plain-trained fine-tunes (ECAPA 0.86 with or without a clip). OURS [R6] `higgs-zero-shot-path.md`.

## Settings

Official numbers disagree:

| Source | temperature | top_p | top_k | other |
|---|---|---|---|---|
| Model card voice-clone example [O1] | 0.8 | not set | 50 | `max_new_tokens` 1024 |
| higgs-tts-3-4b/AGENTS.md "Recommended sampling (voice cloning)" [O3] | 0.8 | not set | 50 | `max_new_tokens` 1024 |
| SGLang cookbook documented defaults [O8] | 0.8 | 0.8 | 30 | `max_new_tokens` 2048 |
| SGLang 0.1.4 `/v1/audio/speech` generic `SamplingParams` [O9] | 0.8 | 0.8 | 30 | repetition_penalty 1.1 |
| SGLang 0.1.4 Higgs stage when a field is unset [O9] | 1.0 | none (1.0) | none (off) | `max_new_tokens` 2048 |
| vLLM-Omni deploy profile, per our notes [R6] | 1.0 | 0.95 | 50 | — |

The two SGLang rows come from one codebase. The Higgs stage builds its state from only the fields the request actually set (`model_fields_set`), so the generic 0.8/0.8/30 never reaches it. Unset means temperature 1.0 and no top-k or top-p. OFFICIAL code [O9]. We measured the result: with no top_k, "one chunk ran to the cap with 80 s of silence". OURS [R2][R4].

Crucible mapping. OURS [R1][R2][R3]:

- `sampling = { temperature = 0.8, top_p = 0.95, top_k = 50 }` on both arms of both base rows. This is the model card's 0.8 and 50, plus a top_p of 0.95 taken from the vLLM-Omni deploy profile, not from the card.
- All three keys are required, and a deviation needs `sampling_reason`.
- `repetition_penalty` is not allowed for Higgs.
- Crucible never reads `generation_config.json`.
- The job's `done` event reports the full triple applied as `sampling`.
- `seed` goes at the top level per take (the `take` rung picks the seed lane).
- `max_new_tokens` is derived by narrator from the text (2.0x expected frames + 150, clipped to the context headroom). The caller does not set it.

## Community findings

- Glitches (screeching, repeats, half a reply then silence) on SGLang got better at temperature 0.7 / top_k 50, and better again at top_k 10. One user's report. COMMUNITY [C1].
- Emotion and style tags shift timbre and F0, with and without cloning. `emotion:arousal` sometimes switched to a different-gender voice. That test used repetition_penalty 1.3. A reply suggested 1.0 to 1.05 for speech. COMMUNITY [C2].
- Periodic clicks ("tupt") every 2-3 s when streaming, also on the hosted API. A Boson-side reply suggested `pcm` output and filing with sglang-omni. COMMUNITY [C3].
- A Japanese reference with Chinese target text came out Japanese. Cross-lingual cloning between CJK languages is unstable. COMMUNITY [C4].
- Writing the onomatopoeia after an sfx tag can collide with real words in other languages (Indonesian "sob" means "friend"). COMMUNITY [C5].
- higgs-voice-kit: reference 10-20 s, single speaker, -20 to -16 LUFS, transcript word-for-word with correct punctuation and capitalisation. It chunks by sentence to at most 200 characters. It clones once to a ~5 s "anchor" and conditions every chunk on that to stop the voice drifting. Its defaults are temperature 0.66, top_k 24, top_p 0.8. It retries a chunk that stops early with new seeds. COMMUNITY [C6].

## Measured in our projects

Text length:

- Zero-shot / base: coverage holds to 600 characters; 898 fails reproducibly (coverage 0.86 and 0.78, tail 0.31 and 0.00). A reference clip does not fix it. [R5] "Every number is measured"; [R4] `v3_served.MAX_CHARS`. Crucible packs zero-shot at `target_chars = 600`, basis `placeholder`. [R1][R2]
- Duration ratio is not a proxy for coverage: one 898-character chunk had ratio 0.99 while dropping 22 % of its text. [R5]
- A fine-tune's stop length follows its training clip length. A fine-tune trained on 8-22 s clips stopped after ~6-10 s on any prompt over ~150 characters. Every fine-tune needs its own measured cap. [R5]
- deathstalker band 500-800 characters, chosen on median longest pause per render (0.50 s at 100, 1.38 at 500, 1.54 at 800, 1.70 at 1400). Below ~300 characters is a defect source; the 200 rung was worst (15.6 % / 12.5 % failures). [R6] `deathstalker-band-500-800-longest-pause.md`
- Caps are characters, not tokens. [R2]
- Book prose tokenised at 3.25 to 4.56 characters per token on our corpus. [R4] `sgl_served.py`

Text form:

- The Higgs voices were trained on `...` and an unspaced `word—word` em dash. A spaced `. . .` or a hyphen for a dash makes them invent words or skip the pause. narrator's `normalize_for_tts` rewrites every ellipsis to `...` and fixes the dash shapes that can only be dashes. It runs on every `tts` batch. [R6] `em-dash-deterministic-where-trustworthy.md`; [R4] `serve/worker.py` `normalize_for_tts`, called from `generate_batch`.
- The same function expands digits, money, percents, ordinals and years to words with num2words. If num2words cannot read a language, the request is refused. [R4]
- narrator strips its own markup (`[break]`, `[pause]`, `[heading]`, `[item]`, `[music]`, `[sfx]`, `[silence]`) before the model sees it. It refuses any `<|...|>` tag outside the vocabulary. [R4] `prompt.clean_text`, `v3_served.validate_control_tokens`
- Asterisk bullets were read aloud until they were stripped at extraction. [R6] `narrator-unspoken-glyphs-and-caps-headings.md`
- An all-caps heading rendered "dues", a ~1 s stall, then the rest. Owen ruled that all-caps runs are folded to Title Case, with an acronym guard. The corpus had caps only as sentence-opening lead-ins. [R6] same file
- A spelled-out "R E U T E R S" was the one trigger for babbling, fixed by a long-caps-name rule. [R6] `higgs-sampling-default-no-deviation.md`
- About 10 % of chunks are damaged on every clean engine for text reasons: em dashes missing from the corpus, a footnote welded onto the head of a chunk, umlauts. [R4] `sgl_served.py`; [R5]
- A doubled terminal period behind a closer (`Christianity.).`) broke alignment. [R6] `higgs-first-inapp-render-findings.md`

Sampling:

- Temperature 0.6 to 0.9 is flat for pace. Babble starts at 1.0, and 0.9 is the highest safe value. Pace rises about 0.15 chars/s per 0.1 of temperature. [R6] `higgs-sampling-default-no-deviation.md`
- Same 88 chunks, mistborn, SGLang: at 0.6 with rep 1.1, 22 guard fires and 4 chunks with real dropped text. At 0.7, 8 fires and 0 drops. At 0.8, 4 fires, 0 drops, and the long-pause tail halved (holes of 5 s or more fell 8 → 7 → 3). The repetition penalty was not the cause. [R6] `higgs-temperature-control-2026-09-11.md`
- On the Mac's WER table, 0.3 truncates and 0.6 is about equal to 1.0. [R6] `higgs-first-inapp-render-findings.md`

References:

- No reference: speaker cosine 0.093. One clip: 0.680. Two clips joined: 0.692. The narrator's own ceiling is 0.766. Same-book clips add +0.076, a second clip only +0.012. [R4] `v3_served.py`
- More than 30 s of reference is refused; 42 s gave HTTP 400 on vllm-omni. [R4]; [R1]
- Training's pick: one clean treated clip of ~15 s (denoised; "the clone copies whatever noise the clip carries"), with a book-exact transcript that has no digits or quotes. 12-16 s works (HIGH confidence); whether it is optimal is MEDIUM, not swept. [R5]; [R6] `higgs-zero-shot-path.md`
- A 15 s clip costs about 430 of the 4096 context positions, a 30 s clip about 760. [R5][R2]
- Expect at least 1 retake in 10 on zero-shot. [R6] `higgs-zero-shot-path.md`

Serving:

- vllm-omni at 16 in flight damaged 13 of 50 chunks (4 early stops). SGLang-Omni at 16 damaged 5 of 50 with 0 early stops, at 2.5x the throughput. [R4] `sgl_served.py`

Guard:

- Pace guard: re-roll a take whose characters per second fall outside 1.3x either side of the book's running median. A truncation showed up as a take running too fast. [R6] `higgs-guard-book-pace.md`

## Conflicts and open questions

**Official vs Crucible.**

1. **top_p.** Crucible sends top_p 0.95. The model card and higgs-tts-3-4b/AGENTS.md send none, and the cookbook's documented default is 0.8. 0.95 comes from the vLLM-Omni deploy profile [R6]. It is not on the card. The memory note calls 0.8/0.95/50 "the boson default"; only 0.8 and 50 are on the card.
2. **Temperature for base weights.** The card says 0.8. Crucible's base rows say 0.8 [R1]. But narrator's `SERVER_DEFAULT_SAMPLING` is 1.0/0.95/50 [R4], and `bookforge/docs/HIGGS_ENGINE.md` still says zero-shot renders at T 1.0 [R5]. Crucible's manifest replaces it [R2]. Which number actually reaches a base-weights request is worth one log check (`done.sampling`).
3. **Chunk length.** Boson says keep `input` to ~300 characters; longer "is not yet reliably supported" [O5]. Crucible packs zero-shot at 600 [R1], measured on ASR coverage [R5]. Our earliest delivered render used 300 (`DELIVERED_MAX_CHARS`) [R4]. Fine-tunes run longer still (deathstalker 500-800). Our 600 is a measured exception, not Boson's guidance.
4. **Text normalisation.** Boson's hosted API expands numbers, units, abbreviations and symbols by default [O5]. Self-hosted, nothing does, so the app owns it. But narrator, inside Crucible, also expands digits, money, percents, ordinals and years and rewrites ellipses and dashes [R4]. This is text preparation inside the serving layer, which the stated division ("apps own their text preparation") says should not happen. It is consistent with Boson's hosted behaviour, though.
5. **Chat template.** higgs-tts-3-4b/AGENTS.md says to use `chat_template.jinja` and never hand-build the prompt [O3]. The template at this revision is generic ChatML, and SGLang-Omni builds the flat `<|tts|>...<|audio|>` prompt itself [O9]. Crucible correctly sends no chat template. The official advice does not apply to `/v1/audio/speech`.
6. **Reference length.** Boson: 5-30 s [O6]. Ours: 12-16 s preferred, 30 s cap [R5][R2]. Community: 10-20 s [C6]. These do not conflict.
7. **Control tags.** Officially supported [O1][O2]. Crucible passes them only if a voice's `allowed_controls` lists them; the base rows list none [R1][R2]. Community reports that tags shift timbre [C2]. Our fine-tunes were never trained on tags. Untested here.

**Open questions (inference, not guidance).**

- Does the base model prefer spoken-form text (no digits, no symbols)? The hosted TN default suggests yes. We have no A/B.
- Is a 300-character ceiling better than 600 for zero-shot on SGLang? Base plus clip at 600 on SGLang is rated MEDIUM, not re-measured [R5].
- Do `<|prosody:pause|>` and `long_pause` give more controllable chapter or scene pauses than silence added at assembly? Not tried.
- Does the "anchor" trick [C6] reduce identity drift between chunks for zero-shot? Drift from chunk to chunk is unmeasured here [R6].
- What do `<|env:music|>` and `<|env:noise|>` do? They are in the vocabulary but in no doc.
- No official statement on numbers, abbreviations or acronyms for self-hosted v3 was found. The v2 script's clean-up (parentheses, degrees, ending period) [O11] may or may not carry over.
