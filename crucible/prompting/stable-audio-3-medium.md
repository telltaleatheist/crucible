# Stable Audio 3.0 Medium: how to prompt it

Model id `stable-audio-3-medium` (Crucible class `music`); upstream `stabilityai/stable-audio-3-medium` @ `27b5a21b791b1b033d193a9e1e3ce78493f102f9` (both arms, from `crucible/audio/stable-audio-3-medium.toml`); inference package `Stability-AI/stable-audio-3` @ `779434a9`; Crucible job type `audio`.

## Sources

All web sources read 2026-10-15. Crucible files read at crucible commit `5a579e7a` (2026-10-13).

- **[S1]** `stable-audio-3/docs/guides/prompting.md`, Stability-AI/stable-audio-3 @ `779434a908193105335fd8d833418603625b2859` (the commit Crucible pins, 2026-09-01). It is byte-identical at main `77e0398c` (2026-10-11). https://github.com/Stability-AI/stable-audio-3/blob/779434a908193105335fd8d833418603625b2859/docs/guides/prompting.md (OFFICIAL)
- **[S2]** `stable-audio-3/docs/workflows/inference.md`, same repo and commit, identical at main. (OFFICIAL)
- **[S3]** `stable-audio-3/docs/guides/model-overview.md`, same repo and commit, identical at main. (OFFICIAL)
- **[S4]** `README.md`, same repo and commit, identical at main. (OFFICIAL)
- **[S5]** `stable_audio_3/interface/reprompt.py` @ `779434a9`. This file holds the system prompts of Stability's own Gradio "Prompt Assistant": a classifier plus Music, Instrument, One-shot and SFX prompt writers, and the `TrackType` prefixes it adds. It is the most concrete official statement of the prompt shape Stability expects. https://github.com/Stability-AI/stable-audio-3/blob/779434a908193105335fd8d833418603625b2859/stable_audio_3/interface/reprompt.py (OFFICIAL)
- **[S6]** `stable_audio_3/interface/diffusion_cond.py` @ `779434a9` (the Gradio UI: its default sliders, and how it handles the assistant's `Length:` suffix). (OFFICIAL)
- **[S7]** `stable_audio_3/model.py`, `inference/sampling.py`, `models/dit.py` @ `779434a9`. Covers how `prompt` and `duration` become conditioning, the 6 s duration padding, the default sampler, and when the CFG branch runs. (OFFICIAL)
- **[S8]** Hugging Face model card, `stabilityai/stable-audio-3-medium` @ `27b5a21b`. At HF head `d4d35793` it differs only in the licence link. https://huggingface.co/stabilityai/stable-audio-3-medium (OFFICIAL)
- **[S9]** `model_config.json` @ `27b5a21b`. The file is gated, so it was read from Crucible's own pulled copy at `~/.crucible/models/stable-audio-3-medium/cuda-linux/` in WSL. It gives the conditioner settings, `sample_size` and the training demo prompts. (OFFICIAL)
- **[S10]** *Stable Audio 3*, tech report, arXiv 2605.17991v1, 18 May 2026, sections 2.2, 3.1, 3.5, 4, 5.1, 5.2, 5.4 and 5.7. https://arxiv.org/abs/2605.17991 (OFFICIAL)
- **[S11]** "Stable Audio 3.0 Prompt Guide", stability.ai/guides, dated "August 18" with no year shown. Mirrored at kb.stability.ai. This guide is written for Stability's products (web app and DAW plugin), not for the open weights specifically. https://stability.ai/guides/stable-audio-3-prompt-guide (OFFICIAL, product-scoped)
- **[S12]** Issue #42, "Limited to 2 minute generations using medium model", 2026-05-29, with a reply from a Stability collaborator (mhrice). https://github.com/Stability-AI/stable-audio-3/issues/42 (COMMUNITY report, confirmed by OFFICIAL maintainer)
- **[S13]** Issue #50, "Prompt to disable guitar behind-the-bridge play", 2026-06-14, unanswered. (COMMUNITY)
- **[S14]** Salomone, Gandhi, Asaria and Primus, "The best-sounding open-source music generator is also the only one you can steer", lab.cloud, 2026-06-25. https://lab.cloud/research/beyond-fad-and-clap/ (COMMUNITY)
- **[S15]** adamdived/stable-audio-3-workflow (a ComfyUI workflow) @ `a67e7f91`, 2026-06-07. https://github.com/adamdived/stable-audio-3-workflow (COMMUNITY)
- **[S16]** ComfyUI docs, "Stable Audio 3 ComfyUI workflow example". https://docs.comfy.org/tutorials/audio/stable-audio/stable-audio-3 (COMMUNITY: Comfy Org, not the trainer)
- **[S17]** Issue #48, "comfyui version very bad", 2026-06-13. (COMMUNITY)
- **[S18]** stableaudio3.com, "How to Use Stable Audio 3". A third-party site that sells credits; it is not on a Stability domain. https://stableaudio3.com/how-to-use-stable-audio-3 (COMMUNITY)
- **[S19]** PR #106, "Optional (phrase:weight) prompt weighting for MLX inference", 2026-09-06, open and not merged. (COMMUNITY)
- **[C1]** `crucible/audio/stable-audio-3-medium.toml`. **[C2]** `docs/AUDIO.md`. **[C3]** `docs/CHANGELOG.md`. **[C4]** `docs/internals/audio.md`. **[C5]** `crucible/jobs/audio/stable_audio_worker.py`. **[C6]** `crucible/playground.py`. (Crucible repo)

## What the model was trained to see

- **Text plus one number.** The model is conditioned on a text prompt and a duration in seconds (`seconds_total`), and on nothing else in a plain text-to-audio call [S3][S7][S9]. Length comes from the duration conditioning, never from the text [S10 §2.2].
- **Text encoder.** The encoder is T5Gemma (`google/t5gemma-b-b-ul2`). Prompts are padded or truncated to **256 tokens** [S10 §2.2][S9 `max_length: 256`].
- **Duration conditioning.** Duration enters every transformer block twice: through AdaLN and through cross-attention next to the text [S10 §2.2]. `seconds_total` is normalised over 0–384 s [S9].
- **Training captions were metadata strings.**
  - AudioSparx tracks carry natural-language descriptions, titles, keywords, and fields such as BPM, genre, moods and instruments.
  - Freesound clips carry titles, tags and descriptions.
  - The training prompt was built from a random subset of those fields. With 50% probability each field kept its label (`Instruments: Guitar, Drums, Bass Guitar, Moods: Uplifting`). The fields were often shuffled or subsampled, joined with commas, and lowercased half the time [S10 §3.5 "Prompt preparation"].
  - Half the Freesound prompts were LLM-rewritten captions [S10 §3.5].
- **Training data.** 806,284 AudioSparx recordings (music, instruments, SFX) plus 472,618 Freesound recordings. Medium was trained on both, so it does music, stems and sound effects in one model [S8][S10 §3.5].
- **Modality prefixes.** The report says every AudioSparx-trained Stable Audio 3 variant prepends `TrackType: Music, VocalType: Instrumental,` for music and `TrackType: SFX,` for sound effects. It says these prefixes "significantly improve generation quality" and recommends them to all users [S10 §5.1].
- **Empty prompts were trained.** CFG dropout replaced the text and timing conditioning with zeros 10% of the time [S9 `cfg_dropout_prob: 0.1`][S10 §3.5]. That is why an empty prompt still produces audio [S1].
- **Clean endings.** Training padded clips with silence (about 4 s on average) to teach the model to end with natural silence rather than an abrupt cut [S10 §3.1][S9 `silence_extension_scale_seconds: 4.0`].
- **English only.** The model was trained on English descriptions; other languages underperform [S3].

## Writing the prompt

**Do**

- **Start with the modality prefix.**
  - Music: `TrackType: Music, VocalType: Instrumental,` [S10 §5.1][S1][S5].
  - Stem or solo instrument: `TrackType: Instrument,` [S1][S5]. Add `Format: Duo` for two instruments [S1].
  - Sound effect on Medium: `TrackType: SFX,` [S1][S5].
  - Stability's own assistant adds these prefixes with a comma and a space, exactly as written above [S5 `TRACK_TYPE_PREFIXES`].
- **Name genre, instruments, mood and energy, and BPM** for music [S1].
  - The assistant's template: genre or style, then main instruments, supporting layers, rhythm and percussion, mood and energy, then BPM, all in one fluid sentence, avoiding semicolons [S5 "Music"].
  - The product guide gives a similar order (genre, instruments and rhythm, mood and arrangement, production character, performance detail). It also suggests era, location or recording-style references, and a usage context such as "the final scene of a slow-burning sci-fi film" [S11].
- **Describe each instrument, not just its name.** The guide asks how each instrument "sound[s] or feel[s]" [S1]. Its examples spell out playing and timbre, e.g. "close-mic'd electric piano plays hard" and "tightly tuned acoustic drums" [S1].
- **Write tempo as a number.** The guide writes "120 BPM" [S1]; the README writes "…with all your friends 124 BPM" [S4]; the assistant writes `BPM: 85.` [S5]. BPM is one of the AudioSparx metadata fields the model saw [S10 §3.5].
- **AudioSparx-style tags help.** For example `Genre: Funk, Genre: Jazz` and `Instruments: Guitar, Saxophone, Bass, Piano`. Genres can be combined, even unrelated ones [S1].
- **Stems.** Say the technique, the recording environment and the effects. "Many of these descriptors are part of the dataset" [S1 tip under stems].
- **Sound effects on Medium.** Name the core source, the action (how it is triggered and how long it lasts) and the production (mic placement, room, processing) [S1]. See `stable-audio-3-small-sfx.md` for the SFX model.
- **Keep it compact.** The assistant rejects its own output above 45 words and retries [S5 `_has_artifacts`]. The encoder truncates past 256 tokens [S9].
- **Match the length to what you describe.** Set `duration_s` to fit the content [S1] (see the next section).

**Don't**

- **Don't ask for words to be sung or spoken.**
  - The models "don't output intelligible vocals" but "sometimes generate unintelligible vocal textures" [S1].
  - The model is "not designed for speech or voice generation" [S3].
  - Stability's assistant rejects any rewrite that contains vocal words (vocals, singing, singer, female, male, voice, chorus, rap, chant, lyrics) [S5 `_VOCAL_RE`].
  - Use `yue2-3b` for songs [C2].
- **Don't put the length in the text.**
  - Stability's assistant writes `Length: N seconds` at the end of its rewrite. The Gradio UI strips it from the prompt and moves the number to the "Seconds total" slider before generating [S5][S6 `_LENGTH_EXTRACT_RE`].
  - In Crucible, the length belongs in `duration_s`.
- **Don't use markup.** The assistant rejects outputs that contain `[`, `]`, `*` or `#` [S5].
- **Don't write in a language other than English** [S3].
- **Don't rely on a negative prompt.** Crucible refuses it (next section).

## Settings that change how the prompt lands

| Upstream control | Crucible param | What the official sources say | Crucible behaviour |
|---|---|---|---|
| `prompt` | `prompt` | required for text-to-audio; empty means unconditional [S1][S2] | required, not blank [C2] |
| `duration` / `seconds_total` | `duration_s` | Length in seconds; up to 380 s on Medium [S4]. "Results tend to be better when you choose a duration that fits what you're describing" [S1] | default 60, max 380 [C1] |
| `steps` | `steps` | default 8. Going above 8 "doesn't necessarily increase quality" unless you use a `-base` checkpoint (about 50) [S2]. The report picked 8 ping-pong steps; 1 step degrades FAD and CLAP [S10 §4, §5.7] | default 8, range 1–50 [C1] |
| `cfg_scale` | not taken | Default 1.0. Stability's docs say cfg_scale and negative_prompt "have no effect on post-trained checkpoints" [S2]. Its own evaluation sampled the post-trained models at 8 PingPong steps, with no CFG scale stated (the base models used CFG 7) [S10 §5.1]; the training demo used `demo_cfg_scales: [1]` [S9] | refused, `audio_param_unsupported` [C1][C2] |
| `negative_prompt` | not taken | Base checkpoints only [S2]. At `cfg_scale == 1.0` the package never runs the guided branch, so a negative prompt is never read [S7 `dit.py`] | refused [C1][C2] |
| `seed` | `seed` | -1 = random [S2] | chosen and reported when not sent [C2] |
| sampler | not exposed | defaults to `pingpong` for post-trained (`rf_denoiser`) models [S7 `sampling.py`]. The card uses `sampler_type="pingpong"`, 8 steps, cfg 1.0 [S8] | the package default, pingpong [C5] |
| `init_audio`, `init_noise_level`, `inpaint_*` | not taken | audio-to-audio and inpainting / continuation [S1][S2] | Crucible's `audio` job does not take them [C2] |

**Duration and the prompt**

- **How the window is sized.** The package generates `duration + 6 s` of latent (`duration_padding_sec`, default 6.0) and trims the output to the requested duration [S7 `model.py`]. Separately, training padded clips with silence (about 4 s on average) [S10 §3.1]. No source links the two numbers.
- **Medium's window.** `sample_size` is 16,777,216 samples, which is 380.4 s at 44.1 kHz [S9]. The package's own `generate()` defaults to a 120 s window; see "Measured in Crucible" [S12][C3].
- **Where Medium is strongest.** The report measured Medium best at 120–190 s [S10 §5.4].
  - At 20 s, FAD and CLAP were worse. The report blames this on training: most short training clips are loops, not full songs.
  - At 380 s, prompt adherence (CLAP) fell. Long training examples are mostly ambient or classical, so a long duration "can bias the model toward generating ambient or classical music, omitting the provided text prompt" [S10 §5.4].
- **Lengths the official assistant suggests** [S5]:
  - Full instrumental tracks: energetic or dance 120–180 s, pop or rock 180–210 s, cinematic or ambient 240–300 s.
  - Loops: 6–20 s. Stems: 20–180 s. One-shots: 1–11 s.
- **BPM and key against duration.** BPM and key live only in the text. Nothing in the official sources ties the BPM to `duration_s` (for example, bar-aligned loop lengths). See open questions.

## Examples

Official, short and attributed:

- Prompting guide (stem): "TrackType: Instrument, a sombre solo acoustic guitar track with cavernous reverb and delicate finger picking." [S1]
- Prompting guide (stem, duo): "TrackType: Instrument, Format: Duo, a dynamic Latin drums and percussion track." [S1]
- README and model card, with `duration=250` and `180`: "House music that encapsulates the feeling of being at a festival in the sunny weather with all your friends 124 BPM" [S4][S8]
- Training demo prompts, at `seconds_total` 190 and 180, 8 steps, cfg 1 [S9]: "Meditative lo-fi ambient piano jazz, soft acoustic drum kit" and "A tropical house track with upbeat melodies, a driving bassline, and cheery vibes". These carry no prefix.
- Assistant examples [S5 "Music"]:
  - "Reggae track with offbeat guitar, warm basslines, snare, kick, congas, and horn stabs giving laid-back groovy energy. BPM: 85. Length: 150 seconds". Here the `Length` part moves to the duration slider.
  - Instrument loop: "Rhodes chord vamp loop with warm electric piano tone. BPM: 100. Length: 12 seconds" [S5 "Instrument"].
- Product guide: "Detroit-influenced techno with a stripped-back arrangement, metallic percussion, deep sub bass, and raw analog synths 130 BPM" [S11]
- The long prose examples in [S1] (tech-house, alt-pop ballad, 70s funk theme) run past the assistant's 45-word cap. Read them at the link rather than here.

Constructed (not from a source; it follows [S5]'s template and [S10]'s prefix), sent as `duration_s: 150`:
`TrackType: Music, VocalType: Instrumental, Melancholic neo-classical piece with soft felt piano lead, sustained cello and viola, distant field-recorded rain, and sparse timpani swells creating a reflective, cinematic mood. BPM: 66.`

## Community findings

- **Tempo and key adherence (Medium).** On 240 prompts naming a style, a tempo (70–150 BPM) and a key, Medium "hits the tempo on 61% of clips and the exact key on 64%" [S14].
  - Three-seed averages were within a point or two (58% tempo, 62% key).
  - The figures are automated estimates; the authors say they ran no listening test. They do not state steps, prefix use or the prompt format.
  - This is the only source found that measures key adherence. The official sources never mention key.
- **The 120 s cap.** Requests longer than 120 s came back cut off at 2:00. A user traced it to `generate()`'s fixed `sample_size` default, and a maintainer confirmed and fixed it in the CLI [S12]. The prompt in that report put a key in the text ("Bass house, 125 BPM, G minor, …").
- **ComfyUI sampler.** For the distilled checkpoint, the ComfyUI docs and workflows use 8+ steps at CFG 1, versus 50+ steps at CFG 7+ for base [S15]. The workflow defaults to the LCM sampler, simple scheduler, 10 steps, CFG 1 [S15].
  - One user found ComfyUI's LCM output noisier than Stability's demo, which uses pingpong; another said the two are the same sampler [S17].
  - Comfy's official Medium workflow has Music / Instrument / SFX / One-shot presets that expand a short description with a Qwen 3.5 2B model [S16]. This matches Stability's assistant [S5].
- **Structured fields and negatives for stems.** The adamdived workflow uses a pipe-separated field prompt (`Format: … | Genre: … | Instruments: … | Mood: … | BPM: …`) and tells users to "use the negative prompt to exclude everything else" for stems [S15].
  - The negative-prompt advice conflicts with [S2] and [S7]: at CFG 1 no negative branch runs.
  - Field values like `Format: Band` and `Drum Stem` are not confirmed as AudioSparx values by any official source. Only `Format: Duo` is [S1].
- **Negation phrases.** One user could not stop a guitar artefact ("behind-the-bridge" playing) with a prompt; the issue is unanswered [S13]. No source says whether "no X" in the text works on its own.
  - Stability's own instrument examples do contain negations, e.g. "…and no hi-hats for a grand orchestral feel" [S5]. There is no evidence either way on whether they are obeyed.
- **Prompt weighting.** An open PR adds `(phrase:weight)` weighting to the MLX path only [S19]. It is not in the package Crucible runs.
- **Third-party advice.** stableaudio3.com suggests a formula of genre + instruments + mood + BPM + optional key + production + duration [S18]. The duration-in-text part is contradicted by how the official UI handles length [S6].

## Measured in Crucible

- **Long requests were truncated (fixed).** Medium requests over 120 s came back 120 s long. The worker now passes the model's own window (`sample_size` from `model_config.json`, 380.4 s), and it refuses audio shorter than the request instead of returning it [C3 "Fix: `stable-audio-3-medium` requests over 120 s came back 120 s long"][C5][C1 `memory_note`].
- **Memory at the longest request.** Measured 2026-10-09 on the RTX 3090 Ti, 380 s at 8 steps: torch reserved 5.87 GB while encoding and denoising, and 6.21 GB while decoding. The card was 6.14 GiB over the desktop. The declared estimate is 6.8 GB [C1 `memory_note`].
- **Loading in float16.** Loading in float32 on the card overran its cap. The worker now halves the weights to float16 in host memory first [C3][C5].
- **No quality measurements yet.** Nothing about prompt adherence, BPM accuracy or sound quality has been measured through Crucible. The Mac (MPS, float32, SDPA fallback) arm's sound and timing are still listed as to be measured [C4 "What the first runs through Crucible must measure"].

## Conflicts and open questions

**Where official guidance and Crucible differ**

- **Prefix punctuation.** Crucible's example writes `TrackType: Music, VocalType: Instrumental. Warm lo-fi …`, with a full stop after the prefix [C2][C6]. Stability's assistant and the report use a trailing comma: `TrackType: Music, VocalType: Instrumental, …` [S5][S10 §5.1]. Training joined fields with commas [S10 §3.5]. Whether the full stop matters is unmeasured.
- **Key in the prompt.** Crucible's manifest says to describe key and tempo in the prompt "as its prompting guide does" [C1 `not_taken.key`]. The guide recommends BPM only and never mentions key [S1][S11]. Key is not among the metadata fields the report lists [S10 §3.5]. The only evidence that key text works is community [S14].
- **Empty prompts.** Crucible refuses a blank `prompt` [C2]. Stability suggests trying an unconditional (empty) generation for inspiration [S1], and the model was trained for it [S9][S10].
- **Wording about cfg.** Crucible's manifest quotes Stability: cfg "has no effect" on post-trained Medium [C1]. Stability's docs say so [S2]. The package code would in fact run a guided branch if `cfg_scale` were not 1.0 [S7 `dit.py`], so "is not meant to be used" is more exact than "has no effect". It does not matter in practice while Crucible refuses `cfg`.

**Upstream inconsistency**

- `model-overview.md` gives Medium's max duration as "~4.75min" [S3]. The README says 380 s [S4], and `model_config.json` gives 380.4 s [S9]. The config is what runs.

**Open questions (inferences, not guidance)**

- **Loop lengths.** Does a `duration_s` equal to a whole number of bars at the prompted BPM give cleaner loops? No source says.
- **Prefix plus metadata vs prose.** Does `TrackType` plus labelled fields (`Genre:`, `Instruments:`) beat plain prose on Medium? Training saw both forms half the time each [S10 §3.5]. No comparison is published.
- **Field order and case.** Do field order or upper/lower case matter? Training shuffled fields and lowercased half the prompts [S10 §3.5], which suggests not, but this is untested.
- **Negations.** Are phrases like "no hi-hats" obeyed with no negative branch? This is untested ([S13] is unanswered).
- **Mac arm quality.** Does the Mac arm (MPS, float32, SDPA sliding-window fallback) sound the same as CUDA? Crucible lists this as unmeasured [C4].
