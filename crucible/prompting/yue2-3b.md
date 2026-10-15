# YuE2 3B: how to prompt it

Crucible id `yue2-3b` · upstream `m-a-p/YuE2-3B` @ `c044757a011169583f363168348ae380946efff8` (VAE `m-a-p/YuE2-Vae` @ `152733a1`), code `yue2-infer` from `multimodal-art-projection/YuE` @ `18a07bb6` · Crucible job type `audio` (class `song`).

Crucible state read: repo @ `5a579e7` (2026-10-13). Everything upstream was read on 2026-10-15.

## Sources

Official means the YuE2 authors: the model card, the repo, its code and skill, and comments by the repo's collaborator account `a43992899`. Community means everyone else.

| # | What | Label | Revision / date read |
|---|---|---|---|
| S1 | HF model card, https://huggingface.co/m-a-p/YuE2-3B | OFFICIAL | `c044757a`, read 2026-10-15 |
| S2 | `yue2_generation_config.json` in the HF repo (sampling defaults) | OFFICIAL | `c044757a` |
| S3 | `examples/tonight-awake.json` in the HF repo (the Mandarin demo request) | OFFICIAL | `c044757a` |
| S4 | Repo README, https://github.com/multimodal-art-projection/YuE | OFFICIAL | `18a07bb6` (2026-09-29) |
| S5 | `YuE/docs/generation.md` (generation guide) | OFFICIAL | `18a07bb6` |
| S6 | `examples/song.json`, `examples/README.md` ("City Lights") | OFFICIAL | `18a07bb6` |
| S7 | `src/yue2/protocol.py` (prompt text, CFG negative, sampling defaults) | OFFICIAL | `18a07bb6` |
| S8 | `skills/yue2-music/SKILL.md` | OFFICIAL | `18a07bb6` |
| S9 | `skills/yue2-music/references/generation-and-covers.md` | OFFICIAL | `18a07bb6` |
| S10 | `skills/yue2-music/instrumental/SKILL.md`, `scripts/instrumental.py`, `assets/example-style.txt` (vendored unchanged in Crucible as `crucible/jobs/audio/yue2music/`) | OFFICIAL | `18a07bb6` |
| S11 | `YuE/docs/covers.md`, `YuE/docs/benchmarks.md` | OFFICIAL | `18a07bb6` |
| S12 | YuE v1 README, "Prompt Engineering Guide" section (branch `YuE-v1`) | OFFICIAL (v1) | `6d4f0b1f` (2026-09-10) |
| S13 | YuE v1 `top_200_tags.json`, `prompt_egs/genre.txt`, `prompt_egs/lyrics.txt` | OFFICIAL (v1) | `6d4f0b1f` |
| S14 | Issue #18 "How to generate just music, no lyrics?" (v1 era; maintainer comments 2025-02-16 and 2026-09-25) | OFFICIAL (maintainer comments) | read 2026-10-15 |
| S15 | Issue #172 "instrumental tracks without lyrics or vocals" (users, plus maintainer comment 2026-09-25) | COMMUNITY + OFFICIAL comment | open; read 2026-10-15 |
| S16 | Issue #208 "Instrumental music ... still contains human voices using instrumental skills" | COMMUNITY | open, no reply; read 2026-10-15 |
| S17 | Issue #216 "per-section vocal control in lyric tags" | COMMUNITY | open; 2026-10-05 |
| S18 | Issues #211 (style adherence vs Suno), #220 (cannot choose timbre), #178 (robotic voice) | COMMUNITY | 2026-09-14 to 10-10 |
| S19 | Issues #184 (Cantonese), #206 (Mandarin polyphonic characters) | COMMUNITY | 2026-09-15 to 09-30 |
| S20 | Issue #214 (three-voice ABC) | COMMUNITY | 2026-10-04 |
| S21 | Issue #186 (lyric-to-score alignment), maintainer reply | OFFICIAL (maintainer comment) | 2026-09-17 |
| S22 | HF discussion #1 (instrumentals) | COMMUNITY | 2026-09-10 to 09-19 |
| S23 | HF discussion #7 (languages) | COMMUNITY | 2026-09-12 to 09-17 |
| S24 | HF discussion #8 (BPM and key conditioning through ABC) | COMMUNITY | 2026-09-12 |
| S25 | HF discussion #10 ("Cannot make rock songs") | COMMUNITY | 2026-09-16 to 09-19 |
| S26 | HF discussion #29 (Ruach Studio measurements: tags, CFG, instruments, long texts) | COMMUNITY | 2026-10-06 |
| S27 | HF discussions #21 (same female voice), #30 (Arabic) | COMMUNITY | 2026-09-22, 10-07 |
| S28 | Comfy-Org `templates/audio_yue2_text2music.json` (linked from S4 as the "official" ComfyUI workflow, but authored by Comfy-Org) | COMMUNITY | `e7cd011d` (2026-09-28) |
| S29 | dav.one, "Generate Music with YuE2" | COMMUNITY | dated 2026-09-12 |
| S30 | songcreator.pro, "YuE2 Prompting Guide" | COMMUNITY | dated 2026-09-17 |
| S31 | chunks.md, "How to Write YuE2 Style Prompts" | COMMUNITY | undated; read 2026-10-15 |
| S32 | Issue #193 (symbolic best-of-N), PR #218 (`instrumental_clip.py`) | COMMUNITY | 2026-09-17, 10-09 |

Crucible files (all @ `5a579e7`): C1 `docs/AUDIO.md`; C2 `docs/CHANGELOG.md`; C3 `crucible/audio/yue2-3b.toml`; C4 `crucible/audio/planning/yue2.toml`; C5 `crucible/audio/tags/song.toml`; C6 `crucible/jobs/audio/params.py` and `yue2_worker.py`; C7 `docs/SCORE.md`.

A Reddit thread on an instrumental LoRA (r/StableDiffusion `1wh10ee`, linked from S22) could not be fetched and is not used.

## What the model was trained to see

**One text prompt, two fields.** The request becomes a fixed text: a mode instruction, then `[Tags]`, the style, `[Lyrics]`, the lyrics [S7]. The mode instruction is chosen by `cot`; you never write it yourself [S9]. Crucible's `tags` is upstream's `style` (upstream also accepts `tags` as an alias) [S9, C1].

**Three modes (`cot`).** `full` plans melody and chords as ABC, then the audio. `melody` plans melody only. `off` skips the score [S4, S7]. `full` is the default for new songs, `melody` is recommended for covers [S1, S4]. Crucible always plans with `full`; it picks `full` or `melody` only for a cover, from whether the score has chords (C1 "Covers"; C6). `off` is not reachable through Crucible.

**Two token stages.** The score is written first (up to 4,096 tokens, sampled at temperature 0.7, top-p 0.9, top-k 30, repetition penalty 1.005). Then codec tokens (up to 9,000, about 25 a second, so about 6 minutes; temperature 1.0, top-p 0.95, top-k 100, repetition penalty 1.2 over a 50-token window; at least 200 tokens). Then a fixed 32-step flow-matching synthesis [S2, S7, C3]. The whole context is 24,576 tokens [S7].

**The score dialect.** Two voices, `Vocal` and `Ins`, with `% section` comments. Native section labels are exactly `intro, verse, pre-chorus, chorus, bridge, interlude, outro, instrumental` [S10 `compile_score.py` SECTIONS; C7]. The model is not built for more voices; a user reports a three-voice ABC switching between parts [S20].

**No audio reference.** YuE2 "exposes no audio-reference, phoneme-alignment, or local-inpainting argument" [S8]. The request has no field for reference audio, phonemes, BPM, a negative prompt or a reference singer [S9]. Voice timbre is not selectable beyond what the style text does [S18 #220]. YuE v1's ICL mode (a 30 s reference clip with separate `-icl` checkpoints [S12]) does not exist in YuE2.

**Languages.** The model card tags `zh` and `en` [S1]. Official examples are English and Mandarin [S3, S6]. Everything else is community report (below).

## Writing the prompt

### Style (`tags`)

- Do write comma-separated phrases. Cover genre, instruments, vocal character, language and tempo [S5, S8]. The official example: "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM" [S6].
- Do describe sound, not story. The official examples name genre, instruments, voice, mood, phrasing and tempo, never the song's subject [S3, S6, S9].
- Do state tempo and meter in the style consistently with the score. Upstream: "Put tempo and meter in the ABC and describe them consistently in the style" [S9]. The official instrumental example also names a key ("G major, 96 BPM") [S10 example-style.txt]. In Crucible, the ABC part is `key` / `meter` / `bpm` (below).
- Do name what to leave out as a plain phrase when it matters. The official jazz edit style includes "no guitar" [S1, S9]; the official instrumental style ends with "no vocals, no singing, no choir, no spoken words" [S10].
- Do not paste a mode instruction or `[Tags]` / `[Lyrics]` headers into the style. The mode selects its own native instruction [S9]; community guides say the same about the outer headers [S31].
- Do not expect the style to fix the singer. Users report which voice sings follows the register of the planned melody, and casting text in the style or section tags does not pin it [S17, S27 #21].
- Language tag: the English example starts with "English" [S6]; the Mandarin demo's style has no language word at all ("City Pop, upbeat, danceable, groovy bass, ...") [S3]. So a language word is used, not required.

**Carried over from YuE v1 (not restated for YuE2):** a stable tag prompt has five parts, genre, instrument, mood, gender and timbre [S12]; tags from v1's top-200 list are "more stable" [S12, S13]; "Mandarin" and "Cantonese" tags separate the two [S12]. v1 separated tags with spaces ("inspiring female uplifting pop airy vocal ...") [S13]. YuE2's own examples use commas and longer phrases [S3, S6], so use commas. The five-part checklist is still a reasonable coverage check, and community YuE2 guides repeat it [S29, S30]. The v1 tag list's categories (genre, instrument, mood, gender, timbre) are a usable vocabulary; whether YuE2 favours those exact words is not documented.

### Lyrics (`lyrics`)

- Do label every section on its own line in square brackets: `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`, and so on [S5, S6]. Use the eight native labels; the score uses exactly those [S10]. Crucible's instrumental and planning-lyrics checks accept only `[Intro] [Verse] [Pre-Chorus] [Chorus] [Bridge] [Interlude] [Outro] [Instrumental]` (C1).
- Do separate sections with one blank line [S6, S3]. Official examples use Title Case tags (`[Verse]`) [S3, S6]; v1 used lower case (`[verse]`) [S13].
- Do write every repeat out in full. The Mandarin demo writes each chorus twice, as two blocks under one `[Chorus]` tag separated by a blank line [S3].
- Empty instrumental sections are allowed in the official demo: `[Intro]` and `[Interlude]` with no lines [S3].
- Do keep only singable words. "Keep implementation notes out of lyrics" [S8]. There is no phoneme field; pronunciation and note alignment are not inputs [S9].
- Do keep lines rhythmically even. The City Lights example uses seven syllables a line, set one line per bar with seven note onsets [S6]. For translated lyrics, adapt syllables, stress, vowels and breath points [S8].
- Do not expect lyrics to lock to notes. Lyric-to-score alignment is "not done yet" per the maintainer [S21].
- Length: the lyrics set the length. Number and length of sections decide how long the song is (C1, S9). A long lyric prefix eats context; one user measured 11.5k characters taking 5.7k tokens of prefix, after which the model stopped early and skipped [S26].

**v1 rules that do not carry over cleanly.** v1 said each section was about 30 s and to not "put too many words in a single segment" [S12]; that was about v1's per-segment decoding with `--max_new_tokens 3000`. YuE2 decodes one pass over the whole song [S7], so the 30 s per section rule has no YuE2 basis. v1 also called `[intro]` "less stable" and advised starting with `[verse]` or `[chorus]` [S12]. YuE2's own demo starts with an empty `[Intro]` [S3]. Some community YuE2 guides still repeat the v1 intro advice [S29].

### Instrumentals

- Official YuE2 workflow: let YuE2 plan a score, move every Vocal note to the Ins voice, then render that fixed score with lyrics that are only the section tags [S4, S10]. Crucible's `instrumental: true` is this workflow, vendored (C1).
- Official instrumental style form: `prepare()` in `instrumental.py` rewrites the style to start with "Instrumental, " and appends "no vocals", "no singing", "no choir", "no spoken words", each only if missing, then a final full stop [S10]. Example: "Instrumental, expressive chamber string ensemble, ..., G major, 96 BPM, ..., no vocals, no singing, no choir, no spoken words." [S10 example-style.txt]. If the style is empty it uses "Expressive instrumental music" [S10]. Crucible does not apply this rewrite; send `tags` already in this form.
- Planning: the skill plans from `--planning-lyrics-file` if given ("never sung in the final request"), else `[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n` [S10]. The plan mode is `full` by default [S10].
- Listen to the result. The maintainer: the conversion "does not guarantee that every generated audio output is free of vocal artifacts" [S15]; "symbolic checks alone cannot rule those out" [S14]. The skill allows at most one retry after a failed plan [S10].
- Older recipes, superseded by the workflow above: v1 used empty sections or several newlines per section [S14]; early YuE2 users used `full` mode, no language or singer in style, empty section tags, and reported it working about half the time ("seed hunting") [S15, S22].

### Covers (`score`)

- Official recipe: transcribe with SheetSage2, review the melody, strip chords, render with `cot="melody"` and new lyrics and style [S1, S4, S11]. Keeping chords (`full`) preserves song identity better; stripping them gives the arrangement more freedom and scored higher on quality [S11].
- Align section tags and lyric order with the score; when translating, match phrasing and syllable counts to the melody [S11].
- Crucible: `score` plus `chords` (`keep` → `full`, `strip` → `melody`); a sung cover strips chords by default (C1).

## Settings that change how the prompt lands

| Upstream | Crucible param | Official guidance | Crucible behaviour |
|---|---|---|---|
| `style` / `tags` | `tags` (required) | Above [S5, S9] | Passed verbatim (C6) |
| `lyrics` | `lyrics` | Above | Required for a sung song; with `instrumental`, only section tags are accepted (C1) |
| `cot` | not exposed | `full` default; `melody` for covers [S1] | Always `full` for planned songs; `full`/`melody` by chords for covers (C1, C6) |
| `cfg_scale` | `cfg` (0-20, default 1.0) | Default 1.0 in full/melody, 1.01 in off. "For an explicit experiment, pass `cfg_scale=1.2`"; "not a guaranteed quality improvement" [S9]. "ABC generation itself has no CFG" [S9]; CFG acts only on codec tokens [S1]. Above 1 the model runs twice per token | Same range and default (C3); applied to composing only (C1) |
| stock negative | `negative_tags`, `negative_lyrics` (need `cfg` > 1) | Upstream's negative keeps the instruction and exact ABC and removes style and lyrics [S7, S9]. Upstream has no field to write a negative and no `negative_prompt` [S9] | Crucible extension: replaces the negative branch with the client's tags and lyrics (C1). No official guidance on what to write there |
| ABC header | `key`, `meter`, `bpm` | Put tempo and meter in the ABC [S9]; no `bpm` field upstream [S9] | Crucible forces the `K:`, `M:`, `Q:1/4=` lines into the score as it is written (C1). Never beside `score` |
| sampling | not exposed | "defaults are ready to use, while changing sampling or guidance may change quality" [S5]; keep default sampling [S10] | Defaults used (C1) |
| ODE steps | not exposed | Fixed preset of 32 midpoint steps; do not lower it to hide a problem [S9, S10] | Fixed (C3 `not_taken.steps`) |
| duration | `min_duration_s`, `max_duration_s` (a nudge only) | No length input; "Strict duration ... not a model guarantee" [S10] | Sizes only a pool-planned instrumental; recorded for every song (C1) |
| seed | `seed` | Seeds help comparison but are not identical across devices or versions [S5, S6] | Same seed, params, model and machine give the same sound (C1) |
| decoder | not exposed | `YuE2-Vae` for listening; `-legacy` only to reproduce benchmarks [S1, S5] | `YuE2-Vae` (C3) |
| candidates | none | Benchmarks pick best of 2 or 8 calls; one call is one candidate [S5, S11] | One pass, one song (C1) |

## Examples

From official sources (short, attributed):

- Style, English pop: "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM" [S6].
- Style, edited jazz version: "Jazz, expressive lead vocal, piano, tenor saxophone, upright bass, brushed drums, no guitar, spacious modern harmony" [S1].
- Style, cover: "Jazz-funk, warm lead vocal, Rhodes piano, electric bass, tight drums" [S1].
- Style, Mandarin demo: "City Pop, upbeat, danceable, groovy bass, electric guitar, synth, energetic, joyful, neon city night" [S3].
- Lyrics shape (City Lights; full text in S6): `[Verse]` then four seven-syllable lines, a blank line, `[Chorus]` then four seven-syllable lines.
- Lyrics shape (Mandarin demo; full text in S3): `[Intro]` (empty), `[Verse]`, `[Pre-Chorus]`, `[Chorus]` with two blocks, `[Interlude]` (empty), `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Chorus]`.
- Instrumental style: see S10 `assets/example-style.txt` (quoted in part above).

Constructed (not from any source), the Crucible request for a song whose key, meter and tempo are stated in both places:

```json
{"type": "audio", "model": "yue2-3b",
 "params": {"tags": "English, slow waltz, warm piano, expressive female voice, E minor, 3/4, 76 BPM",
            "lyrics": "[Verse]\nLine one\nLine two\n\n[Chorus]\nLine one\nLine two\n",
            "key": "Em", "meter": "3/4", "bpm": 76, "seed": 3}}
```

## Community findings

- **Short tags beat long prose.** Most important first: language and genre with era, then instruments and lead voice, then mood and production, BPM last [S26]. Others say earlier words weigh more and suggest 25 to 50 words [S30, S31]. The Comfy-Org template uses a prose sentence instead [S28]; no source compares the two forms officially.
- **CFG.** One studio measured on RTX 3090s: 1.4 to 1.6 steady, 2.0 moves more, 2.4 the most that stays stable, from about 2.6 "the vocals break and music tokens get cut" [S26]. Another user fixed weak rock output by adjusting CFG [S25]. Raising CFG to 2.5 did not pin a voice to a section [S17].
- **Genre drift.** Rock prompts drifting to slow country-like pop; one user got better results with ABC planning off [S25] (not available in Crucible). LoRAs were trained to fix it [S25]. Style adherence is weaker than Suno [S18 #211].
- **Voice control.** Same female voice regardless of prompt [S27 #21]; low lines come out male, high lines female; section suffix tags like `[Chorus - Female]` did not work [S17]. Robotic vocal timbre reported [S18 #178].
- **Instruments.** Not known: shofar, oboe, gong. "orchestral" before an instrument builds a whole orchestra and loses the instrument [S26].
- **BPM and key.** BPM in a score acts as a target, with half- or double-time readings; key and chord anchors are followed loosely, more closely with fewer anchors [S24].
- **Languages beyond en/zh**, all user reports: Japanese, Korean, Russian, Spanish (said to be on the demo page), Polish, French, Portuguese, Romanian, Greek [S23]; Arabic works with English-like scores but broke into garbage words with one ABC melody [S27 #30]. Cantonese is poor and drifts to Mandarin, helped by Cantonese-only words [S19 #184]. For Mandarin polyphonic characters, a user substituted a homophone character (环 for 还) [S19 #206].
- **Instrumentals still leak voice.** With the official skill: 3 of 10 had voices with words [S16]. Earlier recipes gave "vocal chops" [S15].
- **Repetition.** Write every repeat in full; YuE2 will not expand "(repeat chorus)" [S30]. Keep repeated lines at the same syllable count [S29, S30]. Overlong lines spill into the next beat [S29].
- **Selection.** Generate several and keep the best [S30]; a symbolic best-of-N on the ABC stage before audio [S32 #193]. A semantic token budget of `seconds * 25 + 250` for short clips [S32 #218].

## Measured in Crucible

Only what C1, C2 and C3 record.

- **Key, meter, tempo from tags are unreliable** (Victoria's laptop, 2026-10-13): minor-key tags came out major about half the time (planner favours Eb, Bb, D); 3/4 tags gave 6/8 in 5 of 22; 76-84 BPM tags gave `Q:` 124-145. Hence `key`/`meter`/`bpm` (C1, C2).
- **Empty-section instrumentals ran away**: planned from `[Intro] [Verse] [Chorus] [Outro]` with no lines, 4 of 13 instrumentals hit the 4,096-token score cap; the rest scored 1,023 to 3,911 tokens. Sung songs score 1,800 to 2,600 (C1, C4). Planning lyrics of about 95 to 140 syllables fix the range.
- **Score length predicts audio length**: audio ran 0.943 to 1.069 of the score's nominal length (median 0.987) on nine instrumentals; 0.959 to 0.982 on four more (C1).
- **Planning lines to score length is loose**: 4.7 to 14.4 s per planning line (median 8.9; prior 8.6) over 21 scores; tempo and bars per line dominate (C1).
- **Written negative, one seed each at cfg 2.0** (3090 Ti, outside Crucible): Tool hummed with either negative; BB went from sung words to humming (C1, C2).
- **Instrumental LoRA made it worse**: `yue2-3b-instrumental` sang more (Iron Lanterns 37.5 s of voice; real words where the base only hummed) and was removed 2026-10-12 (C2).
- **Runaway composing**: a bad seed ran a song to the 9,000-token cap, 6:00 long, on an RTX 3070, while the same seed ended normally on a 3090 Ti (C1).
- **Some instrumentals fail at transfer**: 2 of about 11 on Victoria's run came back truncated or empty; another failed the skill's checks (bar longer than its meter) (C2).
- Memory and speed figures are in C3 and are not prompting facts.

## Conflicts and open questions

1. **The official instrumental style form is the caller's to write.** Upstream `prepare()` forces "Instrumental, ..., no vocals, no singing, no choir, no spoken words." [S10]. Crucible passes `tags` verbatim, by design: the prompt belongs to the app (Owen, 2026-10-12), so an app asking for `instrumental: true` should send its style already in that form. Whether leaving it out adds to voice leakage is unmeasured.
2. **CFG does not touch the score.** Upstream: ABC sampling uses no CFG [S1, S9]. So `cfg`, `negative_tags` and `negative_lyrics` cannot change key, tempo, form or melody; only `key`/`meter`/`bpm` and the lyrics do. C1 says "composes each codec token twice", which is consistent.
3. **Written negatives are not an upstream feature.** No official guidance on negative wording exists [S9]. Crucible's one measurement is one seed per song (C1).
4. **`[Intro]` first.** v1 advised against it [S12]; YuE2's demo does it [S3]; Crucible's planning pool uses it (C4). Treat the v1 rule as obsolete for YuE2 unless measured.
5. **Languages.** C1 and the lyrics row say "English or Chinese", matching the card [S1]. Users report many more [S23]. Untested in Crucible.
6. **No `cot` choice.** Upstream offers `melody` and `off` for new songs [S4]; Crucible does not. A community fix for rock drift used `off` [S25]. Whether exposing it is wanted is for the app owners.
7. **Five-part tag checklist and the v1 tag list.** Useful but v1 guidance [S12, S13]; YuE2 does not publish a vocabulary (C5 says so too).
8. **Open:** does tag order matter in YuE2? Community says yes [S26, S30]; no official statement. Does the community CFG range 1.4 to 2.4 [S26] hold for Crucible's default sampling and for instrumentals? Unmeasured here.
