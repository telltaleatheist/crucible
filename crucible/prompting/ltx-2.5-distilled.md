# LTX-2.5 Distilled: how to prompt it

Crucible id `ltx-2.5-distilled`. On the PC (cuda-linux) it is `Lightricks/LTX-2.5-Diffusers` @ `426936f8b22dc28e4def61e515478b0b7e4a53cc`, with the transformer taken from `Abiray/LTX-2.5-Distilled-GGUF` @ `7b0c2025441f1bf12c18eac375ad21f5e3d3c9e0` (Q6_K). On the Mac (mlx-darwin) it is `dgrauet/ltx-2.5-mlx-q8` @ `746ca9aacb697d2c739f544d68b584214dedcc75`. Job type `video`.

## Sources

All web sources were read on 2026-10-15.

- **[S1]** `gemma4_i2v_system_prompt.txt`: Lightricks' image-to-video prompt-enhancer system prompt. Path: `packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/` in https://github.com/Lightricks/LTX-2. OFFICIAL. Read at commit `be6d6d0a69d11a0a399c462ed92606692e5f8697` (2026-10-13). The file last changed in `2362161611a6` (2026-08-11).
- **[S2]** `gemma4_t2v_system_prompt.txt`: the text-to-video sibling of [S1], in the same folder. OFFICIAL. Same commit.
- **[S3]** The LTX-2 repo README, section "Prompting for LTX-2" (has the "Keep within 200 words" line), plus the optimization tips. https://github.com/Lightricks/LTX-2/blob/be6d6d0a69d11a0a399c462ed92606692e5f8697/README.md. OFFICIAL. Commit `be6d6d0a`.
- **[S4]** `packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py` in the LTX-2 repo. It holds `DEFAULT_NEGATIVE_PROMPT`, `DISTILLED_SIGMA_VALUES`, `STAGE_2_DISTILLED_SIGMA_VALUES` and the image CRF constants. OFFICIAL. Commit `be6d6d0a`.
- **[S5]** In the LTX-2 repo: `ltx_core/text_encoders/gemma/gemma_assets.py` (`TOKENIZER_MAX_LENGTH = 1024`), `ltx_pipelines/utils/args.py` and `utils/helpers.py`. These hold the frame and size rules: 8k+1 frames, width and height divisible by 32 for one stage and by 64 for two stages. OFFICIAL. Commit `be6d6d0a`.
- **[S6]** Model card, https://huggingface.co/Lightricks/LTX-2.5-Diffusers. OFFICIAL. Read at the repo head `da0b690960026eab794b8ceaf841233cf7064478` (lastModified 2026-10-11) through the rendered page. The repo is gated, so the raw README at Crucible's pinned revision `426936f8` could not be read.
- **[S7]** Model card, https://huggingface.co/Lightricks/LTX-2.5. OFFICIAL. Head `49b093a39c19169a8a81721dcc541e07e945232f` (2026-10-11).
- **[S8]** LTX docs, "Prompting Guide". https://docs.ltx.io/api-documentation/implementation-guides/prompting-guide (markdown via `.md`). OFFICIAL. Read 2026-10-15; the page shows no date. The old URL `docs.ltx.video/api-documentation/prompting-guide` now sits behind a Cloudflare challenge (403). docs.ltx.io serves the same guide.
- **[S9]** LTX blog, "LTX-2.5 Prompt Guide", Rachel Luxemburg (Senior Developer Advocate), 2026-08-10. https://ltx.io/blog/ltx-2-5-prompt-guide. OFFICIAL. Nearly the same text as [S8].
- **[S10]** LTX blog, "Prompting guide for LTX-2", LTX Team, 2025-12-11. https://ltx.io/blog/prompting-guide-for-ltx-2. OFFICIAL, but written for LTX-2.0. [S3] links to it.
- **[S11]** LTX docs, "Image-to-Video Workflow for Beginners" (the LTX-2.5 ComfyUI template). https://docs.ltx.io/open-source-model/usage-guides/image-to-video. OFFICIAL. Read 2026-10-15.
- **[S12]** LTX docs, "Text-to-Video Workflow for Beginners". https://docs.ltx.io/open-source-model/usage-guides/text-to-video. OFFICIAL. Read 2026-10-15.
- **[S13]** LTX docs, LTX-2.5 model page, "Automatic duration" section. https://docs.ltx.io/models/ltx-2-5. OFFICIAL, but about the hosted API. Read 2026-10-15.
- **[S14]** GitHub issue Lightricks/LTX-2 #117, "Static video". It is closed by Lightricks member `michaellightricks` (2026-08-18) with an explanation of why image-to-video stays static. https://github.com/Lightricks/LTX-2/issues/117. The closing comment is a maintainer statement (OFFICIAL). The rest of the thread is COMMUNITY.
- **[S15]** GitHub issue Lightricks/LTX-2 #11, "T2V works well, I2V fails a lot with a frozen or mostly frozen image", 2026-01-06. https://github.com/Lightricks/LTX-2/issues/11. COMMUNITY, plus one maintainer reply (2026-01-07) on image CRF. This is the LTX-2.0 era.
- **[S16]** Comment by `Vesemir` on LTX-2 issue #24, 2026-01-08. https://github.com/Lightricks/LTX-2/issues/24#issuecomment-3724409985. COMMUNITY, LTX-2.0 era, guided pipeline.
- **[S17]** diffusers `pipelines/ltx2/pipeline_ltx2.py` at `5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2`, the commit Crucible's `crucible/envs/video/ltx-cuda-linux.txt` pins. It has `max_sequence_length=1024` with `truncation=True`, and width and height must be divisible by 32. OFFICIAL (Hugging Face's reference implementation).
- **[S18]** The MLX port's card, https://huggingface.co/dgrauet/ltx-2.5-mlx-q8 @ `746ca9aa` (2026-08-28). COMMUNITY port. It carries no prompting guidance, only files and licence.
- **[S19]** "LTX-2.5 Prompt Guide: Prompts, Tips, and Workflows", Aditya Bawankule, dreampixelforge.com, 2026-08-11 (updated 2026-08-29). https://www.dreampixelforge.com/blog/ltx-prompts. COMMUNITY.
- **[S20]** "LTX-2.5 Prompt Guide: Multi-Shot Scenes, Native Audio…", deAPI.ai, 2026-09-04. https://deapi.ai/blog/ltx-2-5-prompting-guide-multi-shot-scenes-native-audio-and-prompts-that-actually-cut. COMMUNITY (a hosted-API vendor).
- **[C1]** Crucible `docs/VIDEO.md`, read 2026-10-15.
- **[C2]** Crucible `docs/internals/video.md`, same repo.
- **[C3]** Crucible manifest `crucible/video/ltx-2.5-distilled.toml`, same repo.
- **[C4]** Crucible `docs/CHANGELOG.md`, same repo.
- **[A1]** Storyboard `storyboard/docs/PROMPTING.md`. These are an app's notes. They were used only as leads, and every claim taken from them was re-checked against the sources above.

## What the model was trained to see

- **Long, single-paragraph audio-visual captions.** The model card says the model was "trained on long, single-paragraph audio-visual captions and degrades on short prompts" [S6].
- **The caption style, as Lightricks' own rewriter writes it.** The enhancer system prompts tell a Gemma model to write "in the EXACT style of the training captions used for this video model" [S1][S2]. That makes them the most direct official description of the input distribution. The style they ask for:
  - one continuous paragraph, with no bullets, headers, or labels such as "Audio:" or "Visual:";
  - roughly 150 to 220 words;
  - it opens straight on the action or a visual detail, never "The scene opens…", "We see…" or "There is…";
  - it describes only what can be seen and heard, with emotion shown through the face and body;
  - full environment and character detail, including where things sit in space;
  - one shot type, the camera motion (stated even when the camera is static) and the viewpoint, all woven into the prose;
  - a complete soundscape: dialogue quoted exactly, tone of voice, music, ambient sound;
  - strict chronological order, with transitions like "Initially…", "A moment later…", "Simultaneously…";
  - English output [S1][S2].
- **The text encoder.** It is Gemma 4 12B [C2][S11]. Prompts are tokenized to at most **1024 tokens**, padded on the left and **truncated silently** past that [S5][S17].
  - Crucible's PC worker calls diffusers' `encode_prompt` with the default length [C2][S17].
  - The Mac worker left-pads to 1024 tokens [C2].
  - A prompt of about 200 words is far below that limit, so the cap only matters for very long screenplay-style prompts.
- **Grid.** Frame counts are 8k+1 (the VAE's causal grid) [S5][S7]. Width and height are multiples of 32 for one-stage generation and 64 for two-stage [S5].
- **Frame rate.** The reference examples run at 24 fps [S6][S12].
- **Sound.** Audio and video are generated jointly in one latent, which is what keeps them in sync [S12]. Every clip gets sound from the same prompt.
- **Training-caption tags.** The official default negative prompt opens with `has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short` [S4]. These look like tags from the training data. That is an inference; see Open questions.

## Writing the prompt

**Do**

- Write one paragraph of plain present-tense sentences. Lightricks says to think "like a cinematographer describing a shot list" [S3]; see [S8] for present tense.
- Open with the main action in one sentence, then add, roughly in this order:
  - movement and gestures;
  - how characters and objects look;
  - background and environment;
  - camera angle and movement;
  - lighting and colour;
  - any changes or sudden events [S3].
- Cover the six elements: shot, scene, action, character, camera movement, audio [S8][S9].
- Give each sentence a verb that does something ("walks, turns, exhales, reaches"). Appearance alone "gives the model little to animate" [S8].
- Name exactly one shot scale (wide, medium, close-up …), the viewpoint (front-facing, side view, over-the-shoulder, low-angle …) and the camera motion. If the camera does not move, say so: "the camera remains static" [S1][S2].
- Describe camera moves relative to the subject. Say how the subject looks after the move, so the model knows where the move ends [S8][S10].
- Match the detail to the shot scale: close-ups need more than wide shots [S8].
- Show emotion through the body. For example, write "his eyebrows angle downward and his lips are pressed together", not "he looks sad" [S1][S2][S8].
- Describe the sound as concretely as the picture:
  - ambience, music (type, mood, volume changes) and effects tied to the action [S1][S8];
  - dialogue in quotation marks, with who says it, the tone, and the language or accent if it matters [S8][S10].
- Keep one coherent light logic per shot [S8][S10].
- Keep the scene focused. A few characters and actions read better than a crowded frame. Start simple and add detail as you iterate [S8][S10].
- If the clip needs pauses, write them in ("she pauses", "a beat of silence"). The duration predictor times the action as written and adds no pause you did not prompt [S8]. Crucible does not use the duration predictor; see Settings and Open questions.
- Several shots in one clip are possible on LTX-2.5:
  - name each cut in prose ("A hard cut transitions to…");
  - re-establish the shot scale and framing after each cut;
  - re-identify recurring people by the same visual markers;
  - say at every cut whether the music, dialogue or ambience continues;
  - use 2 to 4 shots at most [S8][S9].
- Name a stylized look early in the prompt (painterly, noir, claymation, analog film) [S10]. Lightricks says these "work especially well when named early" [S10, LTX-2 era].

**Don't**

- Don't exceed about 200 words. The README says "Keep within 200 words" [S3], and the enhancer aims for 150 to 220 [S1][S2]. The docs ([S8], newer) relax this. They allow a screenplay-style scene to "run longer, provided every sentence adds concrete visual or audio detail". The two pieces of guidance disagree; see Conflicts.
- Don't use tag lists, weights, or a prompt pasted from another model (Kling, Seedance). Tag syntax and shot-list formatting "tend to underperform" [S8].
- Don't use labels such as "Audio:", "Visual:", or "medium shot, static camera —" [S1][S2].
- Don't open with "The scene opens…", "We see…" or "There is…" [S1][S2]. The official sample prompts ([S8][S10]) do use "The shot opens on…", so treat this as a style preference of the training captions, not a hard rule.
- Don't state emotions or intentions as labels [S1][S8].
- Don't ask for readable text, signage or logos [S10, LTX-2]. On LTX-2.5, short text improved, but spelling and consistency "are not guaranteed": keep text short and add critical text in post [S8].
- Don't ask for chaotic or fast-twisting motion such as juggling or jumping. It produces artifacts; dancing is fine [S8][S10].
- Don't mix conflicting light sources unless something motivates them [S8][S10].
- Don't infer ethnicity, nationality, religion or culture. Describe the visible attributes instead [S1][S2]. This is the enhancer's rule, and it shows what the captions contain.

**Text-to-video vs image-to-video**

- Text-to-video: the prompt is the only source of the scene. Describe everything [S12].
- Image-to-video, as the docs put it: the prompt "describes what should *happen*, as the model already knows what the scene looks like". Focus on motion, camera and audio [S11].
- Image-to-video, as the official enhancer does it:
  - the caption's opening "must match the reference image exactly": same subjects, clothing, setting, lighting and composition;
  - then it narrates chronologically from that frame;
  - the shot type and viewpoint at the opening match how the image is framed;
  - it makes "Single continuous take — no hard cuts" [S1].
  - These two official sources pull in different directions; see Conflicts.
- Image-to-video: prefer a single continuous take, unless you deliberately describe a cut away from the opening image [S8][S9].
- Image-to-video: describe the motion and the camera explicitly. A Lightricks maintainer says image-to-video "stays static when the image is locked at high strength and the prompt does not describe motion — zoom/hold is the typical fallback" [S14].
- Image-to-video: never contradict the start image or invent things that clash with it [S1].

## Settings that change how the prompt lands

| Upstream setting | Official value for the distilled model | Crucible param | Crucible behaviour |
| --- | --- | --- | --- |
| Steps / sigmas | Fixed 8-sigma schedule `DISTILLED_SIGMA_VALUES` [S4][S6]. The two-stage DistilledPipeline adds a stage 2 on `STAGE_2_DISTILLED_SIGMA_VALUES` (4 sigma values, which is 3 steps) [S4]; the README says "4 steps stage 2" [S3]. | `steps` | Must be 8; any other value is refused `video_param_unsupported`. The Mac reports `refine_steps` 3 [C1][C3]. |
| CFG / guidance | `guidance_scale=1.0`, `audio_guidance_scale=1.0`, STG 0, modality 1 [S6]; "CFG=1" [S7]; the ComfyUI templates use CFG 1 for video and audio [S11][S12] | none | Runs unguided [C2] |
| Negative prompt | Ignored at CFG 1. The guided pipelines have a long official default negative prompt [S4]. The ComfyUI templates attach `"pc game, console game, video game, cartoon, childish, ugly"` [S11][S12], but they run at CFG 1, where a negative prompt has no effect. | `negative_prompt` | Refused by name. Put what you want into the prompt instead [C1][C3]. |
| Prompt enhancer | Optional Gemma 4 rewrite (`--enhance-prompt`; on by default in the ComfyUI templates) using [S1]/[S2]. Recommended for short or foreign-model prompts; helps least when the prompt already follows the structure [S8]. | none | Not used; `prompt_enhancer/` is not pulled [C2]. The prompt reaches the encoder as the caller wrote it, so the caller must write in the [S1]/[S2] style. |
| Width / height | Divisible by 32 (one stage) or 64 (two stage) [S5]. Template base 768x512, then 2x upscale [S11][S12]. | `width`, `height` | PC: multiples of 32, 256 to 1280 per side, at most 901,120 pixels, default 1280x704. Mac: multiples of 64 [C1][C3]. |
| Frames | 8k+1 [S5][S7]; reference 121 frames at 24 fps [S6]; templates use 97 frames [S11][S12] | `num_frames`, or `duration_s` | 8k+1, at most 145 frames. `duration_s` is rounded to the grid [C1]. |
| Frame rate | 24 fps in the examples [S6]. The template notes say 24 is cinematic, 25 standard, 30 smoother [S12]. | `fps` | 24 or 25 only [C1][C3] |
| Duration predictor | Optional; it sizes the clip to the action in the prompt [S3][S8][S13] | none | Not used; a job always states its length [C2] |
| Image conditioning | Image re-encoded through H.264 at the CRF the model was trained against (`LTX_2_4_IMAGE_CRF = 18`; 33 for older models) [S4]. The ComfyUI i2v template injects the image at strength 0.7 in stage 1 and 1.0 in stage 2 [S11]. | (input `start.png` etc.) | CRF 18; the first latent frame is kept clean (full strength) [C2] |
| Audio | Always generated jointly with the video [S12] | `audio` | `false` only skips decoding. The model still generates sound with the picture [C1]. |

Notes:

- The prompt shapes the sound even when `audio` is `false`, because sound and picture are generated together [C1][S12]. Describe a soundscape anyway.
- A seed reproduces a clip only on the machine (arm) that made it [C1].

## Examples

1. **Official, from the README [S3]; it is also in [S6]'s pipeline usage.** A medium close-up of a bearded man in a green-and-white cap. The prompt describes:
   - his look and where he sits in the frame;
   - "a quick sniff sound is heard";
   - quoted dialogue with the voice described ("a deep male voice and a satisfied tone");
   - "The camera remains static throughout", with a shallow depth of field and a beige wall behind.
   - About 160 words, one paragraph. Full text at the [S3] link.
2. **Official multi-shot example [S8].** "A wide shot frames a rainy city intersection at dusk, neon signs reflecting on wet asphalt…". It goes on through a medium close-up and a hard cut to a low-angle shot. At each cut it states the music, for example "the synth score continues across the cut, traffic muffled". Full text at the [S8] link.
3. **Official image-to-video example [S11].** "The woman turns to face the camera and smiles, a warm breeze moving through her hair. Soft piano music plays in the background." This is a docs example for the template, which turns on the prompt enhancer by default. A prompt this short, sent raw to Crucible (no enhancer), sits below the length [S6] says the model needs.
4. **Crucible's own examples [C1]** (fox in snow; woman turning to a window). They follow the paragraph style and include sound.
5. **Constructed** (not from any source). Text-to-video in the [S2] style, about 120 words:
   > A medium shot frames an elderly fisherman in a yellow oilskin coat as he hauls a dripping net over the side of a small wooden boat, captured from a side view while the camera slowly pushes in. Initially his weathered hands grip the wet rope, knuckles pale, and grey water streams off the mesh. A moment later two silver fish slap against the deck, and he pauses, breath clouding in the cold air, before tossing the net aside. Overcast morning light flattens the sea behind him into soft grey, a gull circling low near the stern. The hull creaks against the swell, waves lap at the planks, the net rasps over the gunwale, and the gull calls twice.

## Community findings

- **"A cinematic scene of …" as an opener reduced static image-to-video results for some users.** It was not reliable [S15]. LTX-2.0 era; ComfyUI.
- **Raising the image compression (CRF) gave more motion in image-to-video.** A maintainer agreed that more CRF means more motion, at that time for LTX-2.0 with a default of 33 [S15]. For LTX-2.5 the official trained value is 18 [S4]; Crucible uses 18 [C2].
- **The default negative prompt may suppress motion in guided image-to-video.** The suggested fix was replacing it with "static image" or leaving it empty [S16]. This is a hypothesis about the guided LTX-2.0 pipeline. It does not apply to the distilled model, which never reads a negative prompt.
- **Some source images move much less than others.** Changing or lightly editing the image helped [S15].
- **Tag-style rewrites put the right nouns on screen, but "nothing happens"** [S19]. In image-to-video, re-describing the scene "fights the conditioning" [S19]. Stating where the camera ends up stops moves from being abandoned halfway [S19].
- **Dialogue budget:** about 2.5 spoken words per second, with a beat of silence at each end [S20].
  - Name concrete sound sources that are visible on screen.
  - Pair ambience with foley tied to the action [S20].
  - Stay at 24 or 25 fps for dialogue [S20].
  - These are a vendor's claims, not measured here.
- **The Gemma 4 encoder keeps long multi-character prompts together better than earlier LTX versions** [S20]. This is the vendor's claim.

## Measured in Crucible

Only what the Crucible docs record:

- **PC arm, first run, 2026-09-30:** a 5 s clip at 1280x704 with sound, in 143 s. Stage peaks:

  | stage | peak |
  | --- | --- |
  | encoding | 15.3 GB |
  | connecting | 8.1 GB |
  | denoising | 19.1 GB |
  | decoding | 3.7 GB |
  | audio | 0.5 GB |

  The manifest figures are still the declared ones [C2].
- **Mac, measured 2026-09-30 on the M1 Ultra.** With tiling, every clip up to 20 s at 1280x704 peaked at 21.7 GB, in the text encoder. The duty-cycle measurements (512x320) are recorded in [C2], "Keeping the desktop responsive".
- **No prompt-quality findings.** Crucible's docs record no measurement of prompt adherence, length effects, audio quality, or Q6_K against reference quality. "Look at the clips" is still listed as owed for both arms [C2].

## Conflicts and open questions

**Official guidance against what Crucible does or documents**

1. **Image-to-video: re-describe the first frame or not?**
   - The LTX ComfyUI docs [S11] and the community [S19] say to describe what happens next rather than re-describe the picture (Crucible's docs said the same until this guide).
   - The official image-to-video enhancer [S1] does the opposite: the caption "must match the reference image exactly" at its opening, then narrates forward.
   - [S1] is the rewriter Lightricks built to produce training-style captions. A caller sending raw prompts (no enhancer, as Crucible does) gets closest to the training distribution by doing what [S1] does: a brief faithful opening on the frame, then the action.
   - Neither approach has been measured in Crucible.
2. **The PC runs one full-size pass; Lightricks' recipe is two-stage.**
   - Lightricks' DistilledPipeline is two-stage: 8 steps at half size, 2x latent upscale, then a short pass at full size [S3][S4]. The ComfyUI templates are two-stage too [S11][S12].
   - The Mac follows it. The PC runs the 8-sigma schedule once, at full size, through diffusers [C2]. That is the recipe the Diffusers card gives [S6], so it is official too.
   - [C2] quotes the port's docstring: running the distilled model directly at full resolution "can produce out-of-distribution artefacts".
   - Whether the PC's 1280x704 single pass shows this has not been looked at ("Look at the clips" is owed [C2]).
3. **Stage-2 steps.** The README says "4 steps stage 2" [S3]. The constant has 4 sigma values, which is 3 steps [S4], and Crucible reports `refine_steps` 3 [C1]. This is a counting difference in the README, not a behaviour difference.
4. **Image conditioning strength.**
   - Crucible keeps the first latent frame clean, i.e. full strength [C2].
   - The ComfyUI image-to-video template uses 0.7 in stage 1 [S11].
   - The maintainer ties static image-to-video to "the image is locked at high strength" plus a prompt without motion [S14].
   - So on Crucible, describing motion and camera explicitly is the main lever. Strength is not exposed. Whether static clips are common on Crucible is unmeasured.

**Open questions (inferences, not guidance)**

- **The tags at the start of `DEFAULT_NEGATIVE_PROMPT` [S4]** (`has_subtitles`, `transition from black`, `speech_ending_short` …) suggest the training captions carried metadata tags. If they did, phrases like "fades from black" or subtitles in a prompt may pull those traits in. Unverified.
- **200 words vs "can run longer".** The README [S3] says 200 words; the newer docs [S8] allow longer screenplay-style prompts. Which holds better for the distilled checkpoint on this hardware is untested. Staying in 150 to 220 words is the safe reading of [S1][S2][S3].
- **Screenplay format.** [S8][S10] show screenplay-style prompts (sluglines, character cues, line breaks). The enhancer [S1][S2] never produces them, and [S8] warns against sluglines in multi-shot prompts unless the cut is also described. Whether the format helps without the enhancer is unknown.
- **The look of the clip.** Both enhancer prompts add "cinematic, film-grade color… crisp high-resolution detail" by default [S1][S2]. The official negative list includes "cartoonish rendering, 3D CGI look, … stylized filters" [S4]. The app notes [A1] infer from this that the model leans photographic, and that a painted style should avoid those words. That inference is reasonable but unmeasured. Note that [S10] lists painterly and stylized looks as working well.
- **Clip length vs prompt length.** Crucible always fixes the length (no duration head) [C2]. [S8] says the predictor times action "as written". It is not known whether a fixed 6 s clip with too little prompted action stretches or idles, or with too much action rushes or drops beats. Writing the beats to fit the duration is the cautious approach.
- **Q6_K and int8 against bf16.** It is not known whether the quantizations change how closely the model follows the prompt. Unmeasured [C2].
