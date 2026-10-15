# Qwen-Image 2.1: how to prompt it

Model id `qwen-image-2.1`; upstream `Qwen/Qwen-Image-2.1` @ `790c92633540aa0cb11d9abf19eb46d861714758` (both backends, `crucible/image/qwen-image-2.1.toml`); Crucible job type `image` (mflux 0.20.0 on `mlx-darwin`, diffusers `5ff8e59f` `QwenImage21Pipeline` on `cuda-linux`).

## Sources

All read 2026-10-15.

- [S1] https://github.com/QwenLM/Qwen-Image-2.1/blob/ab96caa944157ca28a4f36b695c3047549f48f98/prompt_rewrite/prompts/system_prompt_t2i.txt. The system prompt of the official text-to-image prompt rewriter (Qwen-Image-2.1-PE-T2I). OFFICIAL. Commit `ab96caa9` (2026-10-14). The file has been unchanged since it was added; the first version (commit `3eae4184`, 2026-09-20) is byte-identical.
- [S2] https://github.com/QwenLM/Qwen-Image-2.1/blob/ab96caa944157ca28a4f36b695c3047549f48f98/README.md. Official GitHub README: sizes, defaults, RGBA prompt format, editing, rewriter. OFFICIAL. Commit `ab96caa9`.
- [S3] https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/README.md. The model card at the revision Crucible pins. OFFICIAL. Revision `790c9263`.
- [S4] https://huggingface.co/Qwen/Qwen-Image-2.1/blob/790c92633540aa0cb11d9abf19eb46d861714758/scheduler/scheduler_config.json. The scheduler config at the pinned revision. OFFICIAL. Revision `790c9263`.
- [S5] https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I (README). The prompt-rewriter model card. OFFICIAL. Revision `f3ed7985`.
- [S6] https://github.com/QwenLM/Qwen-Image-2.1/blob/ab96caa944157ca28a4f36b695c3047549f48f98/prompt_rewrite/pe_core.py. The rewriter's code (sampling defaults, output record). OFFICIAL. Commit `ab96caa9`.
- [S7] https://github.com/QwenLM/Qwen-Image/blob/6b5e1f5cec987d404be5ac6657db3b9aacb56a89/src/examples/tools/prompt_utils_2512.py. The rewriter for the EARLIER model, Qwen-Image-2512. OFFICIAL, but for a different model. Commit `6b5e1f5c` (2026-02-10). There is no 2512 rewriter in the Qwen-Image-2.1 repo; this is where it lives.
- [S8] https://github.com/QwenLM/Qwen-Image/blob/6b5e1f5cec987d404be5ac6657db3b9aacb56a89/README.md. The README of the earlier Qwen-Image / Qwen-Image-2512 (positive magic, `true_cfg_scale` 4, 50 steps). OFFICIAL, but for a different model. Commit `6b5e1f5c`.
- [S9] https://github.com/huggingface/diffusers/blob/5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py. The diffusers pipeline at the commit Crucible's CUDA arm runs. COMMUNITY (Hugging Face's implementation; Qwen's README points to it as the reference path). Commit `5ff8e59f` (2026-09-28).
- [S10] https://github.com/huggingface/diffusers/blob/5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2/docs/source/en/api/pipelines/qwenimage21.md. diffusers' pipeline doc page. COMMUNITY (Hugging Face). Commit `5ff8e59f`.
- [S11] https://github.com/huggingface/diffusers/blob/5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2/examples/dreambooth/README_qwenimage21.md. diffusers' DreamBooth README for 2.1 (the no-truncation note). COMMUNITY (Hugging Face). Commit `5ff8e59f`.
- [S12] https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1, source at https://github.com/Comfy-Org/docs/blob/660f1c2aa4/tutorials/image/qwen/qwen-image-2-1.mdx. ComfyUI's native-workflow guide (CFG, steps, resolution findings). COMMUNITY. Commit `660f1c2a` (2026-10-15).
- [S13] https://github.com/QwenLM/Qwen-Image-2.1/issues/5 and https://github.com/QwenLM/Qwen-Image-2.1/issues/14. Users report faint purple/pink vertical lines. COMMUNITY. Threads read up to 2026-09-30.
- [S14] https://github.com/QwenLM/Qwen-Image-2.1/issues/9. Users report that editing a generated picture with the seed that made it gives oversaturated, broken output. COMMUNITY. Read up to 2026-09-28.
- [S15] https://github.com/MindrLabs/image-generation-qwen-image-2.1 (README). One team's measured write-up: text rendering by language, CFG for handwriting, steps, the seed effect, the line artifact. COMMUNITY. Commit `45c2ac71` (2026-10-01).
- [S16] Crucible `docs/IMAGE.md`, `docs/internals/image.md`, `docs/CHANGELOG.md`, `crucible/image/qwen-image-2.1.toml`, `crucible/jobs/image/worker.py`. At `5a579e7a` (2026-10-13). Crucible's own docs and code.
- [S17] Storyboard `storyboard/docs/PROMPTING.md`. App-side notes, used only as leads. Not guidance.

Not read: the Qwen blog (https://qwen.ai/blog?id=qwen-image-2.1) renders with JavaScript, and the fetch returned no text. Nothing here is drawn from it.

## What the model was trained to see

- **A Qwen3-VL 8B text encoder reads the prompt.** It is wrapped in a fixed chat template with the system line "Comprehend and analyze the provided prompt." Those system tokens are dropped from the embeddings [S9]. The caller sends plain text only; the template is the pipeline's job [S9].
- **No truncation.** The processor does not truncate. The DreamBooth README says "There is no `--max_sequence_length`: the checkpoint's processor does not truncate" [S11]. Padding is on the left, "as the checkpoint was trained with" [S9].
- **The shape the model is fed in official use is the rewriter's output.** Qwen recommends running every prompt through the official rewriter first [S2]. Its system prompt [S1] produces:
  - one English paragraph, about 20 sentences and 400 to 500 words;
  - present tense, third person, the voice of someone describing a finished picture;
  - an opening sentence of about 20 words naming orientation, style, medium, subject and background or palette, e.g. "The image is a wide ⟨style⟩ ⟨medium⟩ of ⟨subject⟩, ⟨background⟩" [S1];
  - then a walk of the frame: background first, then the top band, then left, centre and right, then the bottom band. For a single subject, it walks the subject instead;
  - 8 to 14 positional phrases that reach the corners and edges, about a third of sentences opening on one;
  - lighting in its own sentence;
  - one closing "The overall composition …" sentence [S1].
  The size stays the same whether the request was short or long [S1].
- **The rewriter's sample inputs are short requests in any language**, e.g. a Chinese one-liner and "minimalist poster for a jazz festival" [S2, S5]. The rewriter always answers in English, except for text shown in the image [S1].
- **Two official prompts in the README are long prose.** One is a dense chemistry poster of about 1,200 words, every visible string quoted. The other is a ~200-word edit instruction [S2]. The model-card example is one short sentence with a quoted sign [S3].
- **Native output is 2K.** The recommended sizes are about 4 MP: 1:1 2048x2048, 4:3 2400x1792, 3:4 1792x2400, 3:2 2528x1696, 2:3 1696x2528, 16:9 2752x1536, 9:16 1536x2752 [S2, S3]. "Both checkpoints default to the 2048 resolution level" [S2]. The scheduler uses dynamic shifting up to `max_image_seq_len` 8192 [S4], which is the token count of a 2048x2048 picture at 16 pixels a token [S9].
- **The VAE is RGBA.** The model can make transparent pictures from a fixed prompt form (below) [S2].
- **One model does generation and editing.** Editing takes condition images (up to 10) that the text encoder sees as vision context [S2, S10]. Crucible's image-to-image and inpainting do not use this path (see Conflicts).

## Writing the prompt

Do:

- Describe the finished picture as an observer would: present tense, declarative [S1].
- Open on the medium and style, e.g. "The image is a vertical realistic photograph of …". The medium noun is "never omitted" [S1].
- Place every element in the frame with positional phrases, covering corners and edges, not only the centre [S1].
- Give lighting its own sentence: source, direction, quality, shadows [S1].
- Name colours with a modifier ("deep navy", "pale cream") and give materials ("brushed metal", "weathered wood") [S1].
- Enumerate. Write small counts as words. Say what each thing is instead of "several items" [S1].
- Describe people by their visible surface: build, posture, gaze, expression, hair, skin tone, each garment's colour and material. Give age as a life stage or decade, never a number of years [S1].
- Use generic objects ("a silver laptop") unless the brand is intended. Photographic and design vocabulary such as depth of field, bokeh, backlit or negative space is welcome [S1].
- Hedge what is uncertain ("appears to be", "a notebook or a tablet"). The rewriter calls this "the natural register here" [S1].
- Write in English. Text inside the image stays in its own script [S1, S5].
- Keep it physically coherent: shadows fall away from the light, and scale stays consistent [S1].
- For a transparent picture, use the recommended frame: "This is an RGBA image with transparency. <your description>. The image has alpha channel and the background is transparent." [S2, S3].

Don't:

- Don't write instructions to the renderer: no "you", "create", "make sure" [S1]. The rewriter treats requests like "4K, no noise" and "make sure the text is sharp" as instructions about the job, not content. It obeys them "silently" and "never echo[es]" them [S1].
- Don't add quality boosters. The rewriter forbids "masterpiece", "8K", "highly detailed" and "award-winning" [S1]. The earlier model's "positive magic" suffix (", Ultra HD, 4K, cinematic composition.") [S8] belongs to Qwen-Image / 2512 and is not part of any 2.1 example [S2, S3].
- Don't write a ratio, a resolution or a pixel count into the description. The rewriter puts the ratio in a separate field, which the caller turns into width and height [S1, S2].
- Don't invent signage for a picture that should have no text. A third of images have none, and the rewriter skips the text step for them [S1].
- Don't use a short prompt and expect the model to fill in the rest. The rewriter's design assumes that "a thin brief never buys a thin description" [S1]. This is how Qwen prepares prompts. It is not a measured statement about how a short prompt renders.

### Text in the picture

- Put each string in straight double quotes, in its own script, copied character for character. Say where it sits, its weight, colour, case and relative size, e.g. a bold black headline across the top reads "…" [S1].
- A line break is described as a second line, not written as a newline in the string [S1].
- Text that should not be readable (distant signs, dense body copy) is described as "blurred, indistinct, or too small to read" [S1].
- Charts and tables count as text: axes, tick labels, legends and cell values are all written out [S1].
- For 2.1, the official rule for a picture with no text is to say nothing about text [S1]. The earlier 2512 rewriter instead closed with "The image contains no recognizable text." or "No other text appears in the image." [S7]. That sentence is a 2512 convention. The 2.1 rewriter does not emit it.

## Settings that change how the prompt lands

| Upstream name | Upstream value | Crucible param | Crucible default |
| --- | --- | --- | --- |
| `true_cfg_scale` (diffusers) / `cfg` (ComfyUI) | 1.0, no guidance: "meant to be sampled without guidance" [S9]; "40 steps and no guidance" [S10] | `guidance` (passed as `true_cfg_scale` on CUDA [S16 worker.py]) | 1.0 |
| `negative_prompt` | none. The official rewriter always writes `negative_prompt: ""` [S6]. It is read only with `true_cfg_scale > 1` [S9] | `negative_prompt` | none. It needs `guidance` above 1.0, and each without the other is refused [S16] |
| `num_inference_steps` | 40 [S2, S3, S9] | `steps` | 40 |
| `width` / `height` | about 4 MP, table above [S2]. diffusers' own fallback is `output_resolution` 1024 [S9] | `width`, `height` | 1024x1024, capped at 1,048,576 pixels [S16] |
| scheduler | FlowMatch Euler, dynamic exponential shift, `shift_terminal` 0.02 [S4] | none (fixed) | the engine's |

- **Guidance.** Two passes per step when above 1 [S10]. ComfyUI's findings: at 1 a negative prompt has no effect. At 2 the model follows dense prompts more closely, "including small text and numbers", but edges are over-sharpened. Higher values shift exposure, "`cfg` 5 degrades quality badly", and "`cfg` 0.5 breaks the image" [S12]. Crucible's floor is 1.0, so the last point cannot happen there.
- **Negative prompt.** Qwen publishes no negative prompt for 2.1 [S2, S3, S6]. The long Chinese negative prompt and `true_cfg_scale=4.0` in the Qwen-Image README are for Qwen-Image-2512 [S8]. Don't copy them to 2.1.
- **Steps.** ComfyUI's template defaults to 25 and notes the published pipeline uses about 40 to 50. They found hands settle "by about 30 steps", and 25 to 40 steps reduces fizzle in detailed areas [S12]. MindrLabs found that at 20 steps "prompt elements such as text or small drawings can go missing" [S15].
- **Size.** ComfyUI says targets "well above the model's 2K trained size, such as 4K, lose prompt adherence" [S12]. None of the sources says anything about quality *below* 2K.
- **Turbo** (`Qwen/Qwen-Image-2.1-Turbo`, 8 steps, its own schedule) is a separate checkpoint [S2]. Crucible does not serve it.

## Examples

From official sources (short, attributed):

- Model card: `A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement`, 2048x2048, 40 steps [S3].
- Transparency: `This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent.` [S2, S3].
- The README's full t2i example is a ~1,200-word chemistry poster. It opens "A vertically oriented educational chemistry study poster fills the frame …" and quotes every string. It is rendered at 1696x2528 [S2]. See the link instead of a copy.
- The rewriter's opening-sentence template: `The image is a ⟨vertical / wide / square / tall⟩ ⟨style⟩ ⟨medium⟩ of ⟨subject⟩, ⟨the background and its palette⟩.` [S1].

Constructed (an illustration of [S1]'s rules, not from any source, and shorter than [S1]'s 400 to 500 words):

> The image is a wide realistic photograph of a weathered wooden kitchen table holding one glossy red apple, against a pale cream plaster wall. In the background, the wall is lit unevenly and a narrow window frame appears at the upper-left edge. On the left side of the frame, a folded coarse linen cloth in muted olive lies partly off the table edge. In the centre, the apple sits slightly below the middle, its stem tilted to the right and a small bruise visible on its lower side. On the right, the grain of the table runs toward the corner, with a few faint scuffs and crumbs. The lighting is soft diffused daylight from the window on the left, leaving a short soft shadow to the right of the apple. The overall composition is calm and sparse, with warm muted colours and plenty of negative space above the apple.

## Community findings

- **Text by language.** Measured with OCR on four handwritten and menu scenes plus short phrases: English and Chinese were "almost never wrong". Korean failed on one or two syllables in longer text, the same way across seeds; only rewording fixed it. Text that was not requested in the background "is always fake text" [S15].
- **Handwriting.** "Neat handwriting" came out looking like a font. With CFG 4 and the negative prompt "font, typed text", most pages stayed readable to the end, at about twice the time [S15].
- **The seed of a generated picture.** Editing a picture the model made, with the seed that made it, at the same size, gives an oversaturated copy instead of the edit. A different seed or size avoids it [S14, S15]. This was seen on the condition-image (edit) path.
- **Faint purple/pink vertical lines at fixed columns** (about x 438, 631, 824 at 1024 wide), most visible on skin and light backgrounds. Several users reported them on CUDA and on Apple Silicon [S13]. The reported cause disagrees. MindrLabs attributes them to tiled VAE decoding and says turning tiling off removes them [S15]. The first reporter in issue 5 says tiling on or off made no difference [S13]. One user says the prompt affects them (strong on portraits and still life, faint on dark scenes) while seed, steps and CFG 4 do not [S13].
- **Hands.** An issue titled "the probability of broken hands is a bit high" exists, with no body [S13 repo, issue 12]. ComfyUI's step finding on hands is above [S12].
- **Many references.** Ten reference images "all make it in" but some resemble their reference less. With three or more, describe each item's colour, material and shape [S15]. This is the edit path; Crucible does not expose it.

## Measured in Crucible

Only what Crucible's docs record [S16]:

- The defaults are 40 steps, guidance 1.0, no negative prompt, and 1024x1024. A negative prompt needs guidance above 1.0, which "runs a second pass per step: twice the time" (docs/IMAGE.md, "Negative prompts").
- Sizes: up to 1,048,576 pixels, the size the memory was measured at. Multiples of 16 on the Mac and 32 on CUDA. Small pictures hold up: covers at 256x256 and thumbnails at 384x512 are in use, and nothing down to 256 has been seen to break (docs/IMAGE.md; qwen-image-2.1.toml `size_note`).
- Owen's prompt lessons from making thumbnails and card art on the Mac (2026-09-28; they lived
  in docs/IMAGE.md until this guide):

  - **Describe the picture, never the genre.** "YouTube thumbnail", "poster" or a quoted title make
    it copy the genre, including gibberish text.
  - **Always add "No text, no letters, no numbers, no logos."** Genre words can still override it.
  - **Fewer objects, fewer tells.** Every extra prop is another chance of a mangled one.
  - **It is weak at spatial layout: spell out angle and scale.** For example "low angle,
    eight-foot ceiling, sofa below for scale".
  - **Avoid the AI look.** "ordinary documentary photograph, natural light, muted colours, film
    grain, real imperfections, not glossy, not CGI", or a plain style (flat graphic, linocut,
    simple editorial cartoon).
  - **Painted card art drifts between images.** Repeat strong style wording in every prompt
    ("gouache, visible brushstrokes, not photorealistic").
  - **Short text renders well, paragraphs do not.** A sign reading "THE LAST CHAPTER" works;
    a paragraph turns to nonsense.
  - **Sizes that work:** 3:4 portrait 384x512 or 768x1024, a thumbnail 1280x720 (on the CUDA
    arm use 1280x704 or 1280x736: 720 is not a multiple of 32).
  - **Timing on the Mac Studio (M1 Ultra):** 512x512 is about a minute, 1280x720 at 40 steps
    about five.
- Image-to-image strength: 0.03 to 0.15 for a real change, 0.15 to 0.3 for a touch-up, and above 0.3 comes back close to a copy. Measured on both machines on 2026-09-29. "The model does not take instructions like 'remove the lamp': describe the picture you want instead." (docs/IMAGE.md, "Image-to-image").
- The PC takes 52 to 73 s per picture at 40 steps, of which 16 to 25 s is the text encoder (qwen-image-2.1.toml `memory_note`). The Mac timings are in Owen's lessons, above.
- The prompt-embedding cache: an identical prompt skips the encoder. A 300-token prompt's embeddings are about 2.5 MB (docs/internals/image.md, "The prompt-embedding cache").

## Conflicts and open questions

1. **Size.** Qwen's native and default size is 2K, about 4 MP [S2, S3]. Crucible caps at 1 MP and defaults to 1024x1024. The cap is a memory budget (docs/IMAGE.md), not a quality finding. They recommend 2K, and none of them discusses quality at 1 MP. Crucible's own record that small sizes hold up is consistent with this, but it does not show that 1 MP is the best size.
2. **"No text, no letters, no numbers, no logos."** (Owen's lessons, above). This is an instruction to the renderer, which the 2.1 rewriter forbids [S1]. Its 2.1 convention for a picture with no text is to say nothing about text [S1]. The descriptive sentence "The image contains no recognizable text." is the 2512 rewriter's [S7], and Storyboard uses it [S17]. Which works better on 2.1 has not been measured in any source I found. Owen's rule is recorded from use. The rewriter's rule is how Qwen prepares prompts, not a measured A/B test.
3. **"Describe the picture, never the genre"** (Owen's lessons, above) partly differs from [S1]. The rewriter's opening sentence names the medium noun ("poster", "infographic", "card"), and the README's flagship example is a poster [S2]. They agree on intent if the genre word comes with a full description of every string. Owen's finding is about a bare genre word or a quoted title. That reading is my inference.
4. **Editing.** Qwen-Image 2.1 edits from instructions and condition images (up to 10, with circles, painted marks or masks) [S2, S10, S12]. Crucible's `image_strength` and `mask` are noise-start and repaint methods on the plain text-to-image path. On CUDA, Crucible builds its own start schedule because the pipeline "takes images only as editing conditions" (docs/internals/image.md). So Crucible's line "The model does not take instructions like 'remove the lamp'" is true of Crucible's image-to-image, not of the model. Crucible has no condition-image parameter. Whether to add one is a Crucible decision, not prompting guidance.
5. **Steps.** Qwen says 40 [S2]. ComfyUI defaults to 25 and says 40 to 50 is published [S12]. Crucible uses 40, which matches Qwen.
6. **The seed effect [S14, S15] in Crucible's image-to-image and inpainting.** If an app feeds a Crucible-made picture back with the same seed and size, does the oversaturation appear? Crucible's path is not the edit path those reports used, so I don't know. It has not been measured.
7. **Line artifact.** Crucible's CUDA arm decodes Qwen-Image with a plain `vae.decode`. From reading `worker.py`, I found no tiling on that path. The Mac arm passes mflux's `tiling_config`. Whether Crucible's pictures show the reported lines is not recorded in its docs. The community disagrees on the cause [S13, S15].
8. **Transparency.** The model can output RGBA [S2]. Crucible's docs do not say whether `image.png` keeps an alpha channel when the RGBA prompt form is used. Unverified.
9. **Mac prompt handling.** [S11] (no truncation) covers diffusers only. I did not check whether mflux 0.20.0's `Qwen21PromptEncoder` truncates long prompts or uses the same template. If it truncates, prompts of 400 to 500 words [S1] could be cut on the Mac.
10. **[S5] holds sampling notes for the rewriter LLM, not the image model:** temperature 1.0, top_p 0.95, top_k 20, thinking on, `max_new_tokens` 16256 [S5, S6]. Storyboard's notes cite [S5] next to the CFG findings [S17]. Those CFG findings come from ComfyUI [S12].
11. **Prompt language.** The rewriter turns any language into English [S1, S5]. No source here says how well the base model follows a non-English prompt sent directly.
