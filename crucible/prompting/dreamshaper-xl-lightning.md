# DreamShaper XL Lightning: how to prompt it

Crucible id `dreamshaper-xl-lightning`; upstream `Lykon/dreamshaper-xl-lightning` @ `561086c22aba3925b1e8803a73857f061635c7c3` (fp16 variant), VAE replaced by `madebyollin/sdxl-vae-fp16-fix` @ `207b116dae70ace3637169f1ddd2434b91b3a8cd`; SDXL family; Crucible job type `image` (CUDA only, the verb's small rung).

## Sources

All read 2026-10-15 unless noted.

- [S1] Hugging Face model card, https://huggingface.co/Lykon/dreamshaper-xl-lightning/blob/561086c22aba3925b1e8803a73857f061635c7c3/README.md. OFFICIAL (Lykon). Revision `561086c2` (2024-03-01, the commit Crucible pins).
- [S2] Same repo, `scheduler/scheduler_config.json`, `model_index.json`, and the file tree with LFS sha256s, at `561086c2`. OFFICIAL.
- [S3] Civitai model page "DreamShaper XL" (model 112902), read through the public API `https://civitai.com/api/v1/models/112902`: the model description and the version "Lightning DPM++ SDE" (version 354657, created 2024-02-21), whose file `dreamshaperXL_lightningDPMSDE.safetensors` has sha256 `FDBE5635...4F64`, the same as `DreamShaperXL_Lightning.safetensors` in [S2]. OFFICIAL (Lykon's own text). Page: https://civitai.com/models/112902?modelVersionId=354657
- [S4] Lykon's 20 example images for version 354657, with their generation metadata, `https://civitai.com/api/v1/images?modelVersionId=354657&username=Lykon&withMeta=true` (images posted 2024-02-21/22). OFFICIAL (the trainer's own settings), though made in Automatic1111 with hires fix, ADetailer and sometimes LoRAs.
- [S5] SDXL paper, Podell et al., "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis", https://arxiv.org/html/2307.01952v1 (v1, July 2023). OFFICIAL for the SDXL base this model is fine-tuned from.
- [S6] SDXL base 1.0 model card, https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 @ `462165984030d82259a11f4367a4eed129e94a7b`, "Limitations". OFFICIAL (Stability AI).
- [S7] diffusers docs, "Stable Diffusion XL" guide, https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl (main, read 2026-10-15). OFFICIAL for the pipeline Crucible runs.
- [S8] diffusers docs, scheduler overview with the A1111-to-diffusers table, https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview (main). OFFICIAL for diffusers.
- [S9] diffusers docs, "Prompting" / prompt weighting, https://huggingface.co/docs/diffusers/main/en/using-diffusers/weighted_prompts (main). OFFICIAL for diffusers.
- [S10] diffusers source, `pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py` at tag v0.35.0 (`d0c01785`), `encode_prompt`. OFFICIAL code. (Crucible runs diffusers at commit `5ff8e59f`; the SDXL `encode_prompt` lines cited are long-standing, but I read them at v0.35.0, not at `5ff8e59f`.)
- [S11] diffusers source, `schedulers/scheduling_dpmsolver_multistep.py` at `5ff8e59ff9fe81c6e2df4fb4c6ea0d97a5df5ab2` (the commit Crucible's recipe pins), `set_timesteps`. OFFICIAL code.
- [S12] k-diffusion source, `k_diffusion/sampling.py` at `4601bf08` (master), `sample_dpmpp_sde` and `sample_dpmpp_2m_sde`. OFFICIAL code of the sampler A1111 calls "DPM++ SDE".
- [S13] diffusers issue #7142, "diffusers inference result is different with webui", https://github.com/huggingface/diffusers/issues/7142 (2024-02-29 to 03-01). COMMUNITY (a user report; answers from diffusers maintainers).
- [S14] diffusers PR #6477, "fix DPM Scheduler with use_karras_sigmas option", https://github.com/huggingface/diffusers/pull/6477 (merged 2024-01-19). OFFICIAL code history.
- [S15] diffusers docs v0.31.0, "Scheduler features", https://huggingface.co/docs/diffusers/v0.31.0/using-diffusers/scheduler_features. OFFICIAL for diffusers.
- [S16] madebyollin/sdxl-vae-fp16-fix model card, https://huggingface.co/madebyollin/sdxl-vae-fp16-fix (the pinned revision is `207b116d`; the card was read at main). COMMUNITY (a third party's retrained VAE, not Lykon's or Stability's).
- [S17] Automatic1111 wiki, "Features", https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features (read 2026-10-15). COMMUNITY (the UI Lykon's examples came from).
- [S18] ByteDance SDXL-Lightning model card, https://huggingface.co/ByteDance/SDXL-Lightning (main). OFFICIAL for ByteDance's Lightning, which is a different model (see Open questions).
- [S19] Félix Sanz, "SDXL-Lightning: quick look and comparison", https://www.felixsanz.dev/articles/sdxl-lightning-quick-look-and-comparison (2024-02-29). COMMUNITY.
- [S20] Comflowy model page, https://www.comflowy.com/model/dream-shaper-xl-lightning (2024-02-22). COMMUNITY (restates [S3]).
- [S21] diffusers source, `schedulers/scheduling_dpmsolver_sde.py` at tag v0.35.0 (`d0c01785`): `DPMSolverSDEScheduler`, whose `set_timesteps` repeats each inner sigma twice. OFFICIAL code.
- Crucible repo, read-only, at `5a579e7a` (2026-10-13): `crucible/image/dreamshaper-xl-lightning.toml`, `crucible/jobs/image/worker.py` (`DiffusersSdxlEngine`, `_sdxl_scheduler`), `crucible/jobs/image/__init__.py`, `docs/IMAGE.md`, `docs/internals/image.md` ("The small rung"), `docs/CHANGELOG.md`, `docs/VERB-SIZING.md` section 10.

## What the model was trained to see

- **An SDXL fine-tune.** The repo calls it a Stable Diffusion model fine-tuned on `stabilityai/stable-diffusion-xl-base-1.0` [S1]. Lykon describes DreamShaper as general purpose, aiming to do "photos, art, anime, manga" well [S3]. Civitai files the Lightning version under base model "SDXL Lightning" [S3].
- **Two CLIP text encoders, read side by side.** SDXL uses CLIP ViT-L and OpenCLIP ViT-bigG. Their penultimate outputs are concatenated along the channel axis, and OpenCLIP's pooled embedding conditions the model too [S5]. The repo ships both (`text_encoder` CLIPTextModel, `text_encoder_2` CLIPTextModelWithProjection) [S2].
- **77 tokens per encoder.** diffusers tokenizes to `tokenizer.model_max_length` with truncation and logs the part it cut: "CLIP can only handle sequences up to 77 tokens" [S10]. Two of those are the start and end tokens, so about 75 tokens of prompt reach the model. The negative prompt is tokenized and truncated the same way [S10].
- **Same text to both encoders unless told otherwise.** diffusers sets `prompt_2 = prompt_2 or prompt` [S10]. The SDXL guide shows sending a different `prompt_2` to OpenCLIP-bigG [S7]. Crucible sends only `prompt` and `negative_prompt` (`DiffusersSdxlEngine._encode`), and refuses unknown params, so both encoders always get the same text.
- **About one megapixel, many shapes.** SDXL's multi-aspect fine-tuning used 40 buckets of roughly 1024² pixels, sides in steps of 64, from 512x2048 to 2048x512 [S5, Appendix I]. Examples (width x height): 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640 and their transposes. The diffusers guide says 1024x1024 is the default for best results and that below 512x512 is "not likely to work" [S7].
- **Size and crop conditioning.** SDXL is told the original size, the target size and the crop corner during training. diffusers defaults them to the requested height and width and a (0, 0) crop, which correlates with centred subjects [S7]. Crucible does not expose these, so the defaults apply.
- **Trained and distilled for a specific sampler.** Lykon says Lightning should be used at CFG 2, about 3 to 6 steps, with DPM++ SDE Karras (or Normal) [S3]. On training: "make sure to use DPM++ SDE Karras at the correct target steps and cfg scale" [S3].

## Writing the prompt

- **Do: comma-separated descriptive phrases work, and so do sentences.** Lykon's own examples mix both: phrase lists like "cinematic film still, close up, photo of …, analog film grain" and full sentences ("photo of a Cat poised gracefully atop an ancient oak tree, autumn leaves fluttering around, golden hour …") [S4]. The HF card's example is a phrase list: "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, …" [S1]. diffusers' general guide recommends subject, then style, then context [S9].
- **Do: put the subject and the medium first, keep it under about 75 tokens.** Anything past 77 tokens is cut off, not chunked, in diffusers [S10]. A1111, where Lykon made his examples, splits long prompts into 75-token chunks instead [S17], so long prompts copied from Civitai lose their tail in Crucible. (Putting the important words first follows from the truncation; it is not stated by Lykon.)
- **Do: name the medium and look.** Lykon's examples lean on medium and camera words: "cinematic film still", "photo of", "analog film grain", "anime style", "in the style of …", lens names [S4].
- **Don't: rely on `(word:1.4)` or `((word))` weighting in Crucible.** That syntax is A1111's [S17], and Lykon uses it in most examples [S4]. diffusers does not parse it; weighting in diffusers goes through precomputed embeddings from `sd_embed` or Compel [S9]. Crucible passes the prompt string straight to `pipe.encode_prompt` with no parser (`worker.py`), so the brackets, colon and number go to the tokenizer as plain text.
- **Don't: use LoRA tags (`<lora:name:0.8>`) or embedding names (`BadDream`, `negativeXL_D`).** Lykon's examples use them [S4], but they are A1111 features that load extra files. Crucible loads neither, so the tags are plain text.
- **Don't: ask for readable text.** The SDXL card says the model "cannot render legible text" [S6]; the paper says it has difficulty "rendering long, legible text" [S5]. Lykon puts "text, signature, watermark" in his negative prompts [S4].
- **Don't: expect clean hands or tight multi-object layouts.** SDXL struggles with "intricate structures, such as human hands" and shows "concept bleeding", where separate elements merge [S5]. The card cites compositional prompts like "A red cube on top of a blue sphere" and says faces and people may not be generated properly [S6]. Lykon's examples were fixed afterwards with ADetailer passes on eyes and hands [S4], which Crucible does not do.
- **Do: use a negative prompt; Lykon always does at CFG 2.** Every one of the 20 examples has one, typically "(low quality, worst quality:1.4), cgi, text, signature, watermark, extra limbs" or "ugly, deformed, noisy, blurry, low contrast" [S4]. Crucible's default negative prompt is in the same spirit (below).
- **Don't: write instructions.** The model gets embeddings of the words; there is no instruction-following text encoder. (This is a general CLIP property; I have no Lykon source saying it. See Open questions.)

## Settings that change how the prompt lands

| what | official guidance | Crucible param and default |
| --- | --- | --- |
| steps | 3 to 6 [S3]; 4 in the HF card's code [S1]; Lykon's examples mostly 4, some 5, 6, 7 [S4] | `steps`, default 6 (`default_steps`), allowed 1 to 100 |
| CFG | 2 [S1][S3]; all 20 examples at 2 [S4] | `guidance`, default 2.0 (`default_guidance`), allowed 1.0 to 10.0 |
| sampler | DPM++ SDE Karras, "or Normal" [S3]; for the Turbo sibling, "only with DPM++ SDE Karras (NOT 2M)" [S3]; the HF card's code uses `DPMSolverMultistepScheduler` with defaults, i.e. DPM++ 2M without Karras [S1][S8] | manifest `scheduler = "dpmpp-sde-karras"`, built as `DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)`. Not a request param. |
| negative prompt | always used at CFG 2 in the examples [S4] | `negative_prompt`; default `"text, letters, words, watermark, signature, logo, blurry, deformed"`; read only with `guidance` > 1.0; refused if blank; `guidance: 1.0` turns CFG and the negative off |
| size | about 1 MP, SDXL buckets [S5]; Lykon: "Avoid going too far above 1024 in either direction for the 1st step" [S3]; examples drawn at 768x1024, 1024x768, 768x1088, 1216x768, then hires-fixed 1.35x to 1.6x [S4] | `width`, `height`, default 1024x1024; multiple of 32; at most 1,048,576 pixels; 256 to 2048 a side. All 40 SDXL buckets pass these rules. No hires fix or img2img on this model. |
| CLIP skip | Lykon's metadata shows `clipSkip 2` on many examples [S4], but A1111 says CLIP skip has no effect on SDXL, which is trained on the penultimate layer [S17] | none; diffusers uses the penultimate layer by default (`hidden_states[-2]`) [S10] |
| VAE | the repo's VAE is SDXL's [S2] | replaced by `madebyollin/sdxl-vae-fp16-fix`, which keeps output "close enough for most purposes" with "slight discrepancies" from SDXL's VAE, and avoids NaN in fp16 [S16] |
| seed | none | `seed`, reproducible within one backend and engine |

**How CFG 2 treats the negative prompt.** In diffusers, CFG computes `uncond + scale * (cond - uncond)`, and a negative prompt replaces the unconditional branch. At scale 2 the push away from the negative is one times the difference; at a typical SDXL scale of 7 it is six times. So negatives still act at CFG 2, but more gently. (The formula is the standard diffusers pipeline; the "more gently" reading is mine and is not measured. See Open questions.) In Crucible the negative costs no extra time at guidance 2.0, since both passes already run (docs/IMAGE.md "Negative prompts").

**Guidance above 2.** Lykon gives no guidance for higher CFG on Lightning; he warns: "please hold your judgement on anything that differs from the suggested settings" [S3]. For the Turbo sibling only, he says it can run as a normal model at CFG 6 and 20 to 40 steps, and says that is "not Lightning" [S3].

## Examples

From official sources (short, attributed):

- HF card [S1]: prompt "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors", `num_inference_steps=4`, `guidance_scale=2`, no negative prompt (diffusers then uses zero embeddings for the unconditional branch [S10]).
- Lykon's Civitai example 6935887 [S4]: prompt starts "photo of a Cat poised gracefully atop an ancient oak tree, autumn leaves fluttering around, golden hour casting long shadows, backlit …"; negative "ugly, deformed, noisy, blurry, low contrast"; 4 steps, CFG 2, DPM++ SDE Karras, drawn at 768x1024, then hires fix 1.5x.
- Lykon's Civitai example 6935888 [S4]: prompt "photo of the warrior Aragorn from Lord of the Rings, film grain, cinematic film still, 8k hd, portrait"; negative "(deformed iris, deformed pupils), text, worst quality, low quality, ugly,"; 4 steps, CFG 2.

Constructed (not from any source; a translation of the above into a Crucible request, untested):

```json
{"type": "image",
 "model": "dreamshaper-xl-lightning",
 "params": {"prompt": "photo of a cat on an ancient oak branch, autumn leaves falling, golden hour, backlit, shallow depth of field, cinematic film still",
            "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, text, signature, watermark",
            "width": 832, "height": 1216, "steps": 6, "guidance": 2.0, "seed": 1}}
```

## Community findings

- **Sampler mismatch made bad pictures in diffusers.** A user ran this exact checkpoint (Civitai version 354657) in diffusers 0.24.0 with `DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="dpmsolver++")` at 4 steps, CFG 2, 512x512, and got strange results [S13]. A diffusers maintainer answered that Lykon says DPM++ SDE Karras, so use `DPMSolverSinglestepScheduler(use_karras_sigmas=True)`, and added that multistep with Karras "generates bad results, it happens with ComfyUI too" [S13]. Two caveats: that report used `dpmsolver++`, not the `sde-dpmsolver++` Crucible uses, and diffusers 0.24.0 (released 2023-11-29) predates PR #6477 (merged 2024-01-19), which fixed the DPM schedulers' Karras last step [S14].
- **Lightning-style models and CFG.** One write-up says SDXL-Lightning, SDXL Turbo and LCM-LoRA "do not use negative prompt or CFG value" [S19]. That describes ByteDance's SDXL-Lightning, whose card says CFG 0 [S18]. It does not match Lykon's own CFG 2 guidance and negative prompts for DreamShaper XL Lightning [S3][S4], so treat it as about a different model.
- **Restatements.** Comflowy repeats Lykon's settings (3 to 6 steps, CFG 2, DPM++ SDE Karras, up to about 1024 for the first pass) and calls the model good at portraits and anime [S20].
- **Karras on DreamShaper.** diffusers' docs say DreamShaper XL "can" use Karras sigmas "since they are trained with Karras sigmas", while base SDXL should not [S15]. That line is about DreamShaper XL 1.0, not Lightning.
- I could not reach Reddit with my tools, so no r/StableDiffusion threads are cited.

## Measured in Crucible

All from Crucible's own docs; nothing here was run for this guide.

- Recipe: the whole pipeline resident, UNet stored as fp8 and computed in fp16, VAE tiled, `dpmpp-sde-karras`, 6 steps, guidance 2.0, the default negative prompt (docs/internals/image.md, "The small rung").
- Measured 2026-10-10 on owens-pc's RTX 3090 Ti held to an 8 GiB card's room, by a script beside Crucible, "not yet through the worker": twelve covers at 1024x1024 (six moods by two seeds); 4.14 GiB resident; 4.99 GiB allocated and 6.15 GiB reserved at peak; 3.2 s a picture warm at 6 steps; 4.7 s to the card (docs/internals/image.md; manifest `memory_note`).
- 512x512 came back "visibly worse" and only 1.6 s faster than 1024x1024, so `native` is 1024x1024 and `quality_min_side` is 768 (docs/internals/image.md "The small rung"; docs/IMAGE.md "Which model, and its sizes").
- The repo's own SDXL VAE "overflows in fp16 and decodes NaN, a black picture"; the fp16-fix VAE replaces it (docs/internals/image.md).
- Text encoding is "a fraction of a second" because both encoders stay resident; the 32-entry prompt cache still applies (docs/IMAGE.md "Many pictures in a row").
- The default negative prompt exists because the album covers carry no writing (docs/internals/image.md, "Backends").
- No measurement of prompt style, weighting, sampler choice, step count or CFG on picture quality is recorded in the repo.

## Conflicts and open questions

1. **Sampler: Crucible runs DPM++ 2M SDE Karras, not DPM++ SDE Karras.** The manifest names it `dpmpp-sde-karras`, but `_sdxl_scheduler` builds `DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)`. diffusers' own table maps exactly that to "DPM++ 2M SDE Karras", and maps "DPM++ SDE Karras" to `DPMSolverSinglestepScheduler(use_karras_sigmas=True)` [S8]. Lykon says Lightning should work "only with DPM++ SDE Karras" [S3] and, for the Turbo sibling, "NOT 2M" [S3]. The two samplers are different algorithms: k-diffusion's `sample_dpmpp_sde` calls the model twice per step, `sample_dpmpp_2m_sde` once [S12]. Crucible's measured covers looked acceptable to whoever chose the recipe, but no side-by-side against the singlestep sampler is recorded. Open: whether Lykon's sampler gives better pictures in Crucible's pipeline. diffusers also has `DPMSolverSDEScheduler` (a port of k-diffusion's DPM++ SDE, needing `torchsde`), which doubles its internal timesteps [S21]; which of it and the singlestep scheduler matches A1111 more closely is not settled here.
2. **The HF card disagrees with the Civitai page.** The HF card's code uses DPM++ 2M without Karras at 4 steps [S1]; the Civitai text says DPM++ SDE Karras at 3 to 6 [S3]. Both are Lykon's. Crucible follows neither exactly.
3. **Steps.** Crucible's default 6 is at the top of Lykon's 3 to 6 range [S3]; Lykon's examples are mostly 4 [S4]. Counting model calls, Lykon's 4 DPM++ SDE steps are roughly 7 to 8 UNet evaluations [S12], close to Crucible's 6 single-call steps. That comparison is my arithmetic, not a measured equivalence.
4. **Size floor vs SDXL buckets.** Crucible's `quality_min_side = 768` comes from a 512x512 test. SDXL was trained at wide and tall shapes with a 512 or 640 side (for example 1536x640, 2048x512) [S5]. Those are about 1 MP and probably fine; the floor reads as an area rule more than a side rule. Not measured in Crucible.
5. **Lykon's examples were drawn below 1 MP and upscaled.** Most were 768x1024 then hires-fixed [S4]; Crucible draws one pass at up to 1 MP with no hires fix. Lykon says the model can be used for hires fix and tiled upscaling [S3]. Whether 1024x1024 in one pass matches his look is not measured.
6. **Owen's prompt lessons are Qwen-Image's.** They live in the Qwen-Image guide. One does not carry over: short text like a sign "renders well" there, but the SDXL card says this model "cannot render legible text" [S6]. His 1280x704 thumbnail is valid on CUDA but is not an SDXL bucket; 1344x768 is [S5].
7. **Weighting syntax in Crucible.** `(word:1.4)` reaches the tokenizer as literal text. What that does to the picture (likely a little noise, maybe nothing visible) is not measured. Apps copying Civitai prompts should strip it.
8. **Truncation is invisible to the caller.** diffusers logs the cut text in the worker [S10], but `done.image` does not report a token count or a truncation. A client cannot tell that its prompt lost its end.
9. **Is this a merge with ByteDance's SDXL-Lightning?** The name and Civitai's "SDXL Lightning" base suggest a Lightning distillation, but none of Lykon's sources say how it was made. ByteDance's model wants CFG 0 and trailing Euler [S18]; Lykon says CFG 2 and DPM++ SDE Karras [S3]. Follow Lykon.
10. **Which variant is the diffusers folder?** The repo holds both `DreamShaperXL_Lightning.safetensors` and `…-SFW.safetensors` [S2]. The diffusers folders were uploaded 2024-02-21, before the SFW file (2024-02-22) [S2 commit list], which suggests they are the non-SFW model. Not verified by hashing the converted UNet.
11. **Timestep spacing.** The repo's scheduler config says `timestep_spacing: "leading"` [S2], and Crucible builds its scheduler from that config. In diffusers at Crucible's commit, Karras sigmas are drawn from the full training sigma range regardless of spacing [S11], so "leading" should not shorten the schedule. This is from reading the source, not a test.
12. **Instruction-style prompts.** No Lykon source says how the model handles sentences like "make the sky darker". CLIP encoders are generally treated as keyword-like; I have no official citation for that, so it stays an open question.
