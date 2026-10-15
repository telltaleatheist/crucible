# Stable Audio 3.0 Small SFX: how to prompt it

Model id `stable-audio-3-small-sfx` (Crucible class `sfx`); upstream `stabilityai/stable-audio-3-small-sfx` @ `ae12755283df9d62ca39a9b050a39a0b607b8c20` (both arms, from `crucible/audio/stable-audio-3-small-sfx.toml`); inference package `Stability-AI/stable-audio-3` @ `779434a9`; Crucible job type `audio`.

## Sources

All web sources read 2026-10-15. Crucible files read at crucible commit `5a579e7a` (2026-10-13).

- **[S1]** `stable-audio-3/docs/guides/prompting.md`, Stability-AI/stable-audio-3 @ `779434a908193105335fd8d833418603625b2859` (Crucible's pin, 2026-09-01). It is byte-identical at main `77e0398c` (2026-10-11). https://github.com/Stability-AI/stable-audio-3/blob/779434a908193105335fd8d833418603625b2859/docs/guides/prompting.md (OFFICIAL)
- **[S2]** `stable-audio-3/docs/workflows/inference.md`, same repo and commit, identical at main. (OFFICIAL)
- **[S3]** `stable-audio-3/docs/guides/model-overview.md`, same repo and commit, identical at main. (OFFICIAL)
- **[S4]** `README.md`, same repo and commit, identical at main. (OFFICIAL)
- **[S5]** `stable_audio_3/interface/reprompt.py` @ `779434a9`: the system prompts of Stability's Gradio "Prompt Assistant". The **SFX** and **One-shot** prompts there are Stability's most detailed statement of SFX prompt shape. https://github.com/Stability-AI/stable-audio-3/blob/779434a908193105335fd8d833418603625b2859/stable_audio_3/interface/reprompt.py (OFFICIAL)
- **[S6]** `stable_audio_3/interface/diffusion_cond.py` @ `779434a9` (Gradio UI defaults; it strips the `Length:` suffix). (OFFICIAL)
- **[S7]** `stable_audio_3/model.py`, `inference/sampling.py`, `models/dit.py` @ `779434a9` (conditioning, duration padding, pingpong default, the CFG branch). (OFFICIAL)
- **[S8]** Hugging Face model card, `stabilityai/stable-audio-3-small-sfx` @ `ae127552`. At HF head `05ca3857` it differs only in the licence link. https://huggingface.co/stabilityai/stable-audio-3-small-sfx (OFFICIAL)
- **[S9]** `model_config.json` @ `ae127552`. The file is gated, so it was read from Crucible's pulled copy at `~/.crucible/models/stable-audio-3-small-sfx/cuda-linux/` in WSL. It gives the conditioner, `sample_size` and the training demo prompts. (OFFICIAL)
- **[S10]** *Stable Audio 3*, tech report, arXiv 2605.17991v1, 18 May 2026, sections 2.2, 3.1, 3.5, 5.1, 5.3, 5.5 and 5.6. https://arxiv.org/abs/2605.17991 (OFFICIAL)
- **[S11]** "Stable Audio 3.0 Prompt Guide", stability.ai/guides, dated "August 18" with no year shown; mirrored at kb.stability.ai. It is written for Stability's products. https://stability.ai/guides/stable-audio-3-prompt-guide (OFFICIAL, product-scoped)
- **[S12]** Issue #109, "Seamless loops on small-sfx not possible", 2026-09-30, unanswered. https://github.com/Stability-AI/stable-audio-3/issues/109 (COMMUNITY)
- **[S13]** jonahjameson, "How to install Stable Audio 3.0 Small SFX safetensors, and run it in ComfyUI", jurn.link, 2026-05-22. https://jurn.link/dazposer/index.php/2026/05/22/how-to-install-stable-audio-3-0-small-sfx-safetensors-and-run-it-in-comfyui/ (COMMUNITY)
- **[S14]** Pinokio, "Stable Audio 3 Small: Ultra Fast Music & Sound Effect Generation", undated. https://pinokio.co/posts/01ks47qre2as3e2s92ty7w9fa4 (COMMUNITY)
- **[C1]** `crucible/audio/stable-audio-3-small-sfx.toml`. **[C2]** `docs/AUDIO.md`. **[C3]** `docs/CHANGELOG.md`. **[C4]** `docs/internals/audio.md`. **[C5]** `crucible/jobs/audio/stable_audio_worker.py`. **[C6]** `crucible/playground.py`. (Crucible repo)

## What the model was trained to see

- **Sound effects only.** Stability's compatibility table marks `small-sfx` for "Samples/SFX" only. It is not marked for Music or for Stems/Solo [S1 "Model Compatibility"].
- **Its own training data.** It shares pre-training on AudioSparx plus Freesound with the other small model. For the final stage of pre-training, distillation warmup and post-training, it used "a higher-quality subset of Freesound" [S10 §3.5].
  - Stability split the small models because, at small size, sound-effects data "degrades musical coherence" [S10 §3.5].
  - So the material this model was finished on is Freesound: titles, tags, descriptions and LLM-rewritten captions [S10 §3.5].
- **Same conditioning as Medium.** T5Gemma text, padded or truncated to **256 tokens** [S9][S10 §2.2], plus `seconds_total` normalised over 0–384 s [S9]. Length comes from that number, not from the text.
- **Window.** `sample_size` is 5,292,032 samples, which is 120.0 s at 44.1 kHz [S9]. Small models generate up to 2 minutes [S1][S4].
- **Clean endings.** Silence padding during training (about 4 s on average) teaches the model to end with natural silence [S10 §3.1][S9 `silence_extension_scale_seconds: 4.0`].
- **Empty prompts were trained.** CFG dropout 0.1 replaced text and timing with zeros, so an empty prompt is in-distribution [S9][S10 §3.5].
- **English only.** Trained on English descriptions; not designed for speech or voice [S3].

## Writing the prompt

**Do**

- **Cover three things** [S1 "Audio Samples & SFX"]:
  - The **source**: what object, instrument or synth makes the sound.
  - The **action**: how the sound is triggered and how long it lasts.
  - The **production**: where the mic is, what the room sounds like, how the sound is processed.
- **Go further with the SFX assistant's checklist** [S5 "SFX"]:
  - **Physical character**: texture, timbre and material ("metal, wood, glass, concrete").
  - **Space**: indoor or outdoor; cave, open field or underwater; dry or reverberant; close-up or distant; echoing or muffled.
  - **Time**: attack, decay, movement and transitions.
  - **Motion**: "passing, approaching, stereo movement".
- **Keep it short and dense.** Write one or two dense sentences in concise technical language. Be specific; prioritise clarity and realism; combine elements into one coherent scene; avoid redundancy [S5 "SFX"].
- **Layered sounds and sequences** are explicitly in the assistant's examples:
  - a "Layered soundscape" of rain, thunder, footsteps and distant sirens;
  - a "Rapid sequence of three impacts", spaced evenly;
  - a left-to-right stereo pass;
  - a close-to-far footstep transition [S5 "SFX"].
  - None of these is a measured success rate, only the format Stability writes. See community findings for one user's view on mixing.
- **One-shots for music production.** Name the source, the playing technique or hit type (pluck, slam, tap, stab), the material or timbre, and the space and production (dry or wet, room, close-mic) [S5 "One-shot"]. Recording-chain detail is in-distribution, e.g. "recorded with a small-diaphragm condenser mic" [S5].
- **The prefix.** The guide says `TrackType: SFX` "tends to produce more semantically reasonable sound effects and samples" [S1]. The report recommends `TrackType: SFX,` to all users [S10 §5.1]. Stability's assistant adds `TrackType: SFX, ` (comma and space) to its sfx rewrites [S5].
  - See the conflicts section: for this checkpoint the sources are mixed.
- **Short durations.** "Most sound effects are brief, so set the length accordingly before generating" [S1].

**Don't**

- **Don't ask it for music, stems or instrument loops.** That is `stable-audio-3-medium`'s job [S1 table][C2].
- **Don't ask for speech or intelligible voices** [S3].
- **Don't put the length in the text.** The assistant ends its rewrites with `Length: N seconds`. The UI strips that and moves the number to the duration slider [S5][S6]. In Crucible, send `duration_s`.
- **Don't use vague terms or markup.** The assistant asks for "specific" wording, "no formatting" [S5], and rejects rewrites containing `[ ] * #` or over 45 words [S5 `_has_artifacts`].

## Settings that change how the prompt lands

| Upstream control | Crucible param | Official | Crucible |
|---|---|---|---|
| `prompt` | `prompt` | the text; empty is unconditional [S1][S2] | required, not blank [C2] |
| `duration` / `seconds_total` | `duration_s` | up to 120 s [S4][S9]; set a realistic, short length [S1] | default 10, max 120 [C1] |
| `steps` | `steps` | default 8; more "doesn't necessarily increase quality" except on `-base` [S2] | default 8, 1–50 [C1] |
| `cfg_scale` | not taken | "no effect on post-trained checkpoints" [S2]; training demos at `demo_cfg_scales: [1]` [S9] | refused, `audio_param_unsupported` [C1] |
| `negative_prompt` | not taken | base checkpoints only [S2]; at cfg 1.0 the guided branch never runs [S7 `dit.py`] | refused [C1] |
| sampler | not exposed | `pingpong` default for post-trained models [S7] | package default [C5] |
| `seed` | `seed` | -1 = random [S2] | chosen and reported [C2] |
| init audio, inpaint, continuation | not taken | [S1][S2] | not in Crucible's `audio` job [C2] |

**Lengths the official sources give**

- **SFX assistant** [S5]:
  - very short sounds (impacts, clicks, gunshots): 1–3 s;
  - medium actions (footsteps, object movement, transitions): 3–6 s;
  - ambience and environments: 6–15 s.
  - Its own examples run longer, up to 120 s for a "Dark eerie drone".
- **One-shot assistant:** 1–11 s [S5].
- **Training demo prompts:** 9, 10, 10 and 35 s [S9].
- **What longer clips trade off.** For sound effects the report found FAD improves as length grows, while CLAP (text adherence) falls. It attributes this to long training clips being mostly field recordings and ambience [S10 §5.5].
- **How the window is sized.** The package generates `duration + 6 s` and trims to `duration` [S7]. Separately, training taught the model to end with natural silence [S10 §3.1]. See community findings on loops.

**Small SFX vs Medium for effects.** At 5 s, the report found "large and medium achieve the best overall performance" on sound effects. All SA3 models beat the open baselines [S10 §5.3]. Medium is also an SFX-capable model [S1 table].

## Examples

Official, short and attributed:

- Model card, with `duration=7`: "chugging train coming into station with horn" [S8]. It has no prefix.
- Training demo prompts, 8 steps, cfg 1, no prefix [S9]:
  - "Dog barking next to a waterfall" (10 s);
  - "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style" (10 s).
- Prompting guide: "A classic, natural recording of 14-inch hi-hats played tightly closed. Zero ring and a very fast decay." [S1]
- SFX assistant examples [S5]:
  - "Glass bottle shattering on concrete, sharp impact, scattered fragments. Length: 2 seconds"
  - "Car speeding past at high velocity, doppler effect, realistic whoosh. Length: 3 seconds"
  - "Gunshot in a large empty warehouse, loud echo decay. Length: 2 seconds"
  - The `Length` part goes to `duration_s`, not into the prompt.
- One-shot assistant: "Dry clap, a crisp, natural single hand clap recorded in a dead room with an extremely sharp transient and no room reflections. Length: 1 second" [S5]
- Longer official SFX prompts, such as the desk-drawer thud and the tape-stop sub drop in [S1], and "Ghostship moan" and "Warp drive" in [S5]: read them at the links.

Constructed (not from a source; it follows [S1]'s source/action/production and [S5]'s checklist), sent as `duration_s: 3`:
`TrackType: SFX, A heavy steel fire door slammed shut in a concrete stairwell, sharp metallic impact with a low resonant boom and a long, cold reverberant tail, close perspective.`

## Community findings

- **Silent tail on loops.** Trying to make seamless loops, one user always got "1-2 seconds of silence at the end" from small-sfx. The issue is unanswered [S12]. See open questions for a possible link to the trained silence padding.
- **Mixing sounds.** One ComfyUI user found that Stable Audio Open 1.0 "seems to handle the instruction to 'mix' sounds better", though it takes about 50 times longer [S13]. This is one person's impression, not a measurement. They ran small-sfx at "Steps = 8. CFG = 1.0. Sampler = Pingpong" [S13].
- **Pinokio's checklist.** It suggests describing "the action, material, impact size, duration, and whether the tail should be short or long" [S14]. This is consistent with [S1]. Its "duration" item is better sent as `duration_s` [S6].

## Measured in Crucible

- **Nothing measured for this model.** Its memory figures on both arms are `memory_basis = "declared"`: estimated from the files and Stability's table, not measured through Crucible [C1 `memory_note`].
- **No quality or adherence records.** Nothing on prompt adherence, sound quality or timing for small-sfx is recorded in `docs/` or `CHANGELOG.md` [C2][C3][C4].
- **Long-request fix applies here too.** The worker passes the model's own window (`sample_size`) and refuses audio shorter than requested [C5][C3]. For small-sfx the window equals the package default of 120 s, so the 120 s truncation bug fixed for Medium could not show here [S9][C3].

## Conflicts and open questions

**Is the `TrackType: SFX` prefix right for this checkpoint?** The sources are mixed.

- For it:
  - The guide recommends it for SFX and marks small-sfx as an SFX model [S1].
  - Stability's assistant adds it for sfx [S5].
  - The report says "all" SA3 users should use it [S10 §5.1].
- Against it:
  - The report ties the prefixes to models "trained on the AudioSparx dataset", and small-sfx's last three training stages were Freesound only [S10 §3.5].
  - The model card example and all four training demo prompts for this checkpoint have no prefix [S8][S9].
- Crucible's examples use the prefix [C2][C6]. Whether it helps or hurts on small-sfx is unmeasured.

**Prefix punctuation.** Crucible writes `TrackType: SFX. A heavy oak door …`, with a full stop [C2][C6]. Stability uses `TrackType: SFX, …`, with a comma [S5][S10 §5.1]. Unmeasured.

**Default length.** Crucible defaults `duration_s` to 10 s [C1]. That fits the training demos (9–10 s) [S9] and the assistant's ambience band (6–15 s) [S5]. It is long for impacts, which the assistant puts at 1–3 s [S5]. A client should always send a length; Crucible's own example sends 4 s [C2].

**Empty prompts.** Crucible refuses a blank `prompt` [C2]. Stability suggests trying unconditional generation [S1].

**Wording about cfg.** The manifest says cfg "has no effect" on the post-trained checkpoint, quoting [S2] [C1]. The code would run a guided branch at any `cfg_scale` other than 1.0 [S7]. This is moot while Crucible refuses `cfg`.

**Open questions (inferences, not guidance)**

- **Silent tails.** Is the 1–2 s tail in [S12] the trained "end with natural silence" behaviour [S10 §3.1]? If so, would a seamless loop need trimming by the client? No source says.
- **Layering.** Does layering several sources in one prompt land better than separate renders mixed by the client? Only an impression exists [S13].
- **Stereo direction.** Is stereo motion ("left to right") reliably followed? It appears in Stability's examples [S5], but no adherence figure is published.
- **Mac arm.** Does the MPS float32 arm sound the same as CUDA float16? Not measured [C4].
