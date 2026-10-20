# LTX-2.3 Dub-It: how to prompt it

Crucible id `ltx-2.3-dubit`, Mac only (mlx-darwin). It is `dgrauet/ltx-2.3-mlx-q8` @ `6671a7572a530862d1d60ce393b5d93491e3f76b` with Lightricks' Dub-It IC-LoRA (`Lightricks/LTX-2.3-22b-IC-LoRA-DubIt` @ `1456334c3d69924de5083e553733b108ed1147f2`, `ltx-2.3-22b-ic-lora-dubit-0.9.safetensors`) fused into the transformer, and `mlx-community/gemma-3-12b-it-4bit` @ `86cc6a8dedbc456dd0e4af01a9d09f396f77e558` as its text encoder. Job type `video`, with one input: the reference video. Beta, as Lightricks labels it.

## Sources

All web sources were read on 2026-10-19.

- **[S1]** LTX docs, "Dub-It (IC-LoRA) Beta". https://docs.ltx.io/open-source-model/advanced-workflows/lip-dub-beta. OFFICIAL. The page shows no date.
- **[S2]** The Dub-It model card, https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-DubIt @ `1456334c`. OFFICIAL. Gated with automatic approval. It points at the JUST-DUB-IT paper (arXiv 2601.22143) and the upstream pipeline `packages/ltx-pipelines/src/ltx_pipelines/lipdub.py` in https://github.com/Lightricks/LTX-2, and says the reference downscale factor is 1.
- **[S3]** `ltx_pipelines_mlx/lipdub.py` in https://github.com/dgrauet/ltx-2-mlx at `1724ca673d59f023a8a95efee06e5d36d61c2765`, the commit Crucible's video env pins. COMMUNITY port, a mirror of [S2]'s upstream pipeline.
- **[C1]** Crucible `docs/VIDEO.md`, "Lip-dub", and the manifest `crucible/video/ltx-2.3-dubit.toml`.
- **[C2]** One run of [S3]'s own CLI on the M1 Ultra, 2026-10-19 (the measurement in [C1]'s manifest note).

## What the model does

- It takes a video of one person talking and re-makes it so the person says the line in the prompt, keeping the picture outside the mouth. The voice is generated, and it imitates the reference's speaker from the reference's own sound [S1][S3]. It takes no audio of the new line: the model speaks it.
- **It does not translate.** It says the words the prompt gives it, in the language they are written in. Write the dialogue in that language's own script (Cyrillic for Russian) [S1].
- **Languages validated by Lightricks:** English, French, Spanish, German, Russian [S1]. Others are untried.
- **One speaker.** The beta LoRA does not tell speakers apart [S1].
- **LTX-2.3 only.** It was validated on LTX-2.3, not on 2.5 [S1]; Crucible runs it on the 2.3 pack, not on `ltx-2.5-distilled`.
- **Two passes, Euler, CFG 1, 8 then 3 steps**, the second at full size on the upscaled latent [S1]. These are fixed in Crucible (`steps` other than 8 is refused), and with CFG 1 there is no negative prompt [C1].

## Writing the prompt

**Do**

- Follow Lightricks' template: `[Speaker] is speaking [Language/Accent], saying: "[Dialogue]"` [S1]. Their examples: `A woman speaking in French saying: "Bonjour, je teste les workflows de doublage avec LTX"` [S1].
- Put the whole new line in quotation marks. The model follows the prompt's content [S1].
- Add emotion or delivery if it matters ("warmly", "in a relaxed, friendly voice") [S1].
- Match the new line to the reference's length: about the same timing and syllable count as what the person says in the reference. "Slightly longer is better than too short" [S1].
- Describe the speaker so the prompt and the picture agree (the person, what they wear, where they sit). The 2026-10-19 run used one sentence of that before the quoted line [C2].

**Avoid**

- A line much longer than the reference: "the model might skip words" [S1].
- A line much shorter: "the output might sound slow and unnatural" [S1].
- Two speakers, or dialogue in a language the reference speaker should "translate" into without writing it out [S1].

## In Crucible

- The clip is as long as the reference, rounded down to 8k+1 frames, at the reference's own frame rate (29.97 stays 30000/1001) [S3][C1]. Trim the reference to the part you want dubbed; a reference past the model's limit is refused, never cut [C1].
- `reference_strength` (above 0, at most 1, default 1.0) is the IC-LoRA reference's conditioning strength [S1][C1]. Lightricks gives no guidance on lower values.
- The sound's loudness follows the reference recording: a quiet raw microphone track gives a quiet clip (−28.4 dB mean on 2026-10-19) [C2]. Crucible does not change it and reports the level in `done.video.reference.audio_level_dbfs` [C1].
- Lightricks suggests cropping to the face to save compute or go higher in resolution [S1]. Crucible's frame ceiling is the measured 768x448 (344,064 pixels) [C1].

## Example

The 2026-10-19 measuring run [C2], from a 6.5 s reference of the speaker at his desk:

> A man with short dark hair, black rectangular glasses and a goatee sits in a black leather chair in front of a black podcast microphone, a fireplace glowing behind him, and speaks directly to the camera in a warm, relaxed, friendly voice: "Hey, if you like what I do, follow me on Patreon, at owenmorgan dot com slash patreon."

## Open questions

- How `reference_strength` below 1 trades the picture's faithfulness against the mouth's freedom: not documented [S1] and not measured here.
- Whether reading a URL or a number aloud needs it spelled as words ("dot com"): the example above did, and it was not compared.
