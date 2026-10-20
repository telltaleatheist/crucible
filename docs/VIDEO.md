# Video generation: the `video` job

Words in, a video clip with its own sound out; or a picture and words in, the picture brought to
life; or, on a Mac, a soundtrack in and a picture made to it (a face speaking the line, lips in
sync), from words alone or from a start picture. One job type, `video`, with one model, `ltx-2.5-distilled` (Lightricks' LTX-2.5, the
distilled checkpoint, 8 steps). How it runs on a 24 GB card and on a Mac, and why every limit
is what it is, is [internals/video.md](internals/video.md); this page is what a caller sends,
what comes back, and how to write a prompt.

| model | makes | class | PC (cuda-linux) | Mac (mlx-darwin) | longest | licence |
| --- | --- | --- | --- | --- | --- | --- |
| `ltx-2.5-distilled` | video with synchronized sound (speech, effects, ambience, music) from a prompt, or from a prompt and a start picture; on the Mac also a picture made to a supplied soundtrack | `video` | yes (24 GB card) | yes (48 GB or more) | 6.04 s at 1280x704 from text on both; from a picture 3.04 s on the PC, 5.04 s on the Mac | LTX-2.x Community License (gated) |
| `ltx-2.3-dubit` | a lip-dub (beta): a video of one person talking, re-made so they say a new line in a voice like theirs; everything but the mouth kept | not a verb's pick (it needs a reference) | no (refused `backend_unsupported`) | yes (64 GB) | 193 frames at 768x448 (6.44 s at 29.97 fps) | LTX-2 Community License; the LoRA gated |

`GET /v1/capability` has a `video` row: "can make video, using ltx-2.5-distilled" on a CUDA card
with room for its largest stage and on a Mac with 48 GB or more of memory, and "cannot make
video" on a smaller card or a smaller Mac, with the reason.

**The same job on either machine.** The PC runs the model through diffusers, quantized and one
component on the card at a time. The Mac runs Lightricks' own distilled recipe through
ltx-2-mlx, a pure-MLX port: 8 steps at half the size, a 2x latent upscale, then 3 steps at full
size. The params, the artifact and the `done.video` fields are the same; what differs is below,
under "The Mac". A seed reproduces a clip on the machine that made it, not across the two.

## Turning it on

```bash
crucible init --enable-video          # or [jobs] enable_video = true
crucible install video                # builds the LTX env and places Crucible's ffmpeg
crucible models pull ltx-2.5-distilled
```

None of these is needed by hand: a job for a missing env or model starts the install and answers
`409 installing`, like every other type. The pull is about 50.7 GB on the PC and 43.5 GB on the
Mac. The one step Crucible cannot take for you is accepting the licence.

### The model is gated

Hugging Face serves `Lightricks/LTX-2.5-Diffusers` only to an account that has accepted the
LTX-2.x Community License (and Lightricks' privacy policy). Until then a job or a pull is refused
`409 model_gated`, and the refusal says exactly what to do:

1. Signed in to Hugging Face, open https://huggingface.co/Lightricks/LTX-2.5-Diffusers and press
   "Agree and Access" (acceptance is immediate).
2. Make a read token at https://huggingface.co/settings/tokens and give it to the server: set
   `HF_TOKEN` in the environment Crucible runs in, or put it under `[hf] token` in the config
   file the refusal names.
3. Run `crucible models pull ltx-2.5-distilled` again, or resend the job.

The transformer itself comes from a second repo, `Abiray/LTX-2.5-Distilled-GGUF` (not gated),
which the same pull fetches and checks by sha256.

On the Mac the repo is `dgrauet/ltx-2.5-mlx-q8` (the same weights converted to MLX at int8,
under the same licence), gated with automatic approval: open
https://huggingface.co/dgrauet/ltx-2.5-mlx-q8, accept, and the same token works. The refusal
names that page.

## The request

Text to video:

```json
{"type": "video",
 "model": "ltx-2.5-distilled",
 "params": {"prompt": "A red fox trots through fresh snow at dawn, the camera tracking alongside at knee height. Low golden light rakes across the drifts, the fox's breath steams, and each step crunches; a crow calls twice from the pines behind.",
            "width": 1280, "height": 704, "duration_s": 5}}
```

From a start picture: the same, with exactly one input (PNG, JPEG or WebP). The picture is the
first frame, cropped to the clip's shape (never stretched) and resized:

```json
{"type": "video",
 "model": "ltx-2.5-distilled",
 "params": {"prompt": "The woman in the photo turns toward the window and smiles as rain starts to tap on the glass; soft daylight, a slow push in, the patter of rain and a distant car passing.",
            "width": 1280, "height": 704, "duration_s": 3},
 "inputs": {"start.png": {"inline_base64": "..."}}}
```

To a soundtrack (the Mac only): send the audio as an input (WAV, FLAC, MP3 or Ogg) and the
picture is made to it, with the sound held exactly as sent; the clip's sound is your file, not
generated. Add a start picture for a talking clip of a given face. Inputs are told apart by their
bytes, so their names are yours; at most one picture and one soundtrack:

```json
{"type": "video",
 "model": "ltx-2.5-distilled",
 "params": {"prompt": "A close-up of the woman in the photo, looking into the camera in a warm kitchen, soft window light. She speaks calmly and smiles at the end: \"I kept every letter you sent.\"",
            "width": 768, "height": 512, "seed": 7},
 "inputs": {"face.png": {"inline_base64": "..."},
            "line.wav": {"inline_base64": "..."}}}
```

With no `duration_s` or `num_frames` the clip is as long as the soundtrack, rounded down onto the
8k+1 frame grid (a 3.0 s line at 24 fps is 65 frames, 2.71 s). Ask for a shorter clip and it is
made to the soundtrack's first part; `done.video.soundtrack` says how much was used. A clip
longer than the soundtrack is refused, never padded or looped. Results vary a lot with the seed:
try a few.

| param | default | what it does |
| --- | --- | --- |
| `prompt` | required | the shot, the motion, the light and the sound, in one paragraph (below) |
| `width`, `height` | 1280 x 704 | multiples of 32 (64 on the Mac), each side 256 to 1280, at most 901,120 pixels; send both or neither |
| `duration_s` | 5, or the soundtrack's length | seconds; rounded to the model's frame grid (8k+1 frames): 5 s at 24 fps is 121 frames, 5.04 s |
| `num_frames` | from `duration_s` | the exact count instead of `duration_s` (not both); must be 8k+1, for example 49, 97, 121, 145 |
| `fps` | 24 | 24 or 25 |
| `seed` | chosen and reported | 0 to 2^32-1; the same seed and params give the same clip |
| `steps` | 8 | the distilled checkpoint runs its own fixed 8-step schedule; any other value is refused (the Mac then refines in 3 more at full size, reported as `refine_steps`) |
| `audio` | `true` | `false` makes a silent clip (the model still generates sound with the picture; it is not decoded); refused with a soundtrack |

`negative_prompt` is refused by name: the distilled checkpoint runs without guidance, so it
would never be read.

### The limits

Every limit is refused by name before anything loads, and each is the size the model's memory
was sized at (declared, not yet measured). The PC's figures; the Mac's differences follow the
list:

- `video_size_not_supported`: a side not a multiple of 32, or under 256.
- `video_too_large`: more than 901,120 pixels or a side over 1280; or more than 16,720 video
  tokens for text-to-video, 8,800 for image-to-video. A video token is
  `((frames - 1) / 8 + 1) x (width / 32) x (height / 32)`; the refusal names the longest clip
  that fits at the size you asked for. At 1280x704: 145 frames (6.04 s) from text, 73 frames
  (3.04 s) from a picture. At 960x544 a picture can run 129 frames (5.4 s).
- `video_frames_not_supported`: `num_frames` not 8k+1 (the refusal names the two nearest).
- `video_too_long`: more than 145 frames.
- `video_param_unsupported`: `negative_prompt`, a `steps` other than 8, an `fps` other than 24
  or 25; `audio: false` with a soundtrack.
- `invalid_inputs`: an input that is neither a PNG, JPEG or WebP picture nor a WAV, FLAC, MP3 or
  Ogg soundtrack; two pictures or two soundtracks; a soundtrack ffprobe cannot read.
- `audio_to_video_unsupported`: a soundtrack sent to the PC, whose arm generates the sound with
  the picture and cannot hold one fixed. Send it to a Mac.
- `video_soundtrack_too_short`: a clip longer than its soundtrack (the refusal names the longest
  clip it covers), or a soundtrack under the shortest clip, 9 frames.
- `video_too_long` also covers a soundtrack longer than the arm makes, when the length came from
  it; send `duration_s` or `num_frames` for a clip of its first part.
- `ffmpeg_missing`: the server has no ffmpeg, or (for a soundtrack) no ffprobe; every
  `crucible install` places both.

### The Mac

- Width and height are multiples of **64**, not 32: the first pass runs at half the size, and
  half must still sit on the VAE's 32-pixel grid. 1280x704, 768x512 and 512x512 are fine;
  1280x720 and 768x544 are refused `video_size_not_supported`.
- From a picture the ceiling is **14,080** video tokens: 1280x704 x 121 frames (5.04 s), where
  the PC stops at 73 frames.
- From text the Mac runs **up to 505 frames and 56,320 video tokens**: 1280x704 for 21 s at
  24 fps (20.2 s at 25). That is because it tiles the full-size pass by default (below), which
  keeps that pass's memory to one tile whatever the clip's length; a 20 s 1280x704 clip
  measured 21.7 GB at its peak (the text encoder) on 2026-09-30. With tiling turned off it is
  the PC's 16,720 (145 frames, 6.04 s), which the declared memory covers untiled.
- It needs 29 GB for its largest stage on top of the 16 GB desktop allowance: a Mac with 48 GB
  or more. A smaller one answers `409 insufficient_memory` naming 29,000,000,000 bytes.
- There is no `conditioning` stage (the picture is encoded while the transformer loads) and
  there is a `refining` stage with steps 1 to 3 after `denoising`.
- It takes a soundtrack. The soundtrack is encoded in its own `audio_encoding` stage after the
  text, both passes hold it fixed and denoise only the picture (Lightricks' audio-to-video,
  frozen audio, plain Euler in both passes), and there is no `audio_decoding`: the clip's sound
  is the soundtrack, cut to the clip. A soundtrack alone has the text-to-video ceiling; with a
  start picture, the image-to-video one (14,080 tokens). Mode `audio-to-video`, or
  `image-audio-to-video` with a picture.
- The Mac keeps its desktop responsive while it renders, by default: the full-size pass is
  tiled, the transformer streams from disk, GPU work goes in small batches, and the GPU is
  left idle for part of every step so the screen can draw. Renders are slower for it.
  `done.video.desktop` says how each clip was split, and `done.video.gpu_busy_*` how busy the
  GPU was. `[video_desktop] enabled = false` in the Mac's config runs it flat out, with the
  shorter limit above. See [internals/video.md](internals/video.md), "Keeping the desktop
  responsive".

## The result

One artifact, `video.mp4`: H.264 video (yuv420p) with AAC stereo sound at 48 kHz, `+faststart`,
so a browser plays it as it downloads. `done.video` carries every effective parameter:

```json
{"model": "ltx-2.5-distilled",
 "hf_repo": "Lightricks/LTX-2.5-Diffusers", "revision": "426936f8…",
 "transformer": {"hf_repo": "Abiray/LTX-2.5-Distilled-GGUF", "revision": "7b0c2025…",
                 "file": "LTX-2.5-Distilled-Q6_K.gguf", "sha256": "ee8835ff…"},
 "backend": "cuda-linux", "engine": "ltx", "dtype": "bfloat16",
 "quantization": {"text_encoder": "torchao int8 weight-only (per row), bfloat16 activations",
                  "transformer": "GGUF Q6_K, dequantized per layer to bfloat16"},
 "mode": "text-to-video", "prompt": "…", "input": null,
 "width": 1280, "height": 704, "num_frames": 121, "fps": 24, "duration_s": 5.042,
 "video_tokens": 14080, "seed": 1234, "steps": 8,
 "audio": true, "audio_seconds": 5.04, "audio_sample_rate": 48000, "audio_channels": 2,
 "artifact": "video.mp4", "bytes": 4812345, "encoder": "libopenh264",
 "seconds": 190.4,
 "stage_seconds": {"encoding": 40.1, "connecting": 12.0, "denoising": 95.2, "decoding": 30.3,
                   "audio_decoding": 1.2, "muxing": 4.1},
 "stage_peak_bytes": {"encoding": …, "connecting": …, "denoising": …, "decoding": …, "audio_decoding": …},
 "peak_bytes": …, "memory_bytes_estimate": 20500000000, "memory_basis": "declared",
 "stage_memory_bytes": {"encoding": 16000000000, "…": "…"},
 "prompt_cache": "miss", "versions": {"diffusers": "…", "torch": "2.14.0", "…": "…"}}
```

With a soundtrack, `mode` is `audio-to-video` (or `image-audio-to-video`), the `audio_*` fields
describe the soundtrack as muxed, and `soundtrack` says what was done with it:

```json
"soundtrack": {"input": "line.wav", "format": "WAV", "total_s": 3.0, "used_s": 2.708,
               "unused_s": 0.292, "covers": "its first 2.708 s", "sound": "supplied",
               "sample_rate": 48000, "channels": 2, "audio_tokens": 68}
```

Without one it is null. (The timings above are placeholders, not measurements.) `refine_steps` is null and `sampling`
names the one full-size pass. Progress events name the stage (`encoding`, `connecting`,
`conditioning` for a start picture, `denoising` with steps 1 to 8, `decoding`,
`audio_decoding`, `muxing`), and a cancel lands between denoising steps.

A Mac clip's `done.video` has the same fields with these values: `hf_repo`
`dgrauet/ltx-2.5-mlx-q8` and its revision, `transformer` null (the transformer is in that repo),
`backend` `mlx-darwin`, `engine` `ltx-2-mlx`, `quantization` MLX int8 for the text encoder and
the transformer, `refine_steps` 3, `sampling` with a half-size pass (8 steps, Euler ancestral)
and a full-size one (3 steps, Euler, after the 2x upscale), `encoder` `h264_videotoolbox`,
`stage_peak_bytes` from MLX's own peak with a `refining` entry and no `connecting` or
`conditioning`, and `memory_bytes_estimate` 29,000,000,000. Its progress runs `encoding`,
`denoising` 1 to 8, `refining` 1 to 3, `decoding`, `audio_decoding`, `muxing`; a cancel lands
between steps of either pass.

## Lip-dub: `ltx-2.3-dubit` (Mac, beta)

Lightricks' Dub-It IC-LoRA on LTX-2.3. Send a video of one person talking (with its sound) and a
prompt that quotes the new line; the clip is the same video with the person saying that line,
in a voice the model makes to sound like theirs from the reference's own sound. It takes no
audio of the line, it does not translate (write the line in the language it is to be spoken
in), and Lightricks has validated English, French, Spanish, German and Russian. How to word the
prompt: `GET /v1/prompting/ltx-2.3-dubit.md`.

```bash
crucible models pull ltx-2.3-dubit     # the Mac only; about 41 GB
```

The pull fetches three repos: the LTX-2.3 MLX int8 pack (`dgrauet/ltx-2.3-mlx-q8`, only the
files the lip-dub pipeline reads), the IC-LoRA (`Lightricks/LTX-2.3-22b-IC-LoRA-DubIt`, gated
with automatic approval: accept it at its page and give the server an HF token, as above) and
the Gemma 3 12B 4-bit text encoder (`mlx-community/gemma-3-12b-it-4bit`). The last two land in
`ic-lora/` and `text-encoder/` beside the pack and are checked by sha256; a file already in
place with the pinned hash is kept, not fetched again.

```json
{"type": "video", "model": "ltx-2.3-dubit",
 "params": {"prompt": "A man in glasses sits in a leather chair and speaks to the camera in a warm, relaxed voice: \"Hey, if you like what I do, follow me on Patreon.\"",
            "seed": 42},
 "inputs": {"talk.mp4": {"blob_id": "…"}}}
```

- **The one input** is the reference: an MP4, MOV, WebM or Matroska file, told apart by its
  bytes (names are free). Anything else beside it, or no reference at all, is refused
  `invalid_inputs`. A reference with no picture is refused `video_reference_no_video`, one with
  no sound `video_reference_no_audio` (the voice comes from it). `ltx-2.5-distilled` refuses a
  reference video `invalid_inputs`.
- **The length is the reference's**: its frame count (as ffprobe reads it) rounded down to 8k+1,
  at its own frame rate, kept exactly (29.97 stays 30000/1001 in the MP4). At most **193
  frames** (6.44 s at 29.97 fps, 8.04 s at 24), the length measured; a longer reference is
  refused `video_reference_too_long` naming how long it may be. It is never cut: trim it to the
  part you want dubbed. Under 9 frames is `video_reference_too_short`. `duration_s`,
  `num_frames`, `fps` and `audio: false` are refused `video_param_unsupported`.
- **The frame** is `width`/`height` in multiples of 64 up to 768 a side and 344,064 pixels (the
  measured 768x448); left out, it is the largest frame within 5% of the reference's shape:
  768x448 for 16:9, 448x768 for 9:16, 576x448 for 4:3, 576x576 square.
- **`reference_strength`**, above 0 and at most 1, default 1.0 (Lightricks' recipe): how strongly
  the reference holds the picture. Every other model refuses it.
- **Steps are fixed**: 8 at half size, then 3 at full size, Euler, CFG 1 (so no
  `negative_prompt`). `steps` other than 8 is refused.
- The transformer always streams from disk block by block, whatever `[video_desktop] low_ram`
  says: unstreamed, the measuring run reached a 67 GB footprint and swapped the 64 GB Mac.
  The rest of `[video_desktop]` (the duty cycle, the MLX buffer limits) applies as for
  `ltx-2.5-distilled`; its tiling does not (the lip-dub passes are not tiled). It declares
  35 GB on top of the 16 GB desktop allowance.

`done.video` has `mode` `lip-dub`, `frame_rate` (`"30000/1001"`), `fps` as a number, `transformer`
null, and two blocks the other modes have as null:

```json
"reference": {"input": "talk.mp4", "format": "MP4", "width": 1920, "height": 1080,
              "duration_s": 6.54, "frame_rate": "30000/1001", "fps": 29.97, "frames": 195,
              "frames_used": 193, "frames_unused": 2, "strength": 1.0,
              "audio_level_dbfs": {"mean": -28.4, "peak": -10.9}},
"lora": {"hf_repo": "Lightricks/LTX-2.3-22b-IC-LoRA-DubIt", "revision": "1456334c…",
         "file": "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors", "sha256": "fc415b12…",
         "strength": 1.0}
```

The sound is the model's, decoded through the audio VAE and vocoder at 48 kHz stereo, and its
loudness follows the reference recording (a raw microphone track at −28 dB gives a clip at
−28 dB). It is not normalized; `audio_level_dbfs` says how loud it came out. Progress runs
`encoding` (the prompt, the reference's sound, the transformer and the LoRA), `denoising` 1 to 8,
`refining` 1 to 3, `decoding`, `audio_decoding`, `muxing`, with a peak for each in
`stage_peak_bytes`; a cancel lands between steps.

## Many clips in a row

Every clip loads the text encoder, the connectors and the transformer from disk one after
another (about 50 GB of reads on the PC, 37 GB on the Mac). A batch should hold the model with a
queue session, like `image` (`POST /v1/queue/sessions` with `{"act": "video"}`; see
[QUEUE.md](QUEUE.md)), optionally warming up with:

```json
{"type": "load-video", "model": "ltx-2.5-distilled"}
```

then send each `video` job as usual. Inside the session the worker stays up between clips, and a
prompt it has already read skips the text stages: the second clip of the same prompt (a new
seed, say) starts at denoising. Closing the session (or `unload-video`) frees the card.

## Writing a prompt

How to word a prompt for LTX-2.5, with a start picture and without, is in its guide:
`GET /v1/prompting/ltx-2.5-distilled.md` (`crucible/prompting/ltx-2.5-distilled.md`).


## Licences

The LTX-2.x Community License Agreement (licence date 2026-08-11). Free, commercial and
production use included, for an entity whose annual revenue (counted with its affiliates) is
under US$10 million; above that a paid licence from Lightricks is needed except for narrow
non-commercial uses. Lightricks claims no rights in the videos you make, and the licence asks
that any provenance or watermark the model applies be kept. The GGUF quantization (PC) and the
MLX int8 conversion (Mac) are redistributed under the same licence; ltx-2-mlx, the library the
Mac runs them with, is MIT. Details in [internals/video.md](internals/video.md),
"Licence".
