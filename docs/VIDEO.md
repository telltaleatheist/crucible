# Video generation: the `video` job

Words in, a video clip with its own sound out; or a picture and words in, the picture brought to
life. One job type, `video`, with one model, `ltx-2.5-distilled` (Lightricks' LTX-2.5, the
distilled checkpoint, 8 steps). How it runs on a 24 GB card and on a Mac, and why every limit
is what it is, is [internals/video.md](internals/video.md); this page is what a caller sends,
what comes back, and how to write a prompt.

| model | makes | class | PC (cuda-linux) | Mac (mlx-darwin) | longest | licence |
| --- | --- | --- | --- | --- | --- | --- |
| `ltx-2.5-distilled` | video with synchronized sound (speech, effects, ambience, music) from a prompt, or from a prompt and a start picture | `video` | yes (24 GB card) | yes (48 GB or more) | 6.04 s at 1280x704 from text on both; from a picture 3.04 s on the PC, 5.04 s on the Mac | LTX-2.x Community License (gated) |

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

| param | default | what it does |
| --- | --- | --- |
| `prompt` | required | the shot, the motion, the light and the sound, in one paragraph (below) |
| `width`, `height` | 1280 x 704 | multiples of 32 (64 on the Mac), each side 256 to 1280, at most 901,120 pixels; send both or neither |
| `duration_s` | 5 | seconds; rounded to the model's frame grid (8k+1 frames): 5 s at 24 fps is 121 frames, 5.04 s |
| `num_frames` | from `duration_s` | the exact count instead of `duration_s` (not both); must be 8k+1, for example 49, 97, 121, 145 |
| `fps` | 24 | 24 or 25 |
| `seed` | chosen and reported | 0 to 2^32-1; the same seed and params give the same clip |
| `steps` | 8 | the distilled checkpoint runs its own fixed 8-step schedule; any other value is refused (the Mac then refines in 3 more at full size, reported as `refine_steps`) |
| `audio` | `true` | `false` makes a silent clip (the model still generates sound with the picture; it is not decoded) |

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
  or 25.
- `ffmpeg_missing`: the server has no ffmpeg (every `crucible install` places one).

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

(The timings above are placeholders, not measurements.) `refine_steps` is null and `sampling`
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

From the Diffusers model card: LTX-2.5 "was trained on long, single-paragraph audio-visual
captions and degrades on short prompts. Describe the shot, the motion, the light and the sound
in one paragraph." So:

- one paragraph of plain sentences, the way a shot list reads, not a list of tags;
- what is in the frame and what moves, and how the camera moves (tracking, a slow push in,
  handheld, a static wide shot);
- the light and the look (dawn, overcast, neon, film grain);
- the sound, as concretely as the picture: speech in quotation marks with who says it, the
  effects that go with the motion, the ambience, any music;
- for a start picture, describe what happens next rather than re-describing the picture.

## Licences

The LTX-2.x Community License Agreement (licence date 2026-08-11). Free, commercial and
production use included, for an entity whose annual revenue (counted with its affiliates) is
under US$10 million; above that a paid licence from Lightricks is needed except for narrow
non-commercial uses. Lightricks claims no rights in the videos you make, and the licence asks
that any provenance or watermark the model applies be kept. The GGUF quantization (PC) and the
MLX int8 conversion (Mac) are redistributed under the same licence; ltx-2-mlx, the library the
Mac runs them with, is MIT. Details in [internals/video.md](internals/video.md),
"Licence".
