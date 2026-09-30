# Video generation: the `video` job

Words in, a video clip with its own sound out; or a picture and words in, the picture brought to
life. One job type, `video`, with one model, `ltx-2.5-distilled` (Lightricks' LTX-2.5, the
distilled checkpoint, 8 steps). How it runs on a 24 GB card and why every limit is what it is
is [internals/video.md](internals/video.md); this page is what a caller sends, what comes back,
and how to write a prompt.

| model | makes | class | PC (cuda-linux) | Mac (mlx-darwin) | longest | licence |
| --- | --- | --- | --- | --- | --- | --- |
| `ltx-2.5-distilled` | video with synchronized sound (speech, effects, ambience, music) from a prompt, or from a prompt and a start picture | `video` | yes | **no** | 6.04 s at 1280x704 (text), 3.04 s at 1280x704 (from a picture) | LTX-2.x Community License (gated) |

`GET /v1/capability` has a `video` row: "can make video, using ltx-2.5-distilled" on a CUDA card
with room for its largest stage, and "cannot make video" on a Mac or a smaller card, with the
reason.

**Not on the Mac.** The model is a 22B-parameter transformer that fits one 24 GB NVIDIA card
only quantized and one component at a time; there is no Apple arm. A Mac asked for it answers
`400 backend_unsupported` naming `cuda-linux`; send video jobs to the PC.

## Turning it on

```bash
crucible init --enable-video          # or [jobs] enable_video = true
crucible install video                # builds the LTX env and places Crucible's ffmpeg
crucible models pull ltx-2.5-distilled
```

None of these is needed by hand: a job for a missing env or model starts the install and answers
`409 installing`, like every other type. The pull is about 50.7 GB. The one step Crucible cannot
take for you is accepting the licence.

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
| `width`, `height` | 1280 x 704 | multiples of 32, each side 256 to 1280, at most 901,120 pixels; send both or neither |
| `duration_s` | 5 | seconds; rounded to the model's frame grid (8k+1 frames): 5 s at 24 fps is 121 frames, 5.04 s |
| `num_frames` | from `duration_s` | the exact count instead of `duration_s` (not both); must be 8k+1, for example 49, 97, 121, 145 |
| `fps` | 24 | 24 or 25 |
| `seed` | chosen and reported | 0 to 2^32-1; the same seed and params give the same clip |
| `steps` | 8 | the distilled checkpoint runs its own fixed 8-step schedule; any other value is refused |
| `audio` | `true` | `false` makes a silent clip (the model still generates sound with the picture; it is not decoded) |
| `lease` | none | keep the model resident across a batch (below) |

`negative_prompt` is refused by name: the distilled checkpoint runs without guidance, so it
would never be read.

### The limits

Every limit is refused by name before anything loads, and each is the size the model's memory
was sized at (declared, not yet measured on the PC):

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

(The timings above are placeholders, not measurements.) Progress events name the stage
(`encoding`, `connecting`, `conditioning` for a start picture, `denoising` with steps 1 to 8,
`decoding`, `audio_decoding`, `muxing`), and a cancel lands between denoising steps.

## Many clips in a row

Every clip loads the text encoder, the connectors and the transformer from disk one after
another (about 50 GB of reads). A batch should hold the model with a lease, like `image`:

```json
{"type": "load-video", "model": "ltx-2.5-distilled", "params": {"lease": {"act": "video", "ttl_seconds": 600}}}
```

then send each `video` job with the same `"lease": {"act": "video", "ttl_seconds": 600}`. Under a
lease the worker stays up between clips, and a prompt it has already read skips the text
stages: the second clip of the same prompt (a new seed, say) starts at denoising.
`unload-video` or letting the lease run out frees the card.

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
that any provenance or watermark the model applies be kept. The GGUF quantization is
redistributed under the same licence. Details in [internals/video.md](internals/video.md),
"Licence".
