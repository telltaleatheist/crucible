# Image generation: the `image` job

One prompt in, one PNG out, on the Mac (mflux, MLX) and on a CUDA card (diffusers). The model
is `qwen-image-2.1` (Qwen-Image 2.1, bf16) on both. How it runs and why is
[internals/image.md](internals/image.md); this page is what a caller sends, what comes back,
and how to write a prompt that works.

## Turning it on

```bash
crucible init --enable-image        # or [jobs] enable_image = true in config.toml
crucible install image              # builds the image env and records whether this host can hold it
crucible models pull qwen-image-2.1 # ~33 GB; on a Mac that already has it in ~/.cache/huggingface, linked, not copied
```

None of these is needed by hand: a job submitted for an env or model that is missing starts the
install and answers `409 installing` (the same install-on-submit every other type has), and
`GET /v1/capability` has an `image` row saying whether this host can make images and with which
model.

## The request

```json
{"type": "image",
 "model": "qwen-image-2.1",
 "params": {"prompt": "An ordinary documentary photograph of a kitchen table with one red apple on it, low angle, natural window light, muted colours, film grain. No text, no letters, no numbers, no logos.",
            "width": 1280, "height": 720, "seed": 1, "steps": 40}}
```

| param | default | rule |
| --- | --- | --- |
| `prompt` | required | not blank |
| `width`, `height` | 1024 | 256 to 2048, a multiple of 16; the CUDA arm needs a multiple of 32 (`image_size_not_supported` names it); width x height at most 1,048,576 pixels (`image_too_large`), the size the memory was sized at |
| `seed` | chosen by the server and reported | 0 to 4294967295; the same seed, prompt and params on the same backend give the same picture |
| `steps` | 40 | 1 to 100; fewer is faster and rougher |
| `guidance` | 1.0 | 1.0 to 10.0; above 1.0 runs true classifier-free guidance (two passes per step, twice the time) and needs `negative_prompt` |
| `negative_prompt` | none | only with `guidance` above 1.0 |
| `image_strength` | none | 0 to 1 exclusive, image-to-image: send exactly one input image (PNG, JPEG or WebP) and this; higher keeps more of the input. Both arms: the input is stretched to width x height and denoising starts at step `max(1, int(steps * image_strength))`. Useful range 0.03 to 0.3; see Image-to-image below. With `mask` it is optional and applies to the masked region only |
| `mask` | none | inpainting and outpainting: the name of the input that carries the mask, for example `"mask.png"`. The job then carries exactly two inputs, the image and the mask; see Inpainting and outpainting below |
| `mask_blur` | 8 when `mask` is sent | 0 to 256 pixels, only with `mask` (`invalid_params` without one): how far inside the mask's edge the new picture fades into the kept one. Nothing outside the mask ever changes |

Unknown params are refused, never ignored. Every refusal names the param and what to send instead.

## Image-to-image

Send one picture as an input and `image_strength` with a new prompt, and the model redraws
the picture toward the prompt. The whole picture is redrawn. To redraw one region, or to extend
the picture past its edges, send a mask as well (Inpainting and outpainting, below). The model
does not take instructions like "remove the lamp": describe the picture you want instead.

```json
{"type": "image",
 "model": "qwen-image-2.1",
 "params": {"prompt": "A cozy log cabin beside a frozen mountain lake on a snowy winter night, snow on the pine trees, warm light in the windows, stars, photographic. No text.",
            "width": 768, "height": 768, "steps": 30, "seed": 11, "image_strength": 0.1},
 "inputs": {"start.png": {"inline_base64": "…"}}}
```

A higher `image_strength` keeps more of the input. The useful range is narrow and near the
bottom, because the schedule is front-loaded: most of what a picture becomes is decided in the
first few steps. Measured on both machines (2026-09-29), a sunny summer cabin sent with the
winter-night prompt above, 30 steps, 768x768:

| `image_strength` | what came back |
| --- | --- |
| 0.05 | a snowy night: new sky, lit windows, snow everywhere; the cabin, trees and framing roughly where they were |
| 0.3 | the original with a few specks of snow |
| 0.6 | the original, near enough pixel for pixel |

- **A real change** (new season, time of day, style): 0.03 to 0.15.
- **A touch-up** (texture, colour, small details): 0.15 to 0.3.
- **Above 0.3** comes back close to a copy.

Put the whole scene in the prompt, not only what should change. Keep the width and height at
the input's aspect ratio: the input is stretched to fit. The PC and the Mac agree on how
strength behaves, but, as with text-to-image, one seed gives different pictures on each.

## Inpainting and outpainting

Send a picture, a mask and a prompt, and only the region the mask marks is regenerated. Every
pixel outside the mask comes back exactly as it was sent. This works like Photoshop's Generative
Fill: select a region, then describe the picture you want.

```json
{"type": "image",
 "model": "qwen-image-2.1",
 "params": {"prompt": "An ordinary documentary photograph of a kitchen table with a blue ceramic vase of white tulips on it, natural window light, muted colours. No text.",
            "width": 1024, "height": 768, "steps": 40, "seed": 3,
            "mask": "mask.png"},
 "inputs": {"photo.png": {"inline_base64": "…"},
            "mask.png": {"inline_base64": "…"}}}
```

- **The mask** is a picture exactly the size of the image (`mask_size_mismatch` names both
  sizes otherwise). It is read as grey: 128 and brighter is regenerated, darker is kept. Paint
  white on black. An 8-bit grey PNG with 255 for the selection (what a `segment` job returns)
  works as it is. Transparency is ignored. PNG is best; JPEG and WebP are read too. A mask with
  no white is refused `mask_empty`.
- **Name the mask input in `mask`.** The other input is the image, whatever its name. A job
  with `mask` carries exactly those two inputs (`invalid_inputs` otherwise, naming what it
  found).
- **Send the image at its own size.** `width` and `height` are the size of the result, and the
  image and mask are stretched to them. Pixels come back exactly as sent only when width x
  height is the image's own size, so the canvas must follow the size rules above: a multiple
  of 16 (32 on the CUDA arm), and at most 1,048,576 pixels. Resize or pad the canvas in the
  app first if it does not.
- **`image_strength` is optional.** Leave it out, and the masked region starts from pure noise:
  new content, drawn to fit what surrounds it. Use this to add, replace or remove something.
  Send it, and the masked region starts from the image at that strength, on the same curve as
  image-to-image (0.03 to 0.3). Use this to change how a region looks while keeping its layout.
- **`mask_blur`** (default 8 pixels) is how far inside the mask's edge the new pixels fade into
  the old ones. `0` gives a hard edge. Use a larger value (24 to 64) for soft areas like sky.
  Make a selection a little larger than the object (8 to 16 pixels), so the fade lands on
  background rather than on the object's edge.
- **Describe the whole picture**, including what is new, not only what goes in the region.

How it works: both arms denoise the whole picture, but after every step everything outside the
mask is replaced by the original, noised to that step. The new region is drawn to fit the
original instead of beside it. The mask is widened to the model's 16-pixel grid for this, and
at the end the original pixels are pasted back outside the mask (with the `mask_blur` fade just
inside its edge). The time and memory are those of a picture of the same size, plus one VAE
encode of the input, the same as image-to-image.

### Outpainting

To extend a picture past its edges, the app makes the canvas bigger and masks the new border:

1. Make the larger canvas and put the picture where it belongs in it. For example, a 768x768
   photo centred on a 1024x768 canvas leaves a 128-pixel strip at each side.
2. Fill the new strips with the picture's own edge pixels stretched outward (edge replicate).
   A flat mid-grey also works. Without `image_strength` the fill is replaced entirely, so it
   hardly matters. Avoid black or transparency if you send `image_strength`, because the
   region then starts from what you filled it with.
3. Make the mask the canvas size: white over the new strips, black over the photo. Let the white
   reach 8 to 16 pixels into the photo so the seam falls inside the fade.
4. Send the canvas as the image, the mask, and a prompt describing the whole wider scene:

```json
{"type": "image",
 "model": "qwen-image-2.1",
 "params": {"prompt": "A wide documentary photograph of a mountain lake at dawn, pine forest on both shores, mist on the water. No text.",
            "width": 1024, "height": 768, "steps": 40, "seed": 5,
            "mask": "border.png", "mask_blur": 16},
 "inputs": {"canvas.png": {"inline_base64": "…"},
            "border.png": {"inline_base64": "…"}}}
```

Extend a picture a strip at a time (a quarter of the width or less). A narrow strip has a lot
of original around it to match, and a very wide one is close to making a new picture. Busy
edges such as trees sometimes fail to line up on one side and match on the other, and which
side fails changes with the seed. If a seam shows, try another seed, or a wider `mask_blur`
(24 to 32) with the white reaching 24 to 32 pixels into the photo.

## The result

The job publishes `image.png` (and, with a mask, `generated.png`). Its `done` event carries
`image`, the effective parameters, so a picture can be made again:

```json
{"artifacts": ["image.png"],
 "image": {"model": "qwen-image-2.1", "hf_repo": "Qwen/Qwen-Image-2.1",
           "revision": "790c92633540aa0cb11d9abf19eb46d861714758",
           "backend": "mlx-darwin", "engine": "mflux", "dtype": "bfloat16",
           "prompt": "…", "negative_prompt": null, "width": 1024, "height": 1024,
           "seed": 1, "steps": 2, "guidance": 1.0, "image_strength": null, "input": null,
           "mask": null, "mask_blur": null, "mask_coverage": null,
           "mask_outside_drift": null, "mask_blend_steps": null,
           "seconds": 28.83,
           "stage_seconds": {"encoding": 3.89, "denoising": 16.29, "decoding": 8.56, "saving": 0.08},
           "peak_bytes": 16441695780,
           "stage_peak_bytes": {"encoding": 2414314312, "denoising": 15502140544, "decoding": 16441695780},
           "memory_bytes_estimate": 17200000000, "memory_basis": "measured",
           "prompt_cache": "miss"},
 "resident": "qwen-image-2.1"}
```

`input` is the image input's name, `mask` the mask input's, `mask_blur` the fade used, and
`mask_coverage` the share of the picture the mask selected (0 to 1).

A masked job also reports two health numbers and publishes a second artifact:

- `mask_blend_steps`: how many denoising steps put the input back outside the mask. It should
  equal the steps run (`steps`, or fewer with `image_strength`).
- `mask_outside_drift`: how far the model's own picture strayed from the input outside the
  mask, before the paste-back (mean absolute difference, 0 to 255). A healthy blend leaves only
  the VAE's round trip, a few units. Tens mean the region was drawn without regard to what
  surrounds it.
- `generated.png` is the model's picture before the paste-back.

The mask fields are `null` without a mask.
`prompt_cache` is `"hit"` when this prompt (and negative prompt) was encoded by an earlier
picture on the same loaded model, so the text encoder was skipped, else `"miss"`.

The same seed reproduces a picture on the same backend and engine. The Mac and the PC run
different engines with different samplers, so one seed does not give one picture across them.

Progress arrives per step: `{"stage": "denoising", "step": 12, "steps": 40}` with `fraction`
12/40, after `encoding` and before `decoding` and `saving`. `DELETE /v1/jobs/{id}` stops the
job between two steps; the model stays loaded if a queue session holds it.

## Many pictures in a row

Like every other resident, the model comes off the card when the job that loaded it ends, unless
something holds it. To make a batch without reloading between pictures, hold the server with a
queue session ([QUEUE.md](QUEUE.md)):

1. Open one:

   ```json
   POST /v1/queue/sessions
   {"act": "image"}
   ```

   It answers `{"session_id": "ses-…", "status": "open"}` at once when the server is free; if
   another app holds it, `status: "queued"` and `GET /v1/queue/sessions/{id}/events` says
   `opened` when your turn comes.
2. Send the batch as ordinary `image` jobs. Every request from your client is an item of your
   session (or name it with `X-Crucible-Session`): the first loads the model, the rest reuse it,
   and nothing from another app runs in between. A job sent while the previous one runs waits
   inside the session, ahead of everyone else.
3. For a long pause on your side with nothing running (a person choosing), send
   `POST /v1/queue/sessions/{id}/touch`; otherwise the session closes after `idle_s`.
4. At the end, `DELETE /v1/queue/sessions/{id}`. The model comes off the card before that
   answers.

If your program stops or crashes without closing it, the session closes `idle_s` after the
last thing it did, and the model is unloaded then.

**Warming up before the first prompt.** A UI can load the model while the user is still typing,
inside its session:

```json
{"type": "load-image", "model": "qwen-image-2.1"}
```

Its `done` event carries `resident`; the first picture then starts at once. Outside a session,
`load-image` leaves the model loaded until the next picture ends.

**Same prompt, different seeds is fastest.** Turning a prompt into embeddings needs the 17.5 GB
text encoder, loaded and freed per picture (17 to 25 s on the PC). The loaded model remembers the
embeddings of its last 32 prompts, so a picture whose prompt and negative prompt were already
encoded skips the encoder entirely (`prompt_cache: "hit"`, `stage_seconds.encoding` near 0).
Change the seed, size or steps freely; change a word of the prompt and it is encoded again. The
memory goes when the model is unloaded.

## Memory

The model is ~33 GB on disk, and it never needs that much at once (measured on the Mac: at
most 16.4 GB, at 1024x1024): its three parts run one after
another (the text encoder, then the transformer, then the VAE), and each is released before the
next is read. The estimate the guard and capability use is the peak of the largest stage at the
largest size, not the sum. On the Mac that peak was measured (`memory_basis: "measured"`); on the
PC it is declared until measured. A card without that much room is refused before anything
loads, by name and with the numbers (`insufficient_memory`); on the Mac, where one model runs at
a time, a resident LLM or voice is unloaded first.

## Writing a prompt (Owen, 2026-09-28)

These are Owen's lessons from making thumbnails and card art with this model on the Mac.

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
