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
| `image_strength` | none | 0 to 1 exclusive, image-to-image: send exactly one input image (PNG, JPEG or WebP) and this; higher keeps more of the input. Both arms: the input is stretched to width x height and denoising starts at step `max(1, int(steps * image_strength))`. Useful range 0.03 to 0.3; see Image-to-image below |
| `lease` | none | `{"act": "image", "ttl_seconds": 30..3600}`: hold the model on the card from the moment it is loaded, for a batch (below). `act` must be `image` (`lease_act_mismatch`); an unknown act is `unknown_act`, a ttl out of range `invalid_ttl` |

Unknown params are refused, never ignored. Every refusal names the param and what to send instead.

## Image-to-image

Send one picture as an input and `image_strength` with a new prompt, and the model redraws
the picture toward the prompt. The whole picture is redrawn: there is no mask, no selected
region, no extending past the edges, and no instructions like "remove the lamp".

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

## The result

The job publishes one artifact, `image.png`. Its `done` event carries `image`, the effective
parameters, so a picture can be made again:

```json
{"artifacts": ["image.png"],
 "image": {"model": "qwen-image-2.1", "hf_repo": "Qwen/Qwen-Image-2.1",
           "revision": "790c92633540aa0cb11d9abf19eb46d861714758",
           "backend": "mlx-darwin", "engine": "mflux", "dtype": "bfloat16",
           "prompt": "…", "negative_prompt": null, "width": 1024, "height": 1024,
           "seed": 1, "steps": 2, "guidance": 1.0, "image_strength": null, "input": null,
           "seconds": 28.83,
           "stage_seconds": {"encoding": 3.89, "denoising": 16.29, "decoding": 8.56, "saving": 0.08},
           "peak_bytes": 16441695780,
           "stage_peak_bytes": {"encoding": 2414314312, "denoising": 15502140544, "decoding": 16441695780},
           "memory_bytes_estimate": 17200000000, "memory_basis": "measured",
           "prompt_cache": "miss"},
 "resident": "qwen-image-2.1",
 "lease_id": null}
```

`lease_id` is the lease the job opened or renewed when it was sent `lease`, else `null`.
`prompt_cache` is `"hit"` when this prompt (and negative prompt) was encoded by an earlier
picture on the same loaded model, so the text encoder was skipped, else `"miss"`.

The same seed reproduces a picture on the same backend and engine. The Mac and the PC run
different engines with different samplers, so one seed does not give one picture across them.

Progress arrives per step: `{"stage": "denoising", "step": 12, "steps": 40}` with `fraction`
12/40, after `encoding` and before `decoding` and `saving`. `DELETE /v1/jobs/{id}` stops the
job between two steps; the model stays loaded if a lease holds it.

## Many pictures in a row

Like every other resident, the model comes off the card when the job that loaded it ends, unless
something holds it. To make a batch without reloading between pictures:

1. Send `lease` on the **first** picture:

   ```json
   {"type": "image", "model": "qwen-image-2.1",
    "params": {"prompt": "…", "seed": 1, "lease": {"act": "image", "ttl_seconds": 300}}}
   ```

   The lease opens the moment the model is loaded, before the picture is made, so nothing can
   take the model off the card in between. The `done` event carries `lease_id`.
2. Send the rest of the batch as ordinary `image` jobs (sending `lease` again is harmless: the
   lease you already hold is renewed and the same `lease_id` comes back, never a second lease).
   The loaded model is reused; nothing reloads.
3. While you work, `POST /v1/leases/{lease_id}/heartbeat` at least once per `ttl_seconds`.
4. At the end, `DELETE /v1/leases/{lease_id}`. The model comes off the card before that answers.

If your program stops or crashes without releasing, the lease runs out `ttl_seconds` after the
last heartbeat and the model is unloaded then. If the first picture fails or is cancelled, the
lease it opened is given back at once (you never got its id). While the lease is open a job that
would load something else (an LLM, a voice) is refused `409 leased`.

**Warming up before the first prompt.** A UI can load the model while the user is still typing:

```json
{"type": "load-image", "model": "qwen-image-2.1",
 "params": {"lease": {"act": "image", "ttl_seconds": 300}}}
```

Its `done` event carries `resident` and `lease_id`; the first picture then starts at once. Sent
with the same `lease`, that picture renews the same lease. Without `lease`, `load-image` leaves
the model loaded until the next picture ends.

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
a time, a resident LLM or voice is unloaded first, unless a lease holds it, in which case the job
is refused `409 leased`.

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
