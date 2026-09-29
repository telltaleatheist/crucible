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
| `image_strength` | none | 0 to 1 exclusive, image-to-image: send exactly one input image (PNG, JPEG or WebP) and this; higher keeps more of the input. Mac only for now: the CUDA arm refuses it with `image_to_image_unsupported` |

Unknown params are refused, never ignored. Every refusal names the param and what to send instead.

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
           "memory_bytes_estimate": 17200000000, "memory_basis": "measured"},
 "resident": "qwen-image-2.1"}
```

The same seed reproduces a picture on the same backend and engine. The Mac and the PC run
different engines with different samplers, so one seed does not give one picture across them.

Progress arrives per step: `{"stage": "denoising", "step": 12, "steps": 40}` with `fraction`
12/40, after `encoding` and before `decoding` and `saving`. `DELETE /v1/jobs/{id}` stops the
job between two steps; the model stays loaded if a lease holds it.

## Many pictures in a row

Like every other resident, the model comes off the card when the job that loaded it ends, unless
something holds it. To make a batch without reloading between pictures, lease it after the first
job, exactly as a book leases its voice:

```http
POST /v1/models/qwen-image-2.1/lease   {"act": "image", "ttl_seconds": 600}
```

Heartbeat while the batch runs and release it at the end (`DELETE /v1/leases/{id}`). While the
lease is open a job that would load something else (an LLM, a voice) is refused `409 leased`.

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
