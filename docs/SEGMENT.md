# Masks: the `segment` job

A picture in, a mask out: the main subject cut out of its background, or the one object a
caller points at. One job type, `segment`, serves two models, and which one you name decides
which you get. Crucible supplies the mask; the app builds the feature around it (a background
remover, a magic-wand selection, the mask for an inpainting job). How it runs and why is
[internals/segment.md](internals/segment.md); this page is what a caller sends and what comes
back.

| model | makes | class | PC (cuda-linux) | Mac (mlx-darwin) | licence |
| --- | --- | --- | --- | --- | --- |
| `birefnet` | the main subject's mask, found by itself (background removal); soft edges | `cutout` | yes, float16 | yes, float32 on Metal | MIT |
| `sam2.1-hiera-large` | the mask of the object under your points and/or inside your box; hard edges | `select` | yes, float32 | yes, float32 on Metal | Apache-2.0 |

`GET /v1/capability` has one row per class: "can cut out a picture's subject, using birefnet"
and "can select what is pointed at in a picture, using sam2.1-hiera-large", each with the fit
reason, or why not. Both models are free for commercial use.

## Turning it on

```bash
crucible init --enable-segment        # or [jobs] enable_segment = true
crucible install segment              # one env for both models
crucible models pull birefnet
crucible models pull sam2.1-hiera-large
```

None of these is needed by hand: a job for a missing env or model starts the install and
answers `409 installing`, like every other type. Neither repo is gated.

## The request

The picture is the job's one input (PNG, JPEG or WebP), sent the way every job sends a file:
inline, as an uploaded blob, or as another job's artifact.

Cut out the subject:

```json
{"type": "segment",
 "model": "birefnet",
 "inputs": {"photo.jpg": {"inline_base64": "…"}}}
```

Select what the user clicked (one click on the object, one on something to leave out):

```json
{"type": "segment",
 "model": "sam2.1-hiera-large",
 "params": {"points": [{"x": 412, "y": 300, "label": 1}, {"x": 520, "y": 610, "label": 0}]},
 "inputs": {"photo.jpg": {"inline_base64": "…"}}}
```

Select what is inside a dragged box (points can be added to refine it):

```json
{"type": "segment",
 "model": "sam2.1-hiera-large",
 "params": {"box": [120, 80, 700, 590]},
 "inputs": {"photo.jpg": {"blob_id": "…"}}}
```

| param | who takes it | rule |
| --- | --- | --- |
| `points` | `sam2.1-hiera-large` | 1 to 64 clicks, `{"x", "y", "label"}`; `label` 1 keeps what is under the point, 0 leaves it out. At least one label-1 point unless there is a box |
| `box` | `sam2.1-hiera-large` | `[x0, y0, x1, y1]`, top-left corner first, `x1 > x0` and `y1 > y0` |
| `lease` | both | `{"act": "<the model's class>", "ttl_seconds": 30..3600}`: `cutout` for `birefnet`, `select` for `sam2.1-hiera-large`. Any other act is refused |

`sam2.1-hiera-large` needs `points`, `box` or both (`segment_param_missing`). `birefnet` takes
neither: it finds the subject by itself, and a `points` or `box` sent to it is refused
`segment_param_unsupported`, naming the model that does take them. Unknown params are refused,
never ignored (`invalid_params`).

**Coordinates are the input's own pixels**, `x` from the left edge and `y` from the top, as
the file stores them. Fractions are fine (`412.5`). A point must lie inside the picture (x at
most width − 1), and a box may end on the edge (x1 up to the width). Anything outside fails
the job `segment_prompt_outside_picture` before a model loads, saying the picture's size.
EXIF orientation is not applied: a phone JPEG stored sideways is segmented sideways, so send
the pixels upright (or rotate the mask the same way the app rotates the picture).

The input must be exactly one PNG, JPEG or WebP (`invalid_inputs` otherwise, also for a file cut
short), of at most 40,000,000 pixels (`image_too_large`: send a smaller copy and scale the mask
up). Both models look at the picture at 1024x1024 whatever its size, then the mask is scaled
back to the input's size, so a larger input gives a larger mask, not a finer one.

## The result

Two artifacts, each exactly the input's width and height:

- **`mask.png`**: 8-bit greyscale (`L`), 255 where the subject or the selected object is, 0
  where it is not. `birefnet`'s mask is soft: hair, fur and edges have the values in between,
  which is what a clean cutout needs. `sam2.1-hiera-large`'s is hard (0 or 255 only). An app can
  hand it straight back to Crucible as another job's mask input.
- **`cutout.png`**: the input as RGBA with the mask as its alpha, so a background remover needs no
  extra step. A pixel the input already made transparent stays transparent (the input's alpha is
  multiplied by the mask).

The `done` event carries `segment`, the effective parameters and measurements:

```json
{"artifacts": ["mask.png", "cutout.png"],
 "segment": {"model": "sam2.1-hiera-large", "kind": "select",
             "hf_repo": "facebook/sam2.1-hiera-large",
             "revision": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
             "backend": "cuda-linux", "engine": "sam2", "dtype": "float32",
             "input": "photo.jpg", "width": 1600, "height": 1200,
             "points": [{"x": 412.0, "y": 300.0, "label": 1}], "box": null,
             "mask": "mask.png", "cutout": "cutout.png",
             "score": 0.97, "multimask": true, "coverage": 0.184,
             "seconds": 0.62,
             "stage_seconds": {"reading": 0.03, "segmenting": 0.41, "saving": 0.18},
             "peak_bytes": 2100000000,
             "stage_peak_bytes": {"segmenting": 2100000000},
             "memory_bytes_estimate": 4000000000, "memory_basis": "declared",
             "versions": {"torch": "2.14.0", "torchvision": "0.29.0", "transformers": "5.17.0"}},
 "resident": "sam2.1-hiera-large",
 "lease_id": null}
```

(The numbers show the shape; neither model has been measured through Crucible yet.)

- `score` is SAM's own estimate of the mask's quality (its predicted IoU, 0 to 1); `null` for
  `birefnet`.
- `multimask` is `true` when the job sent exactly one point and no box. One click is
  ambiguous (the shirt, or the person wearing it), so SAM is asked for three candidates and the
  best-scored one is returned; with more points or a box it makes one. `null` for `birefnet`.
- `coverage` is the mask's mean, 0 to 1: the share of the picture selected. A `coverage` near 0
  means nothing was found; tell the user rather than showing an empty cutout.

Progress events carry `stage` (`reading`, `segmenting`, `saving`) and `fraction`.
`DELETE /v1/jobs/{id}` stops the job between stages; the model stays loaded if a lease holds it.

## Many clicks in a row

A selection tool sends a job per click, and each takes a fraction of a second once the model is
loaded, so hold the model for the session. Load it when the tool opens:

```json
{"type": "load-segment", "model": "sam2.1-hiera-large",
 "params": {"lease": {"act": "select", "ttl_seconds": 300}}}
```

Then send each click as an ordinary `segment` job with the same `lease` (renewing it; the same
`lease_id` comes back, never a second lease), send the whole click history each time (all
points so far, and the box), heartbeat `POST /v1/leases/{lease_id}/heartbeat` within
`ttl_seconds`, and `DELETE /v1/leases/{lease_id}` when the tool closes. Without a lease the model
comes off the card when each job ends and the next click loads it again. `unload-segment` takes
the model off at once when nothing holds it. The flow is the image job's, walked through in more
detail in [IMAGE.md](IMAGE.md), "Many pictures in a row".

Only one model is resident at a time: switching from `birefnet` to `sam2.1-hiera-large` swaps
them (and is refused `409 leased` while a lease holds the other).

## Licences

- **BiRefNet** (`ZhengPeng7/BiRefNet`): MIT, as its model card declares; the MIT text is in the
  code's GitHub repo. Commercial use included, keeping the copyright notice. The datasets it was
  trained on carry their own terms.
- **SAM 2.1 Hiera Large** (`facebook/sam2.1-hiera-large`): Apache-2.0, as its model card
  declares; the text is in Meta's `facebookresearch/sam2` repo. Commercial use included.
