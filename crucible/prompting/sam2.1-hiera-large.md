# SAM 2.1 Hiera Large: how to use it

Crucible id `sam2.1-hiera-large` · upstream `facebook/sam2.1-hiera-large` @ `665f8e2ad61cf5f53d65644ff27c8ee525124610` (transformers `Sam2Model`, float32) · job type `segment`, class `select`.

## Sources

- [S1] OFFICIAL. SAM 2 image predictor notebook, https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/notebooks/image_predictor_example.ipynb (read 2026-10-15)
- [S2] OFFICIAL. Paper, "SAM 2: Segment Anything in Images and Videos", arXiv:2408.00714v2 (2024-10-28)
- [S3] OFFICIAL. Paper, "Segment Anything", arXiv:2304.02643v1 (2023-04-05). SAM 2's image path inherits its prompt design
- [C1] COMMUNITY. "Zero-shot capability of SAM-family models for bone segmentation in CT scans", arXiv:2411.08629 (2024-11-13). Medical CT, so treat it as a hint only
- [K1] Crucible @ `5a579e7`: `docs/SEGMENT.md`, `crucible/segment/sam2.1-hiera-large.toml`, `crucible/jobs/segment/worker.py`

## What the model expects

- Prompts are points in (x, y) with label 1 (foreground) or 0 (background), a box in xyxy, or both together. You can also pass "masks from the previous iteration of prediction" [S1].
- A single point is ambiguous. With `multimask_output=True` the model returns 3 masks with predicted-quality `scores`. "For ambiguous prompts such as a single point, it is recommended to use `multimask_output=True` even if only a single mask is desired", then keep the highest score [S1]. The three nested levels are typically "whole, part, and subpart" [S3].
- When several prompts specify one object, ask for one mask (`multimask_output=False`) [S1].
- Training simulated interaction: the first prompt was a mask (50%), a positive click (25%) or a box (25%), followed by corrective clicks [S2]. In SAM, those corrective points were sampled from the **error region** of the previous prediction [S3].
- The model works at 1024×1024 [S2, K1].

## Placing points and boxes

- DO start with one positive click near the middle of the object. Expect the whole/part/subpart ambiguity, and let the best-scored mask win [S1, S3].
- DO add a **negative** point on what got wrongly included. The official example excludes the car to keep just the window [S1].
- DO add later clicks where the mask is wrong: a positive in a missed area, a negative in a false area. This mirrors how the model was trained [S3] (the placement advice is an inference from the training procedure).
- DO use a box when the object's extent is clear. Combine a box with points to pick a part inside the box, as in the official tire-inside-the-wheel example [S1].
- DO, when nothing is interactive, use a box plus a centre point. That was best in one CT study [C1].
- DON'T expect fine structure. SAM "can miss fine structures, hallucinates small disconnected components at times, and does not produce boundaries as crisply" as heavier methods [S3]. SAM 2 still struggles with "very thin or fine details" and "nearby objects with similar appearance" [S2]. Hair and fur are BiRefNet's job, since its masks are soft [K1].

## Settings that matter (Crucible params)

| Crucible param | Model fact |
| --- | --- |
| `points` [{x, y, label}] (1–64), in the input's own pixels | labels 1/0 as in [S1] |
| `box` [x0, y0, x1, y1] | xyxy box [S1] |
| (automatic) `multimask` | Crucible asks for 3 and keeps the best only for exactly one point and no box. Otherwise it asks for one. This matches [S1] |
| `score` in `done` | the model's own predicted IoU [S1] |
| `coverage` | Crucible's mask mean. Near 0 means nothing was found [K1] |

EXIF orientation is not applied, so send the pixels upright [K1].

## Community findings

- In one CT bone study, a bounding box plus centre points beat any single prompt type [C1]. The domain differs from photos.

## Measured in Crucible

Nothing yet. SEGMENT.md says "neither model has been measured through Crucible yet" [K1].

## Conflicts and open questions

- **No previous-mask input.** The official refinement loop passes the last best mask's low-res logits back as `mask_input` [S1], and the model was trained with iterative prompts [S2, S3]. Crucible's click tool resends the whole click history but no mask [K1 `worker.py`; SEGMENT.md "Many clicks in a row"]. Each click therefore starts fresh. That may give less stable refinement than the official loop. Unmeasured.
- When a box plus exactly one point is sent, Crucible requests a single mask. The notebook does the same for its box-plus-point example [S1], so this is consistent.
