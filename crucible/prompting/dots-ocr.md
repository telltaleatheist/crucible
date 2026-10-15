# dots.ocr: how to use it

Crucible id `dots-ocr` · upstream `dots-studio/dots.ocr` @ `c0111ce6bc07803dbc267932ffef0ae3a51dc951` (cuda-linux, vLLM). The Mac serves `mlx-community/dots.ocr-bf16` @ `98b9cda3` and llama-windows serves `ggml-org/dots.ocr-GGUF` Q8_0 @ `2c093a32`. Job/verb: page reading (`crucible/pages.py`, dialect `dots-json`).

## Sources

- [S1] OFFICIAL. Model card at the pinned revision, https://huggingface.co/dots-studio/dots.ocr/blob/c0111ce6bc07803dbc267932ffef0ae3a51dc951/README.md (last modified 2025-10-31; read 2026-10-15)
- [S2] OFFICIAL. Prompt table, https://github.com/rednote-hilab/dots.ocr/blob/36d7248878f181c6257dc1cbd799ae940970444f/dots_ocr/utils/prompts.py (`dict_promptmode_to_prompt`)
- [S3] OFFICIAL. vLLM client, `dots_ocr/model/inference.py` @ `7c672c9210f4af9e9811ef6bb3db06cd8fc81bb3` (2025-10-11, the dots.ocr-era version)
- [S4] OFFICIAL. `dots_ocr/parser.py` and `dots_ocr/utils/consts.py`, same repo @ `7c672c92` / `36d72488`
- [S5] OFFICIAL. Repo README @ `36d72488`. It notes that on 2026-03-19 "dots.ocr-1.5" was rebranded **dots.mocr**, and `master` now targets that model
- [K1] Crucible @ `5a579e7`: `crucible/pages.py`, `crucible/models/dots-ocr.toml`, `docs/history/CLIENT-SURFACES.md`

## What the model expects

Each task is one image followed by one fixed English prompt. "Switching between tasks is accomplished simply by altering the input prompt" [S1]. The official modes for dots.ocr [S2]:

| mode | prompt (abridged) | output |
| --- | --- | --- |
| `prompt_layout_all_en` | "Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content…", then rules 1–5 (11 categories; Picture has no text; Formula as LaTeX; Table as HTML; others as Markdown; no translation; reading order; one JSON object) | JSON: bbox + category + text |
| `prompt_layout_only_en` | "…including each layout's bbox and its category… Do not output the corresponding text." | JSON layout only |
| `prompt_ocr` | "Extract the text content from this image." | Plain text (the parser's comment says it skips Page-header and Page-footer) |
| `prompt_grounding_ocr` | "Extract text from the given bounding box on the image (format: [x1, y1, x2, y2]).\nBounding Box:\n" + bbox | Text inside the box |

The current repo adds `prompt_web_parsing`, `prompt_scene_spotting`, `prompt_image_to_svg` and `prompt_general`. These belong to dots.mocr / 1.5 [S2, S5], so don't assume the pinned dots.ocr was trained on them.

Official client defaults [S3, S4]: the image part comes first, then the text part sent as `<|img|><|imgpad|><|endofimg|>{prompt}`. The code comment says vLLM v1 otherwise adds a `"\n"` there. Other defaults are `temperature` 0.1, `top_p` 0.9 (client) or 1.0 (parser), and `max_completion_tokens` 32768 (client), 16384 (parser) or 24000 (HF demo `max_new_tokens`) [S1]. The official vLLM serve line adds `--chat-template-content-format string` [S1].

**Resolution** [S1, S4]: render PDFs at 200 DPI ("a setting of 200 is recommended"). The model "performs optimally on images with a resolution under 11289600 pixels". `MIN_PIXELS` is 3136 and `IMAGE_FACTOR` 28 (smart-resize to multiples of 28).

## Writing the instruction / input

- DO send the official prompt **verbatim**. It is a trained task switch, not free text [S1, S2].
- DO raise the render DPI or enlarge the image when glyphs are tiny relative to the page ("character-to-pixel ratio is excessively high"), staying under 11,289,600 px [S1].
- DO switch prompt on a looping page. Long runs of special characters such as `...` and `_` "may cause the prediction output to repeat endlessly". The official fix is to retry with `prompt_layout_only_en`, `prompt_ocr` or `prompt_grounding_ocr` [S1].
- DON'T expect picture content to be read ("Pictures in documents are currently not parsed"). High-complexity tables and formulas are "not yet perfect" [S1].
- DON'T add instructions such as "translate" or "summarise" to the layout prompt. That is not one of the trained modes [S2] (inference from the prompt table).

## Settings that matter (Crucible)

| Crucible (`pages.py`) | Official |
| --- | --- |
| prompt = `DOTS_PROMPT`, the `prompt_layout_all_en` text | the same text, but official ends with a trailing `\n` [S2] |
| image part, then the bare prompt text | image part, then `<|img|><|imgpad|><|endofimg|>` + prompt [S3] |
| `DPI` 200 | 200 [S1, S4] |
| `MAX_PIXELS` 11,289,600 | 11,289,600 [S4] |
| `TEMPERATURE` 0.0 | 0.1 (top_p 0.9 / 1.0) [S3, S4] |
| `MAX_TOKENS` 8192 (truncation detected by `finish_reason == "length"`) | 16,384–32,768 [S3, S4] |
| `PAGE_CONCURRENCY` 12 | parser `num_thread` 64 for PDFs [S4] |
| layout-all only | four modes, plus a fallback mode for loops [S1, S2] |

## Community findings

None gathered that are specific to the pinned dots.ocr beyond the card's own limitations section.

## Measured in Crucible

None in `docs/`. CHANGELOG records only the serving facts: the Mac is 12 pages wide, and the vLLM `--gpu-memory-utilization 0.5` is a budget, not a measurement [K1].

## Conflicts and open questions

- **Temperature 0.0 vs 0.1.** Greedy decoding may make the repeat-loop failure on `...` / `____` stickier than the official sampling would. Unmeasured.
- **The 8192-token cap is below every official budget** (16k–32k). A dense page (big tables in HTML) can hit `length`. Crucible detects this but does not raise the budget.
- **The `<|img|><|imgpad|><|endofimg|>` prefix is missing**, and Crucible's vLLM launch does not pass `--chat-template-content-format string`. Whether vLLM's native dots.ocr support inserts the stray `"\n"` the official comment warns about, and whether it matters, is unverified. The mlx-vlm and llama-server chat templates may differ again.
- The trailing `\n` is dropped from the prompt. Probably harmless, but it is a byte difference from the trained prompt.
- Crucible has no OCR-only or grounding fallback for looping pages, which is the official remedy [S1]. That would be the app's choice (pages.py fixes one prompt).
