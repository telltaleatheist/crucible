from __future__ import annotations

from typing import Any

DOTS_PROMPT = "\n".join(
    [
        "Please output the layout information from the PDF image, including "
        "each layout element's bbox, its category, and the corresponding text "
        "content within the bbox.",
        "",
        "1. Bbox format: [x1, y1, x2, y2]",
        "",
        "2. Layout Categories: The possible categories are ['Caption', "
        "'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', "
        "'Picture', 'Section-header', 'Table', 'Text', 'Title'].",
        "",
        "3. Text Extraction & Formatting Rules:",
        "    - Picture: For the 'Picture' category, the text field should be "
        "omitted.",
        "    - Formula: Format its text as LaTeX.",
        "    - Table: Format its text as HTML.",
        "    - All Others (Text, Title, etc.): Format their text as Markdown.",
        "",
        "4. Constraints:",
        "    - The output text must be the original text from the image, with "
        "no translation.",
        "    - All layout elements must be sorted according to human reading "
        "order.",
        "",
        "5. Final Output: The entire output must be a single JSON object.",
    ]
)

DPI = 200

MAX_PIXELS = 11_289_600

MAX_TOKENS = 8192

TEMPERATURE = 0.0

PAGE_CONCURRENCY = 12

DIALECT = "dots-json"

MODEL_ID = "dots-ocr"

TRUNCATED_FINISH_REASON = "length"


def data_uri(png: bytes) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def request_body(
    image_data_uri: str,
    *,
    model: str = MODEL_ID,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    if max_tokens < 1:
        raise ValueError(
            f"max_tokens is {max_tokens}; a page needs room for an answer. "
            f"The ceiling is {MAX_TOKENS} and a per-page cap is below it, "
            "never at zero"
        )
    return {
        "model": model,
        "temperature": TEMPERATURE,
        "max_tokens": min(max_tokens, MAX_TOKENS),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": DOTS_PROMPT},
                ],
            }
        ],
    }


def was_truncated(choice: dict[str, Any]) -> bool:
    return choice.get("finish_reason") == TRUNCATED_FINISH_REASON


def request_shape() -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "dpi": DPI,
        "max_pixels": MAX_PIXELS,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "prompt": DOTS_PROMPT,
        "dialect": DIALECT,
        "concurrency": PAGE_CONCURRENCY,
        "truncated_finish_reason": TRUNCATED_FINISH_REASON,
    }


def engine_block(
    *, engine: str | None, installed: bool, detail: str
) -> dict[str, Any]:
    return {
        "engine": engine,
        "installed": installed,
        "detail": detail,
        "request": request_shape(),
    }


__all__ = [
    "DIALECT",
    "DOTS_PROMPT",
    "DPI",
    "MAX_PIXELS",
    "MAX_TOKENS",
    "MODEL_ID",
    "PAGE_CONCURRENCY",
    "TEMPERATURE",
    "TRUNCATED_FINISH_REASON",
    "data_uri",
    "engine_block",
    "request_body",
    "request_shape",
    "was_truncated",
]
