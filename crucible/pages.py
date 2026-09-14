"""ONE definition of what a page request IS, for every backend that reads one.

PHASE15-HOST.md 3.10 fact 7, and `C:\\tmp\\foundry-page-reader-spec\\
DOTS-UNDER-VLLM-IN-WSL.md` section 2, which is the engine-side half of the
same spec. Owen's exit condition is that *"an app cannot tell whether a page
was read by vLLM in WSL, llama.cpp on Windows, or mlx-vlm on a Mac, and must
not be able to"* — and that is a statement about the REQUEST, not about the
engine. Two engines handed the same bytes, the same prompt, the same
temperature and the same ceiling answer in the same dialect; two engines
handed nearly-the-same prompt do not, and neither of them errors.

WHY IT IS ON THE WIRE RATHER THAN ONLY IN THE SERVER
------------------------------------------------------
Page reading has no job type of its own: PHASE3-VLM.md section 1 ruled that
dots.ocr is served through the same `llm` proxy as every text model, because
a page is a chat completion in every respect but its content parts. So the
CLIENT builds the request — and today it builds it out of constants pinned
in Foundry's own source (`VLM_DPI`, `maxPixels`, `DOTS_PROMPT`, `maxTokens`),
which is the thing the division-of-knowledge ruling forbids: *the client
knows the ORDER and the SERVER; Crucible knows the ENGINE.* A prompt and a
pixel budget are facts about the weights.

So this module is the owner, and `GET /v1/info`'s `pages_engine.request`
publishes it. Every backend answers from HERE, so the two cannot drift; a
client reads it once and sends what it says. Nothing about the wire changes:
it is still `POST /v1/openai/chat/completions` with one `image_url` part.

WHAT IS PINNED, AND WHERE EACH NUMBER CAME FROM
------------------------------------------------
Every one of these is Foundry's, measured or read by Foundry, and carried
across rather than re-derived — the whole point of the handover:

* **`dpi = 200`** (`VLM_DPI`). The rasterisation is the APP's work
  (3.10: "rasterising, parsing, EPUB assembly stay in the app"), but the
  number is not the app's choice: the parser scales the model's bboxes back
  into render space, so the dpi the client renders at and the budget the
  server's processor uses have to be one pair of facts.
* **`max_pixels = 11_289_600`** — the dots.ocr processor's own limit. A page
  above it is scaled down BY THE PROCESSOR, and `convert.ts` refuses a run
  whose boxes overflow the frame, so a client that assumed a different
  budget produces geometry that does not land on the page.
* **`max_tokens = 8192`**, the CEILING and not a budget. Measured over
  18,202 pages in 55 banks: the longest ACCEPTED page is 7,677 tokens — a
  full-page newspaper facsimile reproduced inside a book — which is 93.7% of
  it. Lowering it would trade a real dense page for a fake one. A client may
  send LESS (a per-page cap derived from the book) and must re-read a page
  that came back `finish_reason: "length"` at the full ceiling.
* **`temperature = 0`**. A layout is not a thing to be creative about.
* **the prompt, VERBATIM from the model card.** Never templated, never
  shortened: a nearly-right prompt answers worse without erroring, which is
  the failure mode that costs a whole book before anybody notices.
* **`dialect = "dots-json"`** — a JSON array, sometimes fenced, of
  `{bbox: [x1, y1, x2, y2], category, text}` in reading order over eleven
  categories. `parseDotsPage` is the reader and it does not change.
"""

from __future__ import annotations

from typing import Any

#: The model card's prompt, byte for byte. `models.ts`' `DOTS_PROMPT`.
#:
#: NOT ASSEMBLED FROM PARTS at run time and not formatted: the newlines and
#: the indentation are part of what was measured, and a `.strip()` or a
#: `textwrap.dedent` here would be this module quietly editing somebody
#: else's prompt. It is a literal for that reason and a test compares it to
#: the handover copy.
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

#: Dots per inch the page is rasterised at. Foundry's `VLM_DPI`.
DPI = 200

#: The processor's own pixel limit for this model. Foundry's `maxPixels`.
MAX_PIXELS = 11_289_600

#: The ceiling, never a budget. See the module docstring for the 18,202-page
#: measurement that says why it is not lower.
MAX_TOKENS = 8192

#: A layout is not a thing to be creative about.
TEMPERATURE = 0.0

#: What the answer is shaped like. `parseDotsPage` is the reader.
DIALECT = "dots-json"

#: The capability class this describes, so a reader can tie the block to a
#: `/v1/capability` row without a second name for one thing.
CLASS_NAME = "pages"

#: The Crucible model id every backend serves pages from. One id, three
#: engines — which is the whole of what `pages_engine` is saying.
MODEL_ID = "dots-ocr"

#: What a truncated page looks like coming back. A client that reads this as
#: "done" silently loses the bottom of a dense page.
TRUNCATED_FINISH_REASON = "length"


def data_uri(png: bytes) -> str:
    """A PNG as the `image_url` a chat content part takes.

    Here rather than in a client because the ENCODING is part of the request
    shape: `data:image/png;base64,` and nothing else — no `charset`, no
    whitespace, no other media type. A JPEG at this dpi costs the accuracy
    the 0.80% CER was measured with.
    """
    import base64

    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def request_body(
    image_data_uri: str,
    *,
    model: str = MODEL_ID,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    """The chat-completions body for ONE page, on ANY backend.

    `max_tokens` may be LOWER than the ceiling — a client derives a per-page
    cap from the book it is reading — and is clamped UP to nothing: a caller
    asking for more than the ceiling is asking for a runaway, and the
    ceiling is what the dense-page measurement says is enough.

    The order of the content parts is the image FIRST and the prompt second,
    which is Foundry's working order and not a detail: the model card's
    examples put the picture first, and the two orders are not measured to
    be equivalent.
    """
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
    """Did this page come back cut off? Then it is RE-READ at the ceiling.

    `finish_reason: "length"` means the model was still writing. A client
    that treats it as a finished page loses the bottom of a dense one
    silently, which is the failure this function exists to make impossible to
    overlook.
    """
    return choice.get("finish_reason") == TRUNCATED_FINISH_REASON


def request_shape() -> dict[str, Any]:
    """What `GET /v1/info`'s `pages_engine.request` publishes.

    Every backend answers from this function, which is the whole of "the
    artifact is byte-shape identical on both backends": there is one place
    the prompt, the dpi, the budget and the ceiling are written down, and a
    client reads it rather than pinning its own copy.
    """
    return {
        "model": MODEL_ID,
        "dpi": DPI,
        "max_pixels": MAX_PIXELS,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "prompt": DOTS_PROMPT,
        "dialect": DIALECT,
        # What a client must do about a page that came back cut off. Named
        # rather than implied, because "re-read it at the full cap" is the
        # part of the contract an app gets wrong by doing nothing.
        "truncated_finish_reason": TRUNCATED_FINISH_REASON,
    }


def engine_block(
    *, engine: str | None, installed: bool, detail: str
) -> dict[str, Any]:
    """`GET /v1/info`'s `pages_engine`. Null `engine` means this host serves none.

    `engine` is the ENGINE NAME (`vllm`, `llama-server`, `mlx-vlm`) and it is
    here so an operator can see which of the three read a page — not so a
    client can branch on it. A client that read `engine` and changed its
    request would be the exact thing 3.10's exit condition forbids, which is
    why `request` sits beside it and says the same thing on every backend.
    """
    return {
        "engine": engine,
        "installed": installed,
        "detail": detail,
        "request": request_shape(),
    }


__all__ = [
    "CLASS_NAME",
    "DIALECT",
    "DOTS_PROMPT",
    "DPI",
    "MAX_PIXELS",
    "MAX_TOKENS",
    "MODEL_ID",
    "TEMPERATURE",
    "TRUNCATED_FINISH_REASON",
    "data_uri",
    "engine_block",
    "request_body",
    "request_shape",
    "was_truncated",
]
