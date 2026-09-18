"""ONE definition of a page request, and `/v1/info` publishes it.

PHASE15-HOST.md 3.10 fact 7 and `DOTS-UNDER-VLLM-IN-WSL.md` section 2.
Owen's exit condition is that *"an app cannot tell whether a page was read by
vLLM in WSL, llama.cpp on Windows, or mlx-vlm on a Mac, and must not be able
to"*, and that is a statement about the REQUEST. Two engines handed the same
bytes, the same prompt and the same ceiling answer in the same dialect; two
engines handed nearly-the-same prompt do not, and neither of them errors.

So what is pinned here is the prompt BYTE FOR BYTE against the handover
copy, the four numbers, and the fact that `GET /v1/info` says the same thing
on every backend.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from crucible import capability, pages

HANDOVER_PROMPT = Path(__file__).resolve().parent / "dots_prompt.txt"


# ------------------------------------------------------------- the constants


def test_the_prompt_is_the_model_card_s_BYTE_FOR_BYTE() -> None:
    """Never templated, never shortened, never stripped.

    A nearly-right prompt answers worse without erroring, which is the
    failure that costs a whole book before anybody notices. The comparison
    is against the copy taken out of Foundry's `models.ts` at handover
    (`tests/dots_prompt.txt`), so a future edit to either has to be a
    deliberate edit to both.
    """
    expected = HANDOVER_PROMPT.read_text(encoding="utf-8")
    assert pages.DOTS_PROMPT == expected
    assert len(pages.DOTS_PROMPT) == 887
    # The shape a reader checks at a glance: the five numbered rules and the
    # eleven categories.
    assert pages.DOTS_PROMPT.startswith("Please output the layout information")
    assert "5. Final Output" in pages.DOTS_PROMPT
    for category in (
        "Caption",
        "Footnote",
        "Formula",
        "List-item",
        "Page-footer",
        "Page-header",
        "Picture",
        "Section-header",
        "Table",
        "Text",
        "Title",
    ):
        assert f"'{category}'" in pages.DOTS_PROMPT


def test_the_four_numbers_are_foundry_s(  ) -> None:
    """Each carried across rather than re-derived, which is the handover."""
    assert pages.DPI == 200
    assert pages.MAX_PIXELS == 11_289_600
    assert pages.MAX_TOKENS == 8192
    assert pages.TEMPERATURE == 0.0
    assert pages.DIALECT == "dots-json"
    assert pages.MODEL_ID == "dots-ocr"


# ---------------------------------------------------------------- the builder


def test_the_body_is_one_image_part_then_the_prompt() -> None:
    """The order is Foundry's working order and not a detail.

    The model card's examples put the picture first, and the two orders are
    not measured to be equivalent.
    """
    body = pages.request_body("data:image/png;base64,AAA")
    assert body["model"] == "dots-ocr"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 8192
    (message,) = body["messages"]
    assert message["role"] == "user"
    first, second = message["content"]
    assert first["type"] == "image_url"
    assert first["image_url"]["url"] == "data:image/png;base64,AAA"
    assert second["type"] == "text"
    assert second["text"] == pages.DOTS_PROMPT


def test_a_per_page_cap_is_honoured_and_the_ceiling_is_a_ceiling() -> None:
    """A client derives a cap from the book; it may not raise the ceiling.

    Asking for more than 8192 is asking for a runaway, and the dense-page
    measurement says 8192 is enough for the longest page anybody has read.
    """
    assert pages.request_body("d", max_tokens=1200)["max_tokens"] == 1200
    assert pages.request_body("d", max_tokens=99_999)["max_tokens"] == 8192
    with pytest.raises(ValueError):
        pages.request_body("d", max_tokens=0)


def test_a_truncated_page_is_named_so_it_cannot_be_read_as_finished() -> None:
    """`finish_reason: length` means the model was still writing."""
    assert pages.was_truncated({"finish_reason": "length"}) is True
    assert pages.was_truncated({"finish_reason": "stop"}) is False
    # An answer with no finish_reason at all is not a truncation claim.
    assert pages.was_truncated({}) is False


def test_the_data_uri_is_png_and_nothing_else() -> None:
    """A JPEG at this dpi costs the accuracy the 0.80% CER was measured with."""
    uri = pages.data_uri(b"\x89PNG\r\n\x1a\n")
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == b"\x89PNG\r\n\x1a\n"
    assert "charset" not in uri


# --------------------------------------------------------- what /v1/info says


def test_info_publishes_the_request_shape_and_it_IS_the_builder_s(
    client: TestClient, auth: dict[str, str]
) -> None:
    """One owner. The block is `pages.request_shape()`, not a copy of it."""
    body = client.get("/v1/info", headers=auth).json()
    block = body["pages_engine"]
    assert block["request"] == pages.request_shape()
    assert block["request"]["prompt"] == pages.DOTS_PROMPT
    assert block["request"]["max_pixels"] == 11_289_600
    assert block["request"]["dpi"] == 200
    assert block["request"]["truncated_finish_reason"] == "length"


def test_the_engine_is_named_for_an_operator_and_the_request_never_varies(
    client: TestClient, auth: dict[str, str]
) -> None:
    """`engine` is for a person; `request` is for a client.

    A client that read `engine` and changed its request would be the exact
    thing 3.10's exit condition forbids, which is why the two sit side by
    side and only one of them is allowed to differ per host.
    """
    block = client.get("/v1/info", headers=auth).json()["pages_engine"]
    # The fake backend is cuda-linux, and dots-ocr's block there is vLLM's.
    assert block["engine"] == "vllm"
    assert block["installed"] is False
    assert "pull" in block["detail"]
    assert block["request"] == pages.request_shape()


def test_a_backend_with_no_page_block_says_so_and_still_publishes_the_request(
    make_client, auth: dict[str, str]
) -> None:
    """The Mac today: `models/dots-ocr.toml` has no `mlx-darwin` block.

    `engine: null` is the honest answer and it is NOT a missing key — a
    client reading the block has to be able to tell "this host reads no
    pages" from "this document predates the field".
    """
    from .conftest import FAKE_MAC_BACKEND

    with make_client(backend=FAKE_MAC_BACKEND) as mac:
        block = mac.get("/v1/info", headers=auth).json()["pages_engine"]
    assert block["engine"] is None
    assert block["installed"] is False
    assert "mlx-darwin" in block["detail"]
    # And the request is still there, because it is not a property of the host.
    assert block["request"] == pages.request_shape()


def test_the_pulled_case_names_the_pin(
    client: TestClient, auth: dict[str, str], fake_weights
) -> None:
    fake_weights("dots-ocr")
    block = client.get("/v1/info", headers=auth).json()["pages_engine"]
    assert block["installed"] is True
    assert "dots-studio/dots.ocr" in block["detail"]


def test_the_block_is_json_serialisable_and_carries_no_bytes(
    client: TestClient, auth: dict[str, str]
) -> None:
    """It is read by two apps and drawn on a page; nothing in it is binary."""
    block = client.get("/v1/info", headers=auth).json()["pages_engine"]
    json.dumps(block)
    assert set(block) == {"engine", "installed", "detail", "request"}


# ------------------------------------------------ the server owns the width


def test_the_request_block_states_how_many_pages_may_be_in_flight(
    client: TestClient, auth: dict[str, str]
) -> None:
    """Ledger N3. A client that picked its own number was the defect: both
    apps sent twelve while `crucible/capability.py` sized the card's
    arithmetic for one, so the guard that decides whether `dots-ocr` FITS was
    reasoning about a twelfth of the work that arrives.

    Twelve is the manifest's own measurement note — *"at the worst case 12
    concurrent pages want about 4.1 GiB of KV"* — and it is the same number
    on every backend, because a client must not be able to tell which engine
    read its page.
    """
    block = client.get("/v1/info", headers=auth).json()["pages_engine"]
    assert block["request"]["concurrency"] == 12
    assert block["request"]["concurrency"] == pages.PAGE_CONCURRENCY


def test_the_fits_arithmetic_asks_for_the_published_number(
    client: TestClient, auth: dict[str, str]
) -> None:
    """One owner, read by both halves. `pages.py` publishes it to clients and
    `capability.py`'s `WorkingContext` — the object `Candidate.need_bytes`
    multiplies by — reads the same name, so there is no second literal that
    could drift from the wire."""
    work = capability.BY_NAME["pages"].work
    assert work is not None
    published = client.get("/v1/info", headers=auth).json()
    assert work.concurrency == published["pages_engine"]["request"]["concurrency"]
    # And the source says where the number came from, which is what stops the
    # next reader inventing a different one.
    assert "pages.py" in work.source


def test_changing_the_published_number_moves_the_arithmetic() -> None:
    """The one-owner claim, made falsifiable.

    In a subprocess because `CAPABILITY_CLASSES` is built at import time: the
    child imports `crucible.pages`, changes the number, and only THEN imports
    `crucible.capability`, so what it prints is what the arithmetic would be
    on a build whose published width were three. Reloading in-process would
    hand every other module in this suite a stale `CLASSES` tuple.
    """
    source = (
        "import crucible.pages as pages\n"
        "pages.PAGE_CONCURRENCY = 3\n"
        "import crucible.capability as capability\n"
        "print(capability.BY_NAME['pages'].work.concurrency)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "3", result.stderr
