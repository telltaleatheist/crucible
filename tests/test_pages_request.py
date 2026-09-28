from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from crucible import capabilityclasses, pages

HANDOVER_PROMPT = Path(__file__).resolve().parent / "dots_prompt.txt"


def test_the_prompt_is_the_model_card_s_BYTE_FOR_BYTE() -> None:
    expected = HANDOVER_PROMPT.read_text(encoding="utf-8")
    assert pages.DOTS_PROMPT == expected
    assert len(pages.DOTS_PROMPT) == 887
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
    assert pages.DPI == 200
    assert pages.MAX_PIXELS == 11_289_600
    assert pages.MAX_TOKENS == 8192
    assert pages.TEMPERATURE == 0.0
    assert pages.DIALECT == "dots-json"
    assert pages.MODEL_ID == "dots-ocr"


def test_the_body_is_one_image_part_then_the_prompt() -> None:
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
    assert pages.request_body("d", max_tokens=1200)["max_tokens"] == 1200
    assert pages.request_body("d", max_tokens=99_999)["max_tokens"] == 8192
    with pytest.raises(ValueError):
        pages.request_body("d", max_tokens=0)


def test_a_truncated_page_is_named_so_it_cannot_be_read_as_finished() -> None:
    assert pages.was_truncated({"finish_reason": "length"}) is True
    assert pages.was_truncated({"finish_reason": "stop"}) is False
    assert pages.was_truncated({}) is False


def test_the_data_uri_is_png_and_nothing_else() -> None:
    uri = pages.data_uri(b"\x89PNG\r\n\x1a\n")
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == b"\x89PNG\r\n\x1a\n"
    assert "charset" not in uri


def test_info_publishes_the_request_shape_and_it_IS_the_builder_s(
    client: TestClient, auth: dict[str, str]
) -> None:
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
    block = client.get("/v1/info", headers=auth).json()["pages_engine"]
    assert block["engine"] == "vllm"
    assert block["installed"] is False
    assert "pull" in block["detail"]
    assert block["request"] == pages.request_shape()


def test_a_backend_whose_weights_are_not_pulled_names_the_engine_and_the_pull(
    make_client, auth: dict[str, str]
) -> None:
    from .conftest import FAKE_MAC_BACKEND

    with make_client(backend=FAKE_MAC_BACKEND) as mac:
        block = mac.get("/v1/info", headers=auth).json()["pages_engine"]
    assert block["engine"] == "mlx-vlm"
    assert block["installed"] is False
    assert "crucible models pull dots-ocr" in block["detail"]
    assert block["request"] == pages.request_shape()


def test_a_backend_with_no_page_block_says_so_and_still_publishes_the_request(
    make_client, auth: dict[str, str], monkeypatch
) -> None:
    import dataclasses

    from crucible import manifests as manifests_module
    from .conftest import FAKE_MAC_BACKEND

    real = manifests_module.load_manifest

    def cuda_only(model_id: str):
        manifest = real(model_id)
        if model_id != pages.MODEL_ID:
            return manifest
        return dataclasses.replace(
            manifest, backends={"cuda-linux": manifest.backends["cuda-linux"]}
        )

    monkeypatch.setattr("crucible.api.routes.info.load_manifest", cuda_only)
    with make_client(backend=FAKE_MAC_BACKEND) as mac:
        block = mac.get("/v1/info", headers=auth).json()["pages_engine"]
    assert block["engine"] is None
    assert block["installed"] is False
    assert "mlx-darwin" in block["detail"]
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
    block = client.get("/v1/info", headers=auth).json()["pages_engine"]
    json.dumps(block)
    assert set(block) == {"engine", "installed", "detail", "request"}


def test_the_request_block_states_how_many_pages_may_be_in_flight(
    client: TestClient, auth: dict[str, str]
) -> None:
    block = client.get("/v1/info", headers=auth).json()["pages_engine"]
    assert block["request"]["concurrency"] == 12
    assert block["request"]["concurrency"] == pages.PAGE_CONCURRENCY


def test_the_fits_arithmetic_asks_for_the_published_number(
    client: TestClient, auth: dict[str, str]
) -> None:
    work = capabilityclasses.BY_NAME["pages"].work
    assert work is not None
    published = client.get("/v1/info", headers=auth).json()
    assert work.concurrency == published["pages_engine"]["request"]["concurrency"]
    assert "pages.py" in work.source


def test_changing_the_published_number_moves_the_arithmetic() -> None:
    source = (
        "import crucible.pages as pages\n"
        "pages.PAGE_CONCURRENCY = 3\n"
        "import crucible.capabilityclasses as capabilityclasses\n"
        "print(capabilityclasses.BY_NAME['pages'].work.concurrency)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "3", result.stderr


SDK_CLIENT_TS = (
    Path(__file__).resolve().parent.parent / "sdk" / "ts" / "src" / "client.ts"
)


def sdk_keys() -> tuple[set[str], set[str]]:
    source = SDK_CLIENT_TS.read_text(encoding="utf-8")
    start = source.index("function readPagesEngine(")
    end = re.search(r"\r?\n\}\r?\n", source[start:])
    assert end is not None, "readPagesEngine has no closing brace at column 0"
    body = source[start : start + end.start()]
    readers = (
        r"(?:str|num|bool|nullableStr|nullableNum|nullableBool|objectField"
        r"|optStr|optNum|optBool|optObject|optArray|optStrArray)"
    )
    block = set(re.findall(readers + r"\(\s*block,\s*'([a-z_]+)'", body))
    request = set(re.findall(readers + r"\(\s*request,\s*'([a-z_]+)'", body))
    return block - {"request"}, request


def test_the_sdk_reads_every_key_the_server_publishes_and_no_other() -> None:
    block_keys, request_keys = sdk_keys()
    assert request_keys == set(pages.request_shape()), (
        "sdk/ts/src/client.ts's readPagesEngine and crucible/pages.py's "
        "request_shape() have drifted. The Python is the owner; the SDK "
        "mirrors it, and a key it does not read is a fact about the weights "
        "that a client goes on pinning for itself."
    )
    published = set(
        pages.engine_block(engine=None, installed=False, detail="")
    ) - {"request"}
    assert block_keys == published, (
        "readPagesEngine and pages.engine_block() name different fields"
    )
