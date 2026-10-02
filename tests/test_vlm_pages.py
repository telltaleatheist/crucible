from __future__ import annotations

import base64
import binascii
import json
import random
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import engines as engines_module
from crucible.manifests import (
    LANGUAGE_MODEL_ONLY,
    MODALITIES,
    SKIP_MM_PROFILING,
    ManifestError,
    load_manifest,
    parse_manifest,
)

from .conftest import FAKE_MAC_BACKEND
from .fake_engine import ANSWER, FakeEngine
from .test_llm_api import llm_client, run_job  # noqa: F401 - a fixture this module uses

PAGE_MODEL = "dots-ocr"
TEXT_MODEL = "qwen3.5-9b"
CUDA_ONLY_VISION_MODEL = "qwen3.8-27b-4bit-vl"

PAGE_WIDTH, PAGE_HEIGHT = 1300, 2112

IN_FLIGHT = 12

BARRIER_SECONDS = 30.0


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def page_png(width: int = PAGE_WIDTH, height: int = PAGE_HEIGHT) -> bytes:
    rng = random.Random(20260913)
    raw = bytearray()
    for _ in range(height):
        raw.append(0)
        raw.extend(rng.randbytes(width * 3))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 0))
        + _chunk(b"IEND", b"")
    )


def page_request(model: str, data_uri: str, prompt: str = "layout-all") -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": 8192,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }


class PageEngines:

    def __init__(self) -> None:
        self.built: list[FakeEngine] = []
        self.on_post: Callable[[], None] | None = None

    def _hook(self) -> None:
        if self.on_post is not None:
            self.on_post()


@pytest.fixture
def page_engines(monkeypatch: pytest.MonkeyPatch) -> PageEngines:
    engines = PageEngines()

    def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
        engine = FakeEngine(python, log_path, on_post=engines._hook)
        engines.built.append(engine)
        return engine

    monkeypatch.setattr(engines_module, "build_engine", build)
    monkeypatch.setattr(
        engines_module,
        "engine_model_name",
        lambda engine_name, model_dir, model_id: model_id,
    )
    return engines


@pytest.fixture
def resident_page_model(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    page_engines: PageEngines,
) -> Iterator[tuple[TestClient, FakeEngine]]:
    fake_weights(PAGE_MODEL)
    events = run_job(llm_client, auth, type="load-model", model=PAGE_MODEL)
    assert events[-1]["event"] == "done", events[-1]
    yield llm_client, page_engines.built[0]


def test_the_page_model_is_the_only_one_offered_for_images() -> None:
    assert "image" in load_manifest(PAGE_MODEL).modalities
    for text_only in ("qwen3.5-9b", "qwen3.8-27b-8bit", "qwen3.8-27b-4bit"):
        assert load_manifest(text_only).modalities == ("text",)


def test_modalities_reaches_the_models_row_and_the_info_capability(
    llm_client: TestClient, auth: dict[str, str]  # noqa: F811
) -> None:
    rows = llm_client.get("/v1/models", headers=auth).json()
    by_id = {row["id"]: row for row in rows}
    assert by_id[PAGE_MODEL]["modalities"] == ["text", "image"]
    assert by_id[TEXT_MODEL]["modalities"] == ["text"]
    capabilities = llm_client.get("/v1/info", headers=auth).json()["capabilities"]
    by_type = {entry["job_type"]: entry for entry in capabilities}
    assert by_type["llm"]["models"] == rows


def test_modalities_is_not_null_on_a_host_that_cannot_serve_the_model(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_env: Path,
) -> None:
    with make_client(enable_llm=True, backend=FAKE_MAC_BACKEND) as client:
        rows = {row["id"]: row for row in client.get("/v1/models", headers=auth).json()}
    row = rows[CUDA_ONLY_VISION_MODEL]
    assert row["backend_supported"] is False
    assert row["revision"] is None
    assert row["memory_bytes_estimate"] is None
    assert row["modalities"] == ["text", "image"]


@pytest.mark.parametrize(
    "declared, expected",
    [
        ("[]", "modalities is empty"),
        ('["text", "sound"]', "modalities[1] is 'sound'"),
        ('["text", "text"]', "lists a modality twice"),
        ("[1]", "modalities[0] must be a string, got int"),
    ],
)
def test_a_modality_the_server_does_not_know_is_refused(
    declared: str, expected: str
) -> None:
    with pytest.raises(ManifestError) as caught:
        parse_manifest(_manifest(modalities=declared), Path("demo-1b.toml"), "demo-1b")
    assert expected in str(caught.value)


def test_the_known_modalities_are_the_two_a_chat_part_can_be() -> None:
    assert MODALITIES == frozenset({"text", "image"})


def test_an_image_model_may_not_skip_multimodal_profiling() -> None:
    with pytest.raises(ManifestError) as caught:
        parse_manifest(
            _manifest(
                modalities='["text", "image"]',
                engine_args=f'["--trust-remote-code", "{SKIP_MM_PROFILING}"]',
            ),
            Path("demo-1b.toml"),
            "demo-1b",
        )
    message = str(caught.value)
    assert SKIP_MM_PROFILING in message
    assert "modalities declares 'image'" in message
    assert "demo-1b.toml [backends.cuda-linux]" in message


def test_a_text_only_model_keeps_the_flag_and_keeps_the_1_9_gib() -> None:
    manifest = parse_manifest(
        _manifest(modalities='["text"]', engine_args=f'["{SKIP_MM_PROFILING}"]'),
        Path("demo-1b.toml"),
        "demo-1b",
    )
    assert manifest.spec("cuda-linux").engine_args == (SKIP_MM_PROFILING,)
    for text_only in ("qwen3.5-9b", "qwen3.8-27b-4bit"):
        args = load_manifest(text_only).spec("cuda-linux").engine_args
        assert SKIP_MM_PROFILING in args


def test_an_image_model_may_not_be_served_language_model_only() -> None:
    with pytest.raises(ManifestError) as caught:
        parse_manifest(
            _manifest(
                modalities='["text", "image"]',
                engine_args=f'["--trust-remote-code", "{LANGUAGE_MODEL_ONLY}"]',
            ),
            Path("demo-1b.toml"),
            "demo-1b",
        )
    message = str(caught.value)
    assert LANGUAGE_MODEL_ONLY in message
    assert "modalities declares 'image'" in message
    assert "demo-1b.toml [backends.cuda-linux]" in message


def test_the_text_models_do_not_load_a_vision_tower_they_never_use() -> None:
    for text_only in ("qwen3.5-9b", "qwen3.8-27b-4bit"):
        manifest = load_manifest(text_only)
        assert "image" not in manifest.modalities, text_only
        assert LANGUAGE_MODEL_ONLY in manifest.spec("cuda-linux").engine_args


def test_the_page_manifest_meets_all_four_hard_requirements() -> None:
    manifest = load_manifest(PAGE_MODEL)
    spec = manifest.spec("cuda-linux")
    args = spec.engine_args

    assert "--max-num-seqs" in args
    assert int(args[args.index("--max-num-seqs") + 1]) >= IN_FLIGHT

    assert manifest.context_for("cuda-linux") == 32768
    assert args[args.index("--max-model-len") + 1] == "32768"

    assert "--trust-remote-code" in args

    assert SKIP_MM_PROFILING not in args


def test_the_page_manifest_pins_a_revision_and_not_a_branch() -> None:
    spec = load_manifest(PAGE_MODEL).spec("cuda-linux")
    assert spec.hf_repo == "dots-studio/dots.ocr"
    assert spec.revision == "c0111ce6bc07803dbc267932ffef0ae3a51dc951"


def test_the_page_manifest_serves_every_backend_and_the_mac_on_mlx_vlm() -> None:
    manifest = load_manifest(PAGE_MODEL)
    assert sorted(manifest.backends) == ["cuda-linux", "llama-windows", "mlx-darwin"]
    assert manifest.spec("mlx-darwin").engine == "mlx-vlm"


def test_the_page_model_fits_a_24_gib_card() -> None:
    spec = load_manifest(PAGE_MODEL).spec("cuda-linux")
    assert spec.memory_bytes_estimate < 24 * 1024 ** 3
    assert spec.memory_bytes_estimate > 6_078_431_736 + 939_524_096


def test_an_image_part_reaches_the_engine_unchanged(
    resident_page_model: tuple[TestClient, FakeEngine], auth: dict[str, str]
) -> None:
    client, engine = resident_page_model
    data_uri = "data:image/png;base64," + base64.b64encode(page_png()).decode("ascii")
    assert len(data_uri) > 10_000_000

    body = page_request(PAGE_MODEL, data_uri)
    response = client.post("/v1/openai/chat/completions", headers=auth, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == ANSWER
    assert response.json()["model"] == PAGE_MODEL

    assert engine.requests == [body]
    parts = engine.requests[0]["messages"][0]["content"]
    assert parts[0]["image_url"]["url"] == data_uri
    assert parts[1] == {"type": "text", "text": "layout-all"}
    assert engine.requests[0]["temperature"] == 0
    assert engine.requests[0]["max_tokens"] == 8192


def test_an_eleven_megabyte_data_uri_is_not_truncated(
    resident_page_model: tuple[TestClient, FakeEngine], auth: dict[str, str]
) -> None:
    client, engine = resident_page_model
    png = page_png()
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    body = page_request(PAGE_MODEL, data_uri)

    response = client.post("/v1/openai/chat/completions", headers=auth, json=body)
    assert response.status_code == 200, response.text

    arrived = engine.requests[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert len(arrived) == len(data_uri)
    assert arrived == data_uri
    assert base64.b64decode(arrived.split(",", 1)[1]) == png

    compact = len(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    assert engine.request_bytes[0] == compact


def test_only_the_model_field_is_rewritten(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[FakeEngine] = []

    def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
        engine = FakeEngine(python, log_path)
        built.append(engine)
        return engine

    monkeypatch.setattr(engines_module, "build_engine", build)
    monkeypatch.setattr(
        engines_module,
        "engine_model_name",
        lambda engine_name, model_dir, model_id: f"/home/owen/.crucible/models/{model_id}",
    )
    fake_weights(PAGE_MODEL)
    run_job(llm_client, auth, type="load-model", model=PAGE_MODEL)

    data_uri = "data:image/png;base64," + base64.b64encode(page_png(64, 64)).decode()
    body = page_request(PAGE_MODEL, data_uri)
    response = llm_client.post("/v1/openai/chat/completions", headers=auth, json=body)
    assert response.status_code == 200, response.text

    arrived = built[0].requests[0]
    assert arrived["model"] == f"/home/owen/.crucible/models/{PAGE_MODEL}"
    assert {k: v for k, v in arrived.items() if k != "model"} == {
        k: v for k, v in body.items() if k != "model"
    }
    assert response.json()["model"] == PAGE_MODEL


def test_twelve_pages_are_in_the_engine_at_once(
    resident_page_model: tuple[TestClient, FakeEngine],
    auth: dict[str, str],
    page_engines: PageEngines,
) -> None:
    client, engine = resident_page_model
    barrier = threading.Barrier(IN_FLIGHT)
    serialised = threading.Event()

    def wait_for_the_others() -> None:
        try:
            barrier.wait(timeout=BARRIER_SECONDS)
        except threading.BrokenBarrierError:
            serialised.set()

    page_engines.on_post = wait_for_the_others

    def read_page(index: int) -> Any:
        data_uri = "data:image/png;base64," + base64.b64encode(
            page_png(64, 64)
        ).decode("ascii")
        return client.post(
            "/v1/openai/chat/completions",
            headers=auth,
            json=page_request(PAGE_MODEL, data_uri, prompt=f"page {index}"),
        )

    with ThreadPoolExecutor(max_workers=IN_FLIGHT) as pool:
        responses = list(pool.map(read_page, range(IN_FLIGHT)))

    assert not serialised.is_set(), (
        f"fewer than {IN_FLIGHT} requests were ever inside the engine at one time; "
        "something between the client and the engine is taking a lock"
    )
    assert [r.status_code for r in responses] == [200] * IN_FLIGHT
    for response in responses:
        assert response.json()["choices"][0]["message"]["content"] == ANSWER
    prompts = sorted(
        request["messages"][0]["content"][1]["text"] for request in engine.requests
    )
    assert prompts == sorted(f"page {index}" for index in range(IN_FLIGHT))


def test_a_page_sent_to_a_text_model_is_still_refused_by_name(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    page_engines: PageEngines,
) -> None:
    fake_weights(TEXT_MODEL)
    run_job(llm_client, auth, type="load-model", model=TEXT_MODEL)
    data_uri = "data:image/png;base64," + base64.b64encode(page_png(64, 64)).decode()
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={**page_request(PAGE_MODEL, data_uri), "queue": False},
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "model_not_resident"
    assert error["details"] == {"requested": PAGE_MODEL, "resident": TEXT_MODEL}


def _manifest(
    *, modalities: str = '["text"]', engine_args: str = '["--dtype", "bfloat16"]'
) -> str:
    return f"""
[model]
id = "demo-1b"
family = "demo"
params_b = 1
context_default = 4096
trained_context = 262144
modalities = {modalities}

[backends.cuda-linux]
engine = "vllm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
engine_args = {engine_args}
"""
