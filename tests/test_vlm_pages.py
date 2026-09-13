"""Page reading: `modalities`, the dots.ocr manifest, and an image through the proxy.

PHASE3-VLM.md section 6. There is no `vlm-pages` job type and nothing here asks
for one — Foundry reads a page by POSTing an ordinary chat completion whose first
content part is a data-URI PNG, so the whole of this phase is: say which models
take pictures, refuse the one engine flag that would quietly break them, and
prove that an image part survives the proxy that already exists.

The proxy is what these tests are really about, and they are deliberately hard to
pass by accident. A data URI is the largest thing Crucible will ever carry — an
11.3 MP page at Foundry's pinned 200 dpi is about 11 MB of base64 — and the
twelve-in-flight test uses a barrier inside the engine rather than a stopwatch,
because nothing else can tell "twelve at once" from "twelve quickly".

No GPU and no weights: the env, the weights and the engine are stood up exactly
as tests/test_llm_api.py stands them up, and the phase-2 fixtures are imported
rather than restated so there is one description of a stamped host and not two.
"""

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

from crucible.jobs.llm import residency as residency_module
from crucible.manifests import (
    MODALITIES,
    SKIP_MM_PROFILING,
    ManifestError,
    load_manifest,
    parse_manifest,
)

from .conftest import FAKE_MAC_BACKEND
from .fake_engine import ANSWER, FakeEngine

# The phase-2 host fixtures, used as they are. Ruff sees them as unused imports;
# pytest sees them as fixtures this module declares, which is what makes
# `llm_client` resolvable here at all.
from .test_llm_api import (  # noqa: F401
    fake_env,
    fake_weights,
    idle_card,
    llm_client,
    run_job,
)

PAGE_MODEL = "dots-ocr"
TEXT_MODEL = "qwen3.5-9b"

#: Foundry's page geometry, from CLIENT-SURFACES.md section 7.2: `VLM_DPI = 200`
#: is pinned in `foundry/src/vlm/read.ts`, and a 468x760 pt page rasterises to
#: exactly this. The number that matters downstream is what it weighs as base64.
PAGE_WIDTH, PAGE_HEIGHT = 1300, 2112

#: Foundry's ungated default (`DEFAULT_VLM_CONCURRENCY`), which BookForge takes by
#: passing `concurrency: 0`. Twelve is therefore the number a Crucible has to
#: carry, not a number chosen here to look like a load test.
IN_FLIGHT = 12

#: How long a request waits at the barrier for the other eleven. Generous, because
#: what it is measuring is the difference between "concurrent" and "never", not a
#: throughput figure — a proxy that serialises does not arrive late, it does not
#: arrive.
BARRIER_SECONDS = 30.0


# --------------------------------------------------------------- a real page


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
    )


def page_png(width: int = PAGE_WIDTH, height: int = PAGE_HEIGHT) -> bytes:
    """A real PNG of the size Foundry actually sends.

    Built rather than checked in, and built honestly: a well-formed 8-bit RGB PNG
    with the signature, IHDR, IDAT and IEND a decoder needs. The pixels are
    pseudo-random from a fixed seed and the deflate level is 0, so the bytes are
    incompressible the way a scanned page's are and the size is the same on every
    machine — about 8.2 MB, which is about 11.0 MB once it is base64 in a data
    URI. A 40-byte stub would prove nothing: the thing under test is size.
    """
    rng = random.Random(20260913)
    raw = bytearray()
    for _ in range(height):
        raw.append(0)  # PNG filter type 0 (None) for this scanline
        raw.extend(rng.randbytes(width * 3))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 0))
        + _chunk(b"IEND", b"")
    )


def page_request(model: str, data_uri: str, prompt: str = "layout-all") -> dict[str, Any]:
    """Foundry's body, field for field (CLIENT-SURFACES.md section 7.1).

    `temperature: 0` because "a layout answer is a measurement of a page", and the
    image part comes first — the order the model's chat template reads.
    """
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


# ------------------------------------------------------------------ fixtures


class PageEngines:
    """The engines a test's residency built, plus a hook inside the completion.

    `on_post` runs on the engine's own serving thread before anything is
    answered, which is where a barrier has to sit if it is going to prove that
    twelve requests were in the engine at the same moment.
    """

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

    monkeypatch.setattr(residency_module, "build_engine", build)
    # vLLM's own behaviour: it takes `--served-model-name`, so the engine answers
    # to the Crucible id and the proxy's one substitution is a no-op.
    # `test_only_the_model_field_is_rewritten` covers the other backend.
    monkeypatch.setattr(
        residency_module,
        "engine_model_name",
        lambda engine_name, model_dir, model_id: model_id,
    )
    return engines


@pytest.fixture
def resident_page_model(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    page_engines: PageEngines,
) -> Iterator[tuple[TestClient, FakeEngine]]:
    """dots.ocr loaded and answering, on a card with room for it."""
    fake_weights(PAGE_MODEL)
    events = run_job(llm_client, auth, type="load-model", model=PAGE_MODEL)
    assert events[-1]["event"] == "done", events[-1]
    yield llm_client, page_engines.built[0]


# ------------------------------------------------------------- `modalities`


def test_the_page_model_is_the_only_one_offered_for_images() -> None:
    """The fact a client picks a page reader by, instead of knowing a name."""
    assert "image" in load_manifest(PAGE_MODEL).modalities
    for text_only in ("qwen3.5-9b", "qwen3.8-27b", "qwen3.8-27b-4bit"):
        assert load_manifest(text_only).modalities == ("text",)


def test_modalities_reaches_the_models_row_and_the_info_capability(
    llm_client: TestClient, auth: dict[str, str]  # noqa: F811
) -> None:
    """One shape, one producer — the `/info` llm rows *are* the `/v1/models` rows."""
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
    fake_env: Path,  # noqa: F811
) -> None:
    """It is a fact about the model, not about the host.

    `revision` and `memory_bytes_estimate` go null when this backend has no block,
    because both of them live in the block. `modalities` lives in `[model]`, so
    the Mac — which has no `dots-ocr` block at all until its 4-bit repo is
    measured — still reports truthfully what the model is offered for. A client
    reading the list to find something that takes pictures gets the same answer
    everywhere, and a separate `backend_supported: false` telling it where.
    """
    with make_client(enable_llm=True, backend=FAKE_MAC_BACKEND) as client:
        rows = {row["id"]: row for row in client.get("/v1/models", headers=auth).json()}
    row = rows[PAGE_MODEL]
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


# --------------------------------------------- the `--skip-mm-profiling` rule


def test_an_image_model_may_not_skip_multimodal_profiling() -> None:
    """PHASE3-VLM.md section 3, made enforceable instead of remembered.

    `qwen3.5-9b`'s manifest has always carried the condition — "if the `llm` proxy
    is ever given image input, this line must come out and the utilisation be
    measured again". This is the loader keeping it, and it names both halves so
    the message says which file and which two lines disagree.
    """
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
    """The rule is about image models; the two manifests that carry it keep it."""
    manifest = parse_manifest(
        _manifest(modalities='["text"]', engine_args=f'["{SKIP_MM_PROFILING}"]'),
        Path("demo-1b.toml"),
        "demo-1b",
    )
    assert manifest.spec("cuda-linux").engine_args == (SKIP_MM_PROFILING,)
    for text_only in ("qwen3.5-9b", "qwen3.8-27b-4bit"):
        args = load_manifest(text_only).spec("cuda-linux").engine_args
        assert SKIP_MM_PROFILING in args


# ------------------------------------------------------- the dots.ocr manifest


def test_the_page_manifest_meets_all_four_hard_requirements() -> None:
    """PHASE3-VLM.md section 4, which states all four exactly."""
    manifest = load_manifest(PAGE_MODEL)
    spec = manifest.spec("cuda-linux")
    args = spec.engine_args

    # Twelve concurrent pages minimum: Foundry's ungated default, which BookForge
    # takes by passing `concurrency: 0`.
    assert "--max-num-seqs" in args
    assert int(args[args.index("--max-num-seqs") + 1]) >= IN_FLIGHT

    # 32k, because 3,450 image tokens plus the prompt plus an 8192-token answer do
    # not fit less. Asserted through `context_for`, which is what vLLM's
    # `--max-model-len` and the `/v1/models` row both read.
    assert manifest.context_for("cuda-linux") == 32768
    assert args[args.index("--max-model-len") + 1] == "32768"

    # dots.ocr ships its modeling class in its own repo.
    assert "--trust-remote-code" in args

    # And the flag that would leave nothing reserved for the page.
    assert SKIP_MM_PROFILING not in args


def test_the_page_manifest_pins_a_revision_and_not_a_branch() -> None:
    """Read from the hub, not invented: `dots-studio/dots.ocr` at `main`.

    The repo was renamed — `rednote-hilab/dots.ocr`, the name both apps still
    send and the name PHASE3-VLM.md was written with, answers a 307 to
    `dots-studio/dots.ocr` and `?author=rednote-hilab` lists nothing. Both names
    report this same commit, so the manifest pins the repo as it is named now
    rather than depending on somebody else's redirect.
    """
    spec = load_manifest(PAGE_MODEL).spec("cuda-linux")
    assert spec.hf_repo == "dots-studio/dots.ocr"
    assert spec.revision == "c0111ce6bc07803dbc267932ffef0ae3a51dc951"


def test_the_page_manifest_has_no_mac_block() -> None:
    """`mlx-local` is the Mac's only page-reading route and this does not touch it."""
    manifest = load_manifest(PAGE_MODEL)
    assert sorted(manifest.backends) == ["cuda-linux"]


def test_the_page_model_fits_a_24_gib_card() -> None:
    """The declared reservation, which is half the card and not the whole of it."""
    spec = load_manifest(PAGE_MODEL).spec("cuda-linux")
    assert spec.memory_bytes_estimate < 24 * 1024 ** 3
    # Above the floor the weights plus one context of KV come to, or the guard
    # would wave through a load the engine cannot complete: 6_078_431_736 B of
    # bf16 safetensors at the pinned sha, plus 28 layers x 2 KV heads x 128 x 2 x
    # 2 bytes x 32768 tokens.
    assert spec.memory_bytes_estimate > 6_078_431_736 + 939_524_096


# ----------------------------------------------------- an image through the proxy


def test_an_image_part_reaches_the_engine_unchanged(
    resident_page_model: tuple[TestClient, FakeEngine], auth: dict[str, str]
) -> None:
    """A list of content parts, with a real page in it, arrives as it was sent.

    The proxy promises to be verbatim but for `model`. A content part is the first
    thing it has ever been asked to carry that is not a string, and a data URI is
    the first thing large enough to meet a limit nobody wrote down.
    """
    client, engine = resident_page_model
    data_uri = "data:image/png;base64," + base64.b64encode(page_png()).decode("ascii")
    # About 11.0 MB. The assertion is on the order of magnitude, not the exact
    # figure, so a change to the PNG helper does not silently shrink the subject.
    assert len(data_uri) > 10_000_000

    body = page_request(PAGE_MODEL, data_uri)
    response = client.post("/v1/openai/chat/completions", headers=auth, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == ANSWER
    assert response.json()["model"] == PAGE_MODEL

    # On cuda-linux the engine answers to the Crucible id, so there is nothing the
    # proxy had to substitute and the body it received is the body that was sent.
    assert engine.requests == [body]
    parts = engine.requests[0]["messages"][0]["content"]
    assert parts[0]["image_url"]["url"] == data_uri
    assert parts[1] == {"type": "text", "text": "layout-all"}
    assert engine.requests[0]["temperature"] == 0
    assert engine.requests[0]["max_tokens"] == 8192


def test_an_eleven_megabyte_data_uri_is_not_truncated(
    resident_page_model: tuple[TestClient, FakeEngine], auth: dict[str, str]
) -> None:
    """Size, asserted on the bytes that arrived and not only on the JSON.

    A truncated body would most often fail to parse and never reach a comparison
    of values at all, so this also counts what the engine actually read.

    "Verbatim" is not "byte for byte" here and the assertion says which: the proxy
    decodes the request and httpx re-encodes it, so the body on the second hop is
    the same JSON with compact separators. On this body that is MEASURED at
    exactly 18 bytes — one per `", "` and `": "` — out of 10_986_363, which is
    every separator in it and nothing else. Asserting against the compact
    encoding rather than against a tolerance is what makes that a statement about
    whitespace instead of a statement about roughly the right size.
    """
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
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mlx-lm case, where the engine answers to a path and not to an id.

    mlx-lm has no `--served-model-name`, so the proxy substitutes `model` on the
    way in and puts Crucible's id back on the way out. This is the test that says
    the substitution is still exactly one field when the body carries an image:
    everything else, content parts included, is the caller's bytes.
    """
    built: list[FakeEngine] = []

    def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
        engine = FakeEngine(python, log_path)
        built.append(engine)
        return engine

    monkeypatch.setattr(residency_module, "build_engine", build)
    monkeypatch.setattr(
        residency_module,
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
    # And the answer names Crucible's id, not the server's disk.
    assert response.json()["model"] == PAGE_MODEL


def test_twelve_pages_are_in_the_engine_at_once(
    resident_page_model: tuple[TestClient, FakeEngine],
    auth: dict[str, str],
    page_engines: PageEngines,
) -> None:
    """The proxy adds no serialisation of its own.

    `load-model` runs on the exclusive lane so a chat can never race a load; chat
    does not, and this is the test that says so. The engine batches — that is what
    vLLM is for — and a lock in front of it that Crucible did not need would turn
    Foundry's twelve workers into one.

    Proved with a barrier and not a stopwatch: every request stops inside the
    engine until all twelve have arrived. A proxy that serialised would not be
    slow here, it would never release the first request at all, and the barrier
    would break rather than the clock running long.
    """
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
    # Twelve distinct pages arrived, none dropped and none sent twice.
    prompts = sorted(
        request["messages"][0]["content"][1]["text"] for request in engine.requests
    )
    assert prompts == sorted(f"page {index}" for index in range(IN_FLIGHT))


def test_a_page_sent_to_a_text_model_is_still_refused_by_name(
    llm_client: TestClient,  # noqa: F811
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],  # noqa: F811
    idle_card: None,  # noqa: F811
    page_engines: PageEngines,
) -> None:
    """`modalities` advertises; residency still decides, and says so.

    A client that reads the model list and picks the page reader is the intended
    path. A client that guesses gets the same 409 by name that phase 2 gives, and
    not an engine error about content parts several layers down.
    """
    fake_weights(TEXT_MODEL)
    run_job(llm_client, auth, type="load-model", model=TEXT_MODEL)
    data_uri = "data:image/png;base64," + base64.b64encode(page_png(64, 64)).decode()
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json=page_request(PAGE_MODEL, data_uri),
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "model_not_resident"
    assert error["details"] == {"requested": PAGE_MODEL, "resident": TEXT_MODEL}


# ------------------------------------------------------------------- helpers


def _manifest(
    *, modalities: str = '["text"]', engine_args: str = '["--dtype", "bfloat16"]'
) -> str:
    """A minimal, valid manifest with the two fields these tests vary."""
    return f"""
[model]
id = "demo-1b"
family = "demo"
params_b = 1
context_default = 4096
modalities = {modalities}

[backends.cuda-linux]
engine = "vllm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
engine_args = {engine_args}
"""
