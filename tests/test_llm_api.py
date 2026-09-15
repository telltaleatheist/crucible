"""`load-model`, `unload-model`, `/v1/models` and the OpenAI proxy.

No GPU and no 19 GB of weights — the env, the weights and the engine are stood
up by the fixtures in conftest.py exactly as the real code paths read them. What
is *not* faked is any of the server's own logic: the preflight refusals, the
exclusive lane, the event stream and the proxy are exactly what runs on the PC.

The one thing this file cannot ask is what happens when the caller goes away
mid-request; `TestClient` has no such state. That is tests/test_proxy_disconnect.py.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator
from crucible import accelerator, jobenv
from crucible.accelerator import GIB, ComputeApp
from crucible import residency as residency_module
from crucible.manifests import load_manifest
from crucible.settle import SETTLEMENT_HOLDER

from .conftest import (
    FAKE_BACKEND,
    FAKE_MAC_BACKEND,
    a_clearance_to_hold,
    parse_sse,
)
from .fake_engine import ANSWER, DELTAS, TOOL_CALL, FakeEngine

MODEL = "qwen3.5-9b"
#: The page reader, which sorts first by id and so leads every listing.
PAGE_MODEL = "dots-ocr"
BIG_MODEL = "qwen3.8-27b"
#: The same 27B at 4 bits: the one that does fit Owen's card.
SMALL_BIG_MODEL = "qwen3.8-27b-4bit"


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def fake_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stamped `~/.crucible/envs/llm` that `env_status` accepts."""
    spec = jobenv.llm_env(FAKE_BACKEND.kind)
    directory = jobenv.env_dir(home, spec)
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (directory / "crucible-env.json").write_text(
        json.dumps(
            {
                "backend": FAKE_BACKEND.kind,
                "recipe": f"{FAKE_BACKEND.kind}.txt",
                "python_version": "3.11.16",
                "seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    pins = jobenv.recipe_pins(jobenv.recipe_for(spec))
    monkeypatch.setattr(jobenv, "installed_packages", lambda _home, _spec: dict(pins))
    return directory


@pytest.fixture
def fake_weights(home: Path) -> Callable[[str], Path]:
    """Stamp a model as pulled at exactly the revision its manifest pins."""

    def stamp(model_id: str) -> Path:
        spec = load_manifest(model_id).spec(FAKE_BACKEND.kind)
        directory = home / "models" / model_id / FAKE_BACKEND.kind
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "crucible-pull.json").write_text(
            json.dumps(
                {
                    "model": model_id,
                    "backend": FAKE_BACKEND.kind,
                    "hf_repo": spec.hf_repo,
                    "revision": spec.revision,
                    "bytes": 19_306_310_880,
                    "seconds": 300.0,
                    "pulled": "2026-09-12T19:00:00+0000",
                }
            ),
            encoding="utf-8",
        )
        return directory

    return stamp


@pytest.fixture
def idle_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (22 * GIB, 24 * GIB))


#: A card big enough for the 27B, so residency can be tested with two real
#: manifests rather than a contrived pair.
ROOMY_BACKEND = replace(
    FAKE_BACKEND,
    gpu=replace(FAKE_BACKEND.gpu, name="NVIDIA H100 80GB HBM3", vram_bytes=80 * GIB),
)


@pytest.fixture
def roomy_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (80 * GIB, 80 * GIB))


@pytest.fixture
def roomy_client(
    make_client: Callable[..., TestClient], fake_env: Path
) -> Iterator[TestClient]:
    with make_client(enable_llm=True, backend=ROOMY_BACKEND) as client:
        yield client


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch) -> list[FakeEngine]:
    """Every engine the residency builds, in order, so a test can inspect them."""
    built: list[FakeEngine] = []

    def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
        engine = FakeEngine(python, log_path)
        built.append(engine)
        return engine

    monkeypatch.setattr(residency_module, "build_engine", build)
    monkeypatch.setattr(
        residency_module,
        "engine_model_name",
        lambda engine_name, model_dir, model_id: model_id,
    )
    return built


@pytest.fixture
def llm_client(
    make_client: Callable[..., TestClient], fake_env: Path
) -> Iterator[TestClient]:
    with make_client(enable_llm=True) as client:
        yield client


def submit(client: TestClient, auth: dict[str, str], **body: Any):
    return client.post("/v1/jobs", headers=auth, json=body)


def _stream_chunks(text: str) -> list[dict[str, Any]]:
    """Every `chat.completion.chunk` of a streamed completion, in order.

    Asserts the stream ended on OpenAI's terminator on the way past, because a
    chunk list read out of a truncated stream would quietly be a shorter one.
    """
    lines = [line for line in text.split("\n") if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]", lines[-3:]
    return [json.loads(line[len("data: ") :]) for line in lines[:-1]]


def run_job(client: TestClient, auth: dict[str, str], **body: Any) -> list[dict]:
    """Submit a job and read its whole event stream. Fails loudly if refused."""
    response = submit(client, auth, **body)
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]
    with client.stream(
        "GET", f"/v1/jobs/{job_id}/events", headers=auth
    ) as stream:
        return parse_sse(line for line in stream.iter_lines())


# -------------------------------------------------------------- /v1/models


def test_models_lists_every_manifest_with_its_standing(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = llm_client.get("/v1/models", headers=auth)
    assert response.status_code == 200
    rows = {row["id"]: row for row in response.json()}
    assert [row["id"] for row in response.json()] == [
        PAGE_MODEL, MODEL, BIG_MODEL, SMALL_BIG_MODEL,
    ]
    row = rows[MODEL]
    assert row["family"] == "qwen3.5"
    assert row["params_b"] == 9
    # The pin for *this* host's backend, verbatim from the manifest: a client
    # that records what it talked to records the same sha the puller used.
    assert row["revision"] == load_manifest(MODEL).spec(FAKE_BACKEND.kind).revision
    assert row["backend_supported"] is True
    assert row["installed"] is False
    assert row["resident"] is False
    assert row["loadable"] is False
    # 16384 on this backend: the context BookForge's launcher served, not the
    # 12288 Foundry pins on Ollama for a model larger than this one.
    assert row["context_default"] == 16384
    assert row["max_model_len"] == 16384
    assert row["memory_bytes_estimate"] > 0
    assert "no weights at" in row["reason"]


def test_models_says_loadable_once_the_weights_are_there(
    llm_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path]
) -> None:
    fake_weights(MODEL)
    rows = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}
    assert rows[MODEL]["installed"] is True
    assert rows[MODEL]["loadable"] is True
    assert "reason" not in rows[MODEL]
    # The 27B has a manifest and is supported here; it is simply not pulled.
    assert rows[BIG_MODEL]["installed"] is False


def test_models_is_refused_when_llm_is_off(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_llm=False) as client:
        response = client.get("/v1/models", headers=auth)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "job_type_disabled"


def test_info_gains_an_llm_capability(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    capabilities = llm_client.get("/v1/info", headers=auth).json()["capabilities"]
    by_type = {entry["job_type"]: entry for entry in capabilities}
    assert "llm" in by_type
    assert [row["id"] for row in by_type["llm"]["models"]] == [
        PAGE_MODEL, MODEL, BIG_MODEL, SMALL_BIG_MODEL,
    ]
    # The two things you can actually POST are in `job_types`, NOT in
    # `capabilities`. They were capabilities of their own until 2026-09-13, and
    # the effect was that one model appeared three times — under `load-model`,
    # under `unload-model` and under `llm` — in two different shapes, which is
    # the thing PHASE2-LLM.md section 5 forbids in as many words.
    info = llm_client.get("/v1/info", headers=auth).json()
    assert "load-model" in info["job_types"]
    assert "unload-model" in info["job_types"]
    assert "load-model" not in by_type
    assert "unload-model" not in by_type
    ids = [row["id"] for entry in capabilities for row in entry["models"]]
    assert len(ids) == len(set(ids)), (
        "a model is described once, in one shape, wherever a client finds it"
    )


def test_the_lifecycle_types_describe_installed_as_the_models_route_does(
    llm_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path]
) -> None:
    """`load-model` and `unload-model` describe their models with DESIGN.md
    section 4's row — the one `resolve_model` and `crucible doctor` read — and
    that row's `installed` is the same stamp `/v1/models` reads, not a second
    opinion (ARCHITECTURE.md R1)."""
    store = llm_client.app.state.store
    for name in ("load-model", "unload-model"):
        rows = {d.id: d.to_dict() for d in store.registry[name].describe_models()}
        assert rows[MODEL]["installed"] is False
        assert rows[BIG_MODEL]["installed"] is False
    fake_weights(MODEL)
    served = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}
    for name in ("load-model", "unload-model"):
        rows = {d.id: d.to_dict() for d in store.registry[name].describe_models()}
        assert rows[MODEL]["installed"] is True
        assert rows[BIG_MODEL]["installed"] is False
        for model_id, row in rows.items():
            assert row["installed"] is served[model_id]["installed"], model_id
            assert row["resident"] is served[model_id]["resident"], model_id


def test_the_llm_capability_rows_are_the_models_rows(
    llm_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path]
) -> None:
    """One shape, one producer: `/info`'s llm rows *are* `/v1/models`' rows."""
    fake_weights(MODEL)
    models = llm_client.get("/v1/models", headers=auth).json()
    capabilities = llm_client.get("/v1/info", headers=auth).json()["capabilities"]
    by_type = {entry["job_type"]: entry for entry in capabilities}
    assert by_type["llm"]["models"] == models
    for row in models:
        assert row["revision"] == load_manifest(row["id"]).spec(
            FAKE_BACKEND.kind
        ).revision


def test_a_model_this_backend_cannot_serve_has_no_revision(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`revision` is the pin for *this* backend, so a model with no block here
    reports null — never the other backend's sha, and never an empty string that
    would read as a pin."""
    fixture = tmp_path / "models"
    fixture.mkdir()
    (fixture / "mac-only.toml").write_text(
        """
[model]
id = "mac-only"
family = "demo"
params_b = 1
context_default = 4096
modalities = ["text"]

[backends.mlx-darwin]
engine = "mlx-lm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CRUCIBLE_MODELS_DIR", str(fixture))
    with make_client(enable_llm=True) as client:
        rows = client.get("/v1/models", headers=auth).json()
        capabilities = client.get("/v1/info", headers=auth).json()["capabilities"]
    assert [row["id"] for row in rows] == ["mac-only"]
    assert rows[0]["backend_supported"] is False
    assert rows[0]["revision"] is None
    assert rows[0]["memory_bytes_estimate"] is None
    # And no max_model_len either: this host would not serve it at any context.
    # `context_default` still answers, because the model's own number is a fact
    # about the model rather than about a backend block that is not there.
    assert rows[0]["max_model_len"] is None
    assert rows[0]["context_default"] == 4096
    by_type = {entry["job_type"]: entry for entry in capabilities}
    assert by_type["llm"]["models"] == rows


# ---------------------------------------------------------------- fingerprint


def test_every_row_carries_the_fingerprint_a_client_records(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    """`<id>@<revision>` — the id alone does not identify bytes.

    Foundry hashes the served model id into its cleanup cache key and BookForge
    stamps it into a book's OPF (CLIENT-SURFACES.md section 6.5). The server
    spells the fingerprint out rather than leaving each client to assemble one,
    because two clients inventing two spellings is two names for one set of
    weights.
    """
    rows = llm_client.get("/v1/models", headers=auth).json()
    for row in rows:
        assert row["fingerprint"] == f"{row['id']}@{row['revision']}"
    # By id, not by position: the rows are sorted by manifest stem, so which one
    # is first changes the moment a manifest is added — `dots-ocr` took the slot
    # from `qwen3.5-9b` the day page reading landed.
    row = next(r for r in rows if r["id"] == MODEL)
    assert row["fingerprint"] == (
        f"{MODEL}@{load_manifest(MODEL).spec(FAKE_BACKEND.kind).revision}"
    )


def test_a_model_this_backend_cannot_serve_has_no_fingerprint(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Null, not the bare id: an unpinned fingerprint would look like a pin."""
    fixture = tmp_path / "models"
    fixture.mkdir()
    (fixture / "mac-only.toml").write_text(
        """
[model]
id = "mac-only"
family = "demo"
params_b = 1
context_default = 4096
modalities = ["text"]

[backends.mlx-darwin]
engine = "mlx-lm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CRUCIBLE_MODELS_DIR", str(fixture))
    with make_client(enable_llm=True) as client:
        row = client.get("/v1/models", headers=auth).json()[0]
    assert row["revision"] is None
    assert row["fingerprint"] is None


def test_the_openai_listing_names_the_weights_the_engine_actually_read(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """That entry describes the engine, so its pin is the loaded one."""
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    entry = llm_client.get("/v1/openai/models", headers=auth).json()["data"][0]
    revision = load_manifest(MODEL).spec(FAKE_BACKEND.kind).revision
    assert entry["revision"] == revision
    assert entry["fingerprint"] == f"{MODEL}@{revision}"


def test_a_provenance_sidecar_names_the_revision_it_was_served_at(
    make_app: Callable[..., Any],
    fake_env: Path,
) -> None:
    """The bug: `revision: null` on every artifact Crucible had ever written.

    The queue holds a model id and nothing that could turn it into a revision, so
    it wrote null and a sidecar named a model while declining to say which one.
    The job type knows; it is asked.

    No `llm` job produces artifacts today, so this reads the document the way the
    artifact writer does rather than fetching a file. That is the point of fixing
    it now: `tts` and `vlm-pages` are the ones that will write it into a book.
    """
    store = make_app(enable_llm=True).state.store
    spec = load_manifest(MODEL).spec(FAKE_BACKEND.kind)

    job = store.create("load-model", MODEL, {})
    assert store.provenance(job)["model"] == {
        "id": MODEL,
        "revision": spec.revision,
        "fingerprint": f"{MODEL}@{spec.revision}",
    }

    # A model-less job type still says `model: null`, which is the honest shape.
    assert store.provenance(store.create("echo", None, {}))["model"] is None


def test_a_provenance_sidecar_names_THIS_host_s_pin(
    make_app: Callable[..., Any],
    fake_env: Path,
) -> None:
    """The same model at two shas, because the weights differ per backend.

    `qwen3.5-9b` is one Crucible id over two HuggingFace repos — Qwen's own on
    cuda-linux, the bf16 conversion on mlx-darwin. A record that named the id
    without the host's pin would say the same thing about two different sets of
    bytes, which is exactly what the fingerprint exists to prevent.
    """
    mac_store = make_app(enable_llm=True, backend=FAKE_MAC_BACKEND).state.store
    mac = mac_store.provenance(mac_store.create("load-model", MODEL, {}))["model"]
    pc_store = make_app(enable_llm=True).state.store
    pc = pc_store.provenance(pc_store.create("load-model", MODEL, {}))["model"]

    assert mac["id"] == pc["id"] == MODEL
    assert mac["revision"] == load_manifest(MODEL).spec(FAKE_MAC_BACKEND.kind).revision
    assert pc["revision"] == load_manifest(MODEL).spec(FAKE_BACKEND.kind).revision
    assert mac["fingerprint"] != pc["fingerprint"]


# ------------------------------------------------------------- max_model_len


def test_the_openai_listing_reports_the_context_the_engine_was_started_with(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """`/v1/openai/models` is the door Foundry reads, so it carries the number.

    Without it `capFor` has no clamp at all and the request goes out unsized
    (CLIENT-SURFACES.md section 6.1).
    """
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    entry = llm_client.get("/v1/openai/models", headers=auth).json()["data"][0]
    assert entry["id"] == MODEL
    assert entry["max_model_len"] == 16384
    # The same number vLLM was handed as --max-model-len.
    assert engines[0].args[-2:] == ["--max-model-len", "16384"]


def test_max_model_len_follows_the_engine_and_context_default_follows_the_manifest(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_env: Path,
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """The one moment the two fields disagree, and which one is which.

    A manifest edited while its engine is up: `context_default` is the manifest's
    intent and moves with the file, `max_model_len` is what is being served and
    stays with the engine. Re-deriving `max_model_len` from the manifest would
    have this row promise a 32768-token context to a client talking to an engine
    that was started at 8192 — and the client would size a request against it and
    be refused by the engine.
    """
    fixture = tmp_path / "models"
    fixture.mkdir()
    manifest = fixture / "shifty.toml"

    def write(context: int) -> None:
        manifest.write_text(
            f"""
[model]
id = "shifty"
family = "demo"
params_b = 1
context_default = {context}
modalities = ["text"]

[backends.cuda-linux]
engine = "vllm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
""",
            encoding="utf-8",
        )

    write(8192)
    monkeypatch.setenv("CRUCIBLE_MODELS_DIR", str(fixture))
    with make_client(enable_llm=True) as client:
        spec = load_manifest("shifty", fixture).spec(FAKE_BACKEND.kind)
        weights_dir = home / "models" / "shifty" / FAKE_BACKEND.kind
        weights_dir.mkdir(parents=True)
        (weights_dir / "crucible-pull.json").write_text(
            json.dumps(
                {
                    "model": "shifty",
                    "backend": FAKE_BACKEND.kind,
                    "hf_repo": spec.hf_repo,
                    "revision": spec.revision,
                    "bytes": 3_000_000_000,
                    "seconds": 1.0,
                    "pulled": "2026-09-12T19:00:00+0000",
                }
            ),
            encoding="utf-8",
        )
        events = run_job(client, auth, type="load-model", model="shifty")
        assert events[-1]["event"] == "done", events[-1]
        assert engines[0].args == ["--max-model-len", "8192"]

        write(32768)  # somebody edits the manifest with the engine still up
        row = client.get("/v1/models", headers=auth).json()[0]
        assert row["resident"] is True
        assert row["context_default"] == 32768, "the manifest's intent moved"
        assert row["max_model_len"] == 8192, "what is being served did not"
        entry = client.get("/v1/openai/models", headers=auth).json()["data"][0]
        assert entry["max_model_len"] == 8192


# -------------------------------------------------- the refusals before queuing


def test_an_unknown_model_is_refused_by_name(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(llm_client, auth, type="load-model", model="llama9000")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_model"


def test_a_model_with_no_block_for_this_backend_is_backend_unsupported(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = tmp_path / "models"
    fixture.mkdir()
    (fixture / "mac-only.toml").write_text(
        """
[model]
id = "mac-only"
family = "demo"
params_b = 1
context_default = 4096
modalities = ["text"]

[backends.mlx-darwin]
engine = "mlx-lm"
hf_repo = "demo/Demo-1B"
revision = "0123456789abcdef0123456789abcdef01234567"
memory_bytes_estimate = 3000000000
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CRUCIBLE_MODELS_DIR", str(fixture))
    with make_client(enable_llm=True) as client:
        response = submit(client, auth, type="load-model", model="mac-only")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "backend_unsupported"
    assert "mlx-darwin" in error["message"]


def test_a_missing_env_is_env_missing(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
) -> None:
    fake_weights(MODEL)
    with make_client(enable_llm=True) as client:  # no fake_env fixture here
        response = submit(client, auth, type="load-model", model=MODEL)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "env_missing"
    assert "crucible install llm" in error["message"]


def test_missing_weights_are_model_not_installed(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(llm_client, auth, type="load-model", model=MODEL)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "model_not_installed"
    assert "crucible models pull qwen3.5-9b" in error["message"]


def test_weights_at_the_wrong_revision_are_not_installed(
    llm_client: TestClient,
    auth: dict[str, str],
    home: Path,
) -> None:
    """A manifest that moved must not serve yesterday's bytes under today's id."""
    directory = home / "models" / MODEL / FAKE_BACKEND.kind
    directory.mkdir(parents=True)
    (directory / "crucible-pull.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "backend": FAKE_BACKEND.kind,
                "hf_repo": "Qwen/Qwen3.5-9B",
                "revision": "f" * 40,
                "bytes": 1,
                "pulled": "2026-01-01T00:00:00+0000",
            }
        ),
        encoding="utf-8",
    )
    response = submit(llm_client, auth, type="load-model", model=MODEL)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "model_not_installed"
    assert "now pins" in response.json()["error"]["message"]


def test_a_busy_card_is_refused_before_the_job_exists(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_weights(MODEL)
    monkeypatch.setattr(
        accelerator,
        "probe_compute_apps",
        lambda: [ComputeApp(pid=12769, name="sgl-omni", used_bytes=17 * GIB)],
    )
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (7 * GIB, 24 * GIB))
    response = submit(llm_client, auth, type="load-model", model=MODEL)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "accelerator_busy"
    assert "sgl-omni" in error["message"]
    # Refused *before* queuing: nothing was created.
    assert llm_client.get("/v1/health", headers=auth).json()["queue_depth"] == 0


def test_the_27b_on_this_card_is_insufficient_memory(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights(BIG_MODEL)
    response = submit(llm_client, auth, type="load-model", model=BIG_MODEL)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "insufficient_memory"
    assert error["details"]["needed_bytes"] == 56_368_313_144
    assert error["details"]["total_bytes"] == FAKE_BACKEND.gpu.vram_bytes


def test_a_model_too_big_for_the_card_is_refused_before_the_download(
    llm_client: TestClient, auth: dict[str, str], idle_card: None
) -> None:
    """52.5 GiB on a 24 GiB card is not a "pull 55 GB first" problem.

    The weights are deliberately NOT stamped here: a refusal that says
    `model_not_installed` would send somebody off to download 55 GB for a model
    that can never load on this host.
    """
    response = submit(llm_client, auth, type="load-model", model=BIG_MODEL)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "insufficient_memory"
    assert "ever" in error["message"]
    assert "52.5 GiB" in error["message"]
    assert "24.0 GiB in total" in error["message"]
    assert "NVIDIA GeForce RTX 3090 Ti" in error["message"]


def test_models_says_why_the_27b_is_not_loadable_here(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    rows = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}
    assert rows[BIG_MODEL]["loadable"] is False
    assert "52.5 GiB" in rows[BIG_MODEL]["reason"]
    assert "24.0 GiB in total" in rows[BIG_MODEL]["reason"]


def test_the_4bit_27b_is_loadable_on_this_card_where_the_bf16_27b_is_not(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    """The whole point of the third manifest, on Owen's own card.

    `llm_client` is the RTX 3090 Ti: 24 GiB in total. Two manifests for the same
    27B — same family, same params_b — and the host answers differently about
    each, because the answer is arithmetic about the weights each one points at
    and not about the model's name.
    """
    fake_weights(SMALL_BIG_MODEL)
    fake_weights(BIG_MODEL)
    rows = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}

    small = rows[SMALL_BIG_MODEL]
    assert small["family"] == rows[BIG_MODEL]["family"] == "qwen3.8"
    assert small["params_b"] == rows[BIG_MODEL]["params_b"] == 27
    assert small["backend_supported"] is True
    assert small["installed"] is True
    assert small["loadable"] is True
    assert "reason" not in small
    # THIS HOST's context, not the model's. The model wants Owen's
    # `qwen3.8:27b-24g` 98304 and gets it on mlx-darwin; on a 24 GB card that is
    # 7.9 GiB of KV the card does not have, so the cuda-linux block carries its
    # own 16384 and that is what vLLM is given as --max-model-len.
    assert load_manifest(SMALL_BIG_MODEL).context_default == 98304
    assert small["context_default"] == 16384
    # Nothing is resident, so what this host WOULD serve it at is the whole
    # answer, and the two fields agree.
    assert small["max_model_len"] == 16384
    assert small["memory_bytes_estimate"] == 21_633_171_456
    assert small["revision"] == (
        load_manifest(SMALL_BIG_MODEL).spec(FAKE_BACKEND.kind).revision
    )

    assert rows[BIG_MODEL]["loadable"] is False
    assert "52.5 GiB" in rows[BIG_MODEL]["reason"]

    # And the refusal the listing predicts is the refusal the load makes.
    response = submit(llm_client, auth, type="load-model", model=BIG_MODEL)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "insufficient_memory"


def test_the_4bit_27b_actually_loads_on_a_free_24_gib_card(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
    engines: list[FakeEngine],
) -> None:
    """Not merely "the card is big enough" — the guard lets it through.

    `loadable` in `/v1/models` compares the estimate against the card's TOTAL, so
    it cannot answer "right now". This does: an empty 24 GiB card, and the live
    guard passes the 23.3 GiB estimate. The margin is 0.7 GiB, which is why the
    `idle_card` fixture — 22 GiB free, the Windows desktop holding the rest — is
    deliberately not used here. On Owen's real card, with his desktop up, this
    load is expected to be tight; see the manifest's comment.
    """
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(
        accelerator,
        "probe_vram",
        lambda: (FAKE_BACKEND.gpu.vram_bytes, FAKE_BACKEND.gpu.vram_bytes),
    )
    fake_weights(SMALL_BIG_MODEL)
    events = run_job(llm_client, auth, type="load-model", model=SMALL_BIG_MODEL)
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["resident"] == SMALL_BIG_MODEL
    assert len(engines) == 1
    # vLLM is told the context the manifest promises, and nothing forces a dtype:
    # compressed-tensors W4A16 carries its own, and `dtype auto` is what it wants.
    # `--max-model-len` is the cuda-linux block's own 16384, not the model's
    # 98304: on a 24 GB card 98304 of KV is 7.9 GiB that is not there.
    # `--language-model-only` is BookForge's `--limit-mm-per-prompt 0`, carried
    # across on 2026-09-15: this checkpoint is multimodal, the `llm` lane sends it
    # nothing but text, and without the flag vLLM reads 921_460_192 B of vision
    # tower onto the card and holds it for the life of the engine.
    assert engines[0].args == [
        "--gpu-memory-utilization", "0.86",
        "--max-num-seqs", "16",
        "--skip-mm-profiling",
        "--language-model-only",
        "--max-model-len", "16384",
    ]


def test_unknown_params_are_refused(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    fake_weights(MODEL)
    response = submit(
        llm_client, auth, type="load-model", model=MODEL, params={"eager": True}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"


# ------------------------------------------------------ the load/unload machine


def test_a_load_warms_then_reports_the_resident_model(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    events = run_job(llm_client, auth, type="load-model", model=MODEL)
    kinds = [event["event"] for event in events]

    assert kinds[0] == "queued"
    assert kinds[-1] == "done"
    assert "warming" in kinds
    # Several warming events, streamed from the engine's readiness (section 5).
    assert kinds.count("warming") >= 3
    # Event ids are strictly increasing from 1, as DESIGN.md section 4 requires.
    assert [event["id"] for event in events] == list(range(1, len(events) + 1))

    warmings = [e["data"]["message"] for e in events if e["event"] == "warming"]
    assert any("checking the accelerator" in m for m in warmings)
    assert any("22.0 GiB free" in m for m in warmings)
    assert any("fake engine warming" in m for m in warmings)
    assert any("is resident at" in m for m in warmings)

    assert events[-1]["data"]["resident"] == MODEL
    assert len(engines) == 1


def test_health_and_models_see_the_resident_model(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)

    health = llm_client.get("/v1/health", headers=auth).json()
    assert health["resident_models"] == [MODEL]
    assert health["status"] == "ok"

    rows = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}
    assert rows[MODEL]["resident"] is True
    assert rows[BIG_MODEL]["resident"] is False

    listed = llm_client.get("/v1/openai/models", headers=auth).json()
    assert listed["object"] == "list"
    assert [entry["id"] for entry in listed["data"]] == [MODEL]


def test_loading_a_second_model_unloads_the_first(
    roomy_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    roomy_card: None,
    engines: list[FakeEngine],
) -> None:
    """Phase 2 residency rule: one resident model at a time (section 3)."""
    llm_client = roomy_client
    fake_weights(MODEL)
    fake_weights(BIG_MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    events = run_job(llm_client, auth, type="load-model", model=BIG_MODEL)

    messages = [e["data"]["message"] for e in events if e["event"] == "warming"]
    assert any("unloading qwen3.5-9b" in m for m in messages)
    assert events[-1]["data"]["resident"] == BIG_MODEL
    assert len(engines) == 2
    assert engines[0].stopped is True
    assert engines[1].stopped is False
    assert llm_client.get("/v1/health", headers=auth).json()["resident_models"] == [
        BIG_MODEL
    ]


def test_unload_frees_the_engine(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    events = run_job(llm_client, auth, type="unload-model", model=MODEL)

    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["resident"] is None
    assert engines[0].stopped is True
    assert llm_client.get("/v1/health", headers=auth).json()["resident_models"] == []
    assert llm_client.get("/v1/openai/models", headers=auth).json()["data"] == []


def test_unloading_a_model_that_is_not_resident_is_refused(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    response = submit(llm_client, auth, type="unload-model", model=MODEL)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "model_not_resident"
    assert "no model is" in error["message"]

    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = submit(llm_client, auth, type="unload-model", model=BIG_MODEL)
    assert response.status_code == 409
    assert response.json()["error"]["details"]["resident"] == MODEL


# ------------------------------- unloading what is already being unloaded
#
# T6, 2026-09-15, on a live card. The page was read, the settlement began
# clearing `dots-ocr` the instant the last chat completion finished, and the
# same client's `unload-model dots-ocr` — a few milliseconds behind it, in a
# `finally` — came back `409 engine_in_use`, *"held by 'the settlement clearing
# the card'"*. The stage failed on its own tidying up and its message overwrote
# the page it had just read.
#
# The three tests below are the whole ruling: the settlement is the same intent
# and is answered; every other holder is a conflict and is still refused, by the
# name it was refused by before.


def test_unloading_what_the_settlement_is_clearing_is_the_same_intent(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """T6's exact race: the clearance is under way and the client's unload lands.

    It is `done` with the card clear, because that is what the client asked for
    and it is what happened. `engine_in_use` would be naming the client's own
    tidying up as somebody else's conversation.
    """
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    reached, release = a_clearance_to_hold(engines[0])

    settlement = llm_client.app.state.settlement
    settled: list[Any] = []
    clearing = threading.Thread(
        target=lambda: settled.append(
            settlement.settle_quietly("the last chat completion finished")
        ),
        name="the-settlement",
        daemon=True,
    )
    clearing.start()
    assert reached.wait(timeout=10), "the settlement never reached the engine"
    # The state T6 arrived in, pinned: the card IS claimed, and by the holder
    # whose name was in the 409. This is what makes the 202 below the fix.
    assert llm_client.app.state.residency.claimed_by == SETTLEMENT_HOLDER

    # THE MOMENT T6 FAILED IN. Admitted, not refused.
    response = submit(llm_client, auth, type="unload-model", model=MODEL)
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]

    release.set()
    clearing.join(timeout=30)
    assert not clearing.is_alive()
    assert settled[0] is not None and settled[0].subject_id == MODEL

    with llm_client.stream(
        "GET", f"/v1/jobs/{job_id}/events", headers=auth
    ) as stream:
        events = parse_sse(line for line in stream.iter_lines())
    assert events[-1]["event"] == "done", events[-1]
    assert events[-1]["data"]["resident"] is None
    # And the card really is clear — the job did not merely say so.
    assert llm_client.get("/v1/health", headers=auth).json()["resident_models"] == []
    assert engines[0].stopped is True


def test_unloading_under_a_holder_that_is_using_the_card_is_still_engine_in_use(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """The refusal keeps its whole meaning for a genuinely different holder.

    A streaming session is on narrator's one stdin and one stdout. Taking the
    engine off the card now ends its conversation mid-sentence, which is exactly
    what `engine_in_use` is for — and it is unchanged.
    """
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    residency = llm_client.app.state.residency
    residency.claim("tts stream abc123", may_mutate=False)
    try:
        response = submit(llm_client, auth, type="unload-model", model=MODEL)
    finally:
        residency.release("tts stream abc123")
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "engine_in_use"
    assert error["details"]["held_by"] == "tts stream abc123"
    # Nothing was taken off the card on the way past.
    assert engines[0].stopped is False


def test_unloading_under_another_client_s_lease_is_still_leased(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """The other holder that refuses an unload, and by its own name.

    A lease is a client saying it has more work on this model. That refusal is
    `leased` and is answered before the type's preflight is even asked, so the
    clearance exception cannot reach it.
    """
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    lease = llm_client.post(
        f"/v1/models/{MODEL}/lease",
        headers=auth,
        json={"act": "clean", "ttl_seconds": 60},
    )
    assert lease.status_code == 201, lease.text

    response = submit(llm_client, auth, type="unload-model", model=MODEL)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "leased"
    assert engines[0].stopped is False


def test_health_says_warming_while_a_load_is_in_flight(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PHASE2-LLM.md section 5: `/health` reports `warming` while a load runs.

    The engine is held inside `ready()` so the server is looked at with a load
    genuinely in flight, rather than racing a sleep. The job's event stream is
    deliberately NOT open while `/health` is read: nesting a request inside a
    TestClient stream answers from a stale view of the store.
    """
    hold = threading.Event()
    built: list[FakeEngine] = []

    def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
        engine = FakeEngine(python, log_path, hold=hold)
        built.append(engine)
        return engine

    monkeypatch.setattr(residency_module, "build_engine", build)
    monkeypatch.setattr(
        residency_module,
        "engine_model_name",
        lambda engine_name, model_dir, model_id: model_id,
    )
    fake_weights(MODEL)

    response = submit(llm_client, auth, type="load-model", model=MODEL)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if built and built[0].warming_started.wait(timeout=0.1):
                break
            time.sleep(0.05)  # the engine is not built yet; let the lane run
        assert built and built[0].warming_started.is_set(), "the lane never started"

        health = llm_client.get("/v1/health", headers=auth).json()
        assert health["status"] == "warming", health
        assert health["queue_depth"] == 1, health
        assert health["resident_models"] == [], health
        assert (
            llm_client.get(f"/v1/jobs/{job_id}", headers=auth).json()["status"]
            == "running"
        )
    finally:
        hold.set()

    with llm_client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        events = parse_sse(line for line in stream.iter_lines())
    assert events[-1]["event"] == "done"
    assert llm_client.get("/v1/health", headers=auth).json() == {
        "status": "ok",
        "queue_depth": 0,
        "resident_models": [MODEL],
        "resident_kind": "llm",
    }


def test_an_engine_that_never_becomes_ready_fails_the_job(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[FakeEngine] = []

    def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
        engine = FakeEngine(
            python, log_path, fail_ready="vllm exited 1 before it was ready"
        )
        built.append(engine)
        return engine

    monkeypatch.setattr(residency_module, "build_engine", build)
    monkeypatch.setattr(
        residency_module,
        "engine_model_name",
        lambda engine_name, model_dir, model_id: model_id,
    )
    fake_weights(MODEL)
    events = run_job(llm_client, auth, type="load-model", model=MODEL)
    assert events[-1]["event"] == "failed"
    assert events[-1]["data"]["error"]["code"] == "engine_failed"
    assert "exited 1" in events[-1]["data"]["error"]["message"]
    # Nothing is left resident and the half-started engine was stopped.
    assert llm_client.get("/v1/health", headers=auth).json()["resident_models"] == []
    assert built[0].stopped is True


# ------------------------------------------------------------------- the proxy


def test_the_proxy_refuses_a_model_that_is_not_resident(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "model_not_resident"
    assert "no model is" in error["message"]
    assert "never loads a model to answer a chat request" in error["message"]
    assert error["details"] == {"requested": MODEL, "resident": None}


def test_the_proxy_names_the_resident_model_in_the_409(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "model_not_resident"
    assert "'qwen3.5-9b' is" in error["message"]
    assert error["details"] == {"requested": "gpt-4", "resident": MODEL}
    # And nothing was loaded to satisfy it.
    assert len(engines) == 1


def test_a_non_streamed_completion_is_passed_through(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is Crucible?"}],
            "temperature": 0.2,
            "max_tokens": 32,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == ANSWER
    assert body["usage"]["total_tokens"] == 12
    # The sampling the client asked for reached the engine unchanged.
    sent = engines[0].last_request
    assert sent["temperature"] == 0.2
    assert sent["max_tokens"] == 32
    assert sent["messages"] == [{"role": "user", "content": "What is Crucible?"}]


def test_a_streamed_completion_keeps_its_sse_framing(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    with llm_client.stream(
        "POST",
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is Crucible?"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())

    chunks = _stream_chunks(text)
    deltas = [
        chunk["choices"][0]["delta"]["content"]
        for chunk in chunks
        if "content" in chunk["choices"][0]["delta"]
    ]
    assert deltas == DELTAS
    assert "".join(deltas) == ANSWER
    # The closing frame carries no delta, only the reason the engine stopped.
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    # Every frame is terminated by a blank line, as SSE requires.
    assert text.endswith("data: [DONE]\n\n")


@pytest.fixture
def engines_under_a_path_name(
    monkeypatch: pytest.MonkeyPatch, home: Path
) -> list[FakeEngine]:
    """Engines that answer to the weights directory, the way mlx-lm does.

    The other `engines` fixture makes the engine's name and the Crucible id the
    same string, which is true of vLLM (`--served-model-name`) and hides the
    substitution entirely. This is the other backend's shape.
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
        lambda engine_name, model_dir, model_id: str(Path(model_dir).resolve()),
    )
    return built


def test_a_completion_comes_back_naming_crucible_s_id_not_the_engine_s(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines_under_a_path_name: list[FakeEngine],
) -> None:
    """The proxy substitutes `model` on the way in; it undoes it on the way out."""
    weights = fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    # The engine was asked by the name it answers to...
    assert engines_under_a_path_name[0].last_request["model"] == str(weights.resolve())
    # ...and the client is answered by the name it asked with.
    assert response.json()["model"] == MODEL
    # Everything else is the engine's own body, untouched.
    assert response.json()["choices"][0]["message"]["content"] == ANSWER


def test_every_streamed_chunk_names_crucible_s_id(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines_under_a_path_name: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    with llm_client.stream(
        "POST",
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())

    chunks = _stream_chunks(text)
    # Every frame, the closing one included — the relabelling reaches all of them.
    assert [chunk["model"] for chunk in chunks] == [MODEL] * (len(DELTAS) + 1)
    # The framing and the content survived the relabelling.
    assert [
        chunk["choices"][0]["delta"]["content"]
        for chunk in chunks
        if "content" in chunk["choices"][0]["delta"]
    ] == DELTAS
    assert text.endswith("data: [DONE]\n\n")


# ------------------------------------------- the constrained transport survives

#: A Foundry analyze verdict, as `askConstrained` builds it (CLIENT-SURFACES.md
#: section 6.2), plus every other knob the OpenAI dialect defines that a client
#: might one day send. The point of the extra fields is that "verbatim" is a rule
#: about the whole body and not about the four keys Crucible happens to know:
#: `logit_bias` and `top_logprobs` are here precisely because nothing in either
#: app sends them today, so nothing in the proxy has ever been taught about them.
CONSTRAINED_BODY: dict[str, Any] = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Does the passage support the claim?"}],
    "temperature": 0,
    "max_tokens": 128,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "verdict",
            "schema": {
                "type": "object",
                "properties": {
                    "supported": {"type": "boolean"},
                    "quote": {"type": "string", "maxLength": 200},
                },
                "required": ["supported", "quote"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    "chat_template_kwargs": {"enable_thinking": False},
    "logprobs": True,
    "top_logprobs": 5,
    "seed": 1729,
    "stop": ["\n\n", "</answer>"],
    "logit_bias": {"15496": -100},
}


def _post_raw(
    client: TestClient, auth: dict[str, str], body: dict[str, Any]
) -> tuple[bytes, Any]:
    """POST a chat body as exact bytes, so byte identity is a question you can ask."""
    payload = json.dumps(body).encode("utf-8")
    response = client.post(
        "/v1/openai/chat/completions",
        headers={**auth, "Content-Type": "application/json"},
        content=payload,
    )
    return payload, response


def test_a_constrained_body_reaches_the_engine_byte_for_byte(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """On vLLM there is nothing to substitute, so nothing is re-encoded.

    `response_format.json_schema.schema` is a grammar the guided-decoding backend
    compiles. Crucible round-tripping it through `json.loads`/`json.dumps` would
    be a re-encoding of somebody else's document on the way past — harmless until
    the day it is not. Here the engine reads the client's own bytes.
    """
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    payload, response = _post_raw(llm_client, auth, CONSTRAINED_BODY)

    assert response.status_code == 200, response.text
    assert engines[0].last_request_bytes == payload


def test_only_the_model_field_changes_when_the_engine_answers_to_a_path(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines_under_a_path_name: list[FakeEngine],
) -> None:
    """The other backend's shape: one field substituted, nothing else touched.

    mlx-lm has no `--served-model-name`, so `model` has to change and the
    document is re-serialised. Everything else — the schema, `strict`, the
    template kwargs, the knobs Crucible has never heard of — arrives with the
    same value in the same position.
    """
    weights = fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    _, response = _post_raw(llm_client, auth, CONSTRAINED_BODY)
    assert response.status_code == 200, response.text

    sent = engines_under_a_path_name[0].last_request
    assert sent == {**CONSTRAINED_BODY, "model": str(weights.resolve())}
    # Order too: `model` keeps the place it held, so nothing is appended or
    # shuffled on the way through.
    assert list(sent) == list(CONSTRAINED_BODY)


@pytest.mark.parametrize("reason", ["stop", "length", "tool_calls"])
def test_finish_reason_comes_back_untouched(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
    reason: str,
) -> None:
    """Foundry turns `length` into a degradation rather than a wrong answer.

    A proxy that normalised or dropped the field would turn a caught truncation
    into silent corruption (CLIENT-SURFACES.md section 6.2), so the engine's own
    word is what comes back — including on a `tool_calls` completion, whose
    `content` is null and whose answer is not text at all.
    """
    engine_factory(finish_reason=reason)
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == reason
    if reason == "tool_calls":
        assert choice["message"]["content"] is None
        assert choice["message"]["tool_calls"] == [TOOL_CALL]


@pytest.mark.parametrize("reason", ["stop", "length", "tool_calls"])
def test_a_streamed_finish_reason_comes_back_untouched(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
    reason: str,
) -> None:
    """The same rule on the streaming half, where it lives in the closing frame."""
    engine_factory(finish_reason=reason)
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    with llm_client.stream(
        "POST",
        "/v1/openai/chat/completions",
        headers=auth,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())

    chunks = _stream_chunks(text)
    assert chunks[-1]["choices"][0]["finish_reason"] == reason
    # And it is the ONLY frame that names one: the deltas keep their null.
    assert [chunk["choices"][0]["finish_reason"] for chunk in chunks[:-1]] == [
        None
    ] * len(DELTAS)


def test_an_engine_s_own_400_is_relayed_rather_than_rewritten(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
) -> None:
    """A schema the engine will not compile is the engine's refusal to explain.

    Rewrapped in Crucible's `{"error": {"code", "message"}}` envelope it would
    read as the server refusing, and the message naming the part of the grammar
    to fix would be gone.
    """
    engine_factory(reject_response_format=True)
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    _, response = _post_raw(llm_client, auth, CONSTRAINED_BODY)

    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "BadRequestError"
    assert "prefixItems" in body["message"]
    # Not Crucible's envelope: this refusal is not Crucible's to make.
    assert "error" not in body


def test_an_engine_s_own_400_is_relayed_on_a_streamed_request_too(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
) -> None:
    """The stream is opened before anything is returned, so a refusal is a refusal.

    It must never come back as a 200 whose stream turns out to be an error.
    """
    engine_factory(reject_response_format=True)
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    _, response = _post_raw(llm_client, auth, {**CONSTRAINED_BODY, "stream": True})

    assert response.status_code == 400
    assert response.json()["type"] == "BadRequestError"


def test_the_proxy_requires_a_model(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_required"


def test_the_proxy_needs_auth(llm_client: TestClient) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers={"X-Crucible-Api": "1"},
        json={"model": MODEL, "messages": []},
    )
    assert response.status_code == 401


def test_a_body_that_is_not_json_is_refused(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers={**auth, "Content-Type": "application/json"},
        content=b"not json",
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


# ------------------------------------- a chat is work, and work must be visible


def test_an_unknown_act_is_refused_BEFORE_the_work_rather_than_mislabelled(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """A wrong act name on a bench is worse than no act name.

    Owen, 2026-09-13: *"they can't lie to the user and say a translate job is
    running when it's actually a simplify job."* A silently-accepted typo would
    do exactly that, so the header is validated against the capability classes
    and refused by name — and refused BEFORE the completion runs, so nobody pays
    for a 27B pass that is then reported under a name nothing knows.
    """
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers={**auth, "X-Crucible-Act": "translat"},
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "unknown_act"
    assert "'translat'" in error["message"]
    # The refusal names the vocabulary rather than leaving a caller to guess it.
    assert "simplify" in error["message"] and "translate" in error["message"]


def test_a_chat_with_no_act_header_records_null_rather_than_a_guess(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """This door is OpenAI-shaped and a generic client cannot know Crucible's
    vocabulary, so the header is optional. What is NOT optional is honesty about
    its absence: the server cannot tell a simplify from a translate — both are a
    chat against the same model, differing only in a prompt it does not own — so
    an absent header is `null`, never an inferred act."""
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200, response.text
    # And the entry is GONE once the completion is over: an entry that outlived
    # its request would make this server look permanently busy with work that
    # stopped.
    body = llm_client.get("/v1/activity", headers=auth).json()
    assert body["chat"] == {"in_flight": 0, "rows": []}


def test_a_chat_in_flight_is_visible_and_still_does_not_take_the_lane(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """The defect, and the shape of the fix.

    A chat took no lane, made no job and left no record, so a server grinding
    through a 27B translation reported `running: []` and read as idle to every
    bench polling it. It is counted now — and it still gates nothing, because a
    vLLM engine BATCHES: two passes on one resident model genuinely run at once,
    and taking the lane to fix a reporting bug would serialise work the engine
    exists to overlap.
    """
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    inflight = llm_client.app.state.inflight

    seen: dict[str, object] = {}
    with inflight.tracked(act="simplify", model=MODEL, client="foundry/0.9.0"):
        body = llm_client.get("/v1/activity", headers=auth).json()
        seen.update(body)

    assert seen["chat"]["in_flight"] == 1
    row = seen["chat"]["rows"][0]
    # The act is named as what it IS. Before Crucible everything ran under
    # "translate"; nothing may report a simplify as one.
    assert row["act"] == "simplify"
    assert row["model"] == MODEL
    assert row["client"] == "foundry/0.9.0"
    assert row["since"]

    # THE OTHER HALF. The lane is free and says so, and the machine still
    # accepts work — because it really does.
    assert seen["slots"]["accelerated"]["busy"] == 0
    assert seen["slots"]["accelerated"]["accepts_work"] is True
    assert seen["running"] == []

def test_the_openai_surface_is_also_mounted_where_openai_clients_look(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    """`/openai/v1/models` and `/openai/v1/chat/completions` are the same doors as
    `/v1/openai/...`, at the path every OpenAI client composes.

    Found 2026-09-13 by the first real Foundry act against a Crucible: its
    engine appends `/v1` to a base that does not end in a version, asked for
    `/v1/openai/v1/models`, and got a 404 from a server that had the door. Same
    handler, same token, same version header — only the path differs, and it is
    the OTHER protocol's convention.
    """
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    ours = llm_client.get("/v1/openai/models", headers=auth).json()
    theirs = llm_client.get("/openai/v1/models", headers=auth).json()
    assert theirs == ours
    assert theirs["data"][0]["id"] == MODEL
    # The same refusals, by the same names, at the new path.
    no_model = llm_client.post("/openai/v1/chat/completions", headers=auth, json={"messages": []})
    assert no_model.status_code == 400
    assert no_model.json()["error"]["code"] == "model_required"
    other = llm_client.post(
        "/openai/v1/chat/completions", headers=auth, json={"model": "not-this-one", "messages": []}
    )
    assert other.status_code == 409
    assert other.json()["error"]["code"] == "model_not_resident"
    # And the same lock: no token, no door.
    assert llm_client.get("/openai/v1/models").status_code == 401

