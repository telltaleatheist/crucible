from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, jobenv, tasks
from crucible import engines as engines_module
from crucible import residency as residency_module
from crucible.accelerator import ComputeApp
from crucible.config import DEFAULT_DESKTOP_ALLOWANCE_BYTES
from crucible.engines import ENGINES
from crucible.engines.vllm import DECIDE_ARGS as VLLM_DECIDE_ARGS
from crucible.manifests import load_manifest
from crucible.memorybudget import GIB
from crucible.settle import SETTLEMENT_HOLDER

from .conftest import (
    FAKE_BACKEND,
    FAKE_MAC_BACKEND,
    a_clearance_to_hold,
    parse_sse,
    write_env_stamp,
)
from .fake_engine import ANSWER, DELTAS, TOOL_CALL, FakeEngine

MODEL = "qwen3.5-9b"
PAGE_MODEL = "dots-ocr"
MAC_ONLY_MODEL = "qwen3.8-27b-8bit"
SMALL_BIG_MODEL = "qwen3.8-27b-4bit"


@pytest.fixture
def fake_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    spec = jobenv.llm_env(FAKE_BACKEND.kind)
    directory = jobenv.env_dir(home, spec)
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    write_env_stamp(home, jobenv.llm_env(FAKE_BACKEND.kind), FAKE_BACKEND.kind)
    pins = jobenv.recipe_pins(jobenv.recipe_for(spec))
    monkeypatch.setattr(jobenv, "installed_packages", lambda _home, _spec: dict(pins))
    return directory


@pytest.fixture
def fake_weights(home: Path) -> Callable[[str], Path]:

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


ROOMY_BACKEND = replace(
    FAKE_BACKEND,
    gpu=replace(FAKE_BACKEND.gpu, name="NVIDIA H100 80GB HBM3", vram_bytes=80 * GIB),
)

SMALL_CARD_BACKEND = replace(
    FAKE_BACKEND,
    gpu=replace(FAKE_BACKEND.gpu, name="NVIDIA GeForce RTX 3060", vram_bytes=12 * GIB),
)


@pytest.fixture
def small_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (11 * GIB, 12 * GIB))


@pytest.fixture
def small_card_client(
    make_client: Callable[..., TestClient], fake_env: Path
) -> Iterator[TestClient]:
    with make_client(enable_llm=True, backend=SMALL_CARD_BACKEND) as client:
        yield client


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
    built: list[FakeEngine] = []

    def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
        engine = FakeEngine(python, log_path)
        built.append(engine)
        return engine

    monkeypatch.setattr(engines_module, "build_engine", build)
    monkeypatch.setattr(
        engines_module,
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
    lines = [line for line in text.split("\n") if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]", lines[-3:]
    return [json.loads(line[len("data: ") :]) for line in lines[:-1]]


def run_job(client: TestClient, auth: dict[str, str], **body: Any) -> list[dict]:
    response = submit(client, auth, **body)
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]
    with client.stream(
        "GET", f"/v1/jobs/{job_id}/events", headers=auth
    ) as stream:
        return parse_sse(line for line in stream.iter_lines())


def test_models_lists_every_manifest_with_its_standing(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = llm_client.get("/v1/models", headers=auth)
    assert response.status_code == 200
    rows = {row["id"]: row for row in response.json()}
    assert [row["id"] for row in response.json()] == [
        PAGE_MODEL, "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", MODEL, "qwen3.5-9b-vl",
        SMALL_BIG_MODEL, "qwen3.8-27b-4bit-vl", MAC_ONLY_MODEL,
    ]
    row = rows[MODEL]
    assert row["family"] == "qwen3.5"
    assert row["params_b"] == 9
    assert row["revision"] == load_manifest(MODEL).spec(FAKE_BACKEND.kind).revision
    assert row["backend_supported"] is True
    assert row["installed"] is False
    assert row["resident"] is False
    assert row["loadable"] is False
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
    assert rows[SMALL_BIG_MODEL]["installed"] is False


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
        PAGE_MODEL, "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", MODEL, "qwen3.5-9b-vl",
        SMALL_BIG_MODEL, "qwen3.8-27b-4bit-vl", MAC_ONLY_MODEL,
    ]
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
    store = llm_client.app.state.store
    for name in ("load-model", "unload-model"):
        rows = {d.id: d.to_dict() for d in store.registry[name].describe_models()}
        assert rows[MODEL]["installed"] is False
        assert rows[SMALL_BIG_MODEL]["installed"] is False
    fake_weights(MODEL)
    served = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}
    for name in ("load-model", "unload-model"):
        rows = {d.id: d.to_dict() for d in store.registry[name].describe_models()}
        assert rows[MODEL]["installed"] is True
        assert rows[SMALL_BIG_MODEL]["installed"] is False
        for model_id, row in rows.items():
            assert row["installed"] is served[model_id]["installed"], model_id
            assert row["resident"] is served[model_id]["resident"], model_id


def test_the_llm_capability_rows_are_the_models_rows(
    llm_client: TestClient, auth: dict[str, str], fake_weights: Callable[[str], Path]
) -> None:
    fake_weights(MODEL)
    models = llm_client.get("/v1/models", headers=auth).json()
    capabilities = llm_client.get("/v1/info", headers=auth).json()["capabilities"]
    by_type = {entry["job_type"]: entry for entry in capabilities}
    assert by_type["llm"]["models"] == models
    for row in models:
        manifest = load_manifest(row["id"])
        if not manifest.supports(FAKE_BACKEND.kind):
            assert row["backend_supported"] is False, row["id"]
            assert row["revision"] is None, row["id"]
            continue
        assert row["revision"] == manifest.spec(FAKE_BACKEND.kind).revision


def test_a_model_this_backend_cannot_serve_has_no_revision(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
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
trained_context = 262144
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
    assert rows[0]["max_model_len"] is None
    assert rows[0]["context_default"] == 4096
    by_type = {entry["job_type"]: entry for entry in capabilities}
    assert by_type["llm"]["models"] == rows


def test_every_row_carries_the_fingerprint_a_client_records(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    rows = llm_client.get("/v1/models", headers=auth).json()
    for row in rows:
        if not row["backend_supported"]:
            assert row["revision"] is None and row["fingerprint"] is None, row
            continue
        assert row["fingerprint"] == f"{row['id']}@{row['revision']}"
    assert any(not row["backend_supported"] for row in rows)
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
    fixture = tmp_path / "models"
    fixture.mkdir()
    (fixture / "mac-only.toml").write_text(
        """
[model]
id = "mac-only"
family = "demo"
params_b = 1
context_default = 4096
trained_context = 262144
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
    store = make_app(enable_llm=True).state.store
    spec = load_manifest(MODEL).spec(FAKE_BACKEND.kind)

    job = store.create("load-model", MODEL, {})
    assert store.provenance(job)["model"] == {
        "id": MODEL,
        "revision": spec.revision,
        "fingerprint": f"{MODEL}@{spec.revision}",
    }

    assert store.provenance(store.create("echo", None, {}))["model"] is None


def test_a_provenance_sidecar_names_THIS_host_s_pin(
    make_app: Callable[..., Any],
    fake_env: Path,
) -> None:
    mac_store = make_app(enable_llm=True, backend=FAKE_MAC_BACKEND).state.store
    mac = mac_store.provenance(mac_store.create("load-model", MODEL, {}))["model"]
    pc_store = make_app(enable_llm=True).state.store
    pc = pc_store.provenance(pc_store.create("load-model", MODEL, {}))["model"]

    assert mac["id"] == pc["id"] == MODEL
    assert mac["revision"] == load_manifest(MODEL).spec(FAKE_MAC_BACKEND.kind).revision
    assert pc["revision"] == load_manifest(MODEL).spec(FAKE_BACKEND.kind).revision
    assert mac["fingerprint"] != pc["fingerprint"]


def test_the_openai_listing_reports_the_context_the_engine_was_started_with(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    entry = llm_client.get("/v1/openai/models", headers=auth).json()["data"][0]
    assert entry["id"] == MODEL
    assert entry["max_model_len"] == 16384
    args = engines[0].args
    assert args[args.index("--max-model-len") + 1] == "16384"


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
# The weights' own wall, well clear of either context this test writes, so
# the two it is actually about are the only ones that move.
trained_context = 262144
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
        assert engines[0].args == ["--max-model-len", "8192", *VLLM_DECIDE_ARGS]

        write(32768)
        row = client.get("/v1/models", headers=auth).json()[0]
        assert row["resident"] is True
        assert row["context_default"] == 32768, "the manifest's intent moved"
        assert row["max_model_len"] == 8192, "what is being served did not"
        entry = client.get("/v1/openai/models", headers=auth).json()["data"][0]
        assert entry["max_model_len"] == 8192


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
trained_context = 262144
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


def test_a_missing_env_is_installed_on_submit(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_weights(MODEL)
    monkeypatch.setattr(tasks, "install_command", lambda: sys.executable)
    with make_client(enable_llm=True) as client:
        response = submit(client, auth, type="load-model", model=MODEL)
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "installing"
        assert "installing the llm environment" in error["message"]
        started = client.app.state.tasks.get(error["details"]["task_id"])
        assert started.request["module"]["job_types"] == [{"type": "llm"}]


def test_missing_weights_are_model_not_installed(
    llm_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(llm_client, auth, type="load-model", model=MODEL)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "installing"
    assert "pulling the model 'qwen3.5-9b'" in error["message"]


def test_weights_at_the_wrong_revision_are_not_installed(
    llm_client: TestClient,
    auth: dict[str, str],
    home: Path,
) -> None:
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
    assert response.json()["error"]["code"] == "installing"
    assert "pulling the model 'qwen3.5-9b'" in response.json()["error"]["message"]


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
    assert llm_client.get("/v1/health", headers=auth).json()["queue_depth"] == 0


def _stamp_without_a_block(home: Path, model_id: str) -> Path:
    directory = home / "models" / model_id / FAKE_BACKEND.kind
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "crucible-pull.json").write_text(
        json.dumps(
            {
                "model": model_id,
                "backend": FAKE_BACKEND.kind,
                "hf_repo": "Qwen/Qwen3.8-27B-FP8",
                "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
                "bytes": 30_866_866_928,
                "seconds": 300.0,
                "pulled": "2026-09-17T19:00:00+0000",
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_the_8bit_27b_is_not_offered_on_cuda_linux_even_with_weights_on_disk(
    llm_client: TestClient,
    auth: dict[str, str],
    home: Path,
    idle_card: None,
) -> None:
    _stamp_without_a_block(home, MAC_ONLY_MODEL)
    response = submit(llm_client, auth, type="load-model", model=MAC_ONLY_MODEL)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "backend_unsupported"
    assert error["details"] == {
        "model": MAC_ONLY_MODEL,
        "backend": FAKE_BACKEND.kind,
        "declared": ["mlx-darwin"],
    }
    rows = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}
    row = rows[MAC_ONLY_MODEL]
    assert row["backend_supported"] is False
    assert row["installed"] is False
    assert row["loadable"] is False
    assert "has no cuda-linux block" in row["reason"]
    assert "mlx-darwin" in row["reason"]


def test_a_model_too_big_for_the_card_is_refused_before_the_download(
    small_card_client: TestClient, auth: dict[str, str], small_card: None
) -> None:
    response = submit(
        small_card_client, auth, type="load-model", model=SMALL_BIG_MODEL
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "insufficient_memory"
    assert "ever" in error["message"]
    assert "20.1 GiB" in error["message"]
    assert "12.0 GiB in total" in error["message"]
    assert "NVIDIA GeForce RTX 3060" in error["message"]
    assert error["details"]["needed_bytes"] == 21_633_171_456
    assert error["details"]["total_bytes"] == SMALL_CARD_BACKEND.gpu.vram_bytes


def test_models_says_why_a_model_is_not_loadable_on_a_small_card(
    small_card_client: TestClient, auth: dict[str, str]
) -> None:
    rows = {
        row["id"]: row
        for row in small_card_client.get("/v1/models", headers=auth).json()
    }
    assert rows[SMALL_BIG_MODEL]["loadable"] is False
    assert "20.1 GiB" in rows[SMALL_BIG_MODEL]["reason"]
    assert "12.0 GiB in total" in rows[SMALL_BIG_MODEL]["reason"]


def test_the_4bit_27b_is_loadable_on_this_card_and_the_8bit_is_not_offered(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    home: Path,
    idle_card: None,
) -> None:
    fake_weights(SMALL_BIG_MODEL)
    _stamp_without_a_block(home, MAC_ONLY_MODEL)
    rows = {row["id"]: row for row in llm_client.get("/v1/models", headers=auth).json()}

    small = rows[SMALL_BIG_MODEL]
    assert small["family"] == rows[MAC_ONLY_MODEL]["family"] == "qwen3.8"
    assert small["params_b"] == rows[MAC_ONLY_MODEL]["params_b"] == 27
    assert small["backend_supported"] is True
    assert small["installed"] is True
    assert small["loadable"] is True
    assert "reason" not in small
    assert load_manifest(SMALL_BIG_MODEL).context_default == 98304
    assert small["context_default"] == 16384
    assert small["max_model_len"] == 16384
    assert small["memory_bytes_estimate"] == 21_633_171_456
    assert small["revision"] == (
        load_manifest(SMALL_BIG_MODEL).spec(FAKE_BACKEND.kind).revision
    )

    assert rows[MAC_ONLY_MODEL]["backend_supported"] is False
    assert rows[MAC_ONLY_MODEL]["loadable"] is False

    response = submit(llm_client, auth, type="load-model", model=MAC_ONLY_MODEL)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "backend_unsupported"


def test_the_4bit_27b_actually_loads_on_a_free_24_gib_card(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
    engines: list[FakeEngine],
) -> None:
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
    terms = load_manifest(SMALL_BIG_MODEL).spec(FAKE_BACKEND.kind).memory
    assert terms is not None
    total = FAKE_BACKEND.gpu.vram_bytes
    budget = total - DEFAULT_DESKTOP_ALLOWANCE_BYTES
    pool = min(budget - terms.fixed_bytes, terms.kv_bytes_per_token * 16384 * 16)
    assert engines[0].args == [
        "--gpu-memory-utilization", "0.86",
        "--max-num-seqs", "16",
        "--skip-mm-profiling",
        "--language-model-only",
        "--max-model-len", "16384",
        *VLLM_DECIDE_ARGS,
        "--kv-cache-memory-bytes", str(pool),
        "--gpu-memory-utilization", f"{budget / total:.4f}",
    ]
    assert pool > terms.kv_bytes_per_token * 16384


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
    assert kinds.count("warming") >= 3
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
    assert rows[SMALL_BIG_MODEL]["resident"] is False

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
    llm_client = roomy_client
    fake_weights(MODEL)
    fake_weights(SMALL_BIG_MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    events = run_job(llm_client, auth, type="load-model", model=SMALL_BIG_MODEL)

    messages = [e["data"]["message"] for e in events if e["event"] == "warming"]
    assert any("unloading qwen3.5-9b" in m for m in messages)
    assert events[-1]["data"]["resident"] == SMALL_BIG_MODEL
    assert len(engines) == 2
    assert engines[0].stopped is True
    assert engines[1].stopped is False
    assert llm_client.get("/v1/health", headers=auth).json()["resident_models"] == [
        SMALL_BIG_MODEL
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
    response = submit(llm_client, auth, type="unload-model", model=SMALL_BIG_MODEL)
    assert response.status_code == 409
    assert response.json()["error"]["details"]["resident"] == MODEL


def test_unloading_what_the_settlement_is_clearing_is_the_same_intent(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
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
    assert llm_client.app.state.residency.claimed_by == SETTLEMENT_HOLDER

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
    assert llm_client.get("/v1/health", headers=auth).json()["resident_models"] == []
    assert engines[0].stopped is True


def test_unloading_under_a_holder_that_is_using_the_card_is_still_engine_in_use(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
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
    assert engines[0].stopped is False


def test_unloading_under_another_client_s_queue_session_is_refused(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    opened = llm_client.post(
        "/v1/queue/sessions",
        headers={**auth, "X-Crucible-Client": "briefcase"},
        json={"act": "clean"},
    )
    assert opened.status_code == 202 and opened.json()["status"] == "open", opened.text

    response = submit(llm_client, auth, type="unload-model", model=MODEL)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "server_busy"
    assert response.json()["error"]["details"]["session_id"] == opened.json()["session_id"]
    assert engines[0].stopped is False


def _a_clearance_under_way(
    client: TestClient, engine: FakeEngine
) -> tuple[threading.Event, threading.Thread, list[Any]]:
    reached, release = a_clearance_to_hold(engine)
    settled: list[Any] = []
    clearing = threading.Thread(
        target=lambda: settled.append(
            client.app.state.settlement.settle_quietly("the lease was released")
        ),
        name="the-settlement",
        daemon=True,
    )
    clearing.start()
    assert reached.wait(timeout=10), "the settlement never reached the engine"
    assert client.app.state.residency.claimed_by == SETTLEMENT_HOLDER
    return release, clearing, settled


def _a_door_waiting(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> threading.Event:
    residency = client.app.state.residency
    waiting = threading.Event()
    wait = residency.await_settled

    def instrumented(what: str, *, timeout: float) -> None:
        waiting.set()
        wait(what, timeout=timeout)

    monkeypatch.setattr(residency, "await_settled", instrumented)
    return waiting


def _sent_during(
    request: Callable[[], Any],
    waiting: threading.Event,
    release: threading.Event,
    clearing: threading.Thread,
) -> Any:
    answers: list[Any] = []
    sender = threading.Thread(target=lambda: answers.append(request()), daemon=True)
    sender.start()
    assert waiting.wait(timeout=10), "the door never waited for the clearance"
    assert not answers, "the door answered before the clearance finished"
    release.set()
    clearing.join(timeout=30)
    assert not clearing.is_alive()
    sender.join(timeout=30)
    assert not sender.is_alive(), "the door is still waiting after the clearance"
    return answers[0]


def test_a_load_submitted_during_a_clearance_waits_it_out_and_ends_done(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    release, clearing, settled = _a_clearance_under_way(llm_client, engines[0])
    waiting = _a_door_waiting(llm_client, monkeypatch)

    response = _sent_during(
        lambda: submit(llm_client, auth, type="load-model", model=MODEL),
        waiting,
        release,
        clearing,
    )
    assert response.status_code == 202, response.json()
    assert settled[0] is not None and settled[0].subject_id == MODEL

    with llm_client.stream(
        "GET", f"/v1/jobs/{response.json()['job_id']}/events", headers=auth
    ) as stream:
        events = parse_sse(line for line in stream.iter_lines())
    assert events[-1]["event"] == "done", events[-1]
    assert engines[0].stopped is True
    assert len(engines) == 2 and engines[1].stopped is False
    assert llm_client.get("/v1/health", headers=auth).json()["resident_models"] == [
        MODEL
    ]


def test_a_session_opened_on_a_model_during_a_clearance_waits_and_loads_it_again(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    release, clearing, _ = _a_clearance_under_way(llm_client, engines[0])
    waiting = _a_door_waiting(llm_client, monkeypatch)

    response = _sent_during(
        lambda: llm_client.post(
            "/v1/queue/sessions", headers=auth, json={"act": "clean", "model": MODEL},
        ),
        waiting,
        release,
        clearing,
    )
    assert response.status_code == 202, response.json()
    session_id = response.json()["session_id"]
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        state = llm_client.get(f"/v1/queue/sessions/{session_id}", headers=auth).json()
        if state["status"] != "queued":
            break
        time.sleep(0.02)
    assert state["status"] == "open", state
    assert state["load_job"] is not None, "the clearance took the model; it was loaded again"
    assert len(engines) == 2 and engines[1].stopped is False


def test_a_chat_sent_during_a_clearance_waits_and_is_not_resident(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    release, clearing, _ = _a_clearance_under_way(llm_client, engines[0])
    waiting = _a_door_waiting(llm_client, monkeypatch)

    response = _sent_during(
        lambda: llm_client.post(
            "/v1/openai/chat/completions",
            headers=auth,
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        ),
        waiting,
        release,
        clearing,
    )
    assert response.status_code == 409, response.json()
    assert response.json()["error"]["code"] == "model_not_resident"
    assert len(llm_client.app.state.inflight) == 0


def test_a_clearance_that_never_finishes_is_a_wedge_not_a_hang(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    release, clearing, _ = _a_clearance_under_way(llm_client, engines[0])
    monkeypatch.setattr(residency_module, "CLEARANCE_TIMEOUT_SECONDS", 0.3)
    try:
        response = submit(llm_client, auth, type="load-model", model=MODEL)
    finally:
        release.set()
        clearing.join(timeout=30)
    assert response.status_code == 409, response.json()
    error = response.json()["error"]
    assert error["code"] == "engine_in_use"
    assert "wedged" in error["message"]
    assert error["details"]["held_by"] == SETTLEMENT_HOLDER
    assert llm_client.app.state.store.queue_depth == 0


def test_health_says_warming_while_a_load_is_in_flight(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hold = threading.Event()
    built: list[FakeEngine] = []

    def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
        engine = FakeEngine(python, log_path, hold=hold)
        built.append(engine)
        return engine

    monkeypatch.setattr(engines_module, "build_engine", build)
    monkeypatch.setattr(
        engines_module,
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
            time.sleep(0.05)
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
        "stopping": None,
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

    monkeypatch.setattr(engines_module, "build_engine", build)
    monkeypatch.setattr(
        engines_module,
        "engine_model_name",
        lambda engine_name, model_dir, model_id: model_id,
    )
    fake_weights(MODEL)
    events = run_job(llm_client, auth, type="load-model", model=MODEL)
    assert events[-1]["event"] == "failed"
    assert events[-1]["data"]["error"]["code"] == "engine_failed"
    assert "exited 1" in events[-1]["data"]["error"]["message"]
    assert llm_client.get("/v1/health", headers=auth).json()["resident_models"] == []
    assert built[0].stopped is True


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
    assert len(engines) == 1


def test_a_serial_engine_refuses_past_its_width_instead_of_queueing(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for cls in ENGINES.values():
        monkeypatch.setattr(cls, "chat_concurrency_flag", None, raising=False)
        monkeypatch.setattr(cls, "chat_concurrency", 1, raising=False)
        monkeypatch.setattr(
            cls, "chat_concurrency_basis", "one generation thread", raising=False
        )
    posts: list[int] = []
    engine_factory(on_post=lambda: posts.append(1))
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)

    activity = llm_client.get("/v1/activity", headers=auth).json()
    assert activity["chat"]["max_in_flight"] == 2
    assert activity["chat"]["max_in_flight_basis"] == "one generation thread"

    inflight = llm_client.app.state.inflight
    held = [
        inflight.open(act=None, model=MODEL, client="a-test") for _ in range(2)
    ]
    try:
        response = llm_client.post(
            "/v1/openai/chat/completions",
            headers=auth,
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
    finally:
        for entry in held:
            inflight.close(entry)

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "chat_queue_full"
    assert error["details"]["max_in_flight"] == 2
    assert posts == []


def test_vllm_admits_its_whole_batch_and_says_so(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)

    activity = llm_client.get("/v1/activity", headers=auth).json()
    assert activity["chat"]["max_in_flight"] == 17
    assert "--max-num-seqs 16" in activity["chat"]["max_in_flight_basis"]

    inflight = llm_client.app.state.inflight
    held = [
        inflight.open(act=None, model=MODEL, client="a-test") for _ in range(12)
    ]
    try:
        response = llm_client.post(
            "/v1/openai/chat/completions",
            headers=auth,
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
    finally:
        for entry in held:
            inflight.close(entry)
    assert response.status_code == 200, response.text


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
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert text.endswith("data: [DONE]\n\n")


@pytest.fixture
def engines_under_a_path_name(
    monkeypatch: pytest.MonkeyPatch, home: Path
) -> list[FakeEngine]:
    built: list[FakeEngine] = []

    def build(engine_name: str, python: Path, log_path: Path) -> FakeEngine:
        engine = FakeEngine(python, log_path)
        built.append(engine)
        return engine

    monkeypatch.setattr(engines_module, "build_engine", build)
    monkeypatch.setattr(
        engines_module,
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
    weights = fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    response = llm_client.post(
        "/v1/openai/chat/completions",
        headers=auth,
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert engines_under_a_path_name[0].last_request["model"] == str(weights.resolve())
    assert response.json()["model"] == MODEL
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
    assert [chunk["model"] for chunk in chunks] == [MODEL] * (len(DELTAS) + 1)
    assert [
        chunk["choices"][0]["delta"]["content"]
        for chunk in chunks
        if "content" in chunk["choices"][0]["delta"]
    ] == DELTAS
    assert text.endswith("data: [DONE]\n\n")


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
    weights = fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    _, response = _post_raw(llm_client, auth, CONSTRAINED_BODY)
    assert response.status_code == 200, response.text

    sent = engines_under_a_path_name[0].last_request
    assert sent == {**CONSTRAINED_BODY, "model": str(weights.resolve())}
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
    engine_factory(reject_response_format=True)
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    _, response = _post_raw(llm_client, auth, CONSTRAINED_BODY)

    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "BadRequestError"
    assert "prefixItems" in body["message"]
    assert "error" not in body


def test_an_engine_s_own_400_is_relayed_on_a_streamed_request_too(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
) -> None:
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


def test_an_unknown_act_is_refused_BEFORE_the_work_rather_than_mislabelled(
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
        headers={**auth, "X-Crucible-Act": "translat"},
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "unknown_act"
    assert "'translat'" in error["message"]
    assert "simplify" in error["message"] and "translate" in error["message"]


def test_generate_is_an_act_and_a_near_miss_is_still_refused(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    refused = llm_client.post(
        "/v1/openai/chat/completions",
        headers={**auth, "X-Crucible-Act": "generat"},
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert refused.status_code == 400, refused.text
    error = refused.json()["error"]
    assert error["code"] == "unknown_act"
    assert "'generat'" in error["message"]
    assert "generate" in error["details"]["known"]

    accepted = llm_client.post(
        "/v1/openai/chat/completions",
        headers={**auth, "X-Crucible-Act": "generate"},
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert accepted.status_code == 200, accepted.text


def test_a_chat_with_no_act_header_records_null_rather_than_a_guess(
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
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200, response.text
    body = llm_client.get("/v1/activity", headers=auth).json()
    assert body["chat"] == {
        "in_flight": 0,
        "max_in_flight": None,
        "max_in_flight_basis": None,
        "rows": [],
    }


def test_a_chat_in_flight_is_visible_and_still_does_not_take_the_lane(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    inflight = llm_client.app.state.inflight

    seen: dict[str, object] = {}
    with inflight.tracked(act="simplify", model=MODEL, client="foundry/0.9.0"):
        body = llm_client.get("/v1/activity", headers=auth).json()
        seen.update(body)

    assert seen["chat"]["in_flight"] == 1
    row = seen["chat"]["rows"][0]
    assert row["act"] == "simplify"
    assert row["model"] == MODEL
    assert row["client"] == "foundry/0.9.0"
    assert row["since"]

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
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    ours = llm_client.get("/v1/openai/models", headers=auth).json()
    theirs = llm_client.get("/openai/v1/models", headers=auth).json()
    assert theirs == ours
    assert theirs["data"][0]["id"] == MODEL
    no_model = llm_client.post("/openai/v1/chat/completions", headers=auth, json={"messages": []})
    assert no_model.status_code == 400
    assert no_model.json()["error"]["code"] == "model_required"
    other = llm_client.post(
        "/openai/v1/chat/completions", headers=auth, json={"model": "not-this-one", "messages": []}
    )
    assert other.status_code == 409
    assert other.json()["error"]["code"] == "model_not_resident"
    assert llm_client.get("/openai/v1/models").status_code == 401


@pytest.fixture
def free_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(
        accelerator,
        "probe_vram",
        lambda: (FAKE_BACKEND.gpu.vram_bytes, FAKE_BACKEND.gpu.vram_bytes),
    )


def _flag(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def test_a_load_that_states_no_context_starts_at_the_default(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    free_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    events = run_job(llm_client, auth, type="load-model", model=MODEL)
    assert events[-1]["event"] == "done", events[-1]
    assert _flag(engines[0].args, "--max-model-len") == "16384"
    row = {r["id"]: r for r in llm_client.get("/v1/models", headers=auth).json()}[MODEL]
    assert row["max_model_len"] == row["context_default"] == 16384


def test_a_stated_context_reaches_the_engine_the_plan_and_the_resident_row(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    free_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    events = run_job(
        llm_client, auth, type="load-model", model=MODEL, params={"context": 65536}
    )
    assert events[-1]["event"] == "done", events[-1]
    args = engines[0].args
    assert args.count("--max-model-len") == 1
    assert _flag(args, "--max-model-len") == "65536"
    terms = load_manifest(MODEL).spec(FAKE_BACKEND.kind).memory
    budget = FAKE_BACKEND.gpu.vram_bytes - DEFAULT_DESKTOP_ALLOWANCE_BYTES
    pool = int(_flag(args, "--kv-cache-memory-bytes"))
    assert pool == min(budget - terms.fixed_bytes, terms.kv_bytes_per_token * 65536 * 16)
    assert pool >= terms.kv_bytes_per_token * 65536
    row = {r["id"]: r for r in llm_client.get("/v1/models", headers=auth).json()}[MODEL]
    assert row["max_model_len"] == 65536
    assert row["context_default"] == 16384
    assert row["max_context"]["tokens"] == 65536
    entry = llm_client.get("/v1/openai/models", headers=auth).json()["data"][0]
    assert entry["max_model_len"] == 65536


def test_a_context_over_the_ceiling_is_refused_before_anything_moves(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    free_card: None,
    engines: list[FakeEngine],
) -> None:
    fake_weights(MODEL)
    fake_weights(SMALL_BIG_MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    assert len(engines) == 1

    refused = submit(
        llm_client, auth, type="load-model", model=SMALL_BIG_MODEL,
        params={"context": 40960},
    )
    assert refused.status_code == 400, refused.text
    error = refused.json()["error"]
    assert error["code"] == "context_over_limit"
    details = error["details"]
    assert details["model"] == SMALL_BIG_MODEL
    assert details["requested"] == {"tokens": 40960, "concurrency": 1}
    assert details["ceiling"]["tokens"] == 32768
    assert details["ceiling"]["bound_by"] == "served"
    assert details["ceiling"]["memory_context"] == 33_945
    assert [c["model"] for c in details["ceilings"]] == [SMALL_BIG_MODEL]
    assert "40960" in error["message"] and "32768" in error["message"]

    assert len(engines) == 1 and engines[0].stopped is False
    assert llm_client.get("/v1/health", headers=auth).json()["resident_models"] == [MODEL]

    over = submit(
        llm_client, auth, type="load-model", model=MODEL, params={"context": 65537}
    )
    assert over.status_code == 400
    assert over.json()["error"]["details"]["ceiling"]["tokens"] == 65536
    assert len(engines) == 1 and engines[0].stopped is False


@pytest.mark.parametrize("context", [2047, 0, -1, "32768", 32768.0, True])
def test_a_context_below_the_floor_or_not_an_int_is_invalid_params(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    free_card: None,
    context: Any,
) -> None:
    fake_weights(MODEL)
    response = submit(
        llm_client, auth, type="load-model", model=MODEL, params={"context": context}
    )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["code"] == "invalid_params"
    assert "context" in response.json()["error"]["message"]


def test_loading_the_resident_model_at_a_new_context_is_a_reload(
    llm_client: TestClient,
    auth: dict[str, str],
    fake_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
    engines: list[FakeEngine],
) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    total = FAKE_BACKEND.gpu.vram_bytes
    resident_estimate = load_manifest(MODEL).spec(FAKE_BACKEND.kind).memory_bytes_estimate
    free = {"bytes": total}
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (free["bytes"], total))
    fake_weights(MODEL)
    run_job(llm_client, auth, type="load-model", model=MODEL)
    free["bytes"] = total - resident_estimate

    events = run_job(
        llm_client, auth, type="load-model", model=MODEL, params={"context": 32768}
    )
    assert events[-1]["event"] == "done", events[-1]
    messages = [e["data"]["message"] for e in events if e["event"] == "warming"]
    assert any(f"unloading {MODEL}" in m for m in messages)
    assert len(engines) == 2
    assert engines[0].stopped is True and engines[1].stopped is False
    assert _flag(engines[0].args, "--max-model-len") == "16384"
    assert _flag(engines[1].args, "--max-model-len") == "32768"
    row = {r["id"]: r for r in llm_client.get("/v1/models", headers=auth).json()}[MODEL]
    assert row["max_model_len"] == 32768
