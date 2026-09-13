"""`load-model`, `unload-model`, `/v1/models` and the OpenAI proxy.

No GPU and no 19 GB of weights: the env, the weights and the engine are all
stood up as the real code paths read them — a stamped venv directory, a stamped
weights directory, and an `Engine` that serves a trivial OpenAI surface on a real
loopback port (tests/fake_engine.py). What is *not* faked is any of the server's
own logic: the preflight refusals, the exclusive lane, the event stream and the
proxy are exactly what runs on the PC.
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

from crucible import accelerator, llmenv
from crucible.accelerator import GIB, ComputeApp
from crucible.jobs.llm import residency as residency_module
from crucible.manifests import load_manifest

from .conftest import FAKE_BACKEND, parse_sse
from .fake_engine import ANSWER, DELTAS, FakeEngine

MODEL = "qwen3.5-9b"
BIG_MODEL = "qwen3.8-27b"


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def fake_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stamped `~/.crucible/envs/llm` that `env_status` accepts."""
    directory = llmenv.llm_env_dir(home)
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
    pins = llmenv.recipe_pins(llmenv.recipe_for(FAKE_BACKEND.kind))
    monkeypatch.setattr(llmenv, "installed_packages", lambda _home: dict(pins))
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
    assert sorted(rows) == [MODEL, BIG_MODEL]
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
    assert row["context_default"] == 12288
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
    assert {row["id"] for row in by_type["llm"]["models"]} == {MODEL, BIG_MODEL}
    # The two things you can actually POST are listed as themselves.
    assert "load-model" in by_type
    assert "unload-model" in by_type


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
    by_type = {entry["job_type"]: entry for entry in capabilities}
    assert by_type["llm"]["models"] == rows


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

    lines = [line for line in text.split("\n") if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    deltas = [
        json.loads(line[len("data: ") :])["choices"][0]["delta"].get("content", "")
        for line in lines[:-1]
    ]
    assert deltas == DELTAS
    assert "".join(deltas) == ANSWER
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

    lines = [line for line in text.split("\n") if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    assert [chunk["model"] for chunk in chunks] == [MODEL] * len(DELTAS)
    # The framing and the content survived the relabelling.
    assert [chunk["choices"][0]["delta"].get("content", "") for chunk in chunks] == DELTAS
    assert text.endswith("data: [DONE]\n\n")


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
