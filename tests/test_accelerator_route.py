"""`GET /v1/accelerator` — the probe (PHASE4-AUDIO.md section 5).

Every probe underneath is monkeypatched, so these assert on what the route
reports rather than on whatever card the suite happens to run on. The case worth
the whole route is `test_somebody_else_s_process_is_reported_and_left_alone`:
BookForge has three incompatible arbitration schemes and a lock file with no
producer because there was no way to ask this question, and the answer has to be
right precisely when the card is NOT free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, jobenv
from crucible.accelerator import GIB, ComputeApp, ProbeError
from crucible import residency as residency_module
from crucible.manifests import load_manifest

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, parse_sse
from .fake_engine import FakeEngine

MODEL = "qwen3.5-9b"

#: The pid the engine double claims. Crucible's own, so it must come back
#: `owned_by_crucible: true` while the stranger beside it does not.
OURS = 31_337


class OwnedEngine(FakeEngine):
    """`FakeEngine` that admits to a pid, which is what the probe reports on."""

    @property
    def pids(self) -> frozenset[int]:
        return frozenset({OURS})


@pytest.fixture
def llm_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
def llm_weights(home: Path) -> Path:
    spec = load_manifest(MODEL).spec(FAKE_BACKEND.kind)
    directory = home / "models" / MODEL / FAKE_BACKEND.kind
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "crucible-pull.json").write_text(
        json.dumps(
            {
                "model": MODEL,
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


@pytest.fixture
def owned_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        residency_module,
        "build_engine",
        lambda engine_name, python, log_path: OwnedEngine(python, log_path),
    )
    monkeypatch.setattr(
        residency_module,
        "engine_model_name",
        lambda engine_name, model_dir, model_id: model_id,
    )


def cuda(
    monkeypatch: pytest.MonkeyPatch,
    *,
    apps: list[ComputeApp],
    free_bytes: int,
    total_bytes: int = 24 * GIB,
) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: list(apps))
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (free_bytes, total_bytes))


def load(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post(
        "/v1/jobs", headers=auth, json={"type": "load-model", "model": MODEL}
    )
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        events = parse_sse(line for line in stream.iter_lines())
    assert events[-1]["event"] == "done", events[-1]


# -------------------------------------------------------------------- access


def test_the_probe_needs_the_bearer_token(client: TestClient) -> None:
    response = client.get("/v1/accelerator")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_the_probe_needs_the_api_version_header(client: TestClient) -> None:
    response = client.get(
        "/v1/accelerator", headers={"Authorization": "Bearer test-token-not-minted"}
    )
    assert response.status_code == 426


# ------------------------------------------------------------------ the shape


def test_an_idle_cuda_card(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cuda(monkeypatch, apps=[], free_bytes=22 * GIB)
    body = client.get("/v1/accelerator", headers=auth).json()
    assert body["backend"] == "cuda-linux"
    assert body["gpu"] == {
        "vendor": "nvidia",
        "name": "NVIDIA GeForce RTX 3090 Ti",
        "total_bytes": 24 * GIB,
    }
    assert body["free_bytes"] == 22 * GIB
    assert body["used_bytes"] == 2 * GIB
    assert body["holders"] == []
    assert body["resident"] is None
    assert body["desktop_allowance_bytes"] == 3 * GIB


def test_somebody_else_s_process_is_reported_and_left_alone(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case the whole route exists for. It reports; it never evicts."""
    cuda(
        monkeypatch,
        apps=[ComputeApp(pid=44503, name="python", used_bytes=17 * GIB)],
        free_bytes=6 * GIB,
    )
    body = client.get("/v1/accelerator", headers=auth).json()
    assert body["holders"] == [
        {
            "pid": 44503,
            "name": "python",
            "bytes": 17 * GIB,
            "owned_by_crucible": False,
        }
    ]
    # A held card is a 200 describing the holding, not a refusal and not an act.
    # Asking twice changes nothing, because asking does nothing.
    assert client.get("/v1/accelerator", headers=auth).json() == body


def test_memory_the_driver_will_not_report_is_null_not_zero(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`[N/A]` from the driver is a refusal to answer, and zero is an answer."""
    cuda(
        monkeypatch,
        apps=[ComputeApp(pid=9, name="python", used_bytes=None)],
        free_bytes=6 * GIB,
    )
    body = client.get("/v1/accelerator", headers=auth).json()
    assert body["holders"][0]["bytes"] is None


def test_unattributed_vram_is_reported_because_wsl2_lists_nothing(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured on Owen's PC: 17 GB held inside WSL2, an empty compute-app list.

    Without this figure the route would report an empty `holders` and 6 GiB free
    and a caller would conclude the card was idle, which is the exact mistake
    BookForge's lock file existed to paper over.
    """
    cuda(monkeypatch, apps=[], free_bytes=4 * GIB)
    body = client.get("/v1/accelerator", headers=auth).json()
    assert body["holders"] == []
    # 20 GiB in use, 3 GiB of it the declared desktop allowance.
    assert body["unattributed_bytes"] == 17 * GIB


def test_the_mac_reports_unified_memory_and_no_holders(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB)
    )
    with make_client(backend=FAKE_MAC_BACKEND) as client:
        body = client.get("/v1/accelerator", headers=auth).json()
    assert body["backend"] == "mlx-darwin"
    assert body["gpu"]["vendor"] == "apple"
    assert body["gpu"]["total_bytes"] == 64 * GIB
    assert body["free_bytes"] == 40 * GIB
    assert body["holders"] == []
    # Not 0: vm_stat cannot attribute unified memory to compute processes, and
    # a zero here would read as "all of it is accounted for".
    assert body["unattributed_bytes"] is None
    assert "unified memory" in body["detail"]


def test_a_probe_that_cannot_answer_never_says_free(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse() -> list[ComputeApp]:
        raise ProbeError("nvidia-smi did not answer the process list within 30s")

    monkeypatch.setattr(accelerator, "probe_compute_apps", refuse)
    response = client.get("/v1/accelerator", headers=auth)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "accelerator_unreadable"
    assert "did not answer" in response.json()["error"]["message"]


# --------------------------------------------------------------- the resident


def test_crucible_s_own_engine_is_named_as_its_own(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    llm_env: Path,
    llm_weights: Path,
    owned_engines: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both halves of the answer: what is resident, and whose the holders are."""
    cuda(monkeypatch, apps=[], free_bytes=22 * GIB)
    with make_client(enable_llm=True) as client:
        load(client, auth)
        cuda(
            monkeypatch,
            apps=[
                ComputeApp(pid=OURS, name="vllm", used_bytes=19 * GIB),
                ComputeApp(pid=44503, name="python", used_bytes=2 * GIB),
            ],
            free_bytes=2 * GIB,
        )
        body = client.get("/v1/accelerator", headers=auth).json()

    assert body["resident"]["kind"] == "llm"
    assert body["resident"]["id"] == MODEL
    assert body["resident"]["since"]
    assert body["resident"]["memory_bytes_estimate"] > 0
    by_pid = {holder["pid"]: holder for holder in body["holders"]}
    assert by_pid[OURS]["owned_by_crucible"] is True
    assert by_pid[44503]["owned_by_crucible"] is False
