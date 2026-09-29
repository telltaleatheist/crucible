from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import jobenv

from .conftest import FAKE_BACKEND, stamp_env
from .test_image_api import (
    MODEL,
    PROMPT,
    _env,
    _weights,
    events_of,
    idle_card,
    loads,
    ready,
    refusal,
    run_job,
    submit,
    transcript,
    wait_until_running,
)
from .test_llm_api import MODEL as LLM_MODEL

__all__ = ["idle_card", "ready", "transcript"]

LEASE = {"act": "image", "ttl_seconds": 120}


def picture(prompt: str = PROMPT, **more: Any) -> dict[str, Any]:
    return {"prompt": prompt, "width": 512, "height": 512, "steps": 4, **more}


def done_of(client: TestClient, auth: dict[str, str], **body: Any) -> dict[str, Any]:
    _, events = run_job(client, auth, **body)
    assert events[-1]["event"] == "done", events[-1]
    return events[-1]["data"]


def generations(transcript: Path) -> list[dict]:
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("op") == "generate"]


def resident_kind(client: TestClient, auth: dict[str, str]) -> str | None:
    return client.get("/v1/health", headers=auth).json()["resident_kind"]


def test_the_first_image_opens_the_lease_and_the_next_reuses_the_worker(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    first = done_of(ready, auth, params=picture(seed=1, lease=LEASE))
    lease_id = first["lease_id"]
    assert isinstance(lease_id, str) and lease_id
    activity = ready.get("/v1/activity", headers=auth).json()
    assert (activity["lease"]["lease_id"], activity["lease"]["act"]) == (lease_id, "image")
    second = done_of(ready, auth, params=picture(seed=2))
    assert second["lease_id"] is None
    assert resident_kind(ready, auth) == "image"
    assert len(loads(transcript)) == 1


def test_asking_again_for_the_lease_it_holds_returns_the_same_lease(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    first = done_of(ready, auth, params=picture(seed=1, lease=LEASE))
    again = done_of(ready, auth, params=picture(seed=2, lease=LEASE))
    assert again["lease_id"] == first["lease_id"]
    assert len(loads(transcript)) == 1


def test_releasing_the_lease_unloads_the_model(
    ready: TestClient, auth: dict[str, str]
) -> None:
    lease_id = done_of(ready, auth, params=picture(lease=LEASE))["lease_id"]
    assert ready.delete(f"/v1/leases/{lease_id}", headers=auth).status_code == 204
    assert resident_kind(ready, auth) is None


def test_an_expired_lease_lets_the_settlement_unload_the_model(
    ready: TestClient, auth: dict[str, str]
) -> None:
    done_of(ready, auth, params=picture(lease=LEASE))
    leases = ready.app.state.leases
    leases._lease = replace(leases._lease, expires_at=leases._lease.since)
    assert ready.app.state.settlement.settle_for_lapsed_lease() is not None
    assert resident_kind(ready, auth) is None


@pytest.fixture
def with_llm(
    make_client: Callable[..., TestClient],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    idle_card: None,
    transcript: Path,
    fake_weights: Callable[[str], Path],
) -> Iterator[TestClient]:
    _env(home, FAKE_BACKEND.kind, monkeypatch)
    _weights(home, FAKE_BACKEND.kind)
    stamp_env(home, jobenv.llm_env(FAKE_BACKEND.kind), FAKE_BACKEND.kind, monkeypatch)
    fake_weights(LLM_MODEL)
    with make_client(enable_image=True, enable_llm=True) as client:
        yield client


def test_a_leased_image_model_refuses_an_llm_load(
    with_llm: TestClient, auth: dict[str, str]
) -> None:
    done_of(with_llm, auth, params=picture(lease=LEASE))
    response = with_llm.post(
        "/v1/jobs", headers=auth, json={"type": "load-model", "model": LLM_MODEL, "params": {}}
    )
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "leased"


def test_a_cancelled_first_image_gives_back_the_lease_it_opened(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_IMAGE_STEP_S", "0.3")
    job_id = submit(ready, auth, params=picture(steps=60, lease=LEASE)).json()["job_id"]
    wait_until_running(ready, auth, job_id)
    assert ready.delete(f"/v1/jobs/{job_id}", headers=auth).status_code == 200
    assert events_of(ready, auth, job_id)[-1]["event"] == "cancelled"
    assert ready.get("/v1/activity", headers=auth).json()["lease"] is None
    assert resident_kind(ready, auth) is None


def test_load_image_warms_the_model_up_under_a_lease(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    warmed = done_of(ready, auth, type="load-image", params={"lease": LEASE})
    assert warmed["resident"] == MODEL and warmed["lease_id"]
    assert resident_kind(ready, auth) == "image"
    made = done_of(ready, auth, params=picture(lease=LEASE))
    assert made["lease_id"] == warmed["lease_id"]
    assert len(loads(transcript)) == 1


def test_load_image_without_a_lease_leaves_it_resident_until_an_image_ends(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    assert done_of(ready, auth, type="load-image", params={})["lease_id"] is None
    assert resident_kind(ready, auth) == "image"
    done_of(ready, auth)
    assert resident_kind(ready, auth) is None
    assert len(loads(transcript)) == 1


@pytest.mark.parametrize(
    ("lease", "code"),
    [
        ({"act": "clean", "ttl_seconds": 120}, "lease_act_mismatch"),
        ({"act": "sorcery", "ttl_seconds": 120}, "unknown_act"),
        ({"act": "image", "ttl_seconds": 5}, "invalid_ttl"),
        ({"act": "image", "ttl_seconds": 7200}, "invalid_ttl"),
        ({"act": "image", "ttl_seconds": 120, "hold": "lane"}, "invalid_params"),
    ],
)
@pytest.mark.parametrize("job_type", ["image", "load-image"])
def test_a_bad_lease_is_refused_by_name_before_anything_loads(
    ready: TestClient,
    auth: dict[str, str],
    transcript: Path,
    lease: dict[str, Any],
    code: str,
    job_type: str,
) -> None:
    params = picture(lease=lease) if job_type == "image" else {"lease": lease}
    error = refusal(submit(ready, auth, type=job_type, params=params))
    assert error["code"] == code, error
    assert loads(transcript) == []


def test_a_repeated_prompt_skips_the_encoder_and_says_so(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_IMAGE_ENCODE_S", "0.5")
    first = done_of(ready, auth, params=picture(seed=1, lease=LEASE))["image"]
    second = done_of(ready, auth, params=picture(seed=2))["image"]
    assert (first["prompt_cache"], second["prompt_cache"]) == ("miss", "hit")
    assert first["stage_seconds"]["encoding"] >= 0.5
    assert second["stage_seconds"]["encoding"] < 0.2
    assert [row["encoded"] for row in generations(transcript)] == [True, False]


def test_a_changed_prompt_or_negative_is_a_miss(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    done_of(ready, auth, params=picture(lease=LEASE))
    other = done_of(ready, auth, params=picture(prompt="A blue door."))["image"]
    negative = done_of(
        ready, auth, params=picture(negative_prompt="blurry", guidance=4.0)
    )["image"]
    assert (other["prompt_cache"], negative["prompt_cache"]) == ("miss", "miss")
    assert [row["encoded"] for row in generations(transcript)] == [True, True, True]


def test_the_prompt_cache_goes_with_the_worker(
    ready: TestClient, auth: dict[str, str]
) -> None:
    assert done_of(ready, auth)["image"]["prompt_cache"] == "miss"
    assert resident_kind(ready, auth) is None
    assert done_of(ready, auth)["image"]["prompt_cache"] == "miss"
