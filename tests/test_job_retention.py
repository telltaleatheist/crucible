from __future__ import annotations

import base64
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from crucible.config import DEFAULT_RETENTION_DAYS, load_config, write_config
from crucible.errors import ConfigError
from crucible.jobs import queue as queue_module
from crucible.jobs.base import Job

from .conftest import FAKE_BACKEND, TOKEN

PAYLOAD = b"a page of something worth keeping" * 8


def a_config(home: Path, **overrides: Any) -> None:
    write_config(
        home,
        name="crucible@test",
        host="127.0.0.1",
        port=7100,
        token=TOKEN,
        backend_kind=FAKE_BACKEND.kind,
        enable_echo=True,
        enable_llm=False,
        enable_asr=False,
        enable_tts=False,
        enable_align=False,
        enable_rvc=False,
        enable_denoise=False,
        desktop_allowance_bytes=3 * 1024 ** 3,
        retention_days=DEFAULT_RETENTION_DAYS,
        desktop_allowance_basis="stated",
        desktop_allowance_note="",
        **overrides,
    )


def submit(
    client: TestClient, auth: dict[str, str], **params: Any
) -> str:
    response = client.post(
        "/v1/jobs",
        json={
            "type": "echo",
            "params": params,
            "inputs": {
                "alpha.bin": {
                    "inline_base64": base64.b64encode(PAYLOAD).decode("ascii")
                }
            },
        },
        headers=auth,
    )
    assert response.status_code == 202, response.text
    return response.json()["job_id"]


def wait_for(
    client: TestClient,
    auth: dict[str, str],
    job_id: str,
    statuses: tuple[str, ...],
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = client.get(f"/v1/jobs/{job_id}", headers=auth).json()
        if last.get("status") in statuses:
            return last
        time.sleep(0.02)
    pytest.fail(f"job {job_id} never reached {statuses} within {timeout_s}s: {last}")


def collect(client: TestClient, auth: dict[str, str], job_id: str, name: str) -> bytes:
    artifact = client.get(f"/v1/jobs/{job_id}/artifacts/{name}", headers=auth)
    assert artifact.status_code == 200, artifact.text
    sidecar = client.get(
        f"/v1/jobs/{job_id}/artifacts/{name}.provenance.json", headers=auth
    )
    assert sidecar.status_code == 200, sidecar.text
    return artifact.content


def test_a_consumed_upload_is_moved_and_not_left_behind(
    client: TestClient, auth: dict[str, str], home: Path
) -> None:
    uploaded = client.post(
        "/v1/uploads", files={"file": ("beta.bin", PAYLOAD)}, headers=auth
    )
    assert uploaded.status_code == 201
    blob_id = uploaded.json()["blob_id"]
    blob = home / "uploads" / blob_id
    meta = home / "uploads" / f"{blob_id}.json"
    assert blob.is_file() and meta.is_file()

    response = client.post(
        "/v1/jobs",
        json={
            "type": "echo",
            "params": {"delay_ms": 0},
            "inputs": {"beta.bin": {"blob_id": blob_id}},
        },
        headers=auth,
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    assert not blob.exists()
    assert not meta.exists()
    store = client.app.state.store
    assert store.get(job_id).inputs_dir.joinpath("beta.bin").read_bytes() == PAYLOAD
    assert wait_for(client, auth, job_id, ("done",))["status"] == "done"


def test_a_second_job_naming_a_consumed_blob_is_told_where_it_went(
    client: TestClient, auth: dict[str, str]
) -> None:
    blob_id = client.post(
        "/v1/uploads", files={"file": ("beta.bin", PAYLOAD)}, headers=auth
    ).json()["blob_id"]
    body = {
        "type": "echo",
        "params": {"delay_ms": 0},
        "inputs": {"beta.bin": {"blob_id": blob_id}},
    }
    first = client.post("/v1/jobs", json=body, headers=auth)
    assert first.status_code == 202, first.text
    wait_for(client, auth, first.json()["job_id"], ("done",))

    second = client.post("/v1/jobs", json=body, headers=auth)
    assert second.status_code == 409, second.text
    error = second.json()["error"]
    assert error["code"] == "blob_consumed"
    assert first.json()["job_id"] in error["message"]


def test_a_job_whose_artifacts_were_all_fetched_is_reaped(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit(client, auth, delay_ms=0)
    wait_for(client, auth, job_id, ("done",))
    store = client.app.state.store
    directory = store.get(job_id).dir
    assert collect(client, auth, job_id, "alpha.bin") == PAYLOAD

    (reaped,) = store.reap()
    assert reaped.job_id == job_id
    assert reaped.why == "fetched"
    assert not directory.exists()


def test_an_artifact_fetched_without_its_sidecar_is_not_yet_collected(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit(client, auth, delay_ms=0)
    wait_for(client, auth, job_id, ("done",))
    store = client.app.state.store
    artifact = client.get(f"/v1/jobs/{job_id}/artifacts/alpha.bin", headers=auth)
    assert artifact.status_code == 200, artifact.text

    assert store.reap() == []
    assert store.get(job_id).dir.is_dir()

    sidecar = client.get(
        f"/v1/jobs/{job_id}/artifacts/alpha.bin.provenance.json", headers=auth
    )
    assert sidecar.status_code == 200, sidecar.text
    assert [record.job_id for record in store.reap()] == [job_id]


def test_a_404_for_a_name_that_is_not_there_is_not_a_collection(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit(client, auth, delay_ms=0)
    wait_for(client, auth, job_id, ("done",))
    store = client.app.state.store
    missing = client.get(f"/v1/jobs/{job_id}/artifacts/nothing.bin", headers=auth)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "unknown_artifact"

    assert store.reap() == []
    assert store.get(job_id).fetched == set()


def test_a_job_that_published_nothing_is_not_vacuously_collected() -> None:
    nothing = Job(
        id="j1",
        type="load-model",
        model="qwen3.5-9b",
        params={},
        dir=Path("/nonexistent"),
        created="2026-09-18T00:00:00+00:00",
    )
    assert nothing.collected is False

    something = Job(
        id="j2",
        type="echo",
        model=None,
        params={},
        dir=Path("/nonexistent"),
        created="2026-09-18T00:00:00+00:00",
        artifacts=["a.bin"],
    )
    assert something.collected is False
    something.fetched.add("a.bin")
    assert something.collected is False
    something.fetched.add("a.bin.provenance.json")
    assert something.collected is True


def test_a_job_nobody_came_back_for_is_reaped_when_it_ages_out(
    client: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = submit(client, auth, delay_ms=0)
    wait_for(client, auth, job_id, ("done",))
    store = client.app.state.store
    directory = store.get(job_id).dir

    assert store.reap() == []
    assert directory.is_dir()

    later = datetime.now(timezone.utc) + timedelta(days=8)
    monkeypatch.setattr(queue_module, "_now", lambda: later)
    (reaped,) = store.reap()
    assert reaped.job_id == job_id
    assert reaped.why == "aged"
    assert "retention_days" in reaped.detail
    assert not directory.exists()


def test_the_window_is_the_configured_one_and_not_a_constant(
    client: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = submit(client, auth, delay_ms=0)
    wait_for(client, auth, job_id, ("done",))
    store = client.app.state.store
    monkeypatch.setattr(
        queue_module, "_now", lambda: datetime.now(timezone.utc) + timedelta(days=6)
    )
    assert store.reap() == []

    store._config = replace(store._config, retention_days=5)
    assert [record.why for record in store.reap()] == ["aged"]


def test_a_running_job_is_never_reaped_however_old_it_looks(
    client: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = submit(client, auth, delay_ms=4000)
    wait_for(client, auth, job_id, ("running",))
    store = client.app.state.store
    directory = store.get(job_id).dir

    monkeypatch.setattr(
        queue_module, "_now", lambda: datetime.now(timezone.utc) + timedelta(days=400)
    )
    assert store.reap() == []
    assert directory.is_dir()
    assert store.get(job_id).status == "running"


def test_a_reaped_job_says_when_and_why_and_is_never_unknown(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit(client, auth, delay_ms=0)
    wait_for(client, auth, job_id, ("done",))
    collect(client, auth, job_id, "alpha.bin")
    client.app.state.store.reap()

    response = client.get(f"/v1/jobs/{job_id}", headers=auth)
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "job_reaped"
    assert error["details"]["why"] == "fetched"
    assert error["details"]["job_id"] == job_id
    assert "fetched" in error["message"]

    unknown = client.get("/v1/jobs/beefbeefbeefbeef", headers=auth)
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "unknown_job"


def test_an_orphan_directory_from_a_dead_process_is_reaped_by_age_alone(
    client: TestClient,
    auth: dict[str, str],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orphan = home / "jobs" / "a-server-that-exited"
    (orphan / "artifacts").mkdir(parents=True)
    (orphan / "artifacts" / "chapter.flac").write_bytes(b"\x00" * 64)

    assert client.app.state.store.reap() == []
    assert orphan.is_dir()

    monkeypatch.setattr(
        queue_module, "_now", lambda: datetime.now(timezone.utc) + timedelta(days=9)
    )
    (reaped,) = client.app.state.store.reap()
    assert reaped.job_id == "a-server-that-exited"
    assert reaped.why == "aged"
    assert not orphan.exists()


def test_the_config_states_the_window_and_refuses_a_zero(home: Path) -> None:
    a_config(home)
    assert load_config(home).retention_days == DEFAULT_RETENTION_DAYS == 7

    path = home / "config.toml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "retention_days = 7", "retention_days = 0"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as refusal:
        load_config(home)
    assert "retention_days" in str(refusal.value)


def test_a_config_written_before_the_key_existed_still_loads(home: Path) -> None:
    a_config(home)
    path = home / "config.toml"
    path.write_text(
        "\n".join(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("retention_days")
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_config(home).retention_days == 7


def test_a_held_job_outlives_its_fetch_and_feeds_a_later_job_by_reference(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit(client, auth, delay_ms=0)
    wait_for(client, auth, job_id, ("done",))
    held = client.post(f"/v1/jobs/{job_id}/hold", headers=auth)
    assert held.status_code == 200, held.text
    assert held.json()["held"] is True and held.json()["artifacts"] == ["alpha.bin"]

    collect(client, auth, job_id, "alpha.bin")
    store = client.app.state.store
    assert store.reap() == []
    assert store.get(job_id).dir.is_dir()

    response = client.post(
        "/v1/jobs",
        json={
            "type": "echo",
            "params": {"delay_ms": 0},
            "inputs": {"again.bin": {"artifact": {"job_id": job_id, "name": "alpha.bin"}}},
        },
        headers=auth,
    )
    assert response.status_code == 202, response.text
    second = response.json()["job_id"]
    assert store.get(second).inputs_dir.joinpath("again.bin").read_bytes() == PAYLOAD
    wait_for(client, auth, second, ("done",))

    directory = store.get(job_id).dir
    assert client.delete(f"/v1/jobs/{job_id}/hold", headers=auth).status_code == 204
    assert not directory.exists()
    assert store.get(second).inputs_dir.joinpath("again.bin").read_bytes() == PAYLOAD


def test_a_reference_to_a_job_that_is_gone_is_artifact_expired_before_the_job_exists(
    client: TestClient, auth: dict[str, str]
) -> None:
    job_id = submit(client, auth, delay_ms=0)
    wait_for(client, auth, job_id, ("done",))
    collect(client, auth, job_id, "alpha.bin")
    client.app.state.store.reap()
    for ref in ({"job_id": job_id, "name": "alpha.bin"}, {"job_id": "0" * 32, "name": "x.bin"}):
        response = client.post(
            "/v1/jobs",
            json={"type": "echo", "params": {"delay_ms": 0}, "inputs": {"a.bin": {"artifact": ref}}},
            headers=auth,
        )
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "artifact_expired"


def test_a_hold_survives_a_restart_and_the_seven_day_collector_still_takes_it(
    client: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = submit(client, auth, delay_ms=0)
    wait_for(client, auth, job_id, ("done",))
    assert client.post(f"/v1/jobs/{job_id}/hold", headers=auth).status_code == 200
    store = client.app.state.store
    directory = store.get(job_id).dir

    del store._jobs[job_id]
    assert job_id in store.restore()
    assert store.get(job_id).held
    collect(client, auth, job_id, "alpha.bin")
    assert store.reap() == [] and directory.is_dir()

    later = datetime.now(timezone.utc) + timedelta(days=8)
    monkeypatch.setattr(queue_module, "_now", lambda: later)
    (reaped,) = store.reap()
    assert reaped.job_id == job_id and not directory.exists()


def test_a_job_held_from_birth_survives_the_fetch_that_races_its_end(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post(
        "/v1/jobs",
        json={
            "type": "echo", "params": {"delay_ms": 0}, "hold": True,
            "inputs": {"alpha.bin": {"inline_base64": base64.b64encode(PAYLOAD).decode("ascii")}},
        },
        headers=auth,
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    record = wait_for(client, auth, job_id, ("done",))
    assert record["held_since"] is not None
    collect(client, auth, job_id, "alpha.bin")
    store = client.app.state.store
    assert store.reap() == [] and store.get(job_id).dir.is_dir()
    again = client.post(f"/v1/jobs/{job_id}/hold", headers=auth).json()
    assert again["held"] is True and again["gc_at"] is not None and again["status"] == "done"
