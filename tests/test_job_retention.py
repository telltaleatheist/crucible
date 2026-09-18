"""A job directory does not live for ever: it is reaped, and the id says so.

Ledger C5, Owen's ruling 2026-09-18. Nothing in Crucible had ever deleted a
finished job's scratch: `store.discard` removed one on the create-then-refused
path and that was all, so `_jobs` grew for the life of the process and the
directories grew for the life of the machine. Measured on the PC the day it was
ruled — 9.3 GB in 88 job directories since 09-12, 60 of them with an empty
`artifacts/`, and 2.4 GB in 2,935 uploads nothing had ever collected.

THE FOUR CLAIMS HELD HERE, in the order a reader wants them:

* an upload is MOVED into the job that names it, so those bytes are on this
  disk once, and a second job naming the same blob is told where they went;
* a job whose every artifact AND sidecar has been fetched is reaped at once,
  because its directory is a second copy of what the client now holds;
* anything else is reaped when it ages past `[jobs] retention_days`, and a
  QUEUED or RUNNING job is never reaped at any age;
* `GET /v1/jobs/{id}` on a reaped job says `job_reaped` with when and why —
  never `unknown_job`, which would send a client looking for a typo.

The clock is injected (`crucible.jobs.queue._now`) rather than waited out, and
every test here runs against pytest's throwaway `CRUCIBLE_HOME`: a reaper is
the one thing in this repo that must never be pointed at a real `~/.crucible`.
"""

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
    """`conftest.make_app`'s config, written without standing a server up."""
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
        desktop_allowance_bytes=3 * 1024 ** 3,
        **overrides,
    )


def submit(
    client: TestClient, auth: dict[str, str], **params: Any
) -> str:
    """One `echo` job with one input, which becomes one artifact."""
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
    """Fetch an artifact and its sidecar, the way `SdkClient` fetches them."""
    artifact = client.get(f"/v1/jobs/{job_id}/artifacts/{name}", headers=auth)
    assert artifact.status_code == 200, artifact.text
    sidecar = client.get(
        f"/v1/jobs/{job_id}/artifacts/{name}.provenance.json", headers=auth
    )
    assert sidecar.status_code == 200, sidecar.text
    return artifact.content


# ------------------------------------------------------- the upload is moved


def test_a_consumed_upload_is_moved_and_not_left_behind(
    client: TestClient, auth: dict[str, str], home: Path
) -> None:
    """One copy of the bytes, which is the 2.4 GB of `uploads/` in one line."""
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

    # Gone from `uploads/`, present in the job, byte for byte. The metadata
    # sidecar describes a blob that is no longer there, so it goes too.
    assert not blob.exists()
    assert not meta.exists()
    store = client.app.state.store
    assert store.get(job_id).inputs_dir.joinpath("beta.bin").read_bytes() == PAYLOAD
    assert wait_for(client, auth, job_id, ("done",))["status"] == "done"


def test_a_second_job_naming_a_consumed_blob_is_told_where_it_went(
    client: TestClient, auth: dict[str, str]
) -> None:
    """`blob_consumed`, not `unknown_blob`.

    This server DID hold those bytes and can say which job has them; telling
    the client they were never here sends it looking for a bug in its own
    upload.
    """
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


# ------------------------------------------------------------ fetched, reaped


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
    """The two requests are in flight together (`SdkClient.#writeArtifact`).

    Reaping on the artifact alone would race the sidecar's own GET and answer
    it `job_reaped` about a job the client was in the middle of collecting.
    """
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
    """The fetch is recorded AFTER the file check, so a miss never reaps."""
    job_id = submit(client, auth, delay_ms=0)
    wait_for(client, auth, job_id, ("done",))
    store = client.app.state.store
    missing = client.get(f"/v1/jobs/{job_id}/artifacts/nothing.bin", headers=auth)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "unknown_artifact"

    assert store.reap() == []
    assert store.get(job_id).fetched == set()


def test_a_job_that_published_nothing_is_not_vacuously_collected() -> None:
    """`load-model` and a job that failed early have an empty artifact list.

    An empty list is vacuously "all fetched", which would reap the job the
    instant it ended — out from under a client still reading its event stream.
    A job with nothing to collect is the retention window's business.
    """
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


# ---------------------------------------------------------------- aged, reaped


def test_a_job_nobody_came_back_for_is_reaped_when_it_ages_out(
    client: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backstop. Seven days by `[jobs] retention_days`, clock injected."""
    job_id = submit(client, auth, delay_ms=0)
    wait_for(client, auth, job_id, ("done",))
    store = client.app.state.store
    directory = store.get(job_id).dir

    # Nothing fetched it, and it is minutes old: it stays.
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
    """Six days is inside a seven-day window and outside a five-day one."""
    job_id = submit(client, auth, delay_ms=0)
    wait_for(client, auth, job_id, ("done",))
    store = client.app.state.store
    monkeypatch.setattr(
        queue_module, "_now", lambda: datetime.now(timezone.utc) + timedelta(days=6)
    )
    assert store.reap() == []

    # The same server, configured for five days instead of seven. `Config` is
    # frozen, so the store is handed a new one rather than having its own
    # edited — which is also the honest shape of the claim: the reaper reads
    # `retention_days` off the config it holds, not a constant.
    store._config = replace(store._config, retention_days=5)
    assert [record.why for record in store.reap()] == ["aged"]


def test_a_running_job_is_never_reaped_however_old_it_looks(
    client: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A render going for eight days is this afternoon's work, not a week-old
    job, and deleting its scratch would take the book with it."""
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


# ------------------------------------------------- the id still answers after


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

    # And an id this server really has never seen is still `unknown_job`: the
    # two are different pieces of news and neither may borrow the other's.
    unknown = client.get("/v1/jobs/beefbeefbeefbeef", headers=auth)
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "unknown_job"


def test_an_orphan_directory_from_a_dead_process_is_reaped_by_age_alone(
    client: TestClient,
    auth: dict[str, str],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_jobs` lives in the process; the directories outlive it.

    That is the whole of the measured 9.3 GB — 88 directories belonging to
    servers that had already exited. It is by AGE and never immediately, so a
    second Crucible sharing this home cannot delete a directory the first one
    is writing into.
    """
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
    """Turning reaping off is not a setting; a long window is how you ask."""
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
    """Absent means the ruled seven, on `_open_pairing`'s terms: every config
    on every machine was written before this key, and demanding it would make
    an upgrade unable to read its own file."""
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
