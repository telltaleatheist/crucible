from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, procgroup, rvcbase, jobenv, tasks
from crucible.accelerator import GIB, ComputeApp
from crucible.errors import ApiError
from crucible.jobs import rvc as rvc_job
from crucible.rvcmodels import load_rvc_manifest

from .conftest import FAKE_BACKEND, end_process_tree, parse_sse

MODEL = "deathstalker-rvc-v1"
FAKE_WORKER = Path(__file__).resolve().parent / "fake_rvc_worker.py"

PARAMS: dict[str, Any] = {
    "index_rate": 0.3,
    "protect_rate": 0.1,
    "n_semitones": -2,
    "f0_method": "rmvpe",
}

SENTENCES = [f"{index}.flac" for index in (41, 42, 43)]
INPUTS = {
    name: {"inline_base64": base64.b64encode(f"audio {name}".encode()).decode("ascii")}
    for name in SENTENCES
}


@pytest.fixture
def rvc_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = home / "envs" / "rvc"
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").symlink_to(sys.executable)
    (directory / "crucible-env.json").write_text(
        json.dumps(
            {
                "job_type": "rvc",
                "backend": FAKE_BACKEND.kind,
                "recipe": f"{FAKE_BACKEND.kind}.txt",
                "python_version": "3.11.16",
                "seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    recipe = jobenv.recipe_for(jobenv.worker_env("rvc", FAKE_BACKEND.kind))
    pins = jobenv.recipe_pins(recipe)
    refs = jobenv.recipe_direct_references(recipe)
    monkeypatch.setattr(
        jobenv, "installed_packages", lambda _home, _type: dict(pins)
    )
    monkeypatch.setattr(
        jobenv, "installed_direct_references", lambda _home, _type: dict(refs)
    )
    return directory


@pytest.fixture
def rvc_weights(home: Path) -> Callable[[str], Path]:

    def stamp(model_id: str) -> Path:
        manifest = load_rvc_manifest(model_id)
        spec = manifest.spec(FAKE_BACKEND.kind)
        directory = home / "rvc" / model_id / FAKE_BACKEND.kind
        model_dir = directory / "rvc" / "voice_models" / manifest.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / f"{manifest.model_name}.pth").write_bytes(b"not a checkpoint")
        (model_dir / f"{manifest.model_name}.index").write_bytes(b"not an index")
        (directory / "crucible-pull.json").write_text(
            json.dumps(
                {
                    "family": "rvc",
                    "id": model_id,
                    "backend": FAKE_BACKEND.kind,
                    "hf_repo": spec.hf_repo,
                    "revision": spec.revision,
                    "archive": spec.archive,
                    "archive_sha256": spec.archive_sha256,
                    "bytes": spec.archive_bytes,
                    "seconds": 12.0,
                    "pulled": "2026-09-13T02:00:00+0000",
                }
            ),
            encoding="utf-8",
        )
        return directory

    return stamp


@pytest.fixture
def base_assets(home: Path) -> Path:
    root = home / "rvc-base"
    for target in rvcbase.load_rvc_base().targets:
        path = root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return root


@pytest.fixture
def idle_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (22 * GIB, 24 * GIB))


@pytest.fixture(autouse=True)
def ffmpeg_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rvc_job,
        "ffmpeg_paths",
        lambda: {"ffmpeg": "/usr/bin/ffmpeg", "ffprobe": "/usr/bin/ffprobe"},
    )


@pytest.fixture
def fake_worker(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(rvc_job, "WORKER_SCRIPT", FAKE_WORKER)
    return FAKE_WORKER


@pytest.fixture
def rvc_client(
    make_client: Callable[..., TestClient], rvc_env: Path
) -> Iterator[TestClient]:
    with make_client(enable_rvc=True) as client:
        yield client


def submit(client: TestClient, auth: dict[str, str], **body: Any):
    body.setdefault("type", "rvc")
    body.setdefault("model", MODEL)
    body.setdefault("params", dict(PARAMS))
    body.setdefault("inputs", dict(INPUTS))
    return client.post("/v1/jobs", headers=auth, json=body)


def run_job(client: TestClient, auth: dict[str, str], **body: Any) -> list[dict]:
    response = submit(client, auth, **body)
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        events = parse_sse(line for line in stream.iter_lines())
    for event in events:
        event["job_id"] = job_id
    return events


def terminal(events: list[dict]) -> dict:
    return events[-1]


@pytest.fixture
def ready(
    rvc_client: TestClient,
    idle_card: None,
    fake_worker: Path,
    base_assets: Path,
    rvc_weights: Callable[[str], Path],
) -> TestClient:
    rvc_weights(MODEL)
    return rvc_client


def test_info_advertises_every_rvc_model(
    rvc_client: TestClient, auth: dict[str, str]
) -> None:
    capabilities = rvc_client.get("/v1/info", headers=auth).json()["capabilities"]
    by_type = {entry["job_type"]: entry for entry in capabilities}
    ids = [row["id"] for row in by_type["rvc"]["models"]]
    assert ids == [
        "deathstalker-rvc-v1",
        "deathstalker-rvc-v3",
        "girlfriend",
        "mistborn-rvc-v1",
        "owen-morgan",
        "sigma",
        "us-female-1",
    ]
    row = next(r for r in by_type["rvc"]["models"] if r["id"] == MODEL)
    assert row["source"] == (
        "owenmorgan/owen-morgan-bookforge:rvc/deathstalker_rvc_v1.tar.gz"
    )
    assert row["installed"] is False
    assert row["resident"] is False


def test_info_says_installed_once_an_rvc_model_is_pulled(
    rvc_client: TestClient, auth: dict[str, str], rvc_weights: Callable[[str], Path]
) -> None:
    rvc_weights(MODEL)
    capabilities = rvc_client.get("/v1/info", headers=auth).json()["capabilities"]
    by_type = {entry["job_type"]: entry for entry in capabilities}
    rows = {row["id"]: row for row in by_type["rvc"]["models"]}
    assert rows[MODEL]["installed"] is True
    assert rows["sigma"]["installed"] is False
    assert rows[MODEL]["resident"] is False


def test_rvc_is_off_unless_the_config_says_otherwise(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_rvc=False) as client:
        response = client.post("/v1/jobs", headers=auth, json={"type": "rvc"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_required"
    assert "requires a model" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "missing", ["index_rate", "protect_rate", "n_semitones"]
)
def test_the_three_numbers_that_change_the_sound_are_required(
    rvc_client: TestClient, auth: dict[str, str], missing: str
) -> None:
    params = {key: value for key, value in PARAMS.items() if key != missing}
    response = submit(rvc_client, auth, params=params)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"
    assert missing in response.json()["error"]["message"]


def test_protect_rate_above_a_half_is_refused_because_the_scale_is_inverted(
    rvc_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(rvc_client, auth, params={**PARAMS, "protect_rate": 0.9})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"
    assert "protect_rate" in response.json()["error"]["message"]


def test_protect_rate_of_exactly_a_half_is_allowed(
    rvc_client: TestClient, auth: dict[str, str]
) -> None:
    assert rvc_job.RvcParams.model_validate(
        {**PARAMS, "protect_rate": 0.5}
    ).protect_rate == 0.5


def test_f0_method_and_hop_length_may_be_absent(
    rvc_client: TestClient, auth: dict[str, str]
) -> None:
    params = {k: v for k, v in PARAMS.items() if k != "f0_method"}
    validated = rvc_job.RvcParams.model_validate(params)
    assert validated.f0_method is None
    assert validated.hop_length is None


def test_batch_size_is_not_a_wire_parameter(
    rvc_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(rvc_client, auth, params={**PARAMS, "batch_size": 4096})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"


def test_a_missing_env_is_installed_on_submit(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "install_command", lambda: sys.executable)
    with make_client(enable_rvc=True) as client:
        response = submit(client, auth)
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "installing"
        assert "installing the rvc environment" in error["message"]
        assert error["details"]["steps"][0] == "install rvc"
        started = client.app.state.tasks.get(error["details"]["task_id"])
        assert started.request["module"]["job_types"] == [{"type": "rvc"}]


def test_the_env_an_env_missing_refusal_names_chooses_the_installer(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "install_command", lambda: sys.executable)
    with make_client(enable_rvc=True) as client:
        plugin = client.app.state.store.registry["rvc"]

        def refuse(model: str | None, params: dict[str, Any]) -> None:
            raise ApiError(
                409,
                "env_missing",
                f"cannot run {model!r}: the align env has gone",
                {"model": model, "env": str(home / "envs" / "align")},
            )

        monkeypatch.setattr(plugin, "preflight", refuse)
        response = submit(client, auth)
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "installing"
        assert error["details"]["steps"][0] == "install align"
        started = client.app.state.tasks.get(error["details"]["task_id"])
        assert started.request["module"]["job_types"] == [{"type": "align"}]


def test_missing_weights_are_named_with_the_pull_command(
    rvc_client: TestClient, auth: dict[str, str], idle_card: None
) -> None:
    response = submit(rvc_client, auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "installing"
    assert "the RVC voice 'deathstalker-rvc-v1'" in (
        response.json()["error"]["message"]
    )


def test_missing_base_assets_are_refused_by_name_with_the_paths(
    rvc_client: TestClient,
    auth: dict[str, str],
    idle_card: None,
    rvc_weights: Callable[[str], Path],
) -> None:
    rvc_weights(MODEL)
    response = submit(rvc_client, auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "installing"
    message = response.json()["error"]["message"]
    assert "pulling its base assets" in message


def test_somebody_else_on_the_card_refuses_by_name(
    rvc_client: TestClient,
    auth: dict[str, str],
    base_assets: Path,
    rvc_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rvc_weights(MODEL)
    monkeypatch.setattr(
        accelerator,
        "probe_compute_apps",
        lambda: [ComputeApp(pid=44503, name="python", used_bytes=17 * GIB)],
    )
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (5 * GIB, 24 * GIB))
    response = submit(rvc_client, auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "accelerator_busy"


def test_a_model_with_no_index_refuses_a_non_zero_index_rate(
    rvc_client: TestClient,
    auth: dict[str, str],
    idle_card: None,
    base_assets: Path,
    rvc_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rvc_weights(MODEL)
    real = load_rvc_manifest

    def indexless(model_id: str, directory=None):
        from dataclasses import replace

        return replace(real(model_id, directory), has_index=False)

    monkeypatch.setattr(rvc_job, "load_all_rvc_manifests", lambda: {MODEL: indexless(MODEL)})
    response = submit(rvc_client, auth)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_has_no_index"
    assert "Send index_rate 0" in response.json()["error"]["message"]


def test_mixed_formats_and_a_name_without_an_extension_are_converted(
    ready: TestClient, auth: dict[str, str]
) -> None:
    events = run_job(
        ready,
        auth,
        inputs={
            **INPUTS,
            "44.wav": {"inline_base64": base64.b64encode(b"w").decode()},
            "c000": {"inline_base64": base64.b64encode(b"c").decode()},
        },
    )
    assert terminal(events)["event"] == "done"


def test_a_job_with_no_inputs_is_refused(
    ready: TestClient, auth: dict[str, str]
) -> None:
    events = run_job(ready, auth, inputs={})
    assert terminal(events)["event"] == "failed"
    assert terminal(events)["data"]["error"]["code"] == "invalid_inputs"


def test_a_run_converts_every_input_and_keeps_its_name(
    ready: TestClient, auth: dict[str, str]
) -> None:
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "done", terminal(events)
    assert terminal(events)["data"]["artifacts"] == SENTENCES
    assert terminal(events)["data"]["files"] == 3
    assert terminal(events)["data"]["model_name"] == "deathstalker_rvc_v1"
    job_id = events[-1]["job_id"]
    for name in SENTENCES:
        response = ready.get(f"/v1/jobs/{job_id}/artifacts/{name}", headers=auth)
        assert response.status_code == 200
        assert response.content.endswith(b"[converted by the fake rvc worker]\n")


def test_a_missing_output_fails_the_job_and_keeps_the_rest(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_RVC_SKIP", "42.flac")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    error = terminal(events)["data"]["error"]
    assert error["code"] == "rvc_output_missing"
    assert "42.flac" in error["message"]
    job_id = events[-1]["job_id"]
    assert ready.get(f"/v1/jobs/{job_id}", headers=auth).json()["artifacts"] == [
        "41.flac",
        "43.flac",
    ]


def test_the_server_and_not_the_client_chooses_how_it_runs(
    ready: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_RVC_TRANSCRIPT", str(transcript))
    run_job(ready, auth)
    lines = transcript.read_text(encoding="utf-8").splitlines()
    sent, environment = json.loads(lines[0]), json.loads(lines[1])

    assert sent["batch_size"] == 96
    assert sent["model_name"] == "deathstalker_rvc_v1"
    assert sent["memory_fraction"] == 0.5
    assert sent["max_batch_audio_s"] == 1800.0
    assert (sent["piece_s"], sent["overlap_s"], sent["crossfade_s"]) == (60.0, 0.5, 0.02)
    assert sent["inputs"] == SENTENCES
    assert sent["index_rate"] == 0.3
    assert sent["protect_rate"] == 0.1
    assert sent["n_semitones"] == -2
    assert sent["f0_method"] == "rmvpe"

    assert environment["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["URVC_SKIP_INIT"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"


def test_an_absent_f0_method_is_absent_from_the_request(
    ready: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_RVC_TRANSCRIPT", str(transcript))
    params = {k: v for k, v in PARAMS.items() if k != "f0_method"}
    run_job(ready, auth, params=params)
    sent = json.loads(transcript.read_text(encoding="utf-8").splitlines()[0])
    assert "f0_method" not in sent
    assert "hop_length" not in sent


def test_the_staged_models_root_holds_this_job_s_model_and_no_other(
    ready: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    home: Path,
) -> None:
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_RVC_TRANSCRIPT", str(transcript))
    run_job(ready, auth)
    sent = json.loads(transcript.read_text(encoding="utf-8").splitlines()[0])
    root = Path(sent["models_dir"])
    voices = sorted(p.name for p in (root / "rvc" / "voice_models").iterdir())
    assert voices == ["deathstalker_rvc_v1"]
    assert (root / "rvc" / "embedders" / "contentvec" / "pytorch_model.bin").is_file()
    assert (root / "rvc" / "predictors" / "rmvpe.pt").is_file()


def test_progress_counts_files_across_batches(
    ready: TestClient, auth: dict[str, str]
) -> None:
    events = run_job(ready, auth)
    rows = [e["data"] for e in events if e["event"] == "progress"]
    converting = [row for row in rows if row.get("stage") == "converting"]
    assert [row["processed"] for row in converting] == [1, 2, 3, 3]
    assert converting[-1]["fraction"] == 1.0


def test_batching_is_the_servers_and_recycles_per_batch(
    ready: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_RVC_TRANSCRIPT", str(transcript))
    monkeypatch.setattr(rvc_job, "BATCH_SIZE", 2)
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "done"
    rows = [e["data"] for e in events if e["event"] == "progress"]
    batched = [row for row in rows if row.get("stage") == "converting"]
    assert [row["processed"] for row in batched] == [1, 2, 3, 3]
    warming = [e["data"]["message"] for e in events if e["event"] == "warming"]
    assert any("2 batch(es) of 2" in message for message in warming)


def test_a_worker_that_dies_fails_the_job_with_its_log(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_RVC_EXIT_CODE", "7")
    events = run_job(ready, auth)
    error = terminal(events)["data"]["error"]
    assert error["code"] == "worker_failed"
    assert "told to exit before saying anything" in error["message"]


def test_a_failed_batch_fails_the_whole_job(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_RVC_BATCH_FAIL", "1")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert "urvc convert-dir exited 1" in terminal(events)["data"]["error"]["message"]


def test_a_short_stream_fails_the_job(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_RVC_SHORT", "1")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert "matched to work by position" in terminal(events)["data"]["error"]["message"]


def test_a_library_printing_to_fd_1_is_refused_not_skipped(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_RVC_JUNK_LINE", "[RVC] loading rmvpe...")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert "not JSON" in terminal(events)["data"]["error"]["message"]


def test_a_cancel_stops_the_worker_and_does_not_sigkill_it(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from crucible import workers

    monkeypatch.setattr(workers, "STOP_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setenv("CRUCIBLE_FAKE_RVC_IGNORE_SIGTERM", "1")
    monkeypatch.setenv("CRUCIBLE_FAKE_RVC_SLOW_S", "30")

    pids: list[int] = []
    real_popen = workers.subprocess.Popen

    def watched(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        pids.append(process.pid)
        return process

    monkeypatch.setattr(workers.subprocess, "Popen", watched)

    job_id = submit(ready, auth).json()["job_id"]
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if ready.get(f"/v1/jobs/{job_id}", headers=auth).json()["status"] == "running":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("the job never started running")
    assert ready.delete(f"/v1/jobs/{job_id}", headers=auth).status_code == 200

    with ready.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        events = parse_sse(line for line in stream.iter_lines())
    if procgroup.platform_kind() == procgroup.WIN32:
        assert terminal(events)["event"] == "cancelled", terminal(events)
    else:
        assert terminal(events)["event"] in ("cancelled", "failed")
        if terminal(events)["event"] == "failed":
            assert "does not SIGKILL" in terminal(events)["data"]["error"]["message"]

    for pid in list(pids):
        end_process_tree(pid)


def test_check_reports_what_is_missing_in_order(
    make_client: Callable[..., TestClient],
    home: Path,
    rvc_env: Path,
    base_assets: Path,
    rvc_weights: Callable[[str], Path],
) -> None:
    from crucible.config import load_config

    with make_client(enable_rvc=True):
        pass
    config = load_config(home)
    job_type = rvc_job.RvcJobType(config, FAKE_BACKEND, frozenset)

    status = job_type.check(FAKE_BACKEND)
    assert status.ready is False
    assert "no RVC model is installed" in status.detail

    rvc_weights(MODEL)
    status = job_type.check(FAKE_BACKEND)
    assert status.ready is True
    assert MODEL in status.detail
    assert "ultimate-rvc" in status.detail


def test_check_names_the_missing_base_assets(
    make_client: Callable[..., TestClient],
    home: Path,
    rvc_env: Path,
    rvc_weights: Callable[[str], Path],
) -> None:
    from crucible.config import load_config

    with make_client(enable_rvc=True):
        pass
    rvc_weights(MODEL)
    job_type = rvc_job.RvcJobType(load_config(home), FAKE_BACKEND, frozenset)
    status = job_type.check(FAKE_BACKEND)
    assert status.ready is False
    assert "base assets are not at" in status.detail
