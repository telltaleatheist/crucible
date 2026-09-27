from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, jobenv, tasks, workers
from crucible.accelerator import GIB
from crucible.denoisemodels import load_denoise_manifest, stamp_name
from crucible.jobs import denoise as denoise_job

from .conftest import FAKE_BACKEND, holding_the_card, parse_sse
from crucible.residency import KIND_DENOISE, Residency

MODEL = "denoise-roformer"
VOCALS_MODEL = "vocals-roformer"
FAKE_WORKER = Path(__file__).resolve().parent / "fake_denoise_worker.py"
BLOCK = "block_00.wav"
INPUTS = {
    BLOCK: {"inline_base64": base64.b64encode(b"44.1 kHz stereo, allegedly").decode()}
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
def model_files(home: Path) -> Path:
    manifest = load_denoise_manifest(MODEL)
    root = denoise_job.denoise_models_dir_for(home)
    root.mkdir(parents=True, exist_ok=True)
    (root / manifest.model_filename).write_bytes(b"not a checkpoint")
    (root / manifest.config_filename).write_bytes(b"not a config")
    return root


@pytest.fixture
def pulled_model(model_files: Path) -> Path:
    manifest = load_denoise_manifest(MODEL)
    spec = manifest.spec(FAKE_BACKEND.kind)
    (model_files / stamp_name(manifest)).write_text(
        json.dumps(
            {
                "label": manifest.id,
                "hf_repo": spec.hf_repo,
                "revision": spec.revision,
                "files": [
                    {
                        "source": spec.model_path,
                        "target": manifest.model_filename,
                        "sha256": spec.model_sha256,
                    },
                    {
                        "source": spec.config_path,
                        "target": manifest.config_filename,
                        "sha256": spec.config_sha256,
                    },
                ],
                "bytes": 28,
                "seconds": 1.0,
                "pulled": "2026-09-14T00:00:00+0000",
            }
        ),
        encoding="utf-8",
    )
    return model_files


@pytest.fixture
def idle_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (22 * GIB, 24 * GIB))


@pytest.fixture
def fake_worker(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(denoise_job, "WORKER_SCRIPT", FAKE_WORKER)
    return FAKE_WORKER


@pytest.fixture
def denoise_client(
    make_client: Callable[..., TestClient], rvc_env: Path
) -> Iterator[TestClient]:
    with make_client(enable_denoise=True) as client:
        yield client


@pytest.fixture
def ready(
    denoise_client: TestClient,
    idle_card: None,
    fake_worker: Path,
    model_files: Path,
) -> TestClient:
    return denoise_client


def submit(client: TestClient, auth: dict[str, str], **body: Any):
    body.setdefault("type", "denoise")
    body.setdefault("model", MODEL)
    body.setdefault("params", {})
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


def test_info_advertises_denoise(
    denoise_client: TestClient, auth: dict[str, str]
) -> None:
    info = denoise_client.get("/v1/info", headers=auth).json()
    assert "denoise" in info["job_types"]
    by_type = {entry["job_type"]: entry for entry in info["capabilities"]}
    rows = by_type["denoise"]["models"]
    assert [row["id"] for row in rows] == [MODEL, VOCALS_MODEL]
    spec = load_denoise_manifest(MODEL).spec(FAKE_BACKEND.kind)
    assert rows[0]["source"] == f"{spec.hf_repo}:{spec.model_path}"
    assert rows[0]["revision"] == spec.revision
    assert rows[0]["installed"] is False
    assert rows[0]["resident"] is False


def test_info_says_installed_only_for_the_puller_s_own_stamp(
    denoise_client: TestClient, auth: dict[str, str], model_files: Path
) -> None:
    info = denoise_client.get("/v1/info", headers=auth).json()
    by_type = {entry["job_type"]: entry for entry in info["capabilities"]}
    assert by_type["denoise"]["models"][0]["installed"] is False


def test_info_says_installed_once_the_separator_is_pulled(
    denoise_client: TestClient, auth: dict[str, str], pulled_model: Path
) -> None:
    info = denoise_client.get("/v1/info", headers=auth).json()
    by_type = {entry["job_type"]: entry for entry in info["capabilities"]}
    row = by_type["denoise"]["models"][0]
    assert row["installed"] is True
    assert row["resident"] is False


def test_denoise_is_off_unless_the_config_says_otherwise(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_denoise=False) as client:
        response = client.post("/v1/jobs", headers=auth, json={"type": "denoise"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_required"
    assert "requires a model" in response.json()["error"]["message"]


def test_rvc_and_denoise_are_separate_flags(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_denoise=True, enable_rvc=False) as client:
        info = client.get("/v1/info", headers=auth).json()
    assert "denoise" in info["job_types"]
    assert "rvc" not in info["job_types"]


def test_denoise_takes_no_params(
    denoise_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(denoise_client, auth, params={"aggressiveness": 0.5})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"
    assert "aggressiveness" in response.json()["error"]["message"]


def test_a_missing_env_installs_rvc_on_submit(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "install_command", lambda: sys.executable)
    with make_client(enable_denoise=True) as client:
        response = submit(client, auth)
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["code"] == "installing"
        assert "installing the rvc environment" in error["message"]
        started = client.app.state.tasks.get(error["details"]["task_id"])
        assert started.request["module"]["job_types"] == [{"type": "rvc"}]


def test_a_missing_checkpoint_names_the_files_and_where_they_are(
    denoise_client: TestClient, auth: dict[str, str], idle_card: None
) -> None:
    response = submit(denoise_client, auth)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "installing"
    assert f"pulling the separator {MODEL!r}" in error["message"]
    assert error["details"]["task_id"]


def test_an_unknown_model_is_refused_by_name(
    denoise_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(denoise_client, auth, model="denoise-9000")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_model"


def test_more_than_one_input_is_refused(
    ready: TestClient, auth: dict[str, str]
) -> None:
    inputs = {
        **INPUTS,
        "block_01.wav": {"inline_base64": base64.b64encode(b"more").decode()},
    }
    events = run_job(ready, auth, inputs=inputs)
    assert terminal(events)["event"] == "failed"
    assert "exactly one audio file" in terminal(events)["data"]["error"]["message"]


def test_no_inputs_is_refused(ready: TestClient, auth: dict[str, str]) -> None:
    events = run_job(ready, auth, inputs={})
    assert terminal(events)["event"] == "failed"
    assert terminal(events)["data"]["error"]["code"] == "invalid_inputs"


def test_a_run_publishes_every_stem_and_names_the_primary(
    ready: TestClient, auth: dict[str, str]
) -> None:
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "done", terminal(events)
    data = terminal(events)["data"]
    assert data["primary_stem"].lower().count("(dry)") == 1
    assert len(data["stems"]) == 2
    assert data["artifacts"] == [data["primary_stem"]]
    assert data["sample_rate"] == 44100
    assert data["frames"] == 441000
    assert data["separate_seconds"] == 8.25
    assert data["load_seconds"] > 0
    assert data["resident"] == MODEL

    job_id = events[0]["job_id"]
    for name in data["artifacts"]:
        got = ready.get(f"/v1/jobs/{job_id}/artifacts/{name}", headers=auth)
        assert got.status_code == 200
        assert name.encode() in got.content
        sidecar = ready.get(
            f"/v1/jobs/{job_id}/artifacts/{name}.provenance.json", headers=auth
        ).json()
        spec = load_denoise_manifest(MODEL).spec(FAKE_BACKEND.kind)
        assert sidecar["model"] == {
            "id": MODEL,
            "revision": spec.revision,
            "fingerprint": f"{MODEL}@{spec.revision}",
        }


def _place_files(home: Path, model_id: str) -> None:
    manifest = load_denoise_manifest(model_id)
    root = denoise_job.denoise_models_dir_for(home)
    root.mkdir(parents=True, exist_ok=True)
    (root / manifest.model_filename).write_bytes(b"not a checkpoint")
    (root / manifest.config_filename).write_bytes(b"not a config")


VOCALS_STEMS = json.dumps(
    [
        {"name": "block_00_(vocals)_vocals_mel_band_roformer.wav"},
        {"name": "block_00_(other)_vocals_mel_band_roformer.wav"},
    ]
)


def test_the_vocals_separator_loads_its_own_checkpoint_and_keeps_vocals(
    ready: TestClient,
    auth: dict[str, str],
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _place_files(home, VOCALS_MODEL)
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_DENOISE_TRANSCRIPT", str(transcript))
    monkeypatch.setenv("CRUCIBLE_FAKE_DENOISE_STEMS", VOCALS_STEMS)
    events = run_job(ready, auth, model=VOCALS_MODEL)
    assert terminal(events)["event"] == "done", terminal(events)
    data = terminal(events)["data"]
    assert data["primary_stem"] == "block_00_(vocals)_vocals_mel_band_roformer.wav"
    assert data["artifacts"] == [data["primary_stem"]]
    assert len(data["stems"]) == 2
    assert data["resident"] == VOCALS_MODEL

    manifest = load_denoise_manifest(VOCALS_MODEL)
    load = json.loads(transcript.read_text(encoding="utf-8").splitlines()[0])
    assert load["op"] == "load"
    assert load["model_filename"] == manifest.model_filename
    assert load["model_filename"] == "vocals_mel_band_roformer.ckpt"

    job_id = events[0]["job_id"]
    sidecar = ready.get(
        f"/v1/jobs/{job_id}/artifacts/{data['primary_stem']}.provenance.json",
        headers=auth,
    ).json()
    assert sidecar["model"]["id"] == VOCALS_MODEL


def test_the_vocals_separator_refuses_a_run_with_no_vocals_stem(
    ready: TestClient,
    auth: dict[str, str],
    home: Path,
) -> None:
    _place_files(home, VOCALS_MODEL)
    events = run_job(ready, auth, model=VOCALS_MODEL)
    assert terminal(events)["event"] == "failed"
    error = terminal(events)["data"]["error"]
    assert error["code"] == "denoise_primary_stem_missing"
    assert "'(vocals)'" in error["message"]


def test_two_separators_each_load_their_own_checkpoint(
    ready: TestClient,
    auth: dict[str, str],
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _place_files(home, VOCALS_MODEL)
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_DENOISE_TRANSCRIPT", str(transcript))
    first = run_job(ready, auth)
    assert terminal(first)["event"] == "done", terminal(first)
    monkeypatch.setenv("CRUCIBLE_FAKE_DENOISE_STEMS", VOCALS_STEMS)
    second = run_job(ready, auth, model=VOCALS_MODEL)
    assert terminal(second)["event"] == "done", terminal(second)
    assert terminal(second)["data"]["resident"] == VOCALS_MODEL
    assert terminal(second)["data"]["load_seconds"] > 0

    loads = [
        json.loads(line)["model_filename"]
        for line in transcript.read_text(encoding="utf-8").splitlines()
        if '"op": "load"' in line
    ]
    assert loads == [
        load_denoise_manifest(MODEL).model_filename,
        "vocals_mel_band_roformer.ckpt",
    ]


def test_the_worker_is_told_the_native_rate_and_the_backend_decides_autocast(
    ready: TestClient,
    auth: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "denoise-request.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_DENOISE_TRANSCRIPT", str(transcript))
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "done"
    lines = transcript.read_text(encoding="utf-8").splitlines()
    load = json.loads(lines[0])
    assert load["op"] == "load"
    assert load["use_autocast"] is True
    assert load["model_filename"].endswith(".ckpt")
    request = json.loads(lines[2])
    assert request["op"] == "separate"
    assert request["sample_rate"] == 44100
    assert request["output_format"] == "WAV"
    environment = json.loads(lines[1])
    assert environment["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert environment["OMP_NUM_THREADS"] == "1"


def test_autocast_is_off_on_the_mac(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    idle_card: None,
    fake_worker: Path,
) -> None:
    from .conftest import FAKE_MAC_BACKEND

    monkeypatch.setattr(
        accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB)
    )
    directory = home / "envs" / "rvc"
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").symlink_to(sys.executable)
    (directory / "crucible-env.json").write_text(
        json.dumps(
            {
                "job_type": "rvc",
                "backend": FAKE_MAC_BACKEND.kind,
                "recipe": f"{FAKE_MAC_BACKEND.kind}.txt",
                "python_version": "3.11.16",
                "seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    recipe = jobenv.recipe_for(jobenv.worker_env("rvc", FAKE_MAC_BACKEND.kind))
    monkeypatch.setattr(
        jobenv, "installed_packages", lambda _h, _t: dict(
            jobenv.recipe_pins(recipe)
        )
    )
    monkeypatch.setattr(
        jobenv, "installed_direct_references", lambda _h, _t: dict(
            jobenv.recipe_direct_references(recipe)
        )
    )
    manifest = load_denoise_manifest(MODEL)
    root = denoise_job.denoise_models_dir_for(home)
    root.mkdir(parents=True, exist_ok=True)
    (root / manifest.model_filename).write_bytes(b"not a checkpoint")
    (root / manifest.config_filename).write_bytes(b"not a config")

    transcript = tmp_path / "mac-request.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_DENOISE_TRANSCRIPT", str(transcript))
    with make_client(enable_denoise=True, backend=FAKE_MAC_BACKEND) as client:
        events = run_job(client, auth)
    assert terminal(events)["event"] == "done", terminal(events)
    assert json.loads(transcript.read_text(encoding="utf-8").splitlines()[0])[
        "use_autocast"
    ] is False


def test_an_input_at_the_wrong_rate_is_refused_and_nothing_is_resampled(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "CRUCIBLE_FAKE_DENOISE_INPUT",
        json.dumps({"sample_rate": 24000, "frames": 240000, "channels": 1}),
    )
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    message = terminal(events)["data"]["error"]["message"]
    assert "24000 Hz" in message and "44100 Hz" in message
    assert "Nothing was resampled" in message


def test_no_primary_stem_fails_the_job(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "CRUCIBLE_FAKE_DENOISE_STEMS",
        json.dumps([{"name": "block_00_(Other)_model.wav"}]),
    )
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert terminal(events)["data"]["error"]["code"] == (
        "denoise_primary_stem_missing"
    )


def test_two_primary_stems_fail_the_job(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "CRUCIBLE_FAKE_DENOISE_STEMS",
        json.dumps(
            [{"name": "block_00_(Dry)_a.wav"}, {"name": "block_00_(dry)_b.wav"}]
        ),
    )
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert terminal(events)["data"]["error"]["code"] == (
        "denoise_primary_stem_missing"
    )


def test_a_resampled_stem_fails_the_job(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "CRUCIBLE_FAKE_DENOISE_STEMS",
        json.dumps([{"name": "block_00_(Dry)_a.wav", "sample_rate": 22050}]),
    )
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert terminal(events)["data"]["error"]["code"] == "denoise_resampled"


def test_a_stem_of_a_different_length_fails_the_job(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "CRUCIBLE_FAKE_DENOISE_STEMS",
        json.dumps([{"name": "block_00_(Dry)_a.wav", "frames": 440999}]),
    )
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert terminal(events)["data"]["error"]["code"] == "denoise_length_changed"


def test_a_second_stem_of_another_length_is_reported_and_not_refused(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "CRUCIBLE_FAKE_DENOISE_STEMS",
        json.dumps(
            [
                {"name": "block_00_(Dry)_a.wav"},
                {"name": "block_00_(Other)_a.wav", "frames": 12},
            ]
        ),
    )
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "done", terminal(events)


def test_a_load_failure_fails_the_job_with_the_engines_words(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_DENOISE_LOAD_FAIL", "1")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert "could not load" in terminal(events)["data"]["error"]["message"]


def test_a_worker_that_dies_fails_the_job(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_DENOISE_EXIT_CODE", "3")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert terminal(events)["data"]["error"]["code"] == "worker_failed"


def test_a_line_that_is_not_a_message_is_a_refusal_naming_it(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "CRUCIBLE_FAKE_DENOISE_JUNK_LINE", "INFO: loading weights, please wait"
    )
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert "please wait" in terminal(events)["data"]["error"]["message"]


def test_a_worker_that_never_says_done_fails_the_job(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_DENOISE_NO_DONE", "1")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"


def test_a_tidy_up_unload_that_also_fails_is_said_and_does_not_win(
    ready: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:

    def would_not_stop(self: Residency, subject_id: str) -> None:
        raise workers.WorkerError(f"{subject_id} did not exit after SIGTERM")

    monkeypatch.setattr(Residency, "unload", would_not_stop)
    monkeypatch.setenv("CRUCIBLE_FAKE_DENOISE_NO_DONE", "1")
    capfd.readouterr()
    events = run_job(ready, auth)

    assert terminal(events)["event"] == "failed"
    assert terminal(events)["data"]["error"]["code"] == "worker_failed"

    said = f"could not take {MODEL} off the card"
    assert said in capfd.readouterr().err
    notes = [row["data"]["message"] for row in events if row["event"] == "note"]
    assert [note for note in notes if said in note and "WorkerError" in note], events

    monkeypatch.undo()


def test_check_reports_what_is_missing_in_order(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    rvc_env: Path,
) -> None:
    from crucible.config import load_config, write_config

    write_config(
        home,
        name="crucible@test",
        host="127.0.0.1",
        port=7100,
        token="t",
        backend_kind=FAKE_BACKEND.kind,
        enable_echo=False,
        enable_llm=False,
        enable_asr=False,
        enable_tts=False,
        enable_align=False,
        enable_rvc=False,
        enable_denoise=True,
        desktop_allowance_bytes=3 * GIB,
        retention_days=7,
        desktop_allowance_basis="stated",
        desktop_allowance_note="",
    )
    config = load_config(home)
    plugin = denoise_job.DenoiseJobType(config, FAKE_BACKEND, frozenset)
    status = plugin.check(FAKE_BACKEND)
    assert status.ready is False
    assert "no separator checkpoint" in status.detail

    manifest = load_denoise_manifest(MODEL)
    root = denoise_job.denoise_models_dir_for(home)
    root.mkdir(parents=True, exist_ok=True)
    (root / manifest.model_filename).write_bytes(b"")
    (root / manifest.config_filename).write_bytes(b"")
    ready_status = plugin.check(FAKE_BACKEND)
    assert ready_status.ready is True
    assert MODEL in ready_status.detail


def test_a_book_denoised_block_by_block_pays_one_load(
    ready: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_DENOISE_TRANSCRIPT", str(transcript))
    with holding_the_card(ready):
        run_job(ready, auth)
        opened = ready.post(
            f"/v1/models/{MODEL}/lease",
            headers=auth,
            json={"act": "denoise", "ttl_seconds": 60},
        )
    assert opened.status_code == 201, opened.text
    lease = opened.json()
    assert lease["subject"] == MODEL
    assert lease["kind"] == KIND_DENOISE

    second = run_job(ready, auth)
    third = run_job(ready, auth)
    assert terminal(second)["event"] == "done", terminal(second)
    assert terminal(third)["event"] == "done", terminal(third)
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == (
        KIND_DENOISE
    )

    assert terminal(second)["data"]["load_seconds"] == 0.0
    assert terminal(third)["data"]["load_seconds"] == 0.0

    assert ready.delete(
        f"/v1/leases/{lease['lease_id']}", headers=auth
    ).status_code == 204
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None

    ops = [
        json.loads(line)["op"]
        for line in transcript.read_text(encoding="utf-8").splitlines()
        if line.startswith("{") and '"op"' in line
    ]
    assert ops == ["load", "separate", "separate", "separate"]


def test_a_denoise_lease_refuses_what_would_evict_it_and_admits_its_own_work(
    ready: TestClient, auth: dict[str, str]
) -> None:
    with holding_the_card(ready):
        run_job(ready, auth)
        opened = ready.post(
            f"/v1/models/{MODEL}/lease",
            headers=auth,
            json={"act": "denoise", "ttl_seconds": 60},
        )
    assert opened.status_code == 201, opened.text

    refused = submit(ready, auth, type="unload-denoiser", model=MODEL, params={})
    assert refused.status_code == 409, refused.text
    error = refused.json()["error"]
    assert error["code"] == "leased"
    assert error["details"]["kind"] == KIND_DENOISE
    assert "the resident separator" in error["message"]

    assert submit(ready, auth).status_code == 202


def test_unload_denoiser_takes_it_off_the_card_and_refuses_when_nothing_is_there(
    ready: TestClient, auth: dict[str, str]
) -> None:
    empty = submit(ready, auth, type="unload-denoiser", model=MODEL, params={})
    assert empty.status_code == 409, empty.text
    assert empty.json()["error"]["code"] == "separator_not_resident"
    assert "no separator is" in empty.json()["error"]["message"]

    with holding_the_card(ready):
        run_job(ready, auth)
        assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == (
            KIND_DENOISE
        )
        events = run_job(ready, auth, type="unload-denoiser", model=MODEL, params={})
    assert terminal(events)["event"] == "done", terminal(events)
    assert terminal(events)["data"]["resident"] is None
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None
