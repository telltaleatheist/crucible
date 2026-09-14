"""The `denoise` job type, end to end through the API.

No GPU, no audio-separator and no 913 MB checkpoint. What stands in for them is
what the real code paths actually read — the stamped rvc venv (denoise has none
of its own), two files in the separator's model directory, monkeypatched
accelerator probes, and `tests/fake_denoise_worker.py` spawned as a real
subprocess in place of the real worker script. Everything else is the server:
the refusals, the shared-env resolution, the engine environment and the three
stem invariants are exactly what would run on the PC.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, workerenv
from crucible.accelerator import GIB
from crucible.denoisemodels import load_denoise_manifest
from crucible.jobs import denoise as denoise_job

from .conftest import FAKE_BACKEND, parse_sse

MODEL = "denoise-roformer"
FAKE_WORKER = Path(__file__).resolve().parent / "fake_denoise_worker.py"
BLOCK = "block_00.wav"
INPUTS = {
    BLOCK: {"inline_base64": base64.b64encode(b"44.1 kHz stereo, allegedly").decode()}
}


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def rvc_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stamped `~/.crucible/envs/rvc` whose python is this interpreter.

    `rvc`, not `denoise`: this type has no env of its own, which is the thing
    this fixture's name exists to keep saying.
    """
    directory = workerenv.worker_env_dir(home, "rvc")
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
    recipe = workerenv.recipe_for("rvc", FAKE_BACKEND.kind)
    pins = workerenv.recipe_pins(recipe)
    refs = workerenv.recipe_direct_refs(recipe)
    monkeypatch.setattr(
        workerenv, "installed_packages", lambda _home, _type: dict(pins)
    )
    monkeypatch.setattr(
        workerenv, "installed_direct_refs", lambda _home, _type: dict(refs)
    )
    return directory


@pytest.fixture
def model_files(home: Path) -> Path:
    """The checkpoint and its config, under the names audio-separator wants.

    Presence is what Crucible checks, so presence is what the fixture provides —
    the same shape as `rvc`'s base assets, and for the same reason: Crucible did
    not put them there and has nothing to check them against yet.
    """
    manifest = load_denoise_manifest(MODEL)
    root = denoise_job.denoise_models_dir_for(home)
    root.mkdir(parents=True, exist_ok=True)
    (root / manifest.model_filename).write_bytes(b"not a checkpoint")
    (root / manifest.config_filename).write_bytes(b"not a config")
    return root


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


# ------------------------------------------------------------------ listing


def test_info_advertises_denoise(
    denoise_client: TestClient, auth: dict[str, str]
) -> None:
    info = denoise_client.get("/v1/info", headers=auth).json()
    assert "denoise" in info["job_types"]
    by_type = {entry["job_type"]: entry for entry in info["capabilities"]}
    rows = by_type["denoise"]["models"]
    assert [row["id"] for row in rows] == [MODEL]
    spec = load_denoise_manifest(MODEL).spec(FAKE_BACKEND.kind)
    # The repo AND the file: one repo holds every UVR model there is.
    assert rows[0]["source"] == f"{spec.hf_repo}:{spec.model_path}"
    assert rows[0]["revision"] == spec.revision
    # Nothing is ever resident: one job, one load, one exit.
    assert rows[0]["resident"] is False


def test_denoise_is_off_unless_the_config_says_otherwise(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_denoise=False) as client:
        response = client.post("/v1/jobs", headers=auth, json={"type": "denoise"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "job_type_disabled"
    assert "enable_denoise" in response.json()["error"]["message"]


def test_rvc_and_denoise_are_separate_flags(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    """One env, two types: a host may have the env and the RVC models and no
    separator checkpoint, or the other way round."""
    with make_client(enable_denoise=True, enable_rvc=False) as client:
        info = client.get("/v1/info", headers=auth).json()
    assert "denoise" in info["job_types"]
    assert "rvc" not in info["job_types"]


# ----------------------------------------------------------------- refusals


def test_denoise_takes_no_params(
    denoise_client: TestClient, auth: dict[str, str]
) -> None:
    """Every separation knob is an engine default this server does not put on
    the wire; a client asking for one is told no by name."""
    response = submit(denoise_client, auth, params={"aggressiveness": 0.5})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"
    assert "aggressiveness" in response.json()["error"]["message"]


def test_a_missing_env_names_the_rvc_install(
    make_client: Callable[..., TestClient], auth: dict[str, str], idle_card: None
) -> None:
    with make_client(enable_denoise=True) as client:
        response = submit(client, auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "env_missing"
    message = response.json()["error"]["message"]
    assert "crucible install rvc" in message
    assert "shares the rvc env" in message


def test_a_missing_checkpoint_names_the_files_and_where_they_are(
    denoise_client: TestClient, auth: dict[str, str], idle_card: None
) -> None:
    response = submit(denoise_client, auth)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "denoise_model_missing"
    manifest = load_denoise_manifest(MODEL)
    spec = manifest.spec(FAKE_BACKEND.kind)
    assert error["details"]["hf_repo"] == spec.hf_repo
    assert error["details"]["revision"] == spec.revision
    assert sorted(error["details"]["missing"]) == sorted(
        [manifest.model_filename, manifest.config_filename]
    )
    # It says why Crucible will not fetch them itself.
    assert "GitHub release" in error["message"]


def test_an_unknown_model_is_refused_by_name(
    denoise_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(denoise_client, auth, model="denoise-9000")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_model"


def test_more_than_one_input_is_refused(
    ready: TestClient, auth: dict[str, str]
) -> None:
    """The blocking is the client's, and a block is one file by the time it is
    sent."""
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


# ---------------------------------------------------------------- a good run


def test_a_run_publishes_every_stem_and_names_the_primary(
    ready: TestClient, auth: dict[str, str]
) -> None:
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "done", terminal(events)
    data = terminal(events)["data"]
    assert data["primary_stem"].lower().count("(dry)") == 1
    assert len(data["stems"]) == 2
    assert sorted(data["artifacts"]) == sorted(data["stems"])
    assert data["sample_rate"] == 44100
    assert data["frames"] == 441000
    assert data["separate_seconds"] == 8.25

    job_id = events[0]["job_id"]
    for name in data["stems"]:
        got = ready.get(f"/v1/jobs/{job_id}/artifacts/{name}", headers=auth)
        assert got.status_code == 200
        assert name.encode() in got.content
        sidecar = ready.get(
            f"/v1/jobs/{job_id}/artifacts/{name}.provenance.json", headers=auth
        ).json()
        # A denoised book that does not name the separator AND its revision is
        # a book nobody can re-derive.
        spec = load_denoise_manifest(MODEL).spec(FAKE_BACKEND.kind)
        assert sidecar["model"] == {
            "id": MODEL,
            "revision": spec.revision,
            "fingerprint": f"{MODEL}@{spec.revision}",
        }


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
    request = json.loads(lines[0])
    assert request["sample_rate"] == 44100
    assert request["output_format"] == "WAV"
    # cuda-linux, so autocast is on: CUDA-only by audio-separator's own docs,
    # and decided by the backend rather than by a manifest or a request.
    assert request["use_autocast"] is True
    assert request["model_filename"].endswith(".ckpt")
    # The OpenMP hardening the shared rvc env needs reached the engine.
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

    # The Mac's guard asks `vm_stat`, which this Linux box does not have.
    monkeypatch.setattr(
        accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB)
    )
    directory = workerenv.worker_env_dir(home, "rvc")
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
    recipe = workerenv.recipe_for("rvc", FAKE_MAC_BACKEND.kind)
    monkeypatch.setattr(
        workerenv, "installed_packages", lambda _h, _t: dict(
            workerenv.recipe_pins(recipe)
        )
    )
    monkeypatch.setattr(
        workerenv, "installed_direct_refs", lambda _h, _t: dict(
            workerenv.recipe_direct_refs(recipe)
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


# ------------------------------------------------------------- what it checks


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
    """Two means nothing here can say which one is the denoised audio."""
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
    """The invariant the client's offset slicing rests on."""
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
    """Only the primary's length is measured, so only the primary's is enforced."""
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


# --------------------------------------------------------- worker behaviour


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
    """fd 1 carries results and nothing else."""
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


# ------------------------------------------------------------------ doctor


def test_check_reports_what_is_missing_in_order(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    rvc_env: Path,
) -> None:
    """The env first, then the checkpoint: an operator fixes them in that order."""
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
