"""The `asr` job type, end to end through the API.

No GPU, no faster-whisper and no 3 GB of weights. What stands in for them is what
the real code paths actually read — a stamped venv whose `bin/python` is a real
interpreter, a stamped weights directory, monkeypatched accelerator probes, and
`tests/fake_asr_worker.py` spawned as a real subprocess in place of the real
worker script. Everything else is the server: the preflight refusals, the
exclusive lane, the worker envelope, the positional shift and the overlap dedup
are exactly what would run on the PC.
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
from crucible.accelerator import GIB, ComputeApp
from crucible.asrmodels import load_asr_manifest
from crucible.jobs import asr as asr_job

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, parse_sse

MODEL = "whisper-tiny"
BIG_MODEL = "whisper-large-v3-turbo"
FAKE_WORKER = Path(__file__).resolve().parent / "fake_asr_worker.py"
FAKE_MLX_WORKER = Path(__file__).resolve().parent / "fake_mlx_asr_worker.py"
#: THE SAME ID as `MODEL` since 2026-09-24 (Owen's asr lineup ruling): one id
#: per transcriber across both backends. Kept as its own name so a Mac test
#: still reads as one.
MAC_MODEL = "whisper-tiny"

#: The three, and nothing else (Owen, 2026-09-24).
ALL_MODELS = ["qwen3-asr-1.7b", "whisper-large-v3-turbo", "whisper-tiny"]

PARAMS = {"language": "en", "vad_filter": True, "word_timestamps": True}

#: Not real audio. Nothing in these tests decodes it — the fake worker is handed
#: the path and reports a duration of its own — but the file must exist, because
#: `ctx.inputs()` lists what is actually on disk.
AUDIO = base64.b64encode(b"not really an m4b").decode("ascii")


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def asr_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stamped `~/.crucible/envs/asr` whose python is this interpreter.

    A symlink and not a stub script: the worker is spawned with it for real, so
    it has to be able to run a Python file.
    """
    directory = workerenv.worker_env_dir(home, "asr")
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").symlink_to(sys.executable)
    (directory / "crucible-env.json").write_text(
        json.dumps(
            {
                "job_type": "asr",
                "backend": FAKE_BACKEND.kind,
                "recipe": f"{FAKE_BACKEND.kind}.txt",
                "python_version": "3.11.16",
                "seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    pins = workerenv.recipe_pins(workerenv.recipe_for("asr", FAKE_BACKEND.kind))
    monkeypatch.setattr(
        workerenv, "installed_packages", lambda _home, _type: dict(pins)
    )
    return directory


@pytest.fixture
def asr_weights(home: Path) -> Callable[[str], Path]:
    """Stamp an ASR model as pulled at exactly the revision its manifest pins."""

    def stamp(model_id: str) -> Path:
        spec = load_asr_manifest(model_id).spec(FAKE_BACKEND.kind)
        directory = home / "models" / model_id / FAKE_BACKEND.kind
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "crucible-pull.json").write_text(
            json.dumps(
                {
                    "model": model_id,
                    "backend": FAKE_BACKEND.kind,
                    "hf_repo": spec.hf_repo,
                    "revision": spec.revision,
                    "bytes": 147_886_409,
                    "seconds": 6.0,
                    "pulled": "2026-09-13T02:00:00+0000",
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


@pytest.fixture
def ffmpeg(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(asr_job, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    return "/usr/bin/ffmpeg"


@pytest.fixture
def fake_worker(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Both engines' scripts, replaced by their own doubles.

    BOTH, in one fixture, because the table is what the server reads and a test
    that replaced only one entry would pass while the server ran the real
    mlx-whisper worker on a machine that has no mlx.
    """
    monkeypatch.setitem(
        asr_job.WORKER_SCRIPT_FOR_ENGINE, "faster-whisper", FAKE_WORKER
    )
    monkeypatch.setitem(
        asr_job.WORKER_SCRIPT_FOR_ENGINE, "mlx-whisper", FAKE_MLX_WORKER
    )
    return FAKE_WORKER


def _mac_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`asr_env`, stamped for `mlx-darwin` instead.

    A function and not a fixture: the Mac tests build their client themselves
    (they pass `backend=FAKE_MAC_BACKEND`), so they need this AFTER the home
    exists and BEFORE the client starts, which is not an ordering a fixture can
    express without a second `home`.
    """
    directory = workerenv.worker_env_dir(home, "asr")
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").symlink_to(sys.executable)
    (directory / "crucible-env.json").write_text(
        json.dumps(
            {
                "job_type": "asr",
                "backend": FAKE_MAC_BACKEND.kind,
                "recipe": f"{FAKE_MAC_BACKEND.kind}.txt",
                "python_version": "3.11.16",
                "seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    pins = workerenv.recipe_pins(
        workerenv.recipe_for("asr", FAKE_MAC_BACKEND.kind)
    )
    monkeypatch.setattr(
        workerenv, "installed_packages", lambda _home, _type: dict(pins)
    )
    return directory


def _mac_weights(home: Path, model_id: str) -> Path:
    spec = load_asr_manifest(model_id).spec(FAKE_MAC_BACKEND.kind)
    directory = home / "models" / model_id / FAKE_MAC_BACKEND.kind
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "crucible-pull.json").write_text(
        json.dumps(
            {
                "model": model_id,
                "backend": FAKE_MAC_BACKEND.kind,
                "hf_repo": spec.hf_repo,
                "revision": spec.revision,
                "bytes": 143_726_326,
                "seconds": 6.0,
                "pulled": "2026-09-14T02:00:00+0000",
            }
        ),
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def asr_client(
    make_client: Callable[..., TestClient], asr_env: Path
) -> Iterator[TestClient]:
    with make_client(enable_asr=True) as client:
        yield client


def submit(client: TestClient, auth: dict[str, str], **body: Any):
    body.setdefault("type", "asr")
    body.setdefault("model", MODEL)
    body.setdefault("params", dict(PARAMS))
    body.setdefault("inputs", {"audio.m4b": {"inline_base64": AUDIO}})
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


# ------------------------------------------------------------------- listing


def test_info_advertises_every_asr_model(
    asr_client: TestClient, auth: dict[str, str]
) -> None:
    capabilities = asr_client.get("/v1/info", headers=auth).json()["capabilities"]
    by_type = {entry["job_type"]: entry for entry in capabilities}
    assert [row["id"] for row in by_type["asr"]["models"]] == ALL_MODELS
    row = next(r for r in by_type["asr"]["models"] if r["id"] == MODEL)
    assert row["revision"] == load_asr_manifest(MODEL).spec(FAKE_BACKEND.kind).revision
    assert row["source"] == "Systran/faster-whisper-tiny"
    assert row["installed"] is False
    # Nothing is ever resident for asr: the worker loads, transcribes, exits.
    assert row["resident"] is False


def test_info_says_installed_once_an_asr_model_is_pulled(
    asr_client: TestClient, auth: dict[str, str], asr_weights: Callable[[str], Path]
) -> None:
    """`installed` is the puller's stamp at the pinned revision, per model: one
    pulled whisper does not make the others installed."""
    asr_weights(MODEL)
    capabilities = asr_client.get("/v1/info", headers=auth).json()["capabilities"]
    by_type = {entry["job_type"]: entry for entry in capabilities}
    rows = {row["id"]: row for row in by_type["asr"]["models"]}
    assert rows[MODEL]["installed"] is True
    assert rows[BIG_MODEL]["installed"] is False
    # Still never resident, pulled or not.
    assert rows[MODEL]["resident"] is False


def test_asr_is_off_unless_the_config_says_otherwise(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_asr=False) as client:
        response = client.post("/v1/jobs", headers=auth, json={"type": "asr"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "job_type_disabled"
    assert "enable_asr" in response.json()["error"]["message"]


# ------------------------------------------------------------------ refusals


def test_a_job_that_names_no_model_is_refused(
    asr_client: TestClient, auth: dict[str, str]
) -> None:
    """There is no default. An ASR pass at the wrong size says nothing about it."""
    response = asr_client.post(
        "/v1/jobs", headers=auth, json={"type": "asr", "params": dict(PARAMS)}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_required"


def test_an_unknown_model_names_the_three(
    asr_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(asr_client, auth, model="whisper-huge")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unknown_model"
    assert error["details"]["offered"] == ALL_MODELS
    for model_id in ALL_MODELS:
        assert model_id in error["message"]


@pytest.mark.parametrize(
    "params, expected",
    [
        ({"vad_filter": True, "word_timestamps": True}, "language"),
        ({"language": "en", "word_timestamps": True}, "vad_filter"),
        ({"language": "en", "vad_filter": True}, "word_timestamps"),
    ],
)
def test_every_param_is_required(
    asr_client: TestClient, auth: dict[str, str], params: dict, expected: str
) -> None:
    """No silent defaults: all three change what whisper is asked for."""
    response = submit(asr_client, auth, params=params)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"
    assert expected in response.json()["error"]["message"]


def test_an_unknown_language_is_refused_before_the_model_loads(
    asr_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(asr_client, auth, params={**PARAMS, "language": "elvish"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"
    assert "not a language faster-whisper knows" in response.json()["error"]["message"]


def test_an_unknown_param_is_refused_not_ignored(
    asr_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(asr_client, auth, params={**PARAMS, "beam_size": 5})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"


def test_compute_type_is_not_a_wire_parameter(
    asr_client: TestClient, auth: dict[str, str]
) -> None:
    """It is the server's, decided by the backend, and a client cannot set it."""
    response = submit(asr_client, auth, params={**PARAMS, "compute_type": "int8"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"


def test_a_missing_env_is_named(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    ffmpeg: str,
    idle_card: None,
) -> None:
    with make_client(enable_asr=True) as client:
        response = submit(client, auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "env_missing"
    assert "crucible install asr" in response.json()["error"]["message"]


def test_missing_weights_are_named_with_the_pull_command(
    asr_client: TestClient, auth: dict[str, str], ffmpeg: str, idle_card: None
) -> None:
    response = submit(asr_client, auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "model_not_installed"
    assert "crucible models pull whisper-tiny" in (
        response.json()["error"]["message"]
    )


def test_no_ffmpeg_is_refused_before_the_job_is_queued(
    asr_client: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    asr_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    asr_weights(MODEL)
    monkeypatch.setattr(asr_job, "ffmpeg_path", lambda: None)
    response = submit(asr_client, auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "ffmpeg_missing"
    assert "silently truncates" in response.json()["error"]["message"]


def test_somebody_else_on_the_card_refuses_by_name(
    asr_client: TestClient,
    auth: dict[str, str],
    ffmpeg: str,
    asr_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asr_weights(MODEL)
    monkeypatch.setattr(
        accelerator,
        "probe_compute_apps",
        lambda: [ComputeApp(pid=44503, name="python", used_bytes=17 * GIB)],
    )
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (5 * GIB, 24 * GIB))
    response = submit(asr_client, auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "accelerator_busy"
    assert "never evicts" in response.json()["error"]["message"]


@pytest.mark.parametrize("backend", [FAKE_BACKEND, FAKE_MAC_BACKEND], ids=lambda b: b.kind)
@pytest.mark.parametrize(
    "old_id, new_id",
    [
        ("faster-whisper-large-v3-turbo", "whisper-large-v3-turbo"),
        ("mlx-whisper-large-v3-turbo", "whisper-large-v3-turbo"),
        ("faster-whisper-tiny", "whisper-tiny"),
        ("mlx-whisper-tiny", "whisper-tiny"),
    ],
)
def test_a_renamed_whisper_id_is_unknown_and_names_its_replacement(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    backend: Any,
    old_id: str,
    new_id: str,
) -> None:
    """Owen, 2026-09-24: the old backend-prefixed ids are GONE, not aliases.
    Refused `unknown_model` on BOTH machines — never quietly run as the new id —
    and the sentence names the id that replaced it, so a client fixes its call
    in one edit."""
    monkeypatch.setattr(
        accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB)
    )
    with make_client(enable_asr=True, backend=backend) as client:
        response = submit(client, auth, model=old_id)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unknown_model"
    assert error["details"] == {"model": old_id, "offered": ALL_MODELS}
    assert f"renamed {new_id!r} on 2026-09-24" in error["message"]
    assert "not an alias" in error["message"]


@pytest.mark.parametrize(
    "old_id",
    [
        "faster-whisper-base",
        "faster-whisper-small",
        "faster-whisper-medium",
        "faster-whisper-large-v3",
        "faster-whisper-distil-large-v3",
        "mlx-whisper-base",
        "mlx-whisper-small",
        "mlx-whisper-medium",
        "mlx-whisper-large-v3",
        "mlx-whisper-distil-large-v3",
    ],
)
def test_a_retired_whisper_size_is_unknown_and_says_it_was_retired(
    asr_client: TestClient, auth: dict[str, str], old_id: str
) -> None:
    """The five sizes Owen removed, on both engines: no replacement is named,
    because none was chosen for them; the three that exist are."""
    response = submit(asr_client, auth, model=old_id)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unknown_model"
    assert f"{old_id!r} was retired on 2026-09-24" in error["message"]
    assert error["details"]["offered"] == ALL_MODELS


def test_a_model_with_no_block_for_this_backend_is_refused_by_name(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`backend_unsupported` outlives the ruling that made every SHIPPED asr id
    span both backends: a manifest may still declare one, and the refusal is
    the manifest's own statement of which it declares."""
    catalog = tmp_path / "asr"
    catalog.mkdir()
    shipped = load_asr_manifest(MODEL).path.read_text(encoding="utf-8")
    pc_only = shipped[: shipped.index("[backends.mlx-darwin]")]
    (catalog / f"{MODEL}.toml").write_text(pc_only, encoding="utf-8")
    monkeypatch.setenv("CRUCIBLE_ASR_DIR", str(catalog))
    monkeypatch.setattr(
        accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB)
    )
    monkeypatch.setattr(asr_job, "ffmpeg_path", lambda: "/opt/homebrew/bin/ffmpeg")
    with make_client(enable_asr=True, backend=FAKE_MAC_BACKEND) as client:
        response = submit(
            client,
            auth,
            params={"language": "en", "vad_filter": False, "word_timestamps": True},
        )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "backend_unsupported"
    assert error["details"] == {
        "model": MODEL,
        "backend": "mlx-darwin",
        "declared": ["cuda-linux"],
    }


def test_the_mac_runs_its_own_worker_with_its_own_device(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of the second engine: a Mac transcribes.

    And it asserts on the envelope, because "the job ran" would also be true if
    the server had spawned the faster-whisper worker with `cuda` in the
    request. `metal` is MLX's own device name and deliberately not `mps`.
    """
    sent = tmp_path / "sent-to-mlx.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_MLX_ASR_TRANSCRIPT", str(sent))
    monkeypatch.setattr(
        accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB)
    )
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(asr_job, "ffmpeg_path", lambda: "/opt/homebrew/bin/ffmpeg")
    monkeypatch.setitem(
        asr_job.WORKER_SCRIPT_FOR_ENGINE, "mlx-whisper", FAKE_MLX_WORKER
    )
    _mac_env(home, monkeypatch)
    _mac_weights(home, MAC_MODEL)
    with make_client(enable_asr=True, backend=FAKE_MAC_BACKEND) as client:
        events = run_job(
            client,
            auth,
            model=MAC_MODEL,
            params={"language": "en", "vad_filter": False, "word_timestamps": True},
        )
    assert terminal(events)["event"] == "done", terminal(events)
    assert terminal(events)["data"]["artifacts"] == ["transcript.json"]
    envelope = json.loads(sent.read_text(encoding="utf-8").splitlines()[0])
    assert envelope["device"] == "metal"
    assert envelope["compute_type"] == "float16"
    assert envelope["window_s"] == 900
    assert envelope["overlap_s"] == 15


def test_vad_on_the_mac_is_refused_by_name_rather_than_ignored(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mlx-whisper has no VAD, and a transcript made without the filter the
    caller asked for is a different transcript with nothing to say so."""
    monkeypatch.setattr(
        accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB)
    )
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(asr_job, "ffmpeg_path", lambda: "/opt/homebrew/bin/ffmpeg")
    _mac_env(home, monkeypatch)
    _mac_weights(home, MAC_MODEL)
    with make_client(enable_asr=True, backend=FAKE_MAC_BACKEND) as client:
        response = submit(
            client,
            auth,
            model=MAC_MODEL,
            params={"language": "en", "vad_filter": True, "word_timestamps": True},
        )
    assert response.status_code == 400, response.json()
    error = response.json()["error"]
    assert error["code"] == "vad_unsupported_by_engine"
    assert "no voice-activity detector" in error["message"]
    assert error["details"]["engine"] == "mlx-whisper"
    assert error["details"]["backend"] == "mlx-darwin"


def test_a_model_bigger_than_the_card_is_refused_before_the_download(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    asr_env: Path,
    ffmpeg: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one refusal no amount of installing or pulling can fix comes first."""
    from dataclasses import replace

    from crucible.backend import Gpu

    tiny_card = replace(
        FAKE_BACKEND, gpu=Gpu(vendor="nvidia", name="GTX 1050", vram_bytes=2 * GIB)
    )
    with make_client(enable_asr=True, backend=tiny_card) as client:
        response = submit(client, auth, model=BIG_MODEL)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "insufficient_memory"
    assert "ever" in response.json()["error"]["message"]


def test_more_than_one_input_is_refused(
    asr_client: TestClient,
    auth: dict[str, str],
    ffmpeg: str,
    idle_card: None,
    asr_weights: Callable[[str], Path],
    fake_worker: Path,
) -> None:
    asr_weights(MODEL)
    events = run_job(
        asr_client,
        auth,
        inputs={
            "a.m4b": {"inline_base64": AUDIO},
            "b.m4b": {"inline_base64": AUDIO},
        },
    )
    assert terminal(events)["event"] == "failed"
    assert terminal(events)["data"]["error"]["code"] == "invalid_inputs"


# ---------------------------------------------------------------- it runs


@pytest.fixture
def ready(
    asr_client: TestClient,
    ffmpeg: str,
    idle_card: None,
    fake_worker: Path,
    asr_weights: Callable[[str], Path],
) -> TestClient:
    asr_weights(MODEL)
    return asr_client


def test_a_run_produces_a_transcript(ready: TestClient, auth: dict[str, str]) -> None:
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "done", terminal(events)
    assert terminal(events)["data"]["artifacts"] == ["transcript.json"]

    job_id = events[-1]["job_id"]
    response = ready.get(f"/v1/jobs/{job_id}/artifacts/transcript.json", headers=auth)
    assert response.status_code == 200
    document = json.loads(response.content)
    assert document["model"] == MODEL
    assert document["revision"] == (
        load_asr_manifest(MODEL).spec(FAKE_BACKEND.kind).revision
    )
    assert document["window_s"] == 900
    assert document["overlap_s"] == 15
    assert document["windows"] == 2
    assert document["duration_s"] == 1800.0
    assert document["vad_filter"] is True
    assert document["word_timestamps"] is True


def test_a_result_is_placed_by_its_position_and_the_overlap_is_dropped(
    ready: TestClient, auth: dict[str, str]
) -> None:
    """The two pieces of arithmetic the server owns, in one assertion each.

    The fake emits the same two window-relative segments for both windows, at
    0-5 and 895-905. Window 1's first segment can only land at 900 if the server
    shifted it by its POSITION — the worker reported no index at all — and it
    then begins inside window 0's straddler, which is exactly the duplicate the
    15 s back-reach produces and exactly what must be dropped.
    """
    events = run_job(ready, auth)
    job_id = events[-1]["job_id"]
    document = json.loads(
        ready.get(f"/v1/jobs/{job_id}/artifacts/transcript.json", headers=auth).content
    )
    starts = [segment["start"] for segment in document["segments"]]
    assert starts == [0.0, 895.0, 1795.0]
    # The dropped one is window 1's 900-905, not window 0's 895-905.
    assert "window 0" in document["segments"][1]["text"]
    assert "window 1" in document["segments"][2]["text"]
    # Words are shifted with their segment, not left in window time.
    assert document["segments"][2]["words"][0]["start"] == 1795.0


def test_progress_carries_seconds_and_a_cue_count(
    ready: TestClient, auth: dict[str, str]
) -> None:
    """PHASE4-AUDIO.md section 3: the app's parser gets what it has today."""
    events = run_job(ready, auth)
    progress = [e["data"] for e in events if e["event"] == "progress"]
    decoding = [row for row in progress if row.get("stage") == "decoding"]
    transcribing = [row for row in progress if row.get("stage") == "transcribing"]
    assert decoding and transcribing
    for row in decoding + transcribing:
        assert {"processed_s", "total_s", "cues"} <= set(row)
    # The decode drives no fraction: none of the transcript exists yet.
    assert all(row["fraction"] == 0.0 for row in decoding)
    assert transcribing[-1]["fraction"] == 1.0
    assert transcribing[-1]["cues"] == 3


def test_the_server_and_not_the_client_chooses_how_it_runs(
    ready: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ffmpeg: str,
) -> None:
    """`compute_type`, the window, the overlap and the device never cross the wire."""
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_ASR_TRANSCRIPT", str(transcript))
    run_job(ready, auth, params={**PARAMS, "language": "auto"})
    sent = json.loads(transcript.read_text(encoding="utf-8").strip())
    assert sent["compute_type"] == "float16"
    assert sent["device"] == "cuda"
    assert sent["window_s"] == 900
    assert sent["overlap_s"] == 15
    assert sent["ffmpeg"] == ffmpeg
    # `auto` reaches faster-whisper as None, which is how it is told to detect.
    assert sent["language"] is None
    assert sent["vad_filter"] is True


def test_auto_keeps_both_the_request_and_the_answer(
    ready: TestClient, auth: dict[str, str]
) -> None:
    events = run_job(ready, auth, params={**PARAMS, "language": "auto"})
    job_id = events[-1]["job_id"]
    document = json.loads(
        ready.get(f"/v1/jobs/{job_id}/artifacts/transcript.json", headers=auth).content
    )
    assert document["language_requested"] == "auto"
    assert document["language"] == "en"
    assert document["language_probability"] == 0.98


def test_words_are_absent_when_they_were_not_asked_for(
    ready: TestClient, auth: dict[str, str]
) -> None:
    events = run_job(ready, auth, params={**PARAMS, "word_timestamps": False})
    job_id = events[-1]["job_id"]
    document = json.loads(
        ready.get(f"/v1/jobs/{job_id}/artifacts/transcript.json", headers=auth).content
    )
    assert all("words" not in segment for segment in document["segments"])


# ---------------------------------------------------------- initial_prompt


def _sent_and_transcript(
    client: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    record: Path,
    params: dict,
) -> tuple[dict, dict]:
    """What the server handed the worker, and the transcript it published."""
    monkeypatch.setenv("CRUCIBLE_FAKE_ASR_TRANSCRIPT", str(record))
    events = run_job(client, auth, params=params)
    assert terminal(events)["event"] == "done", terminal(events)
    job_id = events[-1]["job_id"]
    document = json.loads(
        client.get(f"/v1/jobs/{job_id}/artifacts/transcript.json", headers=auth).content
    )
    sent = json.loads(record.read_text(encoding="utf-8").splitlines()[-1])
    return sent, document


def test_no_initial_prompt_runs_exactly_as_before(
    ready: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The key absent — every client in the fleet today — and the key sent as
    null are one request and one transcript. The worker is still TOLD there is
    no prompt, because its wire has no optional keys, and the transcript says
    so rather than leaving the reader to infer it from a missing field."""
    absent_sent, absent = _sent_and_transcript(
        ready, auth, monkeypatch, tmp_path / "absent.jsonl", dict(PARAMS)
    )
    null_sent, null = _sent_and_transcript(
        ready, auth, monkeypatch, tmp_path / "null.jsonl",
        {**PARAMS, "initial_prompt": None},
    )
    assert "initial_prompt" not in PARAMS
    assert absent_sent["initial_prompt"] is None
    assert absent_sent == {**null_sent, "audio": absent_sent["audio"]}
    assert absent["initial_prompt"] is None
    assert absent == null
    # The rest of the document is what it was before the key existed.
    assert [s["start"] for s in absent["segments"]] == [0.0, 895.0, 1795.0]


def test_an_initial_prompt_reaches_the_worker_and_the_transcript(
    ready: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompt = "The Stormlight Archive: Kaladin, Shallan, Dalinar Kholin."
    sent, document = _sent_and_transcript(
        ready, auth, monkeypatch, tmp_path / "sent.jsonl",
        {**PARAMS, "initial_prompt": prompt},
    )
    # Verbatim: not stripped, not prefixed. The leading space whisper wants is
    # each LIBRARY's own step (`" " + prompt.strip()`), and doing it here too
    # would be a second owner of it.
    assert sent["initial_prompt"] == prompt
    assert document["initial_prompt"] == prompt


def test_an_initial_prompt_reaches_the_mac_worker(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second engine gets the same key, from the same request builder."""
    sent = tmp_path / "sent-to-mlx.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_MLX_ASR_TRANSCRIPT", str(sent))
    monkeypatch.setattr(
        accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB)
    )
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(asr_job, "ffmpeg_path", lambda: "/opt/homebrew/bin/ffmpeg")
    monkeypatch.setitem(
        asr_job.WORKER_SCRIPT_FOR_ENGINE, "mlx-whisper", FAKE_MLX_WORKER
    )
    _mac_env(home, monkeypatch)
    _mac_weights(home, MAC_MODEL)
    with make_client(enable_asr=True, backend=FAKE_MAC_BACKEND) as client:
        events = run_job(
            client,
            auth,
            model=MAC_MODEL,
            params={
                "language": "en",
                "vad_filter": False,
                "word_timestamps": True,
                "initial_prompt": "Mistborn. Vin, Elend, Kelsier.",
            },
        )
    assert terminal(events)["event"] == "done", terminal(events)
    envelope = json.loads(sent.read_text(encoding="utf-8").splitlines()[0])
    assert envelope["initial_prompt"] == "Mistborn. Vin, Elend, Kelsier."
    assert envelope["device"] == "metal"


@pytest.mark.parametrize("value", [5, 1.5, True, ["Kaladin"], {"text": "Kaladin"}])
def test_an_initial_prompt_that_is_not_a_string_is_refused_by_name(
    asr_client: TestClient, auth: dict[str, str], value: Any
) -> None:
    """No coercion: `5` is not quietly the prompt `"5"`."""
    response = submit(asr_client, auth, params={**PARAMS, "initial_prompt": value})
    assert response.status_code == 400, response.json()
    assert response.json()["error"]["code"] == "invalid_params"
    assert "initial_prompt" in response.json()["error"]["message"]


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_a_blank_initial_prompt_is_refused_rather_than_read_as_none(
    asr_client: TestClient, auth: dict[str, str], value: str
) -> None:
    """One spelling of "no prompt": null. faster-whisper would encode `""` as a
    lone space token, which is not the same as no prompt at all."""
    response = submit(asr_client, auth, params={**PARAMS, "initial_prompt": value})
    assert response.status_code == 400, response.json()
    error = response.json()["error"]
    assert error["code"] == "invalid_params"
    assert "initial_prompt is blank; send null" in error["message"]


# ------------------------------------------------------------- it fails well


def test_a_failed_window_fails_the_job_and_publishes_nothing(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hole in a transcript is invisible in the transcript."""
    monkeypatch.setenv("CRUCIBLE_FAKE_ASR_FAIL_WINDOW", "1")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    error = terminal(events)["data"]["error"]
    assert error["code"] == "asr_window_failed"
    assert "window 1 (900s)" in error["message"]
    job_id = events[-1]["job_id"]
    assert ready.get(f"/v1/jobs/{job_id}", headers=auth).json()["artifacts"] == []


def test_a_worker_that_dies_fails_the_job_with_its_log(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_ASR_EXIT_CODE", "7")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    error = terminal(events)["data"]["error"]
    assert error["code"] == "worker_failed"
    assert "exited 7" in error["message"]


def test_a_short_stream_fails_the_job(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_ASR_SHORT", "1")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert "matched to work by position" in terminal(events)["data"]["error"]["message"]


# ----------------------------------------------- the progress event's extras


def test_progress_extras_cannot_shadow_the_fraction_or_the_message() -> None:
    """Two spellings of one field in one event is how a consumer reads the wrong one.

    Python does this one for free — both are named parameters — so there is no
    check in `progress()` to test, only the guarantee, which is worth holding.
    Nothing is emitted, so the store, job and loop are never touched.
    """
    from crucible.jobs.base import JobContext

    context = JobContext(None, None, None)
    with pytest.raises(TypeError):
        context.progress(0.5, "half", message="also half")
    with pytest.raises(TypeError):
        context.progress(0.5, "half", fraction=0.9)


def test_progress_still_refuses_a_fraction_outside_the_range() -> None:
    from crucible.jobs.base import JobContext

    context = JobContext(None, None, None)
    with pytest.raises(ValueError):
        context.progress(1.5, "past the end", stage="transcribing")


# ------------------------------------------------------------------- doctor


def test_check_reports_what_is_missing_in_order(
    make_client: Callable[..., TestClient],
    home: Path,
    asr_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    asr_weights: Callable[[str], Path],
) -> None:
    from crucible.config import load_config

    with make_client(enable_asr=True):
        pass
    config = load_config(home)
    job_type = asr_job.AsrJobType(config, FAKE_BACKEND, frozenset)

    monkeypatch.setattr(asr_job, "ffmpeg_path", lambda: None)
    status = job_type.check(FAKE_BACKEND)
    assert status.ready is False
    assert "no ffmpeg on PATH" in status.detail

    monkeypatch.setattr(asr_job, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    status = job_type.check(FAKE_BACKEND)
    assert status.ready is False
    assert "no ASR model is installed" in status.detail

    asr_weights(MODEL)
    status = job_type.check(FAKE_BACKEND)
    assert status.ready is True
    assert MODEL in status.detail
    assert "faster-whisper 1.2.1" in status.detail
