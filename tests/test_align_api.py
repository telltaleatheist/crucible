"""The `align` job type, end to end through the API.

No GPU, no torch and no 1.7 GB of weights. What stands in for them is what the
real code paths actually read — a stamped venv whose `bin/python` is a real
interpreter, a stamped weights directory, monkeypatched accelerator probes, and
`tests/fake_align_worker.py` spawned as a real subprocess in place of the real
worker script. Everything else is the server: the preflight refusals, the
exclusive lane, the worker envelope, the residency that holds the worker open
across jobs, and the positional matching are exactly what would run on the PC.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, workerenv
from crucible.accelerator import GIB, ComputeApp
from crucible.alignmodels import load_align_manifest
from crucible.jobs import align as align_job
from crucible.residency import KIND_ALIGN

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, parse_sse

MODEL = "qwen3-aligner"
FAKE_WORKER = Path(__file__).resolve().parent / "fake_align_worker.py"

#: Not real audio. Nothing in these tests decodes it — the fake worker never
#: opens the file — but it must exist, because `ctx.inputs()` lists what is
#: actually on disk and `_chunk_inputs` reads the names off it.
AUDIO = base64.b64encode(b"not really a flac").decode("ascii")

CHUNKS = [
    {"index": 41, "text": "He had been walking for some time."},
    {"index": 42, "text": "Then he stopped."},
]
PARAMS: dict[str, Any] = {"language": "en", "chunks": CHUNKS}
INPUTS = {
    "41.flac": {"inline_base64": AUDIO},
    "42.flac": {"inline_base64": AUDIO},
}


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def align_env(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stamped `~/.crucible/envs/align` whose python is this interpreter.

    A symlink and not a stub script: the worker is spawned with it for real, so
    it has to be able to run a Python file.
    """
    directory = workerenv.worker_env_dir(home, "align")
    (directory / "bin").mkdir(parents=True)
    (directory / "bin" / "python").symlink_to(sys.executable)
    (directory / "crucible-env.json").write_text(
        json.dumps(
            {
                "job_type": "align",
                "backend": FAKE_BACKEND.kind,
                "recipe": f"{FAKE_BACKEND.kind}.txt",
                "python_version": "3.11.16",
                "seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    pins = workerenv.recipe_pins(workerenv.recipe_for("align", FAKE_BACKEND.kind))
    monkeypatch.setattr(
        workerenv, "installed_packages", lambda _home, _type: dict(pins)
    )
    return directory


@pytest.fixture
def align_weights(home: Path) -> Callable[[str], Path]:
    """Stamp the aligner as pulled at exactly the revision its manifest pins."""

    def stamp(model_id: str) -> Path:
        spec = load_align_manifest(model_id).spec(FAKE_BACKEND.kind)
        directory = home / "models" / model_id / FAKE_BACKEND.kind
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "crucible-pull.json").write_text(
            json.dumps(
                {
                    "model": model_id,
                    "backend": FAKE_BACKEND.kind,
                    "hf_repo": spec.hf_repo,
                    "revision": spec.revision,
                    "bytes": 1_840_072_459,
                    "seconds": 40.0,
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
    monkeypatch.setattr(align_job, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    return "/usr/bin/ffmpeg"


@pytest.fixture
def fake_worker(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(align_job, "WORKER_SCRIPT", FAKE_WORKER)
    return FAKE_WORKER


@pytest.fixture
def align_client(
    make_client: Callable[..., TestClient], align_env: Path
) -> Iterator[TestClient]:
    with make_client(enable_align=True) as client:
        yield client


def submit(client: TestClient, auth: dict[str, str], **body: Any):
    body.setdefault("type", "align")
    body.setdefault("model", MODEL)
    body.setdefault("params", json.loads(json.dumps(PARAMS)))
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


def artifact(client: TestClient, auth: dict[str, str], events: list[dict]) -> dict:
    job_id = events[-1]["job_id"]
    response = client.get(
        f"/v1/jobs/{job_id}/artifacts/alignment.json", headers=auth
    )
    assert response.status_code == 200, response.text
    return json.loads(response.content)


@pytest.fixture
def ready(
    align_client: TestClient,
    ffmpeg: str,
    idle_card: None,
    fake_worker: Path,
    align_weights: Callable[[str], Path],
) -> TestClient:
    align_weights(MODEL)
    return align_client


# ------------------------------------------------------------------- listing


def test_info_advertises_the_aligner(
    align_client: TestClient, auth: dict[str, str]
) -> None:
    capabilities = align_client.get("/v1/info", headers=auth).json()["capabilities"]
    by_type = {entry["job_type"]: entry for entry in capabilities}
    row = next(r for r in by_type["align"]["models"] if r["id"] == MODEL)
    assert row["revision"] == load_align_manifest(MODEL).spec(FAKE_BACKEND.kind).revision
    assert row["source"] == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert row["resident"] is False


def test_align_is_off_unless_the_config_says_otherwise(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_align=False) as client:
        response = client.post("/v1/jobs", headers=auth, json={"type": "align"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "job_type_disabled"
    assert "enable_align" in response.json()["error"]["message"]


# ------------------------------------------------------------------ refusals


def test_an_unsupported_language_is_refused_before_the_model_loads(
    align_client: TestClient, auth: dict[str, str]
) -> None:
    """It does not fall back to English; it places words badly and says nothing."""
    response = submit(align_client, auth, params={**PARAMS, "language": "cy"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"
    message = response.json()["error"]["message"]
    assert "not a language Qwen3-ForcedAligner supports" in message
    assert "'yue'" in message


def test_every_one_of_the_eleven_languages_is_accepted() -> None:
    """The list is the model's, so it is asserted as a list and not a spot check."""
    assert sorted(align_job.QWEN3_LANGUAGES) == [
        "de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "yue", "zh",
    ]
    for code in align_job.QWEN3_LANGUAGES:
        params = align_job.AlignParams.model_validate({**PARAMS, "language": code})
        assert params.model_language() == align_job.QWEN3_LANGUAGES[code]


def test_an_unknown_param_is_refused_not_ignored(
    align_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(align_client, auth, params={**PARAMS, "device": "cpu"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_params"


def test_a_duplicate_chunk_index_is_refused(
    align_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(
        align_client,
        auth,
        params={
            "language": "en",
            "chunks": [{"index": 41, "text": "one"}, {"index": 41, "text": "two"}],
        },
    )
    assert response.status_code == 400
    assert "appears more than once" in response.json()["error"]["message"]


def test_an_empty_chunk_text_is_refused(
    align_client: TestClient, auth: dict[str, str]
) -> None:
    response = submit(
        align_client,
        auth,
        params={"language": "en", "chunks": [{"index": 41, "text": "   "}]},
    )
    assert response.status_code == 400
    assert "nothing here to place" in response.json()["error"]["message"]


def test_a_missing_env_is_named(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    ffmpeg: str,
    idle_card: None,
) -> None:
    with make_client(enable_align=True) as client:
        response = submit(client, auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "env_missing"
    assert "crucible install align" in response.json()["error"]["message"]


def test_missing_weights_are_named_with_the_pull_command(
    align_client: TestClient, auth: dict[str, str], ffmpeg: str, idle_card: None
) -> None:
    response = submit(align_client, auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "model_not_installed"
    assert "crucible models pull qwen3-aligner" in response.json()["error"]["message"]


def test_no_ffmpeg_is_refused_before_the_job_is_queued(
    align_client: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    align_weights: Callable[[str], Path],
    idle_card: None,
) -> None:
    align_weights(MODEL)
    monkeypatch.setattr(align_job, "ffmpeg_path", lambda: None)
    response = submit(align_client, auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "ffmpeg_missing"
    assert "16 kHz mono float32" in response.json()["error"]["message"]


def test_the_mac_is_refused_because_nobody_has_measured_it(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not "it cannot work" — "nobody has measured it", which is a different refusal."""
    monkeypatch.setattr(
        accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB)
    )
    monkeypatch.setattr(align_job, "ffmpeg_path", lambda: "/opt/homebrew/bin/ffmpeg")
    with make_client(enable_align=True, backend=FAKE_MAC_BACKEND) as client:
        response = submit(client, auth)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "backend_unsupported"
    assert "nobody has measured" in response.json()["error"]["message"]


def test_somebody_else_on_the_card_refuses_by_name(
    align_client: TestClient,
    auth: dict[str, str],
    ffmpeg: str,
    align_weights: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    align_weights(MODEL)
    monkeypatch.setattr(
        accelerator,
        "probe_compute_apps",
        lambda: [ComputeApp(pid=44503, name="python", used_bytes=17 * GIB)],
    )
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (5 * GIB, 24 * GIB))
    response = submit(align_client, auth)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "accelerator_busy"


def test_a_chunk_with_no_audio_is_refused_rather_than_dropped(
    ready: TestClient, auth: dict[str, str]
) -> None:
    """A book aligned with 1,399 of its 1,400 chunks reads as a complete answer."""
    events = run_job(ready, auth, inputs={"41.flac": {"inline_base64": AUDIO}})
    assert terminal(events)["event"] == "failed"
    error = terminal(events)["data"]["error"]
    assert error["code"] == "invalid_inputs"
    assert "chunk(s) [42] have no audio" in error["message"]


def test_an_input_with_no_chunk_is_refused(
    ready: TestClient, auth: dict[str, str]
) -> None:
    events = run_job(
        ready, auth, inputs={**INPUTS, "43.flac": {"inline_base64": AUDIO}}
    )
    assert terminal(events)["event"] == "failed"
    assert "input(s) [43] have no chunk" in terminal(events)["data"]["error"]["message"]


def test_an_input_not_named_by_index_is_refused(
    ready: TestClient, auth: dict[str, str]
) -> None:
    events = run_job(
        ready,
        auth,
        inputs={"41.flac": {"inline_base64": AUDIO}, "two.flac": {"inline_base64": AUDIO}},
    )
    assert terminal(events)["event"] == "failed"
    assert "not named <index>.<ext>" in terminal(events)["data"]["error"]["message"]


# ---------------------------------------------------------------- it runs


def test_a_run_produces_an_alignment(ready: TestClient, auth: dict[str, str]) -> None:
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "done", terminal(events)
    assert terminal(events)["data"]["artifacts"] == ["alignment.json"]
    document = artifact(ready, auth, events)
    assert document["model"] == MODEL
    assert document["revision"] == (
        load_align_manifest(MODEL).spec(FAKE_BACKEND.kind).revision
    )
    assert document["language"] == "en"
    assert document["language_name"] == "English"
    assert document["dtype"] == "bfloat16"
    assert document["max_audio_s"] == 300.0
    assert document["sample_rate"] == 16_000
    assert [row["index"] for row in document["chunks"]] == [41, 42]


def test_a_result_is_placed_by_its_position_and_not_by_an_index(
    ready: TestClient, auth: dict[str, str]
) -> None:
    """The worker reports no index at all, so position is the whole identity.

    The fake derives its items from each chunk's TEXT, so an alignment that put
    chunk 42's words under index 41 would be visible here — which is exactly the
    mistake narrator's aligner made room for until results were dealt by
    position.
    """
    events = run_job(ready, auth)
    document = artifact(ready, auth, events)
    by_index = {row["index"]: row for row in document["chunks"]}
    assert [item["text"] for item in by_index[41]["items"]][:3] == ["He", "had", "been"]
    assert [item["text"] for item in by_index[42]["items"]] == ["Then", "he", "stopped."]


def test_a_cue_lands_per_chunk_before_the_artifact(
    ready: TestClient, auth: dict[str, str]
) -> None:
    """A run killed at chunk 900 of 1,400 costs the 500 it had not reached."""
    events = run_job(ready, auth)
    kinds = [event["event"] for event in events]
    cues = [event["data"] for event in events if event["event"] == "cue"]
    assert [cue["index"] for cue in cues] == [41, 42]
    assert all("items" in cue for cue in cues)
    # Every cue is out before the artifact event, which is the whole point.
    assert max(i for i, k in enumerate(kinds) if k == "cue") < kinds.index("artifact")


def test_the_server_and_not_the_client_chooses_how_it_runs(
    ready: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ffmpeg: str,
) -> None:
    """The dtype, the device, the ceiling and the language NAME never cross the wire."""
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_ALIGN_TRANSCRIPT", str(transcript))
    run_job(ready, auth, params={**PARAMS, "language": "yue"})
    sent = [json.loads(line) for line in transcript.read_text().splitlines()]
    load, align = sent[0], sent[1]
    assert load == {
        "op": "load",
        "model_dir": str(ready.app.state.store._config.home / "models" / MODEL
                         / FAKE_BACKEND.kind),
        "device": "cuda",
        "dtype": "bfloat16",
    }
    assert align["max_audio_s"] == 300.0
    assert align["ffmpeg"] == ffmpeg
    # The ISO code the client sent becomes the model's own English NAME.
    assert align["language"] == "Cantonese"
    # A chunk carries its audio and its text and NOTHING ELSE — in particular no
    # index, because position is the identity in both directions.
    assert all(set(chunk) == {"audio", "text"} for chunk in align["chunks"])


def test_a_failed_chunk_is_reported_and_the_run_continues(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlike `asr`, a hole here is visible: it is named, in the artifact."""
    monkeypatch.setenv("CRUCIBLE_FAKE_ALIGN_FAIL_CHUNK", "0")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "done"
    assert terminal(events)["data"]["failed"] == [41]
    document = artifact(ready, auth, events)
    rows = {row["index"]: row for row in document["chunks"]}
    assert "error" in rows[41] and "items" not in rows[41]
    assert "items" in rows[42]
    # The failure reaches a watching client at the same moment as its neighbours.
    cues = {e["data"]["index"]: e["data"] for e in events if e["event"] == "cue"}
    assert "error" in cues[41]


def test_a_chunk_over_five_minutes_is_refused_and_not_split(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Splitting it would change the alignment and nothing would say so."""
    monkeypatch.setenv("CRUCIBLE_FAKE_ALIGN_LONG_CHUNK", "1")
    events = run_job(ready, auth)
    document = artifact(ready, auth, events)
    rows = {row["index"]: row for row in document["chunks"]}
    assert "places timestamps within 300s" in rows[42]["error"]
    assert "items" in rows[41]


def test_the_items_say_they_are_the_models_tokens(
    ready: TestClient, auth: dict[str, str]
) -> None:
    """Crucible asserts nothing about words; the document says so in words."""
    document = artifact(ready, auth, run_job(ready, auth))
    assert document["items_are"].startswith("the model's own tokenization")


# --------------------------------------------------------------- residency


def test_the_model_is_loaded_once_and_held_across_jobs(
    ready: TestClient,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The whole reason this type has a residency: hundreds of chunks, one load."""
    transcript = tmp_path / "sent.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_ALIGN_TRANSCRIPT", str(transcript))
    run_job(ready, auth)
    run_job(ready, auth)
    ops = [json.loads(line)["op"] for line in transcript.read_text().splitlines()]
    assert ops == ["load", "align", "align"]


def test_a_resident_aligner_lights_up_its_own_row(
    ready: TestClient, auth: dict[str, str]
) -> None:
    run_job(ready, auth)
    capabilities = ready.get("/v1/info", headers=auth).json()["capabilities"]
    by_type = {entry["job_type"]: entry for entry in capabilities}
    row = next(r for r in by_type["align"]["models"] if r["id"] == MODEL)
    assert row["resident"] is True
    health = ready.get("/v1/health", headers=auth).json()
    assert health["resident_kind"] == KIND_ALIGN
    assert health["resident_models"] == [MODEL]


def test_the_accelerator_route_names_the_aligner_as_the_resident_kind(
    ready: TestClient, auth: dict[str, str]
) -> None:
    """It read `resident.model_id` and the literal "llm" until phase 4."""
    run_job(ready, auth)
    resident = ready.get("/v1/accelerator", headers=auth).json()["resident"]
    assert resident["kind"] == KIND_ALIGN
    assert resident["id"] == MODEL


def test_unloading_takes_it_off_the_card(
    ready: TestClient, auth: dict[str, str]
) -> None:
    run_job(ready, auth)
    events = run_job(ready, auth, type="unload-aligner", params={}, inputs={})
    assert terminal(events)["event"] == "done", terminal(events)
    assert terminal(events)["data"]["resident"] is None
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None


def test_unloading_something_that_is_not_resident_is_refused_by_name(
    ready: TestClient, auth: dict[str, str]
) -> None:
    response = ready.post(
        "/v1/jobs",
        headers=auth,
        json={"type": "unload-aligner", "model": MODEL, "params": {}},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "aligner_not_resident"
    assert "no aligner is" in response.json()["error"]["message"]


def test_a_resident_worker_that_died_is_loaded_again_rather_than_written_to(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resident row can outlive its process; the next job must notice."""
    monkeypatch.setenv("CRUCIBLE_FAKE_ALIGN_DIE_AFTER", "1")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert terminal(events)["data"]["error"]["code"] == "worker_failed"
    # And the residency does not go on advertising a worker that is gone.
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None

    monkeypatch.delenv("CRUCIBLE_FAKE_ALIGN_DIE_AFTER")
    assert terminal(run_job(ready, auth))["event"] == "done"


def test_a_load_that_fails_leaves_nothing_resident(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_ALIGN_LOAD_FAIL", "1")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert "told not to load" in terminal(events)["data"]["error"]["message"]
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None


def test_a_load_that_answers_with_results_is_refused(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A load produces no results; a worker that says otherwise is not trusted."""
    monkeypatch.setenv("CRUCIBLE_FAKE_ALIGN_LOAD_RESULT", "1")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert "answered a load request with" in terminal(events)["data"]["error"]["message"]


# ------------------------------------------------------------- it fails well


def test_a_worker_that_dies_fails_the_job_with_its_log(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_ALIGN_EXIT_CODE", "7")
    events = run_job(ready, auth)
    error = terminal(events)["data"]["error"]
    assert error["code"] == "worker_failed"
    # PHASE4-AUDIO.md section 6: the error carries the log, not a pointer to it.
    assert "told to exit before saying anything" in error["message"]


def test_a_short_stream_fails_the_job(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_ALIGN_SHORT", "1")
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    assert "matched to work by position" in terminal(events)["data"]["error"]["message"]


def test_a_library_printing_to_fd_1_is_refused_not_skipped(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whose lesson it is: whisperx's logger, Owen's 401-chunk book, 2026-09-05."""
    monkeypatch.setenv(
        "CRUCIBLE_FAKE_ALIGN_JUNK_LINE", "whisperx.alignment - WARNING - Failed"
    )
    events = run_job(ready, auth)
    assert terminal(events)["event"] == "failed"
    message = terminal(events)["data"]["error"]["message"]
    assert "not JSON" in message
    assert "whisperx.alignment" in message


def test_a_cancel_that_is_honoured_ends_the_job_and_the_residency(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the cancel path: the worker DOES go when asked.

    A cancel stops a held worker mid-exchange, so the session is gone; the
    resident row has to go with it, or `/v1/health` advertises an aligner that is
    not there until some later job happens to notice.
    """
    monkeypatch.setenv("CRUCIBLE_FAKE_ALIGN_SLOW_S", "30")
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
    assert terminal(events)["event"] == "cancelled", terminal(events)
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None
    # And the next job loads it again rather than sending into a closed pipe.
    monkeypatch.delenv("CRUCIBLE_FAKE_ALIGN_SLOW_S")
    assert terminal(run_job(ready, auth))["event"] == "done"


def test_a_cancel_stops_the_worker_and_does_not_sigkill_it(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crucible never SIGKILLs a process that may hold CUDA. Nor does a cancel."""
    from crucible import workers

    monkeypatch.setattr(workers, "STOP_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setenv("CRUCIBLE_FAKE_ALIGN_IGNORE_SIGTERM", "1")
    monkeypatch.setenv("CRUCIBLE_FAKE_ALIGN_SLOW_S", "30")

    pids: list[int] = []
    real_popen = workers.subprocess.Popen

    def watched(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        pids.append(process.pid)
        return process

    monkeypatch.setattr(workers.subprocess, "Popen", watched)

    response = submit(ready, auth)
    job_id = response.json()["job_id"]
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
    # The worker ignores SIGTERM, so the stop times out and says so by name
    # rather than escalating. Either terminal state proves the point; what must
    # never appear is a SIGKILL.
    assert terminal(events)["event"] in ("cancelled", "failed")
    if terminal(events)["event"] == "failed":
        assert "does not SIGKILL" in terminal(events)["data"]["error"]["message"]
    # And a cancelled run leaves nothing advertised as resident — whichever way
    # it ended, the session is gone and the row must go with it.
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None

    # This test made the process; this test cleans it up. Nothing in Crucible
    # will, by design.
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# ------------------------------------------------------------------- doctor


def test_check_reports_what_is_missing_in_order(
    make_client: Callable[..., TestClient],
    home: Path,
    align_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    align_weights: Callable[[str], Path],
) -> None:
    from crucible.config import load_config
    from crucible.residency import Residency

    with make_client(enable_align=True):
        pass
    config = load_config(home)
    job_type = align_job.AlignJobType(config, FAKE_BACKEND, Residency(config))

    monkeypatch.setattr(align_job, "ffmpeg_path", lambda: None)
    status = job_type.check(FAKE_BACKEND)
    assert status.ready is False
    assert "no ffmpeg on PATH" in status.detail

    monkeypatch.setattr(align_job, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    status = job_type.check(FAKE_BACKEND)
    assert status.ready is False
    assert "no aligner is installed" in status.detail

    align_weights(MODEL)
    status = job_type.check(FAKE_BACKEND)
    assert status.ready is True
    assert MODEL in status.detail
    assert "qwen-asr 0.0.6" in status.detail
