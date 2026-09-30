from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, jobenv, tasks, verdict
from crucible.jobs import segment as segment_job
from crucible.memorybudget import GIB
from crucible.segmentmodels import load_segment_manifest

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, parse_sse, stamp_env

Image = pytest.importorskip("PIL.Image")

CUTOUT = "birefnet"
SELECT = "sam2.1-hiera-large"
FAKE_WORKER = Path(__file__).resolve().parent / "fake_segment_worker.py"
WIDTH, HEIGHT = 64, 48


def picture_bytes(fmt: str = "PNG", mode: str = "RGB", size: tuple[int, int] = (WIDTH, HEIGHT)) -> bytes:
    picture = Image.new(mode, size, (200, 120, 40, 255) if mode == "RGBA" else (200, 120, 40))
    buffer = io.BytesIO()
    picture.save(buffer, format=fmt)
    return buffer.getvalue()


def inline(data: bytes) -> dict[str, str]:
    return {"inline_base64": base64.b64encode(data).decode("ascii")}


def _env(home: Path, backend_kind: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    return stamp_env(
        home, jobenv.worker_env("segment", backend_kind), backend_kind, monkeypatch,
        python=Path(sys.executable),
    )


def _weights(home: Path, model: str, backend_kind: str) -> Path:
    spec = load_segment_manifest(model).spec(backend_kind)
    directory = home / "models" / model / backend_kind
    directory.mkdir(parents=True, exist_ok=True)
    for name in spec.files:
        (directory / name).write_bytes(b"weights")
    (directory / "crucible-pull.json").write_text(
        json.dumps({"hf_repo": spec.hf_repo, "revision": spec.revision, "bytes": 900_000_000,
                    "pulled": "2026-09-29T02:00:00+0000"}),
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def idle_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (22 * GIB, 24 * GIB))


@pytest.fixture
def transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "segment-worker.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_SEGMENT_TRANSCRIPT", str(path))
    monkeypatch.setattr(segment_job, "WORKER_SCRIPT", FAKE_WORKER)
    return path


@pytest.fixture
def ready(
    make_client: Callable[..., TestClient],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    idle_card: None,
    transcript: Path,
) -> Iterator[TestClient]:
    _env(home, FAKE_BACKEND.kind, monkeypatch)
    for model in (CUTOUT, SELECT):
        _weights(home, model, FAKE_BACKEND.kind)
    with make_client(enable_segment=True) as client:
        yield client


def submit(client: TestClient, auth: dict[str, str], **body: Any):
    body.setdefault("type", "segment")
    body.setdefault("model", CUTOUT)
    body.setdefault("params", {})
    body.setdefault("inputs", {"photo.png": inline(picture_bytes())})
    return client.post("/v1/jobs", headers=auth, json=body)


def events_of(client: TestClient, auth: dict[str, str], job_id: str) -> list[dict]:
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        return parse_sse(line for line in stream.iter_lines())


def run_job(client: TestClient, auth: dict[str, str], **body: Any) -> tuple[str, list[dict]]:
    response = submit(client, auth, **body)
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]
    return job_id, events_of(client, auth, job_id)


def refusal(response: Any) -> dict:
    assert response.status_code >= 400, response.text
    return response.json()["error"]


def rows(transcript: Path, op: str) -> list[dict]:
    if not transcript.is_file():
        return []
    found = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    return [row for row in found if row.get("op") == op]


def artifact(client: TestClient, auth: dict[str, str], job_id: str, name: str) -> Any:
    response = client.get(f"/v1/jobs/{job_id}/artifacts/{name}", headers=auth)
    assert response.status_code == 200, response.text
    return Image.open(io.BytesIO(response.content))


def wait_until_running(client: TestClient, auth: dict[str, str], job_id: str) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if client.get(f"/v1/jobs/{job_id}", headers=auth).json()["status"] == "running":
            return
        time.sleep(0.01)
    raise AssertionError("the job never started running")


POINT = {"x": 20, "y": 10, "label": 1}


@pytest.mark.parametrize(
    ("model", "params", "code", "words"),
    [
        (CUTOUT, {"points": [POINT]}, "segment_param_unsupported", "does not take 'points'"),
        (CUTOUT, {"box": [1, 1, 9, 9]}, "segment_param_unsupported", "send the job to sam2.1-hiera-large"),
        (SELECT, {}, "segment_param_missing", "points at nothing"),
        (SELECT, {"points": [{"x": 3, "y": 3, "label": 0}]}, "invalid_params", "every point is label 0"),
        (SELECT, {"box": [9, 1, 2, 9]}, "invalid_params", "x1 > x0"),
        (SELECT, {"box": [1, 2, 3]}, "invalid_params", "box"),
        (SELECT, {"points": [{"x": 3, "y": 3, "label": 2}]}, "invalid_params", "label"),
        (SELECT, {"points": [{"x": -1, "y": 3, "label": 1}]}, "invalid_params", "x"),
        (SELECT, {"points": []}, "invalid_params", "points"),
        (CUTOUT, {"threshold": 0.5}, "invalid_params", "Extra inputs are not permitted"),
    ],
)
def test_params_are_refused_by_name_per_model(
    ready: TestClient, auth: dict[str, str], transcript: Path,
    model: str, params: dict, code: str, words: str,
) -> None:
    error = refusal(submit(ready, auth, model=model, params=params))
    assert error["code"] == code, error
    assert words in error["message"], error["message"]
    assert rows(transcript, "load") == []


def test_a_cutout_is_a_mask_and_an_rgba_picture_the_size_of_the_input(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    job_id, events = run_job(ready, auth)
    assert events[-1]["event"] == "done", events[-1]
    done = events[-1]["data"]
    assert done["artifacts"] == ["mask.png", "cutout.png"]
    result = done["segment"]
    spec = load_segment_manifest(CUTOUT).spec(FAKE_BACKEND.kind)
    assert (result["model"], result["kind"], result["engine"], result["revision"]) == (
        CUTOUT, "cutout", "birefnet", spec.revision
    )
    assert (result["width"], result["height"], result["input"]) == (WIDTH, HEIGHT, "photo.png")
    assert (result["points"], result["box"], result["score"]) == (None, None, None)
    assert (result["mask"], result["cutout"]) == ("mask.png", "cutout.png")
    assert result["peak_bytes"] == 7 and result["memory_basis"] == "declared"
    assert 0 < result["coverage"] < 1
    assert set(result["stage_seconds"]) == {"reading", "segmenting", "saving"}
    mask = artifact(ready, auth, job_id, "mask.png")
    cutout = artifact(ready, auth, job_id, "cutout.png")
    assert (mask.mode, mask.size) == ("L", (WIDTH, HEIGHT))
    assert (cutout.mode, cutout.size) == ("RGBA", (WIDTH, HEIGHT))
    assert list(cutout.getchannel("A").getdata()) == list(mask.getdata())
    assert mask.getpixel((WIDTH // 2, HEIGHT // 2)) == 200 and mask.getpixel((0, 0)) == 0
    assert cutout.getpixel((WIDTH // 2, HEIGHT // 2))[:3] == (200, 120, 40)
    fractions = [e["data"]["fraction"] for e in events if e["event"] == "progress"]
    assert fractions == sorted(fractions)
    load = rows(transcript, "load")[0]
    assert (load["engine"], load["device"], load["dtype"], load["working_side"]) == (
        "birefnet", "cuda", "float16", 1024
    )
    assert load["memory_cap_bytes"] == spec.memory_bytes_estimate
    assert load["mps_fallback"] is None


def test_a_selection_reaches_the_worker_as_points_and_a_box(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    points = [{"x": 30, "y": 20, "label": 1}, {"x": 5.5, "y": 40, "label": 0}]
    box = [10, 8, 50, 40]
    job_id, events = run_job(ready, auth, model=SELECT, params={"points": points, "box": box})
    assert events[-1]["event"] == "done", events[-1]
    result = events[-1]["data"]["segment"]
    assert (result["kind"], result["engine"], result["dtype"]) == ("select", "sam2", "float32")
    assert result["points"] == points and result["box"] == box
    assert (result["score"], result["multimask"]) == (0.93, False)
    sent = rows(transcript, "segment")[0]
    assert (sent["points"], sent["box"]) == (points, box)
    mask = artifact(ready, auth, job_id, "mask.png")
    assert set(mask.getdata()) == {0, 255}
    assert mask.getpixel((30, 20)) == 255 and mask.getpixel((2, 2)) == 0
    assert mask.getpixel((5, 40)) == 0


def test_one_click_asks_sam_for_its_candidates(ready: TestClient, auth: dict[str, str]) -> None:
    _, events = run_job(ready, auth, model=SELECT, params={"points": [POINT]})
    assert events[-1]["data"]["segment"]["multimask"] is True


def test_a_point_past_the_edge_is_refused_before_the_model_loads(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    _, events = run_job(ready, auth, model=SELECT, params={"points": [{"x": WIDTH, "y": 3, "label": 1}]})
    assert events[-1]["event"] == "failed"
    error = events[-1]["data"]["error"]
    assert error["code"] == "segment_prompt_outside_picture"
    assert f"{WIDTH}x{HEIGHT}" in error["message"] and "points[0]" in error["message"]
    _, events = run_job(ready, auth, model=SELECT, params={"box": [0, 0, WIDTH, HEIGHT + 1]})
    assert events[-1]["data"]["error"]["code"] == "segment_prompt_outside_picture"
    assert rows(transcript, "load") == []


def test_a_box_may_end_on_the_edge(ready: TestClient, auth: dict[str, str]) -> None:
    _, events = run_job(ready, auth, model=SELECT, params={"box": [0, 0, WIDTH, HEIGHT]})
    assert events[-1]["event"] == "done", events[-1]


@pytest.mark.parametrize("fmt", ["JPEG", "WEBP"])
def test_jpeg_and_webp_are_read_too(ready: TestClient, auth: dict[str, str], fmt: str) -> None:
    job_id, events = run_job(ready, auth, inputs={f"photo.{fmt.lower()}": inline(picture_bytes(fmt))})
    assert events[-1]["event"] == "done", events[-1]
    assert artifact(ready, auth, job_id, "mask.png").size == (WIDTH, HEIGHT)


def test_a_clear_pixel_of_the_input_stays_clear_in_the_cutout(ready: TestClient, auth: dict[str, str]) -> None:
    picture = Image.new("RGBA", (WIDTH, HEIGHT), (10, 20, 30, 255))
    picture.putpixel((WIDTH // 2, HEIGHT // 2), (10, 20, 30, 0))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    job_id, events = run_job(ready, auth, inputs={"layer.png": inline(buffer.getvalue())})
    assert events[-1]["event"] == "done", events[-1]
    cutout = artifact(ready, auth, job_id, "cutout.png")
    assert cutout.getpixel((WIDTH // 2, HEIGHT // 2))[3] == 0
    assert cutout.getpixel((WIDTH // 2 + 2, HEIGHT // 2))[3] == 200


@pytest.mark.parametrize(
    ("inputs", "words"),
    [
        ({}, "exactly one picture"),
        ({"a.png": inline(picture_bytes()), "b.png": inline(picture_bytes())}, "exactly one picture"),
        ({"notes.txt": inline(b"hello, this is not a picture")}, "is not a PNG"),
        ({"cut.png": inline(picture_bytes()[:20])}, "does not say how big it is"),
    ],
)
def test_the_input_must_be_one_whole_picture(
    ready: TestClient, auth: dict[str, str], transcript: Path, inputs: dict, words: str
) -> None:
    _, events = run_job(ready, auth, inputs=inputs)
    assert events[-1]["event"] == "failed"
    error = events[-1]["data"]["error"]
    assert error["code"] == "invalid_inputs" and words in error["message"], error
    assert rows(transcript, "load") == []


def test_a_picture_past_the_pixel_limit_is_refused(
    ready: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    manifest = load_segment_manifest(CUTOUT)
    small = replace(manifest, backends={
        kind: replace(spec, max_pixels=WIDTH * HEIGHT - 1) for kind, spec in manifest.backends.items()
    })
    monkeypatch.setattr(segment_job.MANIFESTS, "all", lambda: {CUTOUT: small, SELECT: load_segment_manifest(SELECT)})
    _, events = run_job(ready, auth)
    assert events[-1]["data"]["error"]["code"] == "image_too_large"


def test_without_a_lease_the_model_is_unloaded_when_the_job_ends(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert len(rows(transcript, "load")) == 2


def test_load_segment_leases_the_model_and_every_click_reuses_it(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    lease = {"act": "select", "ttl_seconds": 60}
    loaded = ready.post("/v1/jobs", headers=auth, json={"type": "load-segment", "model": SELECT, "params": {"lease": lease}})
    assert loaded.status_code == 202, loaded.json()
    done = events_of(ready, auth, loaded.json()["job_id"])[-1]
    assert done["event"] == "done", done
    lease_id = done["data"]["lease_id"]
    assert done["data"]["resident"] == SELECT and lease_id
    for x in (10, 20, 30):
        params = {"points": [{"x": x, "y": 10, "label": 1}], "lease": lease}
        _, events = run_job(ready, auth, model=SELECT, params=params)
        assert events[-1]["data"]["lease_id"] == lease_id
    assert len(rows(transcript, "load")) == 1
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "segment"
    assert ready.delete(f"/v1/leases/{lease_id}", headers=auth).status_code == 204
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None


def test_a_lease_must_name_the_models_own_class(ready: TestClient, auth: dict[str, str]) -> None:
    error = refusal(submit(ready, auth, model=CUTOUT, params={"lease": {"act": "select", "ttl_seconds": 60}}))
    assert error["code"] == "lease_act_mismatch"
    assert '"act": "cutout"' in error["message"]
    error = refusal(submit(ready, auth, params={"lease": {"act": "segment", "ttl_seconds": 60}}))
    assert error["code"] == "unknown_act"


def test_a_cancel_stops_the_job_and_keeps_the_model(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_SEGMENT_PAUSE_S", "20")
    lease = {"act": "cutout", "ttl_seconds": 60}
    loaded = ready.post("/v1/jobs", headers=auth, json={"type": "load-segment", "model": CUTOUT, "params": {"lease": lease}})
    assert events_of(ready, auth, loaded.json()["job_id"])[-1]["event"] == "done"
    job_id = submit(ready, auth).json()["job_id"]
    wait_until_running(ready, auth, job_id)
    deadline = time.monotonic() + 20.0
    while not rows(transcript, "segment") and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.delete(f"/v1/jobs/{job_id}", headers=auth).status_code == 200
    events = events_of(ready, auth, job_id)
    assert events[-1]["event"] == "cancelled", events[-1]
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "segment"
    monkeypatch.setenv("CRUCIBLE_FAKE_SEGMENT_PAUSE_S", "0")
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert len(rows(transcript, "load")) == 1


def test_unload_segment_takes_the_segmenter_off_the_card(ready: TestClient, auth: dict[str, str]) -> None:
    error = refusal(ready.post("/v1/jobs", headers=auth, json={"type": "unload-segment", "model": CUTOUT}))
    assert error["code"] == "segmenter_not_resident"
    loaded = ready.post("/v1/jobs", headers=auth, json={"type": "load-segment", "model": CUTOUT})
    assert events_of(ready, auth, loaded.json()["job_id"])[-1]["event"] == "done"
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "segment"
    response = ready.post("/v1/jobs", headers=auth, json={"type": "unload-segment", "model": CUTOUT})
    assert response.status_code == 202, response.json()
    assert events_of(ready, auth, response.json()["job_id"])[-1]["event"] == "done"
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None


def test_a_card_without_room_is_refused_before_the_worker_starts(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (4 * GIB, 7 * GIB))
    error = refusal(submit(ready, auth))
    assert error["code"] == "insufficient_memory"
    assert error["details"]["needed_bytes"] == 4_500_000_000
    assert rows(transcript, "load") == []


def test_the_mac_runs_both_models_on_metal_with_the_cpu_fallback(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch, transcript: Path,
) -> None:
    monkeypatch.setattr(accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB))
    _env(home, FAKE_MAC_BACKEND.kind, monkeypatch)
    for model in (CUTOUT, SELECT):
        _weights(home, model, FAKE_MAC_BACKEND.kind)
    with make_client(enable_segment=True, backend=FAKE_MAC_BACKEND, desktop_allowance_bytes=16 * GIB) as client:
        cut = run_job(client, auth)[1]
        chosen = run_job(client, auth, model=SELECT, params={"box": [4, 4, 40, 30]})[1]
    assert cut[-1]["event"] == "done", cut[-1]
    assert chosen[-1]["event"] == "done", chosen[-1]
    first, second = rows(transcript, "load")
    assert (first["device"], first["dtype"], first["memory_cap_bytes"]) == ("mps", "float32", None)
    assert (second["engine"], second["device"], second["dtype"]) == ("sam2", "mps", "float32")
    assert first["mps_fallback"] == "1"


def test_a_missing_env_is_installed_on_submit(
    make_client: Callable[..., TestClient], auth: dict[str, str],
    idle_card: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "install_command", lambda: sys.executable)
    with make_client(enable_segment=True) as client:
        error = refusal(submit(client, auth))
        assert error["code"] == "installing"
        assert "installing the segment environment" in error["message"]
        started = client.app.state.tasks.get(error["details"]["task_id"])
        assert started.request["module"]["job_types"] == [{"type": "segment"}]


def test_missing_weights_are_pulled_on_submit(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    idle_card: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _env(home, FAKE_BACKEND.kind, monkeypatch)
    with make_client(enable_segment=True) as client:
        error = refusal(submit(client, auth, model=SELECT, params={"points": [POINT]}))
    assert error["code"] == "installing"
    assert f"pulling the model '{SELECT}'" in error["message"]


def test_the_capability_rows_name_the_model_of_each_class(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    total, allowance = 26 * GIB, 3 * GIB
    decided = verdict.record(
        "cuda-linux",
        total_bytes=total,
        desktop_allowance_bytes=allowance,
        decisions=verdict.decide_all(
            "cuda-linux", total_bytes=total, desktop_allowance_bytes=allowance,
            gpu_vendor="nvidia", chosen={},
        ),
        routes={},
    )
    with make_client(capability=decided) as client:
        record = client.get("/v1/capability", headers=auth).json()
    found = {row["capability"]: row for row in record["classes"]}
    assert found["cutout"]["enabled"] is True and found["cutout"]["selected"] == CUTOUT
    assert found["cutout"]["summary"] == f"can cut out a picture's subject, using {CUTOUT}"
    assert found["select"]["selected"] == SELECT
    assert found["select"]["summary"] == f"can select what is pointed at in a picture, using {SELECT}"
