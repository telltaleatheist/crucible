from __future__ import annotations

import base64
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, jobenv, tasks, verdict
from crucible.errors import ApiError
from crucible.imagemodels import load_image_manifest
from crucible.jobs import image as image_job
from crucible.memorybudget import GIB

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, holding_the_card, parse_sse, stamp_env

MODEL = "qwen-image-2.1"
FAKE_WORKER = Path(__file__).resolve().parent / "fake_image_worker.py"
PROMPT = "A kitchen table with one red apple. No text, no letters, no numbers, no logos."
PNG_HEAD = b"\x89PNG\r\n\x1a\n"


def _env(home: Path, backend_kind: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    return stamp_env(
        home,
        jobenv.worker_env("image", backend_kind),
        backend_kind,
        monkeypatch,
        python=Path(sys.executable),
    )


def _weights(home: Path, backend_kind: str) -> Path:
    spec = load_image_manifest(MODEL).spec(backend_kind)
    directory = home / "models" / MODEL / backend_kind
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "crucible-pull.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "backend": backend_kind,
                "hf_repo": spec.hf_repo,
                "revision": spec.revision,
                "bytes": 33_134_949_212,
                "seconds": 60.0,
                "pulled": "2026-09-28T02:00:00+0000",
            }
        ),
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def idle_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (22 * GIB, 24 * GIB))


@pytest.fixture
def transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "image-worker.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_IMAGE_TRANSCRIPT", str(path))
    monkeypatch.setattr(image_job, "WORKER_SCRIPT", FAKE_WORKER)
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
    _weights(home, FAKE_BACKEND.kind)
    with make_client(enable_image=True) as client:
        yield client


def submit(client: TestClient, auth: dict[str, str], **body: Any):
    body.setdefault("type", "image")
    body.setdefault("model", MODEL)
    body.setdefault("params", {"prompt": PROMPT, "width": 512, "height": 512, "steps": 4})
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


def loads(transcript: Path) -> list[dict]:
    if not transcript.is_file():
        return []
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("op") == "load"]


def leased(client: TestClient, auth: dict[str, str]) -> dict:
    with holding_the_card(client, act="image"):
        assert run_job(client, auth)[1][-1]["event"] == "done"
        opened = client.post(
            f"/v1/models/{MODEL}/lease", headers=auth, json={"act": "image", "ttl_seconds": 60}
        )
    assert opened.status_code == 201, opened.text
    return opened.json()


def wait_until_running(client: TestClient, auth: dict[str, str], job_id: str) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if client.get(f"/v1/jobs/{job_id}", headers=auth).json()["status"] == "running":
            return
        time.sleep(0.01)
    raise AssertionError("the job never started running")


@pytest.mark.parametrize(
    ("params", "words"),
    [
        ({}, "prompt"),
        ({"prompt": "   "}, "the prompt is empty"),
        ({"prompt": PROMPT, "width": 520}, "not a multiple of 16"),
        ({"prompt": PROMPT, "height": 128}, "greater than or equal to 256"),
        ({"prompt": PROMPT, "steps": 0}, "greater than or equal to 1"),
        ({"prompt": PROMPT, "steps": 101}, "less than or equal to 100"),
        ({"prompt": PROMPT, "seed": -1}, "greater than or equal to 0"),
        ({"prompt": PROMPT, "negative_prompt": "blurry"}, "only read when guidance is above 1.0"),
        ({"prompt": PROMPT, "guidance": 4.0}, "needs a negative_prompt"),
        ({"prompt": PROMPT, "image_strength": 1.0}, "less than 1"),
        ({"prompt": PROMPT, "quantize": 8}, "Extra inputs are not permitted"),
        ({"prompt": PROMPT, "mask_blur": 4}, "this job has none"),
        ({"prompt": PROMPT, "mask": " "}, "mask is empty"),
        ({"prompt": PROMPT, "mask": "mask.png", "mask_blur": 999}, "less than or equal to 256"),
    ],
)
def test_params_are_refused_by_name(
    ready: TestClient, auth: dict[str, str], params: dict, words: str
) -> None:
    error = refusal(submit(ready, auth, params=params))
    assert error["code"] == "invalid_params"
    assert words in error["message"]


def test_a_side_the_cuda_engine_cannot_tile_is_refused_with_the_multiple(
    ready: TestClient, auth: dict[str, str]
) -> None:
    error = refusal(submit(ready, auth, params={"prompt": PROMPT, "width": 528, "height": 512}))
    assert error["code"] == "image_size_not_supported"
    assert "multiples of 32" in error["message"]
    assert error["details"]["size_multiple"] == 32


def test_a_picture_larger_than_the_sized_limit_is_refused_before_loading(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    error = refusal(submit(ready, auth, params={"prompt": PROMPT, "width": 2048, "height": 1024}))
    assert error["code"] == "image_too_large"
    assert "1,048,576 pixels" in error["message"]
    assert loads(transcript) == []


def test_image_to_image_is_refused_on_an_arm_that_does_not_declare_it() -> None:
    spec = dataclasses.replace(
        load_image_manifest(MODEL).spec(FAKE_BACKEND.kind), image_to_image=False
    )
    params = image_job.ImageParams(prompt=PROMPT, width=512, height=512, image_strength=0.5)
    with pytest.raises(ApiError) as refused:
        image_job.refuse_what_the_arm_cannot_make(params, spec, MODEL)
    assert refused.value.code == "image_to_image_unsupported"


def test_inpainting_is_refused_on_an_arm_that_does_not_declare_it() -> None:
    spec = dataclasses.replace(load_image_manifest(MODEL).spec(FAKE_BACKEND.kind), inpaint=False)
    params = image_job.ImageParams(prompt=PROMPT, width=512, height=512, mask="mask.png")
    with pytest.raises(ApiError) as refused:
        image_job.refuse_what_the_arm_cannot_make(params, spec, MODEL)
    assert refused.value.code == "inpaint_unsupported"


def test_the_pc_accepts_an_input_image(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    picture = base64.b64encode(PNG_HEAD + b"rest of a picture").decode("ascii")
    _, events = run_job(
        ready,
        auth,
        params={"prompt": PROMPT, "width": 512, "height": 512, "steps": 4, "image_strength": 0.6},
        inputs={"start.png": {"inline_base64": picture}},
    )
    assert events[-1]["event"] == "done", events[-1]
    assert events[-1]["data"]["image"]["input"] == "start.png"


def test_an_image_is_made_with_step_progress_and_a_reproducible_result(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    job_id, events = run_job(
        ready, auth, params={"prompt": PROMPT, "width": 512, "height": 384, "steps": 4, "seed": 7}
    )
    assert events[-1]["event"] == "done", events[-1]
    steps = [e["data"]["step"] for e in events if e["event"] == "progress" and e["data"].get("stage") == "denoising"]
    assert steps[-4:] == [1, 2, 3, 4]
    done = events[-1]["data"]
    assert done["artifacts"] == ["image.png"]
    image = done["image"]
    spec = load_image_manifest(MODEL).spec(FAKE_BACKEND.kind)
    assert (image["seed"], image["steps"], image["width"], image["height"]) == (7, 4, 512, 384)
    assert image["revision"] == spec.revision and image["model"] == MODEL
    assert image["engine"] == "diffusers" and image["memory_basis"] == spec.memory_basis
    png = ready.get(f"/v1/jobs/{job_id}/artifacts/image.png", headers=auth)
    assert png.status_code == 200 and png.content.startswith(PNG_HEAD)
    load = loads(transcript)[0]
    assert (load["engine"], load["device"], load["memory_cap_bytes"]) == (
        "diffusers", "cuda", spec.memory_bytes_estimate
    )


def test_a_seed_left_out_is_chosen_and_reported(ready: TestClient, auth: dict[str, str]) -> None:
    _, events = run_job(ready, auth)
    seed = events[-1]["data"]["image"]["seed"]
    assert isinstance(seed, int) and 0 <= seed <= image_job.MAX_SEED


def test_without_a_lease_the_model_is_unloaded_when_the_job_ends(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert len(loads(transcript)) == 2


def test_a_lease_keeps_the_model_loaded_across_a_batch(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    lease = leased(ready, auth)
    assert (lease["subject"], lease["kind"]) == (MODEL, "image")
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "image"
    assert len(loads(transcript)) == 1
    assert ready.delete(f"/v1/leases/{lease['lease_id']}", headers=auth).status_code == 204
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None


def test_a_cancel_stops_between_steps_and_keeps_the_model(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_IMAGE_STEP_S", "0.3")
    leased(ready, auth)
    job_id = submit(ready, auth, params={"prompt": PROMPT, "width": 512, "height": 512, "steps": 60}).json()["job_id"]
    wait_until_running(ready, auth, job_id)
    time.sleep(1.0)
    deleted = ready.delete(f"/v1/jobs/{job_id}", headers=auth)
    assert deleted.status_code == 200, (deleted.json(), events_of(ready, auth, job_id)[-3:])
    events = events_of(ready, auth, job_id)
    assert events[-1]["event"] == "cancelled", events[-1]
    steps = [e["data"]["step"] for e in events if e["event"] == "progress" and e["data"].get("stage") == "denoising"]
    assert 0 < max(steps) < 60
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "image"
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert len(loads(transcript)) == 1


def test_unload_image_takes_the_generator_off_the_card(
    ready: TestClient, auth: dict[str, str]
) -> None:
    error = refusal(ready.post("/v1/jobs", headers=auth, json={"type": "unload-image", "model": MODEL}))
    assert error["code"] == "generator_not_resident"
    lease = leased(ready, auth)
    assert ready.delete(f"/v1/leases/{lease['lease_id']}", headers=auth).status_code == 204
    with holding_the_card(ready, act="image"):
        assert run_job(ready, auth)[1][-1]["event"] == "done"
        assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "image"
    response = ready.post("/v1/jobs", headers=auth, json={"type": "unload-image", "model": MODEL})
    assert response.status_code == 202, response.json()
    assert events_of(ready, auth, response.json()["job_id"])[-1]["event"] == "done"
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None


def test_a_card_without_room_is_refused_before_the_worker_starts(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (15 * GIB, 16 * GIB))
    error = refusal(submit(ready, auth))
    assert error["code"] == "insufficient_memory"
    assert error["details"]["needed_bytes"] == load_image_manifest(MODEL).spec(FAKE_BACKEND.kind).memory_bytes_estimate
    assert loads(transcript) == []


def test_a_mac_too_small_for_one_stage_is_refused_by_the_measured_number(
    make_client: Callable[..., TestClient],
    home: Path,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    transcript: Path,
) -> None:
    small = FAKE_MAC_BACKEND.__class__(
        **{**FAKE_MAC_BACKEND.__dict__, "gpu": FAKE_MAC_BACKEND.gpu.__class__(
            vendor="apple", name="Apple M2", vram_bytes=16 * GIB
        )}
    )
    monkeypatch.setattr(accelerator, "probe_unified_memory", lambda: (8 * GIB, 16 * GIB))
    _env(home, small.kind, monkeypatch)
    _weights(home, small.kind)
    with make_client(enable_image=True, backend=small, desktop_allowance_bytes=4 * GIB) as client:
        error = refusal(submit(client, auth))
    need = load_image_manifest(MODEL).spec(small.kind).memory_bytes_estimate
    assert error["code"] == "insufficient_memory"
    assert error["details"]["needed_bytes"] == need
    assert loads(transcript) == []


def test_the_mac_sends_the_cache_limit_and_accepts_an_input_image(
    make_client: Callable[..., TestClient],
    home: Path,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    transcript: Path,
) -> None:
    monkeypatch.setattr(accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB))
    _env(home, FAKE_MAC_BACKEND.kind, monkeypatch)
    _weights(home, FAKE_MAC_BACKEND.kind)
    picture = base64.b64encode(PNG_HEAD + b"rest of a picture").decode("ascii")
    with make_client(enable_image=True, backend=FAKE_MAC_BACKEND, desktop_allowance_bytes=16 * GIB) as client:
        _, events = run_job(
            client,
            auth,
            params={"prompt": PROMPT, "width": 528, "height": 512, "steps": 2, "image_strength": 0.6},
            inputs={"start.png": {"inline_base64": picture}},
        )
        assert events[-1]["event"] == "done", events[-1]
        assert events[-1]["data"]["image"]["input"] == "start.png"
        _, refused = run_job(
            client,
            auth,
            params={"prompt": PROMPT, "width": 512, "height": 512, "steps": 2, "image_strength": 0.6},
            inputs={"notes.txt": {"inline_base64": base64.b64encode(b"hello").decode("ascii")}},
        )
    assert refused[-1]["event"] == "failed"
    assert refused[-1]["data"]["error"]["code"] == "invalid_inputs"
    load = loads(transcript)[0]
    spec = load_image_manifest(MODEL).spec(FAKE_MAC_BACKEND.kind)
    assert (load["engine"], load["mlx_cache_limit_bytes"], load["memory_cap_bytes"]) == (
        "mflux", spec.mlx_cache_limit_bytes, None
    )


def test_a_missing_env_is_installed_on_submit(
    make_client: Callable[..., TestClient],
    auth: dict[str, str],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "install_command", lambda: sys.executable)
    with make_client(enable_image=True) as client:
        error = refusal(submit(client, auth))
        assert error["code"] == "installing"
        assert "installing the image environment" in error["message"]
        started = client.app.state.tasks.get(error["details"]["task_id"])
        assert started.request["module"]["job_types"] == [{"type": "image"}]


def test_missing_weights_are_pulled_on_submit(
    make_client: Callable[..., TestClient],
    home: Path,
    auth: dict[str, str],
    idle_card: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _env(home, FAKE_BACKEND.kind, monkeypatch)
    with make_client(enable_image=True) as client:
        error = refusal(submit(client, auth))
    assert error["code"] == "installing"
    assert f"pulling the model '{MODEL}'" in error["message"]


def test_the_capability_row_names_the_model_and_the_fit(
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
    row = next(r for r in record["classes"] if r["capability"] == "image")
    assert row["enabled"] is True
    assert row["selected"] == MODEL
    assert row["summary"] == f"can generate images, using {MODEL}"


def test_a_card_that_cannot_hold_one_stage_is_disabled_with_the_numbers() -> None:
    from crucible.capabilityclasses import BY_NAME

    decided = verdict.decide(
        BY_NAME["image"], "cuda-linux",
        total_bytes=12 * GIB, desktop_allowance_bytes=3 * GIB,
        gpu_vendor="nvidia", chosen=None,
    )
    assert decided.enabled is False
    assert decided.summary.startswith("cannot generate images")
    assert decided.shortfall_bytes > 0


def _png(picture: Any) -> str:
    from io import BytesIO

    buffer = BytesIO()
    picture.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _red(width: int = 512, height: int = 512) -> str:
    from PIL import Image

    return _png(Image.new("RGB", (width, height), (200, 0, 0)))


def _left_half(width: int = 512, height: int = 512) -> str:
    from PIL import Image

    drawn = Image.new("L", (width, height), 0)
    drawn.paste(255, (0, 0, width // 2, height))
    return _png(drawn)


def generates(transcript: Path) -> list[dict]:
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("op") == "generate" and "masked" in row]


def test_a_mask_regenerates_only_its_region_and_keeps_every_other_pixel(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    from io import BytesIO

    import numpy as np
    from PIL import Image

    job_id, events = run_job(
        ready,
        auth,
        params={"prompt": PROMPT, "width": 512, "height": 512, "steps": 4, "mask": "mask.png"},
        inputs={"photo.png": {"inline_base64": _red()}, "mask.png": {"inline_base64": _left_half()}},
    )
    assert events[-1]["event"] == "done", events[-1]
    image = events[-1]["data"]["image"]
    assert (image["input"], image["mask"], image["mask_blur"], image["image_strength"]) == (
        "photo.png", "mask.png", 8, None
    )
    assert image["mask_coverage"] == 0.5
    made = generates(transcript)[-1]
    assert (made["masked"], made["start_step"], made["mask_blur"]) == (True, 0, 8)
    png = ready.get(f"/v1/jobs/{job_id}/artifacts/image.png", headers=auth).content
    pixels = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
    assert (pixels[:, 256:] == [200, 0, 0]).all()
    assert (pixels[:, :240] == [0, 0, 255]).all()


def test_a_mask_with_a_strength_starts_the_region_from_the_input(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    _, events = run_job(
        ready,
        auth,
        params={
            "prompt": PROMPT, "width": 512, "height": 512, "steps": 20,
            "mask": "m.png", "mask_blur": 0, "image_strength": 0.2,
        },
        inputs={"photo.png": {"inline_base64": _red()}, "m.png": {"inline_base64": _left_half()}},
    )
    assert events[-1]["event"] == "done", events[-1]
    assert events[-1]["data"]["image"]["mask_blur"] == 0
    made = generates(transcript)[-1]
    assert (made["start_step"], made["mask_blur"], made["image_strength"]) == (4, 0, 0.2)


@pytest.mark.parametrize(
    ("inputs", "code", "words"),
    [
        ({"photo.png": "red"}, "invalid_inputs", "mask names the input 'mask.png'"),
        ({"mask.png": "half"}, "invalid_inputs", "exactly two inputs"),
        ({"a.png": "red", "b.png": "red", "mask.png": "half"}, "invalid_inputs", "exactly two inputs"),
        ({"photo.png": "red", "mask.png": "text"}, "invalid_inputs", "(the mask) is not a"),
        ({"photo.png": "red", "mask.png": "small"}, "mask_size_mismatch", "256x256"),
        ({"photo.png": "red", "mask.png": "black"}, "mask_empty", "selects nothing"),
    ],
)
def test_a_mask_the_job_cannot_use_is_refused_by_name(
    ready: TestClient, auth: dict[str, str], transcript: Path,
    inputs: dict[str, str], code: str, words: str,
) -> None:
    from PIL import Image

    payloads = {
        "red": _red(),
        "half": _left_half(),
        "small": _left_half(256, 256),
        "black": _png(Image.new("L", (512, 512), 0)),
        "text": base64.b64encode(b"not a picture").decode("ascii"),
    }
    _, events = run_job(
        ready,
        auth,
        params={"prompt": PROMPT, "width": 512, "height": 512, "steps": 2, "mask": "mask.png"},
        inputs={name: {"inline_base64": payloads[kind]} for name, kind in inputs.items()},
    )
    assert events[-1]["event"] == "failed", events[-1]
    error = events[-1]["data"]["error"]
    assert error["code"] == code
    assert words in error["message"]
    if code != "mask_empty":
        assert loads(transcript) == []


def test_an_empty_mask_is_refused_without_losing_the_loaded_model(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    from PIL import Image

    leased(ready, auth)
    _, events = run_job(
        ready,
        auth,
        params={"prompt": PROMPT, "width": 512, "height": 512, "steps": 2, "mask": "mask.png"},
        inputs={
            "photo.png": {"inline_base64": _red()},
            "mask.png": {"inline_base64": _png(Image.new("L", (512, 512), 0))},
        },
    )
    assert events[-1]["data"]["error"]["code"] == "mask_empty"
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "image"
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert len(loads(transcript)) == 1


def test_the_mac_regenerates_a_masked_region_too(
    make_client: Callable[..., TestClient],
    home: Path,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    transcript: Path,
) -> None:
    monkeypatch.setattr(accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB))
    _env(home, FAKE_MAC_BACKEND.kind, monkeypatch)
    _weights(home, FAKE_MAC_BACKEND.kind)
    with make_client(enable_image=True, backend=FAKE_MAC_BACKEND, desktop_allowance_bytes=16 * GIB) as client:
        _, events = run_job(
            client,
            auth,
            params={"prompt": PROMPT, "width": 528, "height": 512, "steps": 2, "mask": "mask.png"},
            inputs={
                "photo.png": {"inline_base64": _red(528, 512)},
                "mask.png": {"inline_base64": _left_half(528, 512)},
            },
        )
    assert events[-1]["event"] == "done", events[-1]
    assert events[-1]["data"]["image"]["engine"] == "mflux"
    assert generates(transcript)[-1]["masked"] is True
