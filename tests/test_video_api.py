from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, hosttools, jobenv, tasks, weights
from crucible.jobs import video as video_job
from crucible.memorybudget import GIB
from crucible.videomodels import load_video_manifest

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, parse_sse, stamp_env

MODEL = "ltx-2.5-distilled"
FAKE_WORKER = Path(__file__).resolve().parent / "fake_video_worker.py"
FFMPEG = "/usr/bin/ffmpeg"
PROMPT = (
    "A red fox trots through fresh snow at dawn, the camera tracking alongside at knee "
    "height; low golden light, breath steaming, snow crunching underfoot and a crow calling"
)
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"rest of a picture").decode("ascii")


def _envs(home: Path, backend_kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for spec in jobenv.video_envs(backend_kind):
        stamp_env(home, spec, backend_kind, monkeypatch, python=Path(sys.executable))


def _weights(home: Path, backend_kind: str = FAKE_BACKEND.kind) -> Path:
    spec = load_video_manifest(MODEL).spec(backend_kind)
    directory = home / "models" / MODEL / backend_kind
    for name in spec.files:
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(b"weights")
    (directory / "crucible-pull.json").write_text(
        json.dumps({"hf_repo": spec.hf_repo, "revision": spec.revision, "bytes": 32_000_000_000,
                    "pulled": "2026-09-29T02:00:00+0000"}),
        encoding="utf-8",
    )
    for companion in spec.companions:
        part = directory / companion.name
        part.mkdir(parents=True, exist_ok=True)
        for entry in companion.files:
            (part / entry.target).write_bytes(b"gguf")
        (part / "crucible-pull.json").write_text(
            json.dumps({"hf_repo": companion.hf_repo, "revision": companion.revision,
                        "bytes": companion.bytes, "pulled": "2026-09-29T02:00:00+0000",
                        "files": [{"target": e.target} for e in companion.files]}),
            encoding="utf-8",
        )
    return directory


@pytest.fixture
def idle_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (23 * GIB, 24 * GIB))


@pytest.fixture
def ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hosttools, "ffmpeg_path", lambda: FFMPEG)


@pytest.fixture
def transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "video-worker.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_VIDEO_TRANSCRIPT", str(path))
    for engine in list(video_job.WORKER_SCRIPTS):
        monkeypatch.setitem(video_job.WORKER_SCRIPTS, engine, FAKE_WORKER)
    return path


@pytest.fixture
def ready(
    make_client: Callable[..., TestClient],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    idle_card: None,
    ffmpeg: None,
    transcript: Path,
) -> Iterator[TestClient]:
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    _envs(home, FAKE_BACKEND.kind, monkeypatch)
    _weights(home)
    with make_client(enable_video=True) as client:
        yield client


def submit(client: TestClient, auth: dict[str, str], **body: Any):
    body.setdefault("type", "video")
    body.setdefault("model", MODEL)
    body.setdefault("params", {"prompt": PROMPT, "width": 768, "height": 512, "duration_s": 2})
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


def wait_until_running(client: TestClient, auth: dict[str, str], job_id: str) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if client.get(f"/v1/jobs/{job_id}", headers=auth).json()["status"] == "running":
            return
        time.sleep(0.01)
    raise AssertionError("the job never started running")


@pytest.mark.parametrize(
    ("params", "code", "words"),
    [
        ({"prompt": PROMPT, "negative_prompt": "blurry"}, "video_param_unsupported", "runs unguided"),
        ({"prompt": PROMPT, "steps": 30}, "video_param_unsupported", "exactly 8 steps"),
        ({"prompt": PROMPT, "fps": 30}, "video_param_unsupported", "24 or 25"),
        ({"prompt": PROMPT, "width": 1280, "height": 720}, "video_size_not_supported", "multiples of 32"),
        ({"prompt": PROMPT, "width": 1920, "height": 1088}, "video_too_large", "no side over 1280"),
        ({"prompt": PROMPT, "width": 224, "height": 224}, "video_size_not_supported", "under 256"),
        ({"prompt": PROMPT, "num_frames": 100}, "video_frames_not_supported", "Send 97 or 105"),
        ({"prompt": PROMPT, "duration_s": 10}, "video_too_long", "at most 145 frames"),
        ({"prompt": PROMPT, "num_frames": 153}, "video_too_long", "6.04 s at 24 fps"),
        ({"prompt": PROMPT, "duration_s": 2, "num_frames": 49}, "invalid_params", "not both"),
        ({"prompt": PROMPT, "width": 768}, "invalid_params", "width and height together"),
        ({"prompt": PROMPT, "quality": "max"}, "invalid_params", "Extra inputs are not permitted"),
        ({"prompt": "   "}, "invalid_params", "is empty"),
    ],
)
def test_params_are_refused_by_name_before_anything_loads(
    ready: TestClient, auth: dict[str, str], transcript: Path, params: dict, code: str, words: str
) -> None:
    error = refusal(submit(ready, auth, params=params))
    assert error["code"] == code, error
    assert words in error["message"], error["message"]
    assert rows(transcript, "load") == []


def test_a_clip_is_made_with_step_progress_and_an_mp4(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    job_id, events = run_job(ready, auth, params={
        "prompt": PROMPT, "width": 768, "height": 512, "duration_s": 2, "seed": 7,
    })
    assert events[-1]["event"] == "done", events[-1]
    steps = [e["data"]["step"] for e in events if e["event"] == "progress" and e["data"].get("stage") == "denoising"]
    assert steps[-8:] == [1, 2, 3, 4, 5, 6, 7, 8]
    fractions = [e["data"]["fraction"] for e in events if e["event"] == "progress"]
    assert fractions == sorted(fractions)
    stages = [e["data"].get("stage") for e in events if e["event"] == "progress"]
    assert {"encoding", "connecting", "denoising", "decoding", "audio_decoding", "muxing"} <= set(stages)
    done = events[-1]["data"]
    assert done["artifacts"] == ["video.mp4"]
    video = done["video"]
    spec = load_video_manifest(MODEL).spec(FAKE_BACKEND.kind)
    assert (video["mode"], video["width"], video["height"]) == ("text-to-video", 768, 512)
    assert (video["num_frames"], video["fps"], video["duration_s"]) == (49, 24, round(49 / 24, 3))
    assert video["video_tokens"] == 7 * 24 * 16
    assert (video["seed"], video["steps"], video["audio"]) == (7, 8, True)
    assert (video["revision"], video["engine"], video["backend"]) == (spec.revision, "ltx", "cuda-linux")
    assert video["transformer"]["file"] == "LTX-2.5-Distilled-Q6_K.gguf"
    assert video["transformer"]["hf_repo"] == "Abiray/LTX-2.5-Distilled-GGUF"
    assert (video["audio_sample_rate"], video["audio_channels"]) == (48000, 2)
    assert video["peak_bytes"] == 19 and video["stage_peak_bytes"] == {"denoising": 19, "decoding": 8}
    assert video["memory_basis"] == "declared"
    assert video["stage_memory_bytes"]["denoising"] == spec.memory_bytes_estimate
    assert (video["encoder"], video["prompt_cache"]) == ("fake-h264", "miss")
    body = ready.get(f"/v1/jobs/{job_id}/artifacts/video.mp4", headers=auth).content
    assert body[4:8] == b"ftyp"
    load = rows(transcript, "load")[0]
    model_dir = Path(load["model_dir"])
    assert (load["engine"], load["device"], load["dtype"]) == ("ltx", "cuda", "bfloat16")
    assert load["memory_cap_bytes"] == spec.memory_bytes_estimate
    assert load["transformer_path"] == str(model_dir / "transformer-gguf" / "LTX-2.5-Distilled-Q6_K.gguf")
    made = rows(transcript, "generate")[0]
    assert (made["ffmpeg"], made["num_frames"], made["image_path"]) == (FFMPEG, 49, None)


def test_defaults_come_from_the_arm_and_audio_can_be_left_off(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    _, events = run_job(ready, auth, params={"prompt": PROMPT, "audio": False})
    assert events[-1]["event"] == "done", events[-1]
    video = events[-1]["data"]["video"]
    assert (video["width"], video["height"], video["num_frames"], video["fps"]) == (1280, 704, 121, 24)
    assert (video["audio"], video["audio_sample_rate"]) == (False, None)
    stages = {e["data"].get("stage") for e in events if e["event"] == "progress"}
    assert "audio_decoding" not in stages


def test_the_same_prompt_twice_in_a_leased_batch_skips_the_text_stages(
    ready: TestClient, auth: dict[str, str]
) -> None:
    params = {"prompt": PROMPT, "width": 768, "height": 512, "duration_s": 2,
              "lease": {"act": "video", "ttl_seconds": 60}}
    first = run_job(ready, auth, params=params)[1][-1]["data"]["video"]
    _, events = run_job(ready, auth, params={**params, "seed": 9})
    second = events[-1]["data"]["video"]
    assert (first["prompt_cache"], second["prompt_cache"]) == ("miss", "hit")
    stages = {e["data"].get("stage") for e in events if e["event"] == "progress"}
    assert "encoding" not in stages and "connecting" not in stages


def test_a_start_image_makes_it_image_to_video(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    _, events = run_job(
        ready, auth,
        params={"prompt": PROMPT, "width": 1280, "height": 704, "duration_s": 3},
        inputs={"start.png": {"inline_base64": PNG}},
    )
    assert events[-1]["event"] == "done", events[-1]
    video = events[-1]["data"]["video"]
    assert (video["mode"], video["input"], video["num_frames"]) == ("image-to-video", "start.png", 73)
    assert "conditioning" in {e["data"].get("stage") for e in events if e["event"] == "progress"}
    made = rows(transcript, "generate")[0]
    assert made["mode"] == "image-to-video" and made["image_path"].endswith("start.png")


def test_image_to_video_has_its_own_lower_ceiling_and_it_is_named_before_the_worker(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    _, events = run_job(
        ready, auth,
        params={"prompt": PROMPT, "width": 1280, "height": 704, "duration_s": 5},
        inputs={"start.png": {"inline_base64": PNG}},
    )
    assert events[-1]["event"] == "failed"
    error = events[-1]["data"]["error"]
    assert error["code"] == "video_too_large"
    assert "8,800" in error["message"] and "73 frames" in error["message"]
    assert rows(transcript, "load") == []


@pytest.mark.parametrize(
    ("inputs", "words"),
    [
        ({"notes.txt": {"inline_base64": base64.b64encode(b"hello").decode("ascii")}}, "not a PNG"),
        (
            {"a.png": {"inline_base64": PNG}, "b.png": {"inline_base64": PNG}},
            "reads at most one",
        ),
    ],
)
def test_a_bad_start_image_is_refused_before_the_worker(
    ready: TestClient, auth: dict[str, str], transcript: Path, inputs: dict, words: str
) -> None:
    _, events = run_job(ready, auth, inputs=inputs)
    assert events[-1]["event"] == "failed"
    assert events[-1]["data"]["error"]["code"] == "invalid_inputs"
    assert words in events[-1]["data"]["error"]["message"]
    assert rows(transcript, "load") == []


def test_load_video_leases_the_model_and_the_batch_reuses_it(
    ready: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    lease = {"act": "video", "ttl_seconds": 60}
    loaded = ready.post("/v1/jobs", headers=auth, json={"type": "load-video", "model": MODEL, "params": {"lease": lease}})
    assert loaded.status_code == 202, loaded.json()
    done = events_of(ready, auth, loaded.json()["job_id"])[-1]
    assert done["event"] == "done", done
    lease_id = done["data"]["lease_id"]
    assert done["data"]["resident"] == MODEL and lease_id
    for _ in range(2):
        _, events = run_job(ready, auth, params={"prompt": PROMPT, "width": 512, "height": 512,
                                                 "num_frames": 17, "lease": lease})
        assert events[-1]["data"]["lease_id"] == lease_id
    assert len(rows(transcript, "load")) == 1
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "video"
    assert ready.delete(f"/v1/leases/{lease_id}", headers=auth).status_code == 204
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None


def test_a_lease_must_name_the_video_act(ready: TestClient, auth: dict[str, str]) -> None:
    error = refusal(submit(ready, auth, params={"prompt": PROMPT, "lease": {"act": "image", "ttl_seconds": 60}}))
    assert error["code"] == "lease_act_mismatch"
    assert '"act": "video"' in error["message"]


def test_a_cancel_stops_between_steps_and_keeps_the_model(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRUCIBLE_FAKE_VIDEO_STEP_S", "0.4")
    lease = {"act": "video", "ttl_seconds": 60}
    loaded = ready.post("/v1/jobs", headers=auth, json={"type": "load-video", "model": MODEL, "params": {"lease": lease}})
    assert events_of(ready, auth, loaded.json()["job_id"])[-1]["event"] == "done"
    job_id = submit(ready, auth).json()["job_id"]
    wait_until_running(ready, auth, job_id)
    time.sleep(1.0)
    assert ready.delete(f"/v1/jobs/{job_id}", headers=auth).status_code == 200
    events = events_of(ready, auth, job_id)
    assert events[-1]["event"] == "cancelled", events[-1]
    steps = [e["data"]["step"] for e in events if e["event"] == "progress" and e["data"].get("stage") == "denoising"]
    assert 0 < max(steps) < 8
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "video"
    monkeypatch.setenv("CRUCIBLE_FAKE_VIDEO_STEP_S", "0")
    assert run_job(ready, auth)[1][-1]["event"] == "done"
    assert len(rows(transcript, "load")) == 1


def test_unload_video_takes_the_generator_off_the_card(ready: TestClient, auth: dict[str, str]) -> None:
    error = refusal(ready.post("/v1/jobs", headers=auth, json={"type": "unload-video", "model": MODEL}))
    assert error["code"] == "video_generator_not_resident"
    loaded = ready.post("/v1/jobs", headers=auth, json={"type": "load-video", "model": MODEL})
    assert events_of(ready, auth, loaded.json()["job_id"])[-1]["event"] == "done"
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] == "video"
    response = ready.post("/v1/jobs", headers=auth, json={"type": "unload-video", "model": MODEL})
    assert response.status_code == 202, response.json()
    assert events_of(ready, auth, response.json()["job_id"])[-1]["event"] == "done"
    assert ready.get("/v1/health", headers=auth).json()["resident_kind"] is None


def test_a_card_without_room_is_refused_before_the_worker_starts(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (18 * GIB, 19 * GIB))
    error = refusal(submit(ready, auth))
    assert error["code"] == "insufficient_memory"
    assert error["details"]["needed_bytes"] == 20_500_000_000
    assert rows(transcript, "load") == []


def test_no_ffmpeg_is_refused_at_submit(
    ready: TestClient, auth: dict[str, str], transcript: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hosttools, "ffmpeg_path", lambda: None)
    error = refusal(submit(ready, auth))
    assert error["code"] == "ffmpeg_missing"
    assert "video.mp4" in error["message"]
    assert rows(transcript, "load") == []


@pytest.fixture
def mac(
    make_client: Callable[..., TestClient],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    ffmpeg: None,
    transcript: Path,
) -> Iterator[TestClient]:
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    monkeypatch.setattr(accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB))
    _envs(home, FAKE_MAC_BACKEND.kind, monkeypatch)
    _weights(home, FAKE_MAC_BACKEND.kind)
    with make_client(enable_video=True, backend=FAKE_MAC_BACKEND, desktop_allowance_bytes=16 * GIB) as client:
        yield client


def _in_order(events: list[dict]) -> list[str]:
    stages: list[str] = []
    for event in events:
        stage = event["data"].get("stage") if event["event"] == "progress" else None
        if stage is not None and stage not in stages[-1:]:
            stages.append(stage)
    return stages


def test_the_mac_makes_a_clip_in_a_half_size_and_a_full_size_pass(
    mac: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    _, events = run_job(mac, auth, params={
        "prompt": PROMPT, "width": 768, "height": 512, "duration_s": 2, "seed": 5,
    })
    assert events[-1]["event"] == "done", events[-1]
    assert _in_order(events) == [
        "encoding", "denoising", "refining", "decoding", "audio_decoding", "muxing", "done",
    ]
    steps = {
        stage: [e["data"]["step"] for e in events if e["event"] == "progress" and e["data"].get("stage") == stage]
        for stage in ("denoising", "refining")
    }
    assert steps["denoising"][-8:] == list(range(1, 9)) and steps["refining"][-3:] == [1, 2, 3]
    fractions = [e["data"]["fraction"] for e in events if e["event"] == "progress"]
    assert fractions == sorted(fractions)
    video = events[-1]["data"]["video"]
    spec = load_video_manifest(MODEL).spec(FAKE_MAC_BACKEND.kind)
    assert (video["engine"], video["backend"], video["hf_repo"], video["revision"]) == (
        "ltx-2-mlx", "mlx-darwin", "dgrauet/ltx-2.5-mlx-q8", spec.revision,
    )
    assert video["transformer"] is None
    assert (video["steps"], video["refine_steps"], video["seed"]) == (8, 3, 5)
    assert (video["width"], video["height"], video["num_frames"], video["video_tokens"]) == (768, 512, 49, 7 * 24 * 16)
    assert video["stage_peak_bytes"] == {"denoising": 19, "refining": 23, "decoding": 8}
    assert video["memory_bytes_estimate"] == 29_000_000_000 == video["stage_memory_bytes"]["refining"]
    assert video["memory_basis"] == "declared" and video["sampling"] == {"passes": ["half", "full"]}
    assert (video["audio_sample_rate"], video["audio_channels"]) == (48000, 2)
    load = rows(transcript, "load")[0]
    assert (load["engine"], load["device"], load["transformer_path"]) == ("ltx-2-mlx", "metal", None)
    assert (load["mlx_cache_limit_bytes"], load["memory_cap_bytes"]) == (4_000_000_000, None)
    made = rows(transcript, "generate")[0]
    assert (made["steps"], made["refine_steps"], made["ffmpeg"]) == (8, 3, FFMPEG)


def test_the_pc_sends_no_refining_pass(ready: TestClient, auth: dict[str, str], transcript: Path) -> None:
    _, events = run_job(ready, auth)
    video = events[-1]["data"]["video"]
    assert video["refine_steps"] is None and "refining" not in _in_order(events)
    assert rows(transcript, "load")[0]["mlx_cache_limit_bytes"] is None


@pytest.mark.parametrize(
    ("params", "has_picture", "code", "words"),
    [
        ({"prompt": PROMPT, "width": 768, "height": 544}, False, "video_size_not_supported",
         "multiples of 64"),
        ({"prompt": PROMPT, "width": 1280, "height": 704, "duration_s": 6}, True, "video_too_large",
         "14,080"),
        ({"prompt": PROMPT, "negative_prompt": "blurry"}, False, "video_param_unsupported",
         "without classifier-free guidance"),
    ],
)
def test_the_mac_refuses_by_its_own_limits_before_the_worker(
    mac: TestClient, auth: dict[str, str], transcript: Path,
    params: dict, has_picture: bool, code: str, words: str,
) -> None:
    inputs = {"start.png": {"inline_base64": PNG}} if has_picture else {}
    response = submit(mac, auth, params=params, inputs=inputs)
    if response.status_code == 202:
        error = events_of(mac, auth, response.json()["job_id"])[-1]["data"]["error"]
    else:
        error = refusal(response)
    assert error["code"] == code, error
    assert words in error["message"], error["message"]
    assert "mlx-darwin" in error["message"]
    assert rows(transcript, "load") == []


def test_the_mac_starts_a_5_s_clip_from_a_picture_which_the_pc_would_refuse(
    mac: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    _, events = run_job(
        mac, auth,
        params={"prompt": PROMPT, "width": 1280, "height": 704, "duration_s": 5},
        inputs={"start.png": {"inline_base64": PNG}},
    )
    assert events[-1]["event"] == "done", events[-1]
    video = events[-1]["data"]["video"]
    assert (video["mode"], video["input"], video["num_frames"], video["video_tokens"]) == (
        "image-to-video", "start.png", 121, 14_080,
    )
    assert "conditioning" not in _in_order(events)
    assert rows(transcript, "generate")[0]["image_path"].endswith("start.png")


def test_a_mac_too_small_for_the_refining_pass_is_refused_before_the_worker(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch, ffmpeg: None, transcript: Path,
) -> None:
    small = FAKE_MAC_BACKEND.__class__(
        **{**FAKE_MAC_BACKEND.__dict__, "gpu": FAKE_MAC_BACKEND.gpu.__class__(
            vendor="apple", name="Apple M2 Max", vram_bytes=32 * GIB
        )}
    )
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    monkeypatch.setattr(accelerator, "probe_unified_memory", lambda: (20 * GIB, 32 * GIB))
    _envs(home, small.kind, monkeypatch)
    _weights(home, small.kind)
    with make_client(enable_video=True, backend=small, desktop_allowance_bytes=16 * GIB) as client:
        error = refusal(submit(client, auth))
    assert error["code"] == "insufficient_memory", error
    assert error["details"]["needed_bytes"] == 29_000_000_000
    assert rows(transcript, "load") == []


def test_the_mac_pack_is_gated_and_the_refusal_names_its_page(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    ffmpeg: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB))
    _envs(home, FAKE_MAC_BACKEND.kind, monkeypatch)
    with make_client(enable_video=True, backend=FAKE_MAC_BACKEND, desktop_allowance_bytes=16 * GIB) as client:
        error = refusal(submit(client, auth))
    assert error["code"] == "model_gated"
    assert error["details"]["accept_url"] == "https://huggingface.co/dgrauet/ltx-2.5-mlx-q8"


def test_a_gated_model_without_a_token_is_refused_with_the_page_to_accept(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    idle_card: None, ffmpeg: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    _envs(home, FAKE_BACKEND.kind, monkeypatch)
    with make_client(enable_video=True) as client:
        error = refusal(submit(client, auth))
    assert error["code"] == "model_gated"
    accept = "https://huggingface.co/Lightricks/LTX-2.5-Diffusers"
    assert error["details"]["accept_url"] == accept
    assert accept in error["message"] and "https://huggingface.co/settings/tokens" in error["message"]
    assert f"`crucible models pull {MODEL}`" in error["message"]


def test_the_pull_refuses_the_gated_repo_before_downloading(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crucible import videoweights
    from crucible.config import load_config

    from .conftest import configure_box

    monkeypatch.delenv("HF_TOKEN", raising=False)
    configure_box(home)
    config = load_config(home)
    manifest = load_video_manifest(MODEL)
    fetched: list[Any] = []
    monkeypatch.setattr(weights, "_snapshot", lambda *a, **k: fetched.append(a))
    monkeypatch.setattr(weights, "pull_files", lambda *a, **k: fetched.append(a))
    with pytest.raises(weights.WeightsError) as caught:
        videoweights.pull(config, manifest, manifest.spec(FAKE_BACKEND.kind))
    assert "is gated" in str(caught.value)
    assert "https://huggingface.co/Lightricks/LTX-2.5-Diffusers" in str(caught.value)
    assert fetched == []


def test_missing_weights_are_pulled_on_submit_when_a_token_is_set(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    idle_card: None, ffmpeg: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_tests")
    _envs(home, FAKE_BACKEND.kind, monkeypatch)
    with make_client(enable_video=True) as client:
        error = refusal(submit(client, auth))
    assert error["code"] == "installing"
    assert f"pulling the model '{MODEL}'" in error["message"]


def test_a_missing_env_is_installed_on_submit(
    make_client: Callable[..., TestClient], auth: dict[str, str],
    idle_card: None, ffmpeg: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tasks, "install_command", lambda: sys.executable)
    with make_client(enable_video=True) as client:
        error = refusal(submit(client, auth))
        assert error["code"] == "installing"
        assert "installing the video environment" in error["message"]
        started = client.app.state.tasks.get(error["details"]["task_id"])
        assert started.request["module"]["job_types"] == [{"type": "video"}]


def test_a_fresh_box_without_ffmpeg_still_installs_on_submit(
    make_client: Callable[..., TestClient], auth: dict[str, str],
    idle_card: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hosttools, "ffmpeg_path", lambda: None)
    monkeypatch.setattr(tasks, "install_command", lambda: sys.executable)
    with make_client(enable_video=True) as client:
        error = refusal(submit(client, auth))
    assert error["code"] == "installing", error
    assert "installing the video environment" in error["message"]


def test_a_video_trial_table_lifts_the_limits_for_measuring(
    ready: TestClient, auth: dict[str, str], home: Path, transcript: Path
) -> None:
    long_clip = {"prompt": PROMPT, "width": 768, "height": 512, "duration_s": 10}
    assert refusal(submit(ready, auth, params=long_clip))["code"] == "video_too_long"
    config = home / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8")
        + "\n[video_trial]\nmax_frames = 481\nmax_video_tokens = 60000\nsilence_timeout_s = 7200\n",
        encoding="utf-8",
    )
    _, events = run_job(ready, auth, params=long_clip)
    assert events[-1]["event"] == "done", events[-1]
    assert events[-1]["data"]["video"]["num_frames"] == 241


def test_the_trial_table_reads_only_positive_limits(tmp_path: Path) -> None:
    from types import SimpleNamespace

    config = tmp_path / "config.toml"
    config.write_text(
        '[video_trial]\nmax_side = 1920\nmax_frames = -1\nmax_pixels = "lots"\nsilence_timeout_s = 900\n',
        encoding="utf-8",
    )
    found = video_job.trial_settings(SimpleNamespace(path=config))
    assert found == {"max_side": 1920, "silence_timeout_s": 900.0}
    assert video_job.trial_settings(SimpleNamespace(path=tmp_path / "missing.toml")) == {}


def _add_to_config(home: Path, text: str) -> None:
    config = home / "config.toml"
    config.write_text(config.read_text(encoding="utf-8") + text, encoding="utf-8")


def test_without_a_video_desktop_table_the_mac_runs_as_before(
    mac: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    _, events = run_job(mac, auth)
    assert events[-1]["data"]["video"]["desktop"] is None
    assert rows(transcript, "load")[0]["desktop"] is None


def test_a_video_desktop_table_reaches_the_mac_worker_and_the_done_event(
    mac: TestClient, auth: dict[str, str], home: Path, transcript: Path
) -> None:
    _add_to_config(home, "\n[video_desktop]\nmax_tile_tokens = 12000\nlow_ram = false\n")
    _, events = run_job(mac, auth)
    assert events[-1]["event"] == "done", events[-1]
    desktop = events[-1]["data"]["video"]["desktop"]
    expected = {**video_job.DESKTOP_DEFAULTS, "max_tile_tokens": 12000, "low_ram": False}
    assert {key: desktop[key] for key in expected} == expected
    assert desktop["environment"] == {
        "MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "40", "LTX2_DIT_EVAL_EVERY": "1",
    }
    assert rows(transcript, "load")[0]["desktop"] == expected


def test_the_pc_ignores_a_video_desktop_table(
    ready: TestClient, auth: dict[str, str], home: Path, transcript: Path
) -> None:
    _add_to_config(home, "\n[video_desktop]\ndit_eval_every = 1\n")
    _, events = run_job(ready, auth)
    assert events[-1]["data"]["video"]["desktop"] is None
    assert rows(transcript, "load")[0]["desktop"] is None


def test_the_desktop_table_fills_defaults_and_ignores_bad_values(tmp_path: Path) -> None:
    from types import SimpleNamespace

    config = tmp_path / "config.toml"
    config.write_text(
        "[video_desktop]\ntile_spatial = 0\nmax_tile_tokens = 0\ndit_eval_every = true\n"
        'mlx_max_ops_per_buffer = 8\nlow_ram = "yes"\ntile_overlap = 4\n',
        encoding="utf-8",
    )
    found = video_job.desktop_settings(SimpleNamespace(path=config))
    assert found == {
        **video_job.DESKTOP_DEFAULTS, "max_tile_tokens": 0, "mlx_max_ops_per_buffer": 8,
        "tile_overlap": 4,
    }
    assert video_job.desktop_environment(found) == {
        "MLX_MAX_OPS_PER_BUFFER": "8", "MLX_MAX_MB_PER_BUFFER": "40", "LTX2_DIT_EVAL_EVERY": "1",
    }
    assert video_job.desktop_settings(SimpleNamespace(path=tmp_path / "missing.toml")) is None
    assert video_job.desktop_environment(None) == {}
    (tmp_path / "plain.toml").write_text("[video_trial]\nmax_frames = 481\n", encoding="utf-8")
    assert video_job.desktop_settings(SimpleNamespace(path=tmp_path / "plain.toml")) is None


@pytest.fixture(autouse=True)
def no_gpu_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test here reads the machine's real GPU; the ones that want samples script them."""
    from crucible import gpubusy

    monkeypatch.setattr(gpubusy, "reader_for", lambda backend_kind: (None, None))


def _scripted_probe(monkeypatch: pytest.MonkeyPatch, values: list[float | None]) -> None:
    from crucible import gpubusy

    queue = list(values)

    def read() -> float | None:
        return queue.pop(0) if queue else values[-1]

    monkeypatch.setattr(gpubusy, "reader_for", lambda backend_kind: (read, "scripted"))
    monkeypatch.setattr(gpubusy, "INTERVAL_S", 0.01)


def test_a_run_without_a_gpu_probe_reports_null_busy_fields_and_still_finishes(
    mac: TestClient, auth: dict[str, str]
) -> None:
    _, events = run_job(mac, auth)
    assert events[-1]["event"] == "done", events[-1]
    video = events[-1]["data"]["video"]
    assert (video["gpu_busy_mean_pct"], video["gpu_busy_max_pct"], video["gpu_busy_samples"]) == (None, None, 0)
    assert video["gpu_busy_source"] is None and video["gpu_busy_stages"] == {}
    assert (video["gpu_busy_target_pct"], video["gpu_busy_target_met"]) == (90.0, None)


def test_the_mac_records_gpu_busy_per_stage_and_judges_it_against_the_target(
    mac: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted_probe(monkeypatch, [60.0, 70.0, 80.0])
    monkeypatch.setenv("CRUCIBLE_FAKE_VIDEO_STEP_S", "0.03")
    _, events = run_job(mac, auth)
    assert events[-1]["event"] == "done", events[-1]
    video = events[-1]["data"]["video"]
    assert video["gpu_busy_samples"] >= 3 and video["gpu_busy_source"] == "scripted"
    assert video["gpu_busy_max_pct"] == 80.0 and 60.0 <= video["gpu_busy_mean_pct"] <= 80.0
    assert video["gpu_busy_pinned_samples"] == 0
    assert "denoising" in video["gpu_busy_stages"]
    assert video["gpu_busy_stages"]["denoising"]["samples"] >= 1
    assert (video["gpu_busy_target_pct"], video["gpu_busy_target_met"]) == (90.0, True)


def test_a_pinned_gpu_misses_the_target_the_table_sets(
    mac: TestClient, auth: dict[str, str], home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_to_config(home, "\n[video_desktop]\ngpu_busy_target_pct = 75\n")
    _scripted_probe(monkeypatch, [100.0])
    monkeypatch.setenv("CRUCIBLE_FAKE_VIDEO_STEP_S", "0.03")
    _, events = run_job(mac, auth)
    video = events[-1]["data"]["video"]
    assert video["gpu_busy_mean_pct"] == 100.0 and video["gpu_busy_pinned_samples"] == video["gpu_busy_samples"]
    assert (video["gpu_busy_target_pct"], video["gpu_busy_target_met"]) == (75.0, False)


def test_a_probe_that_keeps_failing_gives_up_without_failing_the_job(
    mac: TestClient, auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from crucible import gpubusy

    calls: list[int] = []

    def broken() -> float:
        calls.append(1)
        raise OSError("ioreg vanished")

    monkeypatch.setattr(gpubusy, "reader_for", lambda backend_kind: (broken, "broken"))
    monkeypatch.setattr(gpubusy, "INTERVAL_S", 0.01)
    monkeypatch.setenv("CRUCIBLE_FAKE_VIDEO_STEP_S", "0.03")
    _, events = run_job(mac, auth)
    assert events[-1]["event"] == "done", events[-1]
    video = events[-1]["data"]["video"]
    assert video["gpu_busy_samples"] == 0 and video["gpu_busy_mean_pct"] is None
    assert len(calls) == gpubusy.FAILURES_BEFORE_GIVING_UP


def test_the_pc_has_no_busy_target_by_default(ready: TestClient, auth: dict[str, str]) -> None:
    _, events = run_job(ready, auth)
    video = events[-1]["data"]["video"]
    assert (video["gpu_busy_target_pct"], video["gpu_busy_target_met"]) == (None, None)


def test_the_probes_parse_what_ioreg_and_nvidia_smi_print() -> None:
    from crucible import gpubusy

    ioreg = (
        '+-o AGXAcceleratorG13X  <class AGXAcceleratorG13X, id 0x1000003a1, registered>\n'
        '    {\n'
        '      "PerformanceStatistics" = {"In use system memory"=123,"Tiler Utilization %"=31,'
        '"Renderer Utilization %"=44,"Device Utilization %"=57,"Alloc system memory"=9}\n'
        '    }\n'
        '+-o AGXAcceleratorG13X  <class AGXAcceleratorG13X>\n'
        '      "PerformanceStatistics" = {"Device Utilization %"=12}\n'
    )
    assert gpubusy.parse_ioreg(ioreg) == 57.0
    assert gpubusy.parse_ioreg("no accelerator here") is None
    assert gpubusy.parse_nvidia_smi("97\n") == 97.0
    assert gpubusy.parse_nvidia_smi("[N/A]\n") is None
    assert gpubusy.verdict({"gpu_busy_mean_pct": 90.0}, 90.0) is True
    assert gpubusy.verdict({"gpu_busy_mean_pct": None}, 90.0) is None
    assert gpubusy.verdict({"gpu_busy_mean_pct": 50.0}, None) is None
