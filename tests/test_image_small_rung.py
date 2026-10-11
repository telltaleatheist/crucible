"""The image verb's small rung: dreamshaper-xl-lightning (SDXL) for 8 GiB CUDA cards.

The pick by fit, a request that names no model, per-model defaults and the negative-prompt
rule, the refusals of what the rung does not do, its weights (repo files plus a VAE
companion) and the facts the playground publishes. The worker is the fake one
(tests/fake_image_worker.py); nothing here touches a card.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, imageweights, jobenv, verdict
from crucible.backend import Backend, Gpu
from crucible.capabilityclasses import BY_NAME
from crucible.imagemodels import load_image_manifest
from crucible.jobs import image as image_job
from crucible.memorybudget import GIB
from crucible.weights import STAMP_NAME

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, parse_sse, stamp_env

SMALL = "dreamshaper-xl-lightning"
BIG = "qwen-image-2.1"
FAKE_WORKER = Path(__file__).resolve().parent / "fake_image_worker.py"
PROMPT = "album cover art, watercolour folk landscape, rolling green hills, warm morning light"
PNG_HEAD = b"\x89PNG\r\n\x1a\n"

LAPTOP = Backend(
    kind="cuda-linux",
    platform="linux",
    arch="x86_64",
    gpu=Gpu(vendor="nvidia", name="NVIDIA GeForce RTX 3070 Laptop GPU", vram_bytes=8 * GIB),
    detail="test double",
)


def _record(backend: Backend, allowance: int) -> Any:
    return verdict.record(
        backend.kind,
        total_bytes=backend.gpu.vram_bytes,
        desktop_allowance_bytes=allowance,
        decisions=verdict.decide_all(
            backend.kind, total_bytes=backend.gpu.vram_bytes, desktop_allowance_bytes=allowance,
            gpu_vendor=backend.gpu.vendor, chosen={}, audio_low_vram=False,
        ),
        routes={},
    )


def _env(home: Path, backend_kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    stamp_env(
        home, jobenv.worker_env("image", backend_kind), backend_kind, monkeypatch,
        python=Path(sys.executable),
    )


def _stamp(directory: Path, record: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / STAMP_NAME).write_text(json.dumps(record), encoding="utf-8")


def _small_weights(home: Path, *, with_vae: bool = True) -> Path:
    spec = load_image_manifest(SMALL).spec("cuda-linux")
    directory = home / "models" / SMALL / "cuda-linux"
    for name in spec.files:
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(b"w")
    _stamp(directory, {
        "family": "models", "id": SMALL, "backend": "cuda-linux",
        "hf_repo": spec.hf_repo, "revision": spec.revision, "files": list(spec.files),
        "bytes": 6_771_000_000, "seconds": 60.0, "pulled": "2026-10-10T02:00:00+0000",
    })
    vae = spec.vae
    assert vae is not None
    if with_vae:
        for entry in vae.files:
            (directory / "vae" / entry.target).parent.mkdir(parents=True, exist_ok=True)
            (directory / "vae" / entry.target).write_bytes(b"v")
        _stamp(directory / "vae", {
            "label": "vae", "hf_repo": vae.hf_repo, "revision": vae.revision,
            "files": [{"source": e.source, "target": e.target, "sha256": e.sha256} for e in vae.files],
            "bytes": vae.bytes, "seconds": 1.0, "pulled": "2026-10-10T02:00:00+0000",
        })
    return directory


@pytest.fixture
def laptop_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (int(7.6 * GIB), 8 * GIB))


@pytest.fixture
def transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "image-worker.jsonl"
    monkeypatch.setenv("CRUCIBLE_FAKE_IMAGE_TRANSCRIPT", str(path))
    monkeypatch.setattr(image_job, "WORKER_SCRIPT", FAKE_WORKER)
    return path


@pytest.fixture
def laptop(
    make_client: Callable[..., TestClient],
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    laptop_card: None,
    transcript: Path,
) -> Iterator[TestClient]:
    _env(home, LAPTOP.kind, monkeypatch)
    _small_weights(home)
    with make_client(enable_image=True, backend=LAPTOP, capability=_record(LAPTOP, 1 * GIB)) as client:
        yield client


def submit(client: TestClient, auth: dict[str, str], **body: Any) -> Any:
    body.setdefault("type", "image")
    body.setdefault("params", {"prompt": PROMPT})
    return client.post("/v1/jobs", headers=auth, json=body)


def run_job(client: TestClient, auth: dict[str, str], **body: Any) -> list[dict]:
    response = submit(client, auth, **body)
    assert response.status_code == 202, response.json()
    job_id = response.json()["job_id"]
    with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=auth) as stream:
        return parse_sse(line for line in stream.iter_lines())


def refusal(response: Any) -> dict:
    assert response.status_code >= 400, response.text
    return response.json()["error"]


def worker_rows(transcript: Path, op: str) -> list[dict]:
    rows = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row.get("op") == op]


# --- the pick ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "allowance", "picked"),
    [
        (LAPTOP, 1 * GIB, SMALL),
        (FAKE_BACKEND, 3 * GIB, BIG),
        (FAKE_MAC_BACKEND, 16 * GIB, BIG),
    ],
)
def test_the_biggest_rung_that_fits_is_picked(backend: Backend, allowance: int, picked: str) -> None:
    decided = verdict.decide(
        BY_NAME["image"], backend.kind,
        total_bytes=backend.gpu.vram_bytes, desktop_allowance_bytes=allowance,
        gpu_vendor=backend.gpu.vendor, chosen=None, audio_low_vram=False,
    )
    assert decided.enabled is True
    assert decided.selected == picked


def test_the_mac_has_no_small_rung() -> None:
    assert not load_image_manifest(SMALL).supports("mlx-darwin")


# --- a request that names no model ------------------------------------------------------


def test_a_job_with_no_model_is_made_by_the_registered_rung_with_its_own_defaults(
    laptop: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    events = run_job(laptop, auth, params={"prompt": PROMPT, "seed": 3})
    assert events[-1]["event"] == "done", events[-1]
    image = events[-1]["data"]["image"]
    spec = load_image_manifest(SMALL).spec("cuda-linux")
    assert image["model"] == SMALL
    assert (image["width"], image["height"]) == (1024, 1024)
    assert (image["steps"], image["guidance"]) == (6, 2.0)
    assert image["negative_prompt"] == spec.default_negative_prompt
    assert image["negative_prompt_is_default"] is True
    assert (image["pipeline"], image["storage_dtype"], image["scheduler"]) == (
        "sdxl", "float8_e4m3fn", "dpmpp-sde-karras",
    )
    (load,) = worker_rows(transcript, "load")
    assert (load["engine"], load["pipeline"], load["variant"]) == ("diffusers", "sdxl", "fp16")
    assert load["vae_dir"].endswith(str(Path(SMALL) / "cuda-linux" / "vae"))
    assert load["memory_cap_bytes"] == spec.memory_bytes_estimate


def test_a_negative_prompt_sent_replaces_the_default(
    laptop: TestClient, auth: dict[str, str]
) -> None:
    events = run_job(laptop, auth, params={"prompt": PROMPT, "negative_prompt": "people"})
    image = events[-1]["data"]["image"]
    assert (image["negative_prompt"], image["negative_prompt_is_default"]) == ("people", False)


def test_guidance_one_reads_no_negative_prompt(laptop: TestClient, auth: dict[str, str]) -> None:
    events = run_job(laptop, auth, params={"prompt": PROMPT, "guidance": 1.0})
    image = events[-1]["data"]["image"]
    assert (image["guidance"], image["negative_prompt"]) == (1.0, None)
    error = refusal(submit(laptop, auth, params={"prompt": PROMPT, "guidance": 1.0, "negative_prompt": "x"}))
    assert error["code"] == "invalid_params"
    assert "only read when guidance is above 1.0" in error["message"]


def test_one_side_alone_is_refused(laptop: TestClient, auth: dict[str, str]) -> None:
    error = refusal(submit(laptop, auth, params={"prompt": PROMPT, "width": 768}))
    assert error["code"] == "invalid_params"
    assert "width and height together" in error["message"]


def test_no_model_without_a_record_is_refused_by_name(
    make_client: Callable[..., TestClient],
    home: Path,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    laptop_card: None,
    transcript: Path,
) -> None:
    _env(home, LAPTOP.kind, monkeypatch)
    _small_weights(home)
    with make_client(enable_image=True, backend=LAPTOP) as client:
        error = refusal(submit(client, auth))
    assert error["code"] == "capability_undecided"
    assert "crucible capability --write" in error["message"]


def test_naming_the_big_rung_on_the_laptop_is_refused_before_anything_loads(
    laptop: TestClient, auth: dict[str, str], transcript: Path
) -> None:
    error = refusal(submit(laptop, auth, model=BIG, params={"prompt": PROMPT, "width": 512, "height": 512}))
    assert error["code"] == "insufficient_memory"
    assert not transcript.exists() or worker_rows(transcript, "load") == []


# --- what the rung does not do ----------------------------------------------------------


def test_image_to_image_and_inpainting_are_refused_naming_the_model_that_does(
    laptop: TestClient, auth: dict[str, str]
) -> None:
    picture = base64.b64encode(PNG_HEAD + b"rest").decode("ascii")
    error = refusal(submit(
        laptop, auth, params={"prompt": PROMPT, "image_strength": 0.2},
        inputs={"start.png": {"inline_base64": picture}},
    ))
    assert error["code"] == "image_to_image_unsupported"
    assert error["details"]["models"] == [BIG]
    error = refusal(submit(
        laptop, auth, params={"prompt": PROMPT, "mask": "mask.png"},
        inputs={"start.png": {"inline_base64": picture}, "mask.png": {"inline_base64": picture}},
    ))
    assert error["code"] == "inpaint_unsupported"
    assert error["details"]["models"] == [BIG]


def test_the_mac_refuses_the_small_rung_by_name(
    make_client: Callable[..., TestClient],
    home: Path,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    transcript: Path,
) -> None:
    monkeypatch.setattr(accelerator, "probe_unified_memory", lambda: (40 * GIB, 64 * GIB))
    _env(home, FAKE_MAC_BACKEND.kind, monkeypatch)
    with make_client(enable_image=True, backend=FAKE_MAC_BACKEND, desktop_allowance_bytes=16 * GIB) as client:
        error = refusal(submit(client, auth, model=SMALL))
    assert error["code"] in ("unknown_model", "backend_unsupported")
    assert SMALL in error["message"]


# --- weights ----------------------------------------------------------------------------


def test_the_rung_is_installed_only_with_its_vae(home: Path) -> None:
    from crucible.config import load_config
    from .conftest import configure_box

    configure_box(home, backend=LAPTOP, enable_image=True)
    config = load_config(home)
    manifest = load_image_manifest(SMALL)
    spec = manifest.spec("cuda-linux")
    _small_weights(home, with_vae=False)
    assert imageweights.installed(config, manifest, spec) is None
    _small_weights(home)
    found = imageweights.installed(config, manifest, spec)
    assert found is not None and found.bytes == 6_771_000_000 + spec.vae.bytes


def test_missing_weights_pull_only_the_small_rung(
    make_client: Callable[..., TestClient],
    home: Path,
    auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    laptop_card: None,
) -> None:
    _env(home, LAPTOP.kind, monkeypatch)
    with make_client(enable_image=True, backend=LAPTOP, capability=_record(LAPTOP, 1 * GIB)) as client:
        error = refusal(submit(client, auth))
    assert error["code"] == "installing"
    assert f"pulling the model '{SMALL}'" in error["message"]
    assert BIG not in error["message"]


# --- what a client reads ----------------------------------------------------------------


def test_the_playground_publishes_each_rungs_facts_and_puts_the_pick_first(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch, laptop_card: None,
) -> None:
    _env(home, LAPTOP.kind, monkeypatch)
    _small_weights(home)
    with make_client(enable_image=True, backend=LAPTOP, capability=_record(LAPTOP, 1 * GIB)) as client:
        pages = client.get("/v1/playground", headers=auth).json()["pages"]
    images = [page for page in pages if page["job_type"] == "image"]
    assert [page["id"] for page in images] == [SMALL, BIG]
    small, big = images
    assert (small["registered"], big["registered"]) == (True, False)
    assert small["standing"] == "ready"
    assert small["facts"]["native_width"] == 1024 and small["facts"]["quality_min_side"] == 768
    assert small["facts"]["negative_prompt"]["costs_extra_pass"] is False
    assert big["facts"]["negative_prompt"]["costs_extra_pass"] is True
    fields = {field["name"]: field for field in small["fields"]}
    assert (fields["width"]["default"], fields["steps"]["default"]) == (1024, 6)
    assert all(page["facts"] is None for page in pages if page["job_type"] != "image")


def test_on_a_big_card_with_both_installed_the_big_rung_is_first(
    make_client: Callable[..., TestClient], home: Path, auth: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B-Sides takes the first ready image page: it must be the server's own pick."""
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: [])
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (22 * GIB, 24 * GIB))
    _env(home, FAKE_BACKEND.kind, monkeypatch)
    _small_weights(home)
    big = load_image_manifest(BIG).spec("cuda-linux")
    _stamp(home / "models" / BIG / "cuda-linux", {
        "hf_repo": big.hf_repo, "revision": big.revision, "bytes": 33_134_949_212,
        "seconds": 60.0, "pulled": "2026-09-28T02:00:00+0000",
    })
    with make_client(enable_image=True, capability=_record(FAKE_BACKEND, 3 * GIB)) as client:
        pages = client.get("/v1/playground", headers=auth).json()["pages"]
    images = [(page["id"], page["standing"], page["registered"]) for page in pages if page["job_type"] == "image"]
    assert images == [(BIG, "ready", True), (SMALL, "ready", False)]
