"""Decide with images, and the API-key recommendation below 4B (docs/VERB-SIZING.md
section 8, Owen 2026-10-09): a decision that carries images and names no model is served
by the vision form of decide's registered model when it fits, else the largest model that
reads images and fits at or below the 9B goal, else refused by name; a small automatic
pick of a routable text verb recommends an API key, and changes nothing else."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from crucible import accelerator, capabilitystore, installplan
from crucible.backend import CUDA_LINUX, MLX_DARWIN, Backend, Gpu
from crucible.capabilityclasses import API_KEY_ADVICE_BELOW_PARAMS_B, BY_NAME, CLASSES
from crucible.capabilityrecord import CapabilityRecord
from crucible.capabilitywords import UPSTREAM_SETTINGS_BLOCK
from crucible.config import (
    ConfigError,
    _capability_record,
    default_desktop_allowance_bytes,
    load_config,
)
from crucible.fit import Candidate
from crucible.memorybudget import GIB
from crucible.verdict import Decision, decide, decide_all

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND, configure_box
from .fake_engine import FakeEngine
from .live_server import run_job, serve
from .test_decide_api import EXAMPLE, PNG, example_probs

PC = (CUDA_LINUX, FAKE_BACKEND.gpu.vram_bytes, 3 * GIB, "nvidia")
STUDIO = (
    MLX_DARWIN,
    FAKE_MAC_BACKEND.gpu.vram_bytes,
    default_desktop_allowance_bytes(MLX_DARWIN, FAKE_MAC_BACKEND.gpu.vram_bytes, "apple"),
    "apple",
)
MAC_16 = (MLX_DARWIN, 16 * GIB, default_desktop_allowance_bytes(MLX_DARWIN, 16 * GIB, "apple"), "apple")
CUDA_8 = (CUDA_LINUX, 8 * GIB, default_desktop_allowance_bytes(CUDA_LINUX, 8 * GIB, "nvidia"), "nvidia")
CUDA_12 = (CUDA_LINUX, 12 * GIB, default_desktop_allowance_bytes(CUDA_LINUX, 12 * GIB, "nvidia"), "nvidia")
CUDA_16 = (CUDA_LINUX, 16 * GIB, default_desktop_allowance_bytes(CUDA_LINUX, 16 * GIB, "nvidia"), "nvidia")

ROOMY_BACKEND = Backend(
    kind=CUDA_LINUX,
    platform="linux",
    arch="x86_64",
    gpu=Gpu(vendor="nvidia", name="a 32 GiB test card", vram_bytes=32 * GIB),
    detail="test double",
)

WITH_IMAGES = {**{k: v for k, v in EXAMPLE.items() if k != "model"}, "images": [PNG]}
NO_MODEL = {k: v for k, v in EXAMPLE.items() if k != "model"}


def on(card: tuple[str, int, int, str], name: str, chosen: str | None = None) -> Decision:
    kind, total, allowance, vendor = card
    return decide(
        BY_NAME[name], kind, total_bytes=total, desktop_allowance_bytes=allowance,
        gpu_vendor=vendor, chosen=chosen, audio_low_vram=False,
    )


def record_for(card: tuple[str, int, int, str]) -> CapabilityRecord:
    kind, total, allowance, vendor = card
    decisions = decide_all(
        kind, total_bytes=total, desktop_allowance_bytes=allowance, gpu_vendor=vendor,
        chosen={}, audio_low_vram=False,
    )
    return capabilitystore.record_of(
        kind, total_bytes=total, desktop_allowance_bytes=allowance,
        decisions=decisions, routes={},
    )


def with_decide_row(record: CapabilityRecord, **changes: Any) -> CapabilityRecord:
    rows = tuple(
        replace(row, **changes) if row.capability == "decide" else row
        for row in record.rows
    )
    return replace(record, rows=rows)


# --- the pick for a decision with images ------------------------------------------------


def test_the_pc_decides_images_on_the_4b_because_the_9b_vision_form_does_not_fit() -> None:
    verdict = on(PC, "decide")
    assert verdict.selected == "qwen3.5-9b"
    assert verdict.with_images is not None
    assert verdict.with_images.selected == "qwen3.5-4b"
    assert "qwen3.5-9b-vl, the vision form of qwen3.5-9b, needs" in verdict.with_images.reason
    assert "the largest model that reads images and fits at or below the 9B goal" in (
        verdict.with_images.reason
    )
    assert verdict.summary.endswith("; with images, qwen3.5-4b")
    assert "With images: qwen3.5-4b" in verdict.reason


def test_the_studio_decides_images_on_the_vision_form_of_the_same_weights() -> None:
    verdict = on(STUDIO, "decide")
    assert verdict.selected == "qwen3.5-9b"
    assert verdict.with_images is not None
    assert verdict.with_images.selected == "qwen3.5-9b-vl"
    assert "the vision form of qwen3.5-9b (the same weights)" in verdict.with_images.reason
    assert verdict.summary.endswith("; with images, qwen3.5-9b-vl")


def test_a_roomy_cuda_card_decides_images_on_the_9b_vision_form() -> None:
    verdict = on((CUDA_LINUX, 32 * GIB, 3 * GIB, "nvidia"), "decide")
    assert (verdict.selected, verdict.with_images.selected) == ("qwen3.5-9b", "qwen3.5-9b-vl")


def test_a_model_that_reads_images_itself_is_its_own_image_pick() -> None:
    verdict = on(CUDA_8, "decide")
    assert verdict.selected == "qwen3.5-0.8b"
    assert verdict.with_images.selected == "qwen3.5-0.8b"
    assert "qwen3.5-0.8b itself, which reads images on cuda-linux" in verdict.with_images.reason


def test_nothing_that_reads_images_fits_is_said_with_what_would() -> None:
    verdict = on(MAC_16, "decide")
    assert verdict.enabled and verdict.selected == "qwen3.5-4b"
    pick = verdict.with_images
    assert pick is not None and pick.selected == ""
    assert "qwen3.5-4b has no form that reads images on mlx-darwin" in pick.reason
    assert "the smallest model that reads images, qwen3.5-9b-vl, needs" in pick.reason
    assert "more than there is" in pick.reason
    assert verdict.summary.endswith("; with images, nothing fits this card")


def test_a_settings_choice_is_served_with_images_by_its_own_form() -> None:
    verdict = on(PC, "decide", chosen="qwen3.5-2b")
    assert verdict.chosen and verdict.selected == "qwen3.5-2b"
    assert verdict.with_images.selected == "qwen3.5-2b"


def test_only_decide_records_an_image_pick() -> None:
    for decision in decide_all(
        CUDA_LINUX, total_bytes=PC[1], desktop_allowance_bytes=PC[2], gpu_vendor="nvidia",
        chosen={}, audio_low_vram=False,
    ):
        if decision.capability == "decide":
            assert decision.with_images is not None
        else:
            assert decision.with_images is None, decision.capability
    assert [entry.name for entry in CLASSES if entry.takes_images] == ["decide"]


def test_a_refused_decide_records_no_image_pick() -> None:
    verdict = on((CUDA_LINUX, 4 * GIB, GIB, "nvidia"), "decide")
    assert verdict.enabled is False and verdict.with_images is None


# --- the record shows both ------------------------------------------------------------


def test_the_record_keeps_both_forms_through_config_toml(home: Path) -> None:
    record = record_for(PC)
    configure_box(home, enable_llm=True, capability=record)
    row = load_config(home).capability.row("decide")
    assert (row.selected, row.with_images) == ("qwen3.5-9b", "qwen3.5-4b")
    assert row.with_images_reason == on(PC, "decide").with_images.reason
    assert load_config(home).capability.row("translate").with_images is None
    assert "with_images" not in record.row("translate").to_dict()


def test_a_row_with_half_the_image_pick_is_refused() -> None:
    row = {
        "capability": "decide", "enabled": True, "selected": "qwen3.5-9b",
        "reason": "r", "summary": "s", "shortfall_bytes": 0, "with_images": "qwen3.5-4b",
    }
    table = {"capability": {
        "backend_kind": CUDA_LINUX, "total_bytes": 1, "desktop_allowance_bytes": 1,
        "classes": [row],
    }}
    with pytest.raises(ConfigError, match="written together"):
        _capability_record(table)


def test_v1_capability_shows_decide_with_and_without_images(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    with make_client(enable_llm=True, capability=record_for(PC)) as client:
        rows = {
            row["capability"]: row
            for row in client.get("/v1/capability", headers=auth).json()["classes"]
        }
    assert rows["decide"]["selected"] == "qwen3.5-9b"
    assert rows["decide"]["with_images"] == "qwen3.5-4b"
    assert "qwen3.5-9b-vl, the vision form of qwen3.5-9b" in rows["decide"]["with_images_reason"]
    assert "; with images, qwen3.5-4b" in rows["decide"]["summary"]
    assert rows["translate"]["with_images"] is None
    assert rows["translate"]["with_images_reason"] is None


def test_the_install_plan_names_the_image_model() -> None:
    kind, total, allowance, vendor = PC
    decisions = decide_all(
        kind, total_bytes=total, desktop_allowance_bytes=allowance, gpu_vendor=vendor,
        chosen={}, audio_low_vram=False,
    )
    plan = installplan.install_plan("llm", decisions, card=None, total_bytes=total, pool="card")
    row = next(r for r in plan["classes"] if r["capability"] == "decide")
    assert row["with_images"] == "qwen3.5-4b"
    assert row["line"].endswith(" With images, qwen3.5-4b.")


# --- the door ---------------------------------------------------------------------------


@pytest.fixture
def decide_server(
    make_app: Callable[..., Any],
    fake_env: Path,
    fake_weights: Callable[[str], Path],
    idle_card: None,
    engine_factory: Callable[..., list[FakeEngine]],
):
    def start(record: CapabilityRecord, *weights: str, backend: Backend = FAKE_BACKEND):
        engines = engine_factory(probs_for=example_probs)
        for model_id in weights:
            fake_weights(model_id)
        return engines, serve(make_app(enable_llm=True, capability=record, backend=backend))

    return start


def _decide(base: str, auth: dict[str, str], body: dict[str, Any]) -> httpx.Response:
    return httpx.post(f"{base}/v1/decide", headers=auth, json=body, timeout=60.0)


def test_images_with_no_model_switch_to_the_image_pick_and_back(
    decide_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    engines, server = decide_server(record_for(PC), "qwen3.5-9b", "qwen3.5-4b")
    with server as base:
        run_job(base, auth, type="load-model", model="qwen3.5-9b")
        seen = _decide(base, auth, WITH_IMAGES)
        assert seen.status_code == 200, seen.text
        assert seen.json()["model"]["id"] == "qwen3.5-4b"
        assert seen.json()["tokens"]["images"] == 1
        read = _decide(base, auth, NO_MODEL)
        assert read.status_code == 200, read.text
        assert read.json()["model"]["id"] == "qwen3.5-9b"
    assert len(engines) == 3, "the 9B, the 4B for the images, the 9B again: one resident at a time"


def test_the_vision_form_is_served_from_the_texts_own_download(
    decide_server: Callable[..., Any], auth: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (30 * GIB, 32 * GIB))
    card = (CUDA_LINUX, ROOMY_BACKEND.gpu.vram_bytes, 3 * GIB, "nvidia")
    record = record_for(card)
    assert record.row("decide").with_images == "qwen3.5-9b-vl"
    _, server = decide_server(record, "qwen3.5-9b", backend=ROOMY_BACKEND)
    with server as base:
        refused = _decide(base, auth, {**WITH_IMAGES, "queue": False})
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "model_not_resident"
        assert "qwen3.5-9b-vl" in refused.json()["error"]["message"]
        answered = _decide(base, auth, WITH_IMAGES)
        assert answered.status_code == 200, answered.text
        assert answered.json()["model"]["id"] == "qwen3.5-9b-vl"


def test_a_named_text_only_model_is_still_refused_with_images(
    decide_server: Callable[..., Any], auth: dict[str, str]
) -> None:
    engines, server = decide_server(record_for(PC), "qwen3.5-9b")
    with server as base:
        run_job(base, auth, type="load-model", model="qwen3.5-9b")
        refused = _decide(base, auth, {**WITH_IMAGES, "model": "qwen3.5-9b"})
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "model_text_only"
    assert engines[-1].requests == []


def test_no_image_model_that_fits_is_refused_by_name(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    reason = on(MAC_16, "decide").with_images.reason
    record = with_decide_row(record_for(PC), with_images="", with_images_reason=reason)
    with make_client(enable_llm=True, capability=record) as client:
        refused = client.post("/v1/decide", headers=auth, json=WITH_IMAGES)
    assert refused.status_code == 409
    error = refused.json()["error"]
    assert error["code"] == "no_image_model_fits"
    assert reason in error["message"]
    assert error["details"]["text_model"] == "qwen3.5-9b"


def test_a_record_from_before_the_image_pick_names_the_command(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    record = with_decide_row(record_for(PC), with_images=None, with_images_reason=None)
    with make_client(enable_llm=True, capability=record) as client:
        refused = client.post("/v1/decide", headers=auth, json=WITH_IMAGES)
    assert refused.status_code == 503
    assert refused.json()["error"]["code"] == "capability_undecided"
    assert "crucible capability --write" in refused.json()["error"]["message"]


def test_a_card_that_cannot_decide_refuses_a_decision_with_no_model(
    make_client: Callable[..., TestClient], auth: dict[str, str]
) -> None:
    record = with_decide_row(
        record_for(PC), enabled=False, selected="", reason="disabled: too small",
        with_images=None, with_images_reason=None,
    )
    with make_client(enable_llm=True, capability=record) as client:
        refused = client.post("/v1/decide", headers=auth, json=NO_MODEL)
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "capability_disabled"
    assert "disabled: too small" in refused.json()["error"]["message"]


# --- an API key below 4B ----------------------------------------------------------------


def _advised(decision: Decision) -> bool:
    in_reason = UPSTREAM_SETTINGS_BLOCK in decision.reason
    in_summary = UPSTREAM_SETTINGS_BLOCK in decision.summary
    assert in_reason == in_summary, "the reason and the summary say it together"
    return in_reason


def test_the_threshold_is_4b() -> None:
    assert API_KEY_ADVICE_BELOW_PARAMS_B == 4
    translate, decide_class = BY_NAME["translate"], BY_NAME["decide"]
    small = Candidate(id="x", memory_bytes_estimate=1, params_b=3.9, bits=16)
    assert translate.advises_api_key(small)
    assert not translate.advises_api_key(replace(small, params_b=4))
    assert not decide_class.advises_api_key(small), "decide reads logprobs no upstream returns"


@pytest.mark.parametrize("name", ["clean", "translate", "simplify", "analysis", "generate"])
def test_a_pick_under_4b_recommends_an_api_key_and_names_where(name: str) -> None:
    verdict = on(CUDA_12, name)
    assert verdict.selected == "qwen3.5-2b"
    assert _advised(verdict)
    assert (
        f"Models under 4B give weaker results, so for better ones add an API key for "
        f"Anthropic or OpenAI in Settings, under \"{UPSTREAM_SETTINGS_BLOCK}\", and send "
        f"{name} to it in the same page (or set [upstreams] and [routes] in config.toml)."
    ) in verdict.reason


@pytest.mark.parametrize("card,model", [(CUDA_16, "qwen3.5-4b"), (PC, "qwen3.8-27b-4bit")])
def test_a_pick_at_4b_or_above_recommends_nothing(card: tuple, model: str) -> None:
    verdict = on(card, "translate")
    assert verdict.selected == model
    assert not _advised(verdict)


def test_decide_is_never_advised_an_api_key() -> None:
    verdict = on(CUDA_8, "decide")
    assert verdict.selected == "qwen3.5-0.8b"
    assert not _advised(verdict)


def test_a_settings_choice_under_4b_is_not_advised() -> None:
    verdict = on(PC, "translate", chosen="qwen3.5-2b")
    assert verdict.chosen and verdict.selected == "qwen3.5-2b"
    assert not _advised(verdict)


def test_the_advice_changes_neither_the_pick_nor_whether_it_runs() -> None:
    verdict = on(CUDA_8, "translate")
    assert verdict.enabled and verdict.selected == "qwen3.5-0.8b"
    assert verdict.shortfall_bytes == 0


def test_the_install_plan_line_carries_the_advice() -> None:
    kind, total, allowance, vendor = CUDA_12
    decisions = decide_all(
        kind, total_bytes=total, desktop_allowance_bytes=allowance, gpu_vendor=vendor,
        chosen={}, audio_low_vram=False,
    )
    plan = installplan.install_plan("llm", decisions, card=None, total_bytes=total, pool="card")
    lines = {row["capability"]: row["line"] for row in plan["classes"]}
    assert UPSTREAM_SETTINGS_BLOCK in lines["translate"]
    assert UPSTREAM_SETTINGS_BLOCK not in lines["decide"]


def test_the_doctor_line_is_the_recorded_reason_with_the_advice() -> None:
    from crucible.cli.doctor import lines_capability

    record = record_for(CUDA_12)
    report = {
        "capability": {**record.to_dict(), "could_enable": []},
        "audio_low_vram": None,
    }
    lines = [line for line in lines_capability(report) if line.startswith("capability translate")]
    assert len(lines) == 1 and UPSTREAM_SETTINGS_BLOCK in lines[0]
