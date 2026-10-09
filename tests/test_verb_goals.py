"""Every verb sized to the card, phase 1 (docs/VERB-SIZING.md): a goal per verb, the
automatic pick under it (most parameters, then most bits), every text verb reaching the
smallest model, and a person's choice still winning."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from crucible import capabilitystore, installplan
from crucible.backend import CUDA_LINUX, LLAMA_WINDOWS, MLX_DARWIN, Backend, Gpu
from crucible.capabilityclasses import BY_NAME, CLASSES
from crucible.cli import doctor
from crucible.config import default_desktop_allowance_bytes, load_config
from crucible.fit import Candidate
from crucible.manifests import ManifestError, parse_manifest
from crucible.memorybudget import GIB
from crucible.verdict import Decision, decide

from .conftest import FAKE_MAC_BACKEND, configure_box

TEXT_VERBS = ("decide", "clean", "translate", "simplify", "analysis", "generate")

PC = (CUDA_LINUX, 24 * GIB, 3 * GIB, "nvidia")
STUDIO = (
    MLX_DARWIN,
    FAKE_MAC_BACKEND.gpu.vram_bytes,
    default_desktop_allowance_bytes(MLX_DARWIN, FAKE_MAC_BACKEND.gpu.vram_bytes, "apple"),
    "apple",
)
EIGHT_UNMEASURED = (CUDA_LINUX, 8 * GIB, default_desktop_allowance_bytes(CUDA_LINUX, 8 * GIB, "nvidia"), "nvidia")
EIGHT_FLAT_3 = (CUDA_LINUX, 8 * GIB, 3 * GIB, "nvidia")


def on(card: tuple[str, int, int, str], name: str, chosen: str | None = None) -> Decision:
    kind, total, allowance, vendor = card
    return decide(
        BY_NAME[name], kind, total_bytes=total, desktop_allowance_bytes=allowance,
        gpu_vendor=vendor, chosen=chosen, audio_low_vram=False,
    )


def fitting_ids(decision: Decision, name: str) -> list[str]:
    work = BY_NAME[name].work
    return [c.id for c in decision.candidates if c.holds(work, decision.available_bytes)]


# --- the goal caps the pick -----------------------------------------------------------


def test_decide_on_the_24_gib_pc_takes_the_9b_though_the_27b_fits() -> None:
    verdict = on(PC, "decide")
    assert verdict.selected == "qwen3.5-9b"
    assert "qwen3.8-27b-4bit" in fitting_ids(verdict, "decide"), (
        "the 27B fits this card, so it is the goal, not the fit, that passes it over"
    )
    assert verdict.summary.startswith("can decide, using qwen3.5-9b (goal 9B; bf16 fits with ")
    assert "to spare)" in verdict.summary
    assert "qwen3.8-27b-4bit also fits, and is above the 9B goal" in verdict.reason


def test_decide_on_the_mac_studio_takes_the_9b_not_the_8bit_27b() -> None:
    verdict = on(STUDIO, "decide")
    assert verdict.selected == "qwen3.5-9b"
    assert {"qwen3.8-27b-8bit", "qwen3.8-27b-4bit", "qwen3.5-9b-vl"} <= set(
        fitting_ids(verdict, "decide")
    )
    assert "(goal 9B; 16-bit fits with " in verdict.summary


@pytest.mark.parametrize("card", [PC, STUDIO], ids=["pc", "studio"])
def test_clean_never_goes_above_its_9b_goal(card: tuple[str, int, int, str]) -> None:
    assert on(card, "clean").selected == "qwen3.5-9b"


# --- params first, then bits ------------------------------------------------------------


def candidate(model_id: str, params_b: float, bits: int, need: int, alias: bool = False) -> Candidate:
    return Candidate(id=model_id, memory_bytes_estimate=need, params_b=params_b, bits=bits, alias=alias)


def test_more_parameters_beat_more_bits_and_bits_settle_a_size() -> None:
    entry = BY_NAME["translate"]
    found = (
        candidate("4b-16", 4, 16, 9 * GIB),
        candidate("9b-4", 9, 4, 6 * GIB),
        candidate("27b-4", 27, 4, 18 * GIB),
        candidate("27b-8", 27, 8, 30 * GIB),
        candidate("70b-4", 70, 4, 40 * GIB),
    )
    order = [c.id for c in entry.pick_order(found)]
    assert order == ["27b-8", "27b-4", "9b-4", "4b-16"], (
        "above the goal is out; a 9B at 4-bit beats a 4B at 16-bit; the 27B at 8-bit "
        "beats the 27B at 4-bit"
    )


def test_a_models_own_form_is_taken_before_its_alias() -> None:
    entry = BY_NAME["decide"]
    found = (candidate("9b-vl", 9, 16, 22 * GIB, alias=True), candidate("9b", 9, 16, 20 * GIB))
    assert [c.id for c in entry.pick_order(found)] == ["9b", "9b-vl"]


def test_the_pc_translates_on_the_4bit_27b_over_the_bf16_9b_that_also_fits() -> None:
    verdict = on(PC, "translate")
    assert {"qwen3.8-27b-4bit", "qwen3.5-9b"} <= set(fitting_ids(verdict, "translate"))
    assert verdict.selected == "qwen3.8-27b-4bit"
    assert "(goal 27B; 4-bit fits with " in verdict.summary


def test_the_studio_translates_on_the_8bit_27b_over_the_4bit() -> None:
    verdict = on(STUDIO, "translate")
    assert {"qwen3.8-27b-8bit", "qwen3.8-27b-4bit"} <= set(fitting_ids(verdict, "translate"))
    assert verdict.selected == "qwen3.8-27b-8bit"


def test_a_goal_class_candidate_without_bits_is_refused_by_name() -> None:
    with pytest.raises(ValueError) as caught:
        BY_NAME["generate"].pick_order((Candidate(id="mystery", memory_bytes_estimate=GIB, params_b=9),))
    assert "mystery" in str(caught.value) and "bits" in str(caught.value)


def test_every_goal_class_candidate_states_params_and_bits_on_every_backend() -> None:
    for entry in CLASSES:
        if entry.goal is None:
            continue
        for kind in (CUDA_LINUX, MLX_DARWIN, LLAMA_WINDOWS):
            for c in entry.candidates(kind):
                assert c.params_b is not None and c.bits is not None, (entry.name, kind, c.id)
            assert entry.pick_order(entry.candidates(kind)), (entry.name, kind)


# --- the 4-bit floor ---------------------------------------------------------------------

MANIFEST = """
[model]
id = "demo"
family = "demo"
params_b = 1
context_default = 4096
trained_context = 262144
modalities = ["text"]

[backends.cuda-linux]
engine = "vllm"
hf_repo = "demo/{repo}"
revision = "0123456789abcdef0123456789abcdef01234567"
bits = {bits}
memory_bytes_estimate = 3000000000
"""


def test_a_stated_bits_under_four_is_refused() -> None:
    with pytest.raises(ManifestError) as caught:
        parse_manifest(MANIFEST.format(repo="Demo-1B", bits=3), Path("demo.toml"), "demo")
    assert "nothing under 4 bits" in str(caught.value)


def test_stated_bits_must_agree_with_the_repo() -> None:
    with pytest.raises(ManifestError) as caught:
        parse_manifest(MANIFEST.format(repo="Demo-1B-4bit", bits=8), Path("demo.toml"), "demo")
    assert "says 4-bit" in str(caught.value)
    spec = parse_manifest(MANIFEST.format(repo="Demo-1B", bits=8), Path("demo.toml"), "demo")
    assert spec.spec(CUDA_LINUX).bits == 8


# --- every verb, always ------------------------------------------------------------------


@pytest.mark.parametrize("card", [EIGHT_UNMEASURED, EIGHT_FLAT_3], ids=["scaled", "flat-3"])
@pytest.mark.parametrize("name", TEXT_VERBS)
def test_an_8_gib_card_runs_every_text_verb_on_a_smaller_model(
    card: tuple[str, int, int, str], name: str
) -> None:
    verdict = on(card, name)
    assert verdict.enabled is True
    assert verdict.selected == "qwen3.5-0.8b"
    assert verdict.summary.endswith("(goal " + BY_NAME[name].goal.words + "; the largest that fits this card)")


@pytest.mark.parametrize("name", TEXT_VERBS)
def test_a_card_too_small_for_the_0_8b_is_refused_by_name(name: str) -> None:
    verdict = on((CUDA_LINUX, 4 * GIB, GIB, "nvidia"), name)
    assert verdict.enabled is False
    assert "the smallest of" in verdict.reason and "is qwen3.5-0.8b" in verdict.reason
    assert verdict.shortfall_bytes > 0


def test_an_unmeasured_8_gib_card_keeps_one_gib_for_its_desktop() -> None:
    assert EIGHT_UNMEASURED[2] == GIB
    assert on(EIGHT_UNMEASURED, "decide").available_bytes == 7 * GIB


# --- a person's choice wins -------------------------------------------------------------


def test_a_chosen_model_above_the_goal_still_wins() -> None:
    verdict = on(PC, "decide", chosen="qwen3.8-27b-4bit")
    assert verdict.selected == "qwen3.8-27b-4bit"
    assert verdict.chosen is True
    assert "goal" not in verdict.summary
    assert on(STUDIO, "decide", chosen="qwen3.8-27b-8bit").selected == "qwen3.8-27b-8bit"


def test_a_chosen_model_below_the_goal_still_wins() -> None:
    verdict = on(PC, "translate", chosen="qwen3.5-4b")
    assert (verdict.selected, verdict.chosen) == ("qwen3.5-4b", True)


def test_the_store_passes_the_settings_choice_over_the_pick() -> None:
    kind, total, allowance, vendor = PC
    decisions = capabilitystore.decide_on(
        kind, total_bytes=total, desktop_allowance_bytes=allowance, gpu_vendor=vendor,
        card=None, chosen={"decide": "qwen3.8-27b-4bit"}, audio_low_vram=False,
    )
    by_name = {d.capability: d for d in decisions}
    assert by_name["decide"].selected == "qwen3.8-27b-4bit"
    assert by_name["clean"].selected == "qwen3.5-9b"


# --- the surfaces say it --------------------------------------------------------------------


def test_the_install_plan_names_the_goal_and_its_best_within_it() -> None:
    kind, total, allowance, vendor = EIGHT_UNMEASURED
    decisions = capabilitystore.decide_on(
        kind, total_bytes=total, desktop_allowance_bytes=allowance, gpu_vendor=vendor,
        card=None, chosen={}, audio_low_vram=False,
    )
    plan = installplan.install_plan(
        "llm", decisions, card=None, total_bytes=total, pool="card",
        desktop_allowance_bytes=allowance, desktop_basis="declared",
    )
    rows = {row["capability"]: row for row in plan["classes"]}
    assert rows["decide"]["best"] == "qwen3.5-9b"
    assert rows["decide"]["goal"]["params_b"] == 9
    assert rows["decide"]["line"].startswith(
        "Will decide with qwen3.5-0.8b (goal 9B; the largest that fits this card). "
        "The best, qwen3.5-9b, needs "
    )
    assert rows["translate"]["best"] == "qwen3.8-27b-4bit"
    assert rows["pages"]["goal"] is None


def test_doctor_says_a_record_from_before_the_goal_is_stale(home: Path) -> None:
    pc = Backend(
        kind=CUDA_LINUX, platform="linux", arch="x86_64",
        gpu=Gpu(vendor="nvidia", name="NVIDIA GeForce RTX 3090 Ti", vram_bytes=24 * GIB),
        detail="test double",
    )
    configure_box(home, enable_llm=True, backend=pc)  # the voices this host serves read it
    decisions = capabilitystore.decide_on(
        CUDA_LINUX, total_bytes=24 * GIB, desktop_allowance_bytes=3 * GIB,
        gpu_vendor="nvidia", card=None, chosen={}, audio_low_vram=False,
    )
    record = capabilitystore.record_of(
        CUDA_LINUX, total_bytes=24 * GIB, desktop_allowance_bytes=3 * GIB,
        decisions=decisions, routes={},
    )
    configure_box(home, enable_llm=True, backend=pc, capability=record)
    assert doctor._pick_findings(load_config(home), pc) == []

    old = replace(
        record,
        rows=tuple(
            replace(row, selected="qwen3.8-27b-4bit") if row.capability == "decide" else row
            for row in record.rows
        ),
    )
    configure_box(home, enable_llm=True, backend=pc, capability=old)
    (finding,) = doctor._pick_findings(load_config(home), pc)
    assert finding.code == "capability_stale"
    assert finding.message.startswith(
        "the record says decide: qwen3.8-27b-4bit, and this build decides qwen3.5-9b"
    )
