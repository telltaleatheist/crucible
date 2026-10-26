from __future__ import annotations

from crucible.capabilityclasses import CLASSES
from crucible.capabilitywords import UPSTREAM_OFFER
from crucible.classnames import ROUTABLE_CLASSES
from crucible.verdict import decide, decide_all

THREE_NINETY_TI = 25_757_220_864
TWELVE_GIG = 12 * 1024 ** 3
FOUR_GIG = 4 * 1024 ** 3
DESKTOP = 3 * 1024 ** 3


def a_class(name: str):
    return next(entry for entry in CLASSES if entry.name == name)


def test_generate_reaches_down_to_the_nine_b() -> None:
    offered = {c.id for c in a_class("generate").candidates("cuda-linux")}
    assert "qwen3.5-9b" in offered
    assert "qwen3.8-27b-4bit" in offered


def test_the_best_model_is_still_chosen_by_default_on_a_card_that_holds_it() -> None:
    decisions = {
        d.capability: d
        for d in decide_all(
            "cuda-linux",
            total_bytes=THREE_NINETY_TI,
            desktop_allowance_bytes=DESKTOP,
            gpu_vendor="nvidia",
            chosen={},
            audio_low_vram=False,
        )
    }
    assert decisions["generate"].selected == "qwen3.8-27b-4bit"
    assert decisions["decide"].selected == "qwen3.5-9b"


def test_a_nine_b_can_be_chosen_for_chat() -> None:
    decision = decide(
        a_class("generate"),
        "cuda-linux",
        total_bytes=THREE_NINETY_TI,
        desktop_allowance_bytes=DESKTOP,
        gpu_vendor="nvidia",
        chosen="qwen3.5-9b",
        audio_low_vram=False,
    )
    assert decision.enabled
    assert decision.selected == "qwen3.5-9b"


def test_the_noun_names_the_set_it_counts() -> None:
    decision = decide(
        a_class("generate"),
        "cuda-linux",
        total_bytes=TWELVE_GIG,
        desktop_allowance_bytes=DESKTOP,
        gpu_vendor="nvidia",
        chosen=None,
        audio_low_vram=False,
    )
    assert "qwen3.8 and qwen3.5 variants" in decision.reason
    assert decision.selected in decision.reason


def test_a_refused_routable_class_names_the_upstream() -> None:
    for name in ROUTABLE_CLASSES:
        decision = decide(
            a_class(name),
            "cuda-linux",
            total_bytes=FOUR_GIG,
            desktop_allowance_bytes=DESKTOP,
            gpu_vendor="nvidia",
            chosen=None,
            audio_low_vram=False,
        )
        assert not decision.enabled, name
        assert UPSTREAM_OFFER.strip() in decision.reason, name


def test_pages_is_refused_without_the_offer() -> None:
    decision = decide(
        a_class("pages"),
        "cuda-linux",
        total_bytes=TWELVE_GIG,
        desktop_allowance_bytes=DESKTOP,
        gpu_vendor="nvidia",
        chosen=None,
        audio_low_vram=False,
    )
    assert not decision.enabled
    assert UPSTREAM_OFFER.strip() not in decision.reason
    assert "pages" not in ROUTABLE_CLASSES


def test_an_enabled_class_is_not_offered_an_upstream() -> None:
    decision = decide(
        a_class("generate"),
        "cuda-linux",
        total_bytes=THREE_NINETY_TI,
        desktop_allowance_bytes=DESKTOP,
        gpu_vendor="nvidia",
        chosen=None,
        audio_low_vram=False,
    )
    assert decision.enabled
    assert UPSTREAM_OFFER.strip() not in decision.reason


def test_the_offer_never_claims_a_key_is_already_there() -> None:
    assert "add an API key" in UPSTREAM_OFFER
