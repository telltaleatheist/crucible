from __future__ import annotations

from crucible.capability import (
    CLASSES,
    ROUTABLE_CLASSES,
    UPSTREAM_OFFER,
    decide,
    decide_all,
)

THREE_NINETY_TI = 25_757_220_864
TWELVE_GIG = 12 * 1024 ** 3
DESKTOP = 3 * 1024 ** 3


def a_class(name: str):
    return next(entry for entry in CLASSES if entry.name == name)


def test_translate_reaches_down_to_the_nine_b() -> None:
    offered = {c.id for c in a_class("translate").candidates("cuda-linux")}
    assert "qwen3.5-9b" in offered
    assert "qwen3.8-27b-4bit" in offered


def test_simplify_and_analysis_moved_with_it() -> None:
    translate = {c.id for c in a_class("translate").candidates("cuda-linux")}
    for name in ("simplify", "analysis"):
        assert {c.id for c in a_class(name).candidates("cuda-linux")} == translate


def test_clean_did_not_move() -> None:
    assert {c.id for c in a_class("clean").candidates("cuda-linux")} == {"qwen3.5-9b"}


def test_the_best_model_is_still_chosen_by_default_on_a_card_that_holds_it() -> None:
    decisions = {
        d.capability: d
        for d in decide_all(
            "cuda-linux",
            total_bytes=THREE_NINETY_TI,
            desktop_allowance_bytes=DESKTOP,
            gpu_vendor="nvidia",
            chosen={},
        )
    }
    assert decisions["translate"].selected == "qwen3.8-27b-4bit"
    assert decisions["clean"].selected == "qwen3.5-9b"


def test_a_nine_b_can_be_chosen_for_translation() -> None:
    decision = decide(
        a_class("translate"),
        "cuda-linux",
        total_bytes=THREE_NINETY_TI,
        desktop_allowance_bytes=DESKTOP,
        gpu_vendor="nvidia",
        chosen="qwen3.5-9b",
    )
    assert decision.enabled
    assert decision.selected == "qwen3.5-9b"


def test_the_noun_names_the_set_it_counts() -> None:
    decision = decide(
        a_class("translate"),
        "cuda-linux",
        total_bytes=TWELVE_GIG,
        desktop_allowance_bytes=DESKTOP,
        gpu_vendor="nvidia",
        chosen=None,
    )
    assert "qwen3.8 and qwen3.5 variants" in decision.reason
    assert "qwen3.5-9b" in decision.reason


def test_a_refused_routable_class_names_the_upstream() -> None:
    for name in ("clean", "translate", "simplify", "analysis"):
        decision = decide(
            a_class(name),
            "cuda-linux",
            total_bytes=TWELVE_GIG,
            desktop_allowance_bytes=DESKTOP,
            gpu_vendor="nvidia",
            chosen=None,
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
    )
    assert not decision.enabled
    assert UPSTREAM_OFFER.strip() not in decision.reason
    assert "pages" not in ROUTABLE_CLASSES


def test_an_enabled_class_is_not_offered_an_upstream() -> None:
    decision = decide(
        a_class("translate"),
        "cuda-linux",
        total_bytes=THREE_NINETY_TI,
        desktop_allowance_bytes=DESKTOP,
        gpu_vendor="nvidia",
        chosen=None,
    )
    assert decision.enabled
    assert UPSTREAM_OFFER.strip() not in decision.reason


def test_the_offer_never_claims_a_key_is_already_there() -> None:
    assert "add an API key" in UPSTREAM_OFFER
