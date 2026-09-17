"""The floor for translation is the 9B, and a refusal names the door that is open.

docs/MODEL-CHOICE.md. Owen, 2026-09-16, reversing his own ruling of 2026-09-13:

    "the user can pick a model to use for translate/simplify/etc, but they cant
    pick smaller than 9b. this is a reversal from what it was previously -
    before, it was never smaller than 27b. i think 9b could do an ok job at
    translation."

and, for the second half:

    "if nothing fits their card, it should give them the option of using api keys
    for claude or openai"
"""

from __future__ import annotations

from crucible.capability import (
    CLASSES,
    ROUTABLE_CLASSES,
    UPSTREAM_OFFER,
    decide,
    decide_all,
)

THREE_NINETY_TI = 25_757_220_864
#: A card that holds nothing in this catalog. The interesting host: under the old
#: ruling it could clean and not translate, and under this one it can do neither
#: locally and must be told about the upstream instead of being left at a
#: full stop.
TWELVE_GIG = 12 * 1024 ** 3
DESKTOP = 3 * 1024 ** 3


def a_class(name: str):
    return next(entry for entry in CLASSES if entry.name == name)


def test_translate_reaches_down_to_the_nine_b() -> None:
    """The reversal itself, as a set rather than as prose.

    `translate` read `qwen3.8` alone, which is what made the 27B a floor. It now
    reads both families, so the smallest thing it can be asked to run is the
    smallest 9B in the catalog.
    """
    offered = {c.id for c in a_class("translate").candidates("cuda-linux")}
    assert "qwen3.5-9b" in offered
    assert "qwen3.8-27b-4bit" in offered


def test_simplify_and_analysis_moved_with_it() -> None:
    """Three acts, one ruling — so a reversal that reached only one of them
    would leave a host able to translate and unable to simplify, which is the
    kind of split nothing in the ruling asks for."""
    translate = {c.id for c in a_class("translate").candidates("cuda-linux")}
    for name in ("simplify", "analysis"):
        assert {c.id for c in a_class(name).candidates("cuda-linux")} == translate


def test_clean_did_not_move() -> None:
    """`clean` was always 9B-class and the ruling does not touch it.

    Named because the obvious mistake when widening three classes is to widen
    the fourth for symmetry, which would put a 27B in front of cleanup — an act
    Owen runs on every book and has pinned to the 9B since before Crucible.
    """
    assert {c.id for c in a_class("clean").candidates("cuda-linux")} == {"qwen3.5-9b"}


def test_the_best_model_is_still_chosen_by_default_on_a_card_that_holds_it() -> None:
    """Widening the floor must not lower the ceiling.

    The walk is still best-first by declared size, so Owen's own card still
    selects the 27B for translation. His 9B preference is a CHOICE he makes in
    settings, not a default this table makes for him — no arithmetic here can
    rank a 9B at bf16 against a 27B at 4-bit, because the ranking depends on
    what the person is doing.
    """
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
    """The point of the reversal: the choice is honoured, not overruled.

    A chosen model that fits is never quietly replaced by the one the walk would
    have picked — that would make the settings document a suggestion.
    """
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
    """"the smallest of 3 qwen3.8 variants is qwen3.5-9b" counts one set and
    names another, which is how a reader concludes the server is confused."""
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


# ------------------------------------------------------------- the open door


def test_a_refused_routable_class_names_the_upstream() -> None:
    """A full stop is the wrong last word when another door is open."""
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
    """Deliberately not routable: sending page IMAGES to Anthropic is a
    different feature with a different body that nobody has asked for.

    This is the test that makes `routable` worth being a declared field rather
    than `job_type == "llm"` — `pages` IS an llm job type, and a derivation would
    have offered to send Owen's scanned pages to a third party.
    """
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
    """The offer belongs to a refusal. A host that can do the work locally being
    told to go and buy an API key is noise on every settings page that works."""
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
    """It says "add an API key", because at refusal time the server does not
    know whether one exists — and a sentence that said "this will be routed"
    would be a promise made on a table it did not read."""
    assert "add an API key" in UPSTREAM_OFFER
