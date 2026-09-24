"""A class asks its OWN question, and the server publishes the ceiling it found.

docs/FITS-AND-THE-CARD.md sections 3 and 6. Two claims are held here.

**A class's working context belongs to the class.** Owen, 2026-09-16:
*"translate/simplify/etc dont actually need that much kv cache because it's
batched with small blocks. it isnt sending in the entire book to be translated,
its only sending it in one block (roughly a paragraph) at a time."* Before this,
every class was made to ask the model's `context_default` — so a 27B could be
refused on a card that had ample room for the work actually about to run.

**The ceiling is published rather than discovered as a 400.** Owen, the same
day: *"crucible should have the upper limit for each crucible server. if
bookforge tries to send an entire book through on one translate call, that would
work on the mac but not on the PC."* The measured counter-example is Ollama,
which answers 200 to `num_ctx: 1_000_000`, silently clamps to what the checkpoint
supports, silently truncates an over-long prompt, and never tells the caller
either thing.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from crucible.backend import Backend, Gpu
from crucible.capability import CLASSES, Candidate, WorkingContext, decide, decide_all
from crucible.manifests import MemoryTerms
from crucible.pages import PAGE_CONCURRENCY

#: Owen's card, and the one every measured number in the catalog came off.
THREE_NINETY_TI = 25_757_220_864
#: His Mac Studio: enough unified memory that the WEIGHTS, not the machine, are
#: what stop a long request. That asymmetry is the whole of section 6.2.
M1_ULTRA = 68_719_476_736
DESKTOP = 3 * 1024 ** 3


def a_candidate(**overrides: Any) -> Candidate:
    terms = MemoryTerms(
        weights_bytes=18_983_441_367,
        overhead_bytes=1_546_188_226,
        kv_bytes_per_token=86_251,
        basis="measured",
        measured_at_context=16384,
    )
    return Candidate(
        id="probe",
        memory_bytes_estimate=overrides.get("estimate", 21_633_171_456),
        memory=overrides.get("memory", terms),
    )


# ------------------------------------------------- the class states its work


def test_every_context_shaped_class_declares_its_work_with_a_source() -> None:
    """A context with no stated origin is the defect this catches.

    12288 sat on the 9B for months with no source; chased, it turned out to be
    BookForge's 32B tier reaching a 9B — the right function's wrong tier. A
    number nobody can trace costs a day to disprove, so `source` is not optional
    and this test is what makes it so in practice rather than in a docstring.
    """
    declared = [entry for entry in CLASSES if entry.work is not None]
    assert {entry.name for entry in declared} == {
        "clean",
        "translate",
        "simplify",
        "analysis",
        # The generic chat-shaped act; its default is 8192 x 1 and a client
        # may state its own (2026-09-23).
        "generate",
        # The decision door's (PHASE22-DECIDE.md section 2.9, 2026-09-23).
        "decide",
        "pages",
    }
    for entry in declared:
        assert entry.work.tokens > 0
        assert entry.work.concurrency > 0
        assert len(entry.work.source) > 40, entry.name


def test_the_classes_that_are_not_token_shaped_declare_nothing() -> None:
    """A voice holds one reservation whatever the sentence is.

    Stated because the temptation is to give every class a number for symmetry,
    and a working context on `tts` would be a fact about narration that nothing
    in narration means.
    """
    for entry in CLASSES:
        if entry.job_type in {"tts", "asr", "align", "rvc", "denoise", "echo"}:
            assert entry.work is None, entry.name


def test_translate_and_simplify_and_analysis_share_one_ruling() -> None:
    """Three acts, one sentence of Owen's — so one working context, not three."""
    work = {
        entry.name: entry.work
        for entry in CLASSES
        if entry.name in {"translate", "simplify", "analysis"}
    }
    assert len({(w.tokens, w.concurrency) for w in work.values()}) == 1
    assert work["translate"].tokens == 4096


# ----------------------------------------------------- the need is arithmetic


def test_a_candidate_costs_what_the_class_asks_for() -> None:
    candidate = a_candidate()
    translate = WorkingContext(tokens=4096, concurrency=4, source="a test")
    long_one = WorkingContext(tokens=32768, concurrency=1, source="a test")
    assert candidate.need_bytes(translate) < candidate.need_bytes(long_one)


def test_a_candidate_with_no_terms_answers_the_way_it_always_did() -> None:
    """The one guarantee that makes this safe to land mid-catalog.

    Three shipped blocks have no `[memory]` table and cannot honestly get one
    (dots-ocr's estimate is a BUDGET, not a sum; the 27B-4bit on MLX has a
    residual nothing measured says is flat or rising). Those blocks must decide
    today what they decided yesterday, whatever a class asks.
    """
    bare = a_candidate(memory=None)
    for work in (
        None,
        WorkingContext(tokens=4096, concurrency=4, source="a test"),
        WorkingContext(tokens=131072, concurrency=8, source="a test"),
    ):
        assert bare.need_bytes(work) == bare.memory_bytes_estimate


def test_the_refusal_names_all_four_terms() -> None:
    """`fits` states itself, or a reader cannot tell which term to act on.

    Of the four, two belong to the model and two belong to the WORK — and the
    two that belong to the work are the two an app can change. A sentence that
    gives only the total hides which half of it is negotiable.
    """
    entry = next(e for e in CLASSES if e.name == "translate")
    decision = decide(
        entry,
        "cuda-linux",
        total_bytes=THREE_NINETY_TI,
        desktop_allowance_bytes=DESKTOP,
        gpu_vendor="nvidia",
        chosen=None,
    )
    assert decision.enabled
    for phrase in ("weights", "overhead", "KV for", "4096 tokens x 4 in flight"):
        assert phrase in decision.reason, decision.reason


def test_a_class_with_no_work_gets_the_sentence_it_always_got() -> None:
    """`pages` reads a catalog with no terms, so its reason must not grow a
    breakdown made of numbers nobody has."""
    entry = next(e for e in CLASSES if e.name == "pages")
    decision = decide(
        entry,
        "cuda-linux",
        total_bytes=THREE_NINETY_TI,
        desktop_allowance_bytes=DESKTOP,
        gpu_vendor="nvidia",
        chosen=None,
    )
    assert "weights +" not in decision.reason


def test_the_card_still_decides_every_llm_class_on_owens_pc() -> None:
    """A regression fence, not a feature.

    The split changes HOW a verdict is reached; on the machine every number in
    this catalog was measured on it must not change WHAT the verdict is. If this
    ever goes red, the arithmetic has started disagreeing with the hand-tuned
    contexts it was derived from and one of the two is wrong.
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
    assert decisions["clean"].selected == "qwen3.5-9b"
    assert decisions["translate"].selected == "qwen3.8-27b-4bit"
    assert decisions["pages"].selected == "dots-ocr"
    for name in ("clean", "translate", "simplify", "analysis", "pages"):
        assert decisions[name].enabled, name


# ------------------------------------------------------- the published ceiling


def rows_for(backend: Backend) -> dict[str, dict[str, Any]]:
    from crucible.config import load_config, write_config
    from crucible.jobs.llm import model_rows
    from crucible.residency import Residency

    home = Path(tempfile.mkdtemp(prefix="crucible-ceiling-"))
    os.environ["CRUCIBLE_HOME"] = str(home)
    write_config(
        home,
        name="crucible@test",
        host="127.0.0.1",
        port=7100,
        token="not-minted",
        backend_kind=backend.kind,
        enable_echo=True,
        enable_llm=True,
        enable_asr=False,
        enable_tts=False,
        enable_align=False,
        enable_rvc=False,
        desktop_allowance_bytes=DESKTOP,
    )
    config = load_config(home)
    return {row["id"]: row for row in model_rows(config, backend, Residency(config))}


@pytest.fixture
def pc() -> Backend:
    return Backend(
        kind="cuda-linux",
        platform="linux",
        arch="x86_64",
        gpu=Gpu(vendor="nvidia", name="RTX 3090 Ti", vram_bytes=THREE_NINETY_TI),
        detail="test double",
    )


@pytest.fixture
def mac() -> Backend:
    return Backend(
        kind="mlx-darwin",
        platform="darwin",
        arch="arm64",
        gpu=Gpu(vendor="apple", name="M1 Ultra", vram_bytes=M1_ULTRA),
        detail="test double",
    )


def test_the_ceiling_is_the_lower_of_the_card_and_the_manifest_maximum(
    mac: Backend,
) -> None:
    """Owen, 2026-09-16: *"the max should obviously be the model's max, not the
    technical memory max"* — and 2026-09-23, *"set the max to something that
    makes sense … something that wont page/thrash/OOM"*.

    His Mac affords over a million tokens of the 9B. The checkpoint stops at
    262144, and the manifest's `max_context` for mlx-darwin stops at 131072
    (a longer prefill is not worth its time). The row is the SAME ceiling the
    load door refuses against, so it publishes 131072, not the wall behind it.
    """
    ceiling = rows_for(mac)["qwen3.5-9b"]["max_context"]
    assert ceiling["card_affords"] > ceiling["weights_allow"] > ceiling["max_context"]
    assert ceiling["tokens"] == ceiling["max_context"] == 131072
    assert ceiling["limited_by"] == "max_context"


def test_on_the_pc_the_card_binds_where_the_manifest_maximum_does_not(
    pc: Backend,
) -> None:
    """The other half of the asymmetry, and why the number has to be per server.

    The 9B and the 27B-4bit are held to the maxima their manifests computed
    for this card (65536 and 32768), each just under what the card affords at
    one in flight; the 9B-vl's intercept leaves the card room for only ~1,700
    tokens, so there the CARD binds, below even its default.
    """
    rows = rows_for(pc)
    nine = rows["qwen3.5-9b"]["max_context"]
    assert (nine["tokens"], nine["limited_by"]) == (65536, "max_context")
    assert nine["card_affords"] == 74_887
    big = rows["qwen3.8-27b-4bit"]["max_context"]
    assert (big["tokens"], big["limited_by"]) == (32768, "max_context")
    assert big["card_affords"] == 33_945
    vision = rows["qwen3.5-9b-vl"]["max_context"]
    assert vision["limited_by"] == "card"
    assert vision["tokens"] == vision["card_affords"] < 16384


def test_both_walls_are_always_published(pc: Backend, mac: Backend) -> None:
    """Not just the answer: which wall, and how far away the other one is.

    One of these two is worth buying a bigger card for and the other is not, and
    a single number cannot say which.
    """
    for backend in (pc, mac):
        for row in rows_for(backend).values():
            ceiling = row["max_context"]
            if ceiling is None:
                continue
            assert ceiling["tokens"] == min(
                ceiling["card_affords"], ceiling["max_context"]
            )
            assert ceiling["max_context"] <= ceiling["weights_allow"]
            assert ceiling["limited_by"] in {"card", "max_context"}
            assert ceiling["basis"] in {"measured", "computed", "declared"}


def test_a_model_the_card_cannot_hold_affords_no_context_at_all(pc: Backend) -> None:
    """Zero, and it stays a different answer from "your request is too long".

    The 4-bit 27B's weights are 17.68 GiB; on a 12 GiB card that is a refusal
    about the MODEL, and rounding it into a context refusal would send somebody
    off to shorten a paragraph that was never the problem.

    (This read the 8-bit 27B on the 3090 Ti until 2026-09-23, when the 8-bit
    lost its cuda-linux block — Owen: *"we shouldnt have an 8 bit 27b on here.
    waste of space, wont fit in the gpu"*. Every cuda-linux block now holds its
    weights on that card, so the too-small card is a smaller one.)
    """
    small = replace(pc, gpu=replace(pc.gpu, name="RTX 3060", vram_bytes=12 * 1024**3))
    ceiling = rows_for(small)["qwen3.8-27b-4bit"]["max_context"]
    assert ceiling["tokens"] == 0
    assert ceiling["card_affords"] == 0
    # And on the 3090 Ti the 8-bit has no row to publish a ceiling in at all.
    eight = rows_for(pc)["qwen3.8-27b-8bit"]
    assert eight["backend_supported"] is False
    assert eight["max_context"] is None


def test_a_block_with_no_terms_publishes_no_ceiling(pc: Backend) -> None:
    """Null, not a guess.

    `dots-ocr`'s cuda-linux estimate is a budget rather than a sum of terms, so
    this server cannot say what another context would cost. Saying so is the
    honest answer; inventing one is the Ollama failure.
    """
    assert rows_for(pc)["dots-ocr"]["max_context"] is None
    assert rows_for(pc)["dots-ocr"]["memory_terms"] is None


def test_the_trained_context_is_published_even_where_the_backend_is_not(
    pc: Backend,
) -> None:
    """A property of the weights, so it is the same on every machine.

    Unlike `revision` and `memory_bytes_estimate`, which are per-host and go
    null when this backend has no block.
    """
    for row in rows_for(pc).values():
        assert isinstance(row["trained_context"], int)
        assert row["trained_context"] > 0


def test_the_pages_class_asks_for_the_width_its_own_module_publishes() -> None:
    """Ledger N3: `concurrency=1` here was the stale copy.

    The comment beside it said "one page at a time is what the guard and the
    lease already assume", and both apps were sending twelve — the number
    `models/dots-ocr.toml`'s KV note is written against. So the arithmetic
    that decides whether `dots-ocr` fits was sized for a twelfth of the work.
    `crucible/pages.py` owns the number now, on the wire and here.
    """
    pages_class = next(entry for entry in CLASSES if entry.name == "pages")
    assert pages_class.work is not None
    assert pages_class.work.concurrency == PAGE_CONCURRENCY == 12
    assert pages_class.work.tokens == 32768

    # It reaches the arithmetic, which is the whole reason it matters: a
    # candidate with terms costs twelve pages' KV, not one page's.
    candidate = a_candidate()
    one = WorkingContext(tokens=32768, concurrency=1, source="a test")
    assert candidate.need_bytes(pages_class.work) > candidate.need_bytes(one)
