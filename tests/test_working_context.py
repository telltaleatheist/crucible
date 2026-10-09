from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from crucible.backend import Backend, Gpu
from crucible.capabilityclasses import CLASSES
from crucible.fit import Candidate, WorkingContext
from crucible.manifests import MemoryTerms
from crucible.pages import PAGE_CONCURRENCY
from crucible.verdict import decide, decide_all

THREE_NINETY_TI = 25_757_220_864
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


def test_every_context_shaped_class_declares_its_work_with_a_source() -> None:
    declared = [entry for entry in CLASSES if entry.work is not None]
    assert {entry.name for entry in declared} == {
        "clean",
        "translate",
        "simplify",
        "analysis",
        "generate",
        "decide",
        "pages",
    }
    for entry in declared:
        assert entry.work.tokens > 0
        assert entry.work.concurrency > 0
        assert len(entry.work.source) > 40, entry.name


def test_the_classes_that_are_not_token_shaped_declare_nothing() -> None:
    for entry in CLASSES:
        if entry.job_type in {"tts", "asr", "align", "rvc", "denoise", "echo"}:
            assert entry.work is None, entry.name


def test_translate_and_simplify_and_analysis_share_one_ruling() -> None:
    work = {
        entry.name: entry.work
        for entry in CLASSES
        if entry.name in {"translate", "simplify", "analysis"}
    }
    assert len({(w.tokens, w.concurrency) for w in work.values()}) == 1
    assert work["translate"].tokens == 4096


def test_a_candidate_costs_what_the_class_asks_for() -> None:
    candidate = a_candidate()
    translate = WorkingContext(tokens=4096, concurrency=4, source="a test")
    long_one = WorkingContext(tokens=32768, concurrency=1, source="a test")
    assert candidate.need_bytes(translate) < candidate.need_bytes(long_one)


def test_a_candidate_with_no_terms_answers_the_way_it_always_did() -> None:
    bare = a_candidate(memory=None)
    for work in (
        None,
        WorkingContext(tokens=4096, concurrency=4, source="a test"),
        WorkingContext(tokens=131072, concurrency=8, source="a test"),
    ):
        assert bare.need_bytes(work) == bare.memory_bytes_estimate


def test_the_refusal_names_all_four_terms() -> None:
    entry = next(e for e in CLASSES if e.name == "translate")
    decision = decide(
        entry,
        "cuda-linux",
        total_bytes=THREE_NINETY_TI,
        desktop_allowance_bytes=DESKTOP,
        gpu_vendor="nvidia",
        chosen=None,
        audio_low_vram=False,
    )
    assert decision.enabled
    for phrase in ("weights", "overhead", "KV for", "4096 tokens x 4 in flight"):
        assert phrase in decision.reason, decision.reason


def test_a_class_with_no_work_gets_the_sentence_it_always_got() -> None:
    entry = next(e for e in CLASSES if e.name == "pages")
    decision = decide(
        entry,
        "cuda-linux",
        total_bytes=THREE_NINETY_TI,
        desktop_allowance_bytes=DESKTOP,
        gpu_vendor="nvidia",
        chosen=None,
        audio_low_vram=False,
    )
    assert "weights +" not in decision.reason


def test_the_card_still_decides_every_llm_class_on_owens_pc() -> None:
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
    assert decisions["clean"].selected == "qwen3.5-9b"
    assert decisions["translate"].selected == "qwen3.8-27b-4bit"
    assert decisions["pages"].selected == "dots-ocr"
    for name in ("clean", "translate", "simplify", "analysis", "pages"):
        assert decisions[name].enabled, name


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
        enable_denoise=False,
        retention_days=7,
        desktop_allowance_basis="stated",
        desktop_allowance_note="",
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
    ceiling = rows_for(mac)["qwen3.5-9b"]["max_context"]
    assert ceiling["card_affords"] > ceiling["weights_allow"] > ceiling["max_context"]
    assert ceiling["tokens"] == ceiling["max_context"] == 131072
    assert ceiling["limited_by"] == "max_context"


def test_on_the_pc_the_card_binds_where_the_manifest_maximum_does_not(
    pc: Backend,
) -> None:
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
    small = replace(pc, gpu=replace(pc.gpu, name="RTX 3060", vram_bytes=12 * 1024**3))
    ceiling = rows_for(small)["qwen3.8-27b-4bit"]["max_context"]
    assert ceiling["tokens"] == 0
    assert ceiling["card_affords"] == 0
    eight = rows_for(pc)["qwen3.8-27b-8bit"]
    assert eight["backend_supported"] is False
    assert eight["max_context"] is None


def test_a_block_with_no_terms_publishes_no_ceiling(pc: Backend) -> None:
    assert rows_for(pc)["dots-ocr"]["max_context"] is None
    assert rows_for(pc)["dots-ocr"]["memory_terms"] is None


def test_the_trained_context_is_published_even_where_the_backend_is_not(
    pc: Backend,
) -> None:
    for row in rows_for(pc).values():
        assert isinstance(row["trained_context"], int)
        assert row["trained_context"] > 0


def test_the_pages_class_asks_for_the_width_its_own_module_publishes() -> None:
    pages_class = next(entry for entry in CLASSES if entry.name == "pages")
    assert pages_class.work is not None
    assert pages_class.work.concurrency == PAGE_CONCURRENCY == 12
    assert pages_class.work.tokens == 32768

    candidate = a_candidate()
    one = WorkingContext(tokens=32768, concurrency=1, source="a test")
    assert candidate.need_bytes(pages_class.work) > candidate.need_bytes(one)
