from __future__ import annotations

import pytest

from crucible.accelerator import AcceleratorState
from crucible.engines import engine_load_args
from crucible.manifests import BackendSpec, KvBlocks, MemoryTerms, load_manifest
from crucible.vram import (
    VLLM_CUDA_CONTEXT_BYTES,
    engine_budget_bytes,
    max_num_seqs,
    plan_vllm_memory,
)

MIB = 1024**2
GIB = 1024**3

TOTAL = 24_564 * MIB
ALLOWANCE = 3 * GIB


def card(free_mib: int) -> AcceleratorState:
    return AcceleratorState(
        backend="cuda-linux",
        total_bytes=TOTAL,
        free_bytes=free_mib * MIB,
        compute_apps=(),
        detail="fixture",
    )


def spec(
    *,
    engine: str = "vllm",
    memory: MemoryTerms | None = None,
    args: tuple[str, ...] = ("--max-num-seqs", "16"),
) -> BackendSpec:
    return BackendSpec(
        backend="cuda-linux",
        engine=engine,
        hf_repo="fixture/repo",
        revision="0" * 40,
        memory_bytes_estimate=20_950_548_480,
        engine_args=args,
        context_default=None,
        memory=memory,
    )


# vLLM 0.29's KV layout for Qwen3.5-4B and -9B (the manifests' kv_blocks).
FOUR_B_BLOCKS = KvBlocks(tokens=528, bytes=17_301_504, state_blocks=6)

NINE_B = MemoryTerms(
    weights_bytes=18_038_862_643,
    overhead_bytes=1_476_395_008,
    kv_bytes_per_token=None,
    kv_blocks=FOUR_B_BLOCKS,
    basis="measured",
    measured_at_context=16384,
)


def test_the_measurement_wins_when_the_desktop_is_over_its_allowance():
    free = 21_234 * MIB
    assert engine_budget_bytes(TOTAL, ALLOWANCE, free) == free
    assert free < TOTAL - ALLOWANCE, "this case is only interesting while free is the smaller"


def test_the_allowance_holds_the_room_open_when_the_desktop_is_under_it():
    free = 23_372 * MIB
    assert engine_budget_bytes(TOTAL, ALLOWANCE, free) == TOTAL - ALLOWANCE


def test_there_is_no_third_term():
    for free_mib in (0, 5_000, 21_234, 21_492, 23_372, 24_564):
        budget = engine_budget_bytes(TOTAL, ALLOWANCE, free_mib * MIB)
        assert budget in (free_mib * MIB, TOTAL - ALLOWANCE)


def test_an_allowance_larger_than_the_card_is_zero_and_not_negative():
    assert engine_budget_bytes(4 * GIB, 8 * GIB, 4 * GIB) == 0


def test_a_block_with_no_terms_is_left_exactly_as_its_manifest_states():
    plan = plan_vllm_memory(
        model_id="dots-ocr",
        spec=spec(memory=None),
        context=32768,
        card=card(21_300),
        desktop_allowance_bytes=ALLOWANCE,
        reclaimable_bytes=0,
    )
    assert plan is None


def test_a_backend_that_is_not_vllm_is_left_alone_even_with_terms():
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=spec(engine="llama-server", memory=NINE_B),
        context=16384,
        card=card(21_300),
        desktop_allowance_bytes=ALLOWANCE,
        reclaimable_bytes=0,
    )
    assert plan is None


@pytest.mark.parametrize(
    "args, expected",
    [
        (("--max-num-seqs", "16"), 16),
        (("--max-num-seqs=16",), 16),
        (("--dtype", "bfloat16"), None),
        ((), None),
    ],
)
def test_max_num_seqs_is_read_from_the_block_not_assumed(args, expected):
    assert max_num_seqs(spec(args=args)) == expected


def test_the_pool_never_exceeds_what_the_engine_could_reach():
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=spec(memory=NINE_B),
        context=4096,
        card=AcceleratorState(
            backend="cuda-linux",
            total_bytes=80 * GIB,
            free_bytes=79 * GIB,
            compute_apps=(),
            detail="fixture",
        ),
        desktop_allowance_bytes=ALLOWANCE,
        reclaimable_bytes=0,
    )
    assert plan is not None
    # Each in-flight request's whole blocks and the state blocks vLLM keeps for it inside
    # the pool, and the one null block the pool holds back.
    blocks = 16 * (-(-4096 // 528) + 6) + 1
    assert plan.pool_bytes == blocks * 17_301_504


def test_one_request_at_its_full_context_gets_the_room_vllm_asks_for():
    """Victoria's RTX 3070, 2026-10-09: qwen3.5-4b-bside-4bit, 8192 tokens x 1, was given
    exactly 8192 x 40,337 B (0.31 GiB) and vLLM refused: 0.35 GiB needed, 0.29 usable."""
    plan = plan_vllm_memory(
        model_id="qwen3.5-4b-bside-4bit",
        spec=spec(memory=MemoryTerms(
            weights_bytes=3_339_000_000, overhead_bytes=1_600_000_000,
            kv_bytes_per_token=None, kv_blocks=FOUR_B_BLOCKS, basis="measured", measured_at_context=8192,
        ), args=("--max-num-seqs", "1")),
        context=8192,
        card=AcceleratorState(
            backend="cuda-linux", total_bytes=8 * GIB, free_bytes=int(7.83 * GIB),
            compute_apps=(), detail="fixture",
        ),
        desktop_allowance_bytes=GIB,
        reclaimable_bytes=0,
    )
    assert plan is not None and plan.fits
    needed, state = int(0.35 * GIB), int(0.019 * GIB)
    assert plan.pool_bytes >= needed + state
    # vLLM's own check: the pool less its null block holds the request's blocks.
    blocks = FOUR_B_BLOCKS.request_blocks(8192)
    assert plan.pool_bytes - FOUR_B_BLOCKS.bytes >= blocks * FOUR_B_BLOCKS.bytes


def test_a_block_that_states_no_concurrency_may_have_the_whole_budget():
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=spec(memory=NINE_B, args=("--dtype", "bfloat16")),
        context=16384,
        card=card(21_300),
        desktop_allowance_bytes=ALLOWANCE,
        reclaimable_bytes=0,
    )
    assert plan is not None
    assert plan.concurrency is None
    assert plan.pool_bytes == plan.budget_bytes - NINE_B.fixed_bytes


def test_the_card_that_refused_on_2026_09_17_is_sized_and_not_refused():
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=spec(memory=NINE_B),
        context=16384,
        card=card(21_234),
        desktop_allowance_bytes=ALLOWANCE,
        reclaimable_bytes=0,
    )
    assert plan is not None
    assert plan.fits, plan.sentence()
    assert plan.pool_bytes > 0
    assert plan.affordable_context > 16384


def test_the_pool_is_the_budget_less_weights_and_overhead():
    free_mib = 21_234
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=spec(memory=NINE_B),
        context=16384,
        card=card(free_mib),
        desktop_allowance_bytes=ALLOWANCE,
        reclaimable_bytes=0,
    )
    assert plan is not None
    # The free memory the probe reports, less what vLLM's own CUDA context holds.
    from crucible.vram import VLLM_CUDA_CONTEXT_BYTES

    reachable = free_mib * MIB - VLLM_CUDA_CONTEXT_BYTES
    assert plan.budget_bytes == reachable
    assert plan.pool_bytes == reachable - NINE_B.fixed_bytes


def test_a_card_too_full_refuses_by_name_with_every_term_in_the_sentence():
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=spec(memory=NINE_B),
        context=16384,
        card=card(19_000),
        desktop_allowance_bytes=ALLOWANCE,
        reclaimable_bytes=0,
    )
    assert plan is not None
    assert not plan.fits
    said = plan.sentence()
    for term in ("qwen3.5-9b", "16384", "528-token blocks", "measured", "free", "budget"):
        assert term in said, f"the refusal does not name {term!r}: {said}"


def test_both_flags_go_on_and_the_pool_is_stated_in_bytes():
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=spec(memory=NINE_B),
        context=16384,
        card=card(21_234),
        desktop_allowance_bytes=ALLOWANCE,
        reclaimable_bytes=0,
    )
    assert plan is not None
    flags = plan.flags()
    assert flags[0] == "--kv-cache-memory-bytes"
    assert int(flags[1]) == plan.pool_bytes
    assert flags[2] == "--gpu-memory-utilization"
    assert 0.0 < float(flags[3]) <= 1.0


def test_the_plan_flags_come_after_the_manifest_so_they_win():
    manifest = load_manifest("qwen3.5-9b")
    block = manifest.backends["cuda-linux"]
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=block,
        context=16384,
        card=card(21_234),
        desktop_allowance_bytes=ALLOWANCE,
        reclaimable_bytes=0,
    )
    assert plan is not None
    args = engine_load_args(
        manifest, block, __import__("pathlib").Path("/w"), plan, context=16384
    )
    assert "--kv-cache-memory-bytes" in args
    last = len(args) - 1 - args[::-1].index("--gpu-memory-utilization")
    assert args[last + 1] == f"{plan.budget_bytes / plan.total_bytes:.4f}"
    assert args.count("--gpu-memory-utilization") == 2, (
        "the manifest's own value should still be visible on the line, "
        "overridden rather than edited out"
    )


def test_none_leaves_the_manifest_line_untouched():
    from pathlib import Path

    manifest = load_manifest("qwen3.5-9b")
    block = manifest.backends["cuda-linux"]
    args = engine_load_args(
        manifest, block, Path("/w"), None, context=manifest.context_for("cuda-linux")
    )
    assert "--kv-cache-memory-bytes" not in args
    assert args.count("--gpu-memory-utilization") == 1


def test_the_calibrated_terms_in_this_file_are_the_manifests_own():
    assert load_manifest("qwen3.5-9b").backends["cuda-linux"].memory == NINE_B


def test_the_terms_are_measured_and_the_layout_is_vllms():
    terms = load_manifest("qwen3.5-9b").backends["cuda-linux"].memory
    assert terms is not None
    assert terms.basis == "measured"
    assert terms.kv_blocks == FOUR_B_BLOCKS


@pytest.mark.parametrize(
    ("model", "pool", "context", "concurrency"),
    [
        # Each load's own "Maximum concurrency for 16,384 tokens per request" on owens-pc.
        ("qwen3.5-2b", 4_724_621_312, 16384, 19.08),
        ("qwen3.5-4b", 9_802_918_407, 16384, 14.89),
        ("qwen3.5-9b", 2_036_124_877, 16384, 3.08),
        ("qwen3.5-9b", 1_735_183_565, 16384, 2.63),
        ("qwen3.8-27b-4bit", 1_880_298_567, 16384, 1.33),
        ("qwen3.8-27b-4bit", 1_655_903_303, 16384, 1.19),
    ],
)
def test_the_layout_gives_the_concurrency_vllm_reported(model, pool, context, concurrency):
    blocks = load_manifest(model).backends["cuda-linux"].memory.kv_blocks
    assert blocks is not None
    assert round((pool // blocks.bytes) / blocks.request_blocks(context), 2) == concurrency


def test_the_27b_pool_vllm_refused_on_2026_11_11_affords_what_vllm_said():
    """BookForge's Simplify, owens-pc 2026-11-11 23:03: qwen3.8-27b-4bit planned at 11,317
    tokens in a pool of 1,070,797,895 B, and vLLM refused: "1.0 GiB KV cache is needed ...
    available KV cache memory (0.95 GiB) ... the estimated maximum model length is 10192"."""
    terms = load_manifest("qwen3.8-27b-4bit").backends["cuda-linux"].memory
    assert terms is not None
    pool = 1_070_797_895
    assert terms.context_in(pool, concurrency=1) == 10_192
    assert terms.kv_bytes(context=11_317, concurrency=1) > pool
    assert terms.kv_bytes(context=10_192, concurrency=1) <= pool


def test_a_27b_plan_on_that_card_starts_at_a_context_its_pool_holds():
    terms = load_manifest("qwen3.8-27b-4bit").backends["cuda-linux"].memory
    assert terms is not None
    budget = terms.fixed_bytes + 1_070_797_895
    plan = plan_vllm_memory(
        model_id="qwen3.8-27b-4bit",
        spec=spec(memory=terms),
        context=16384,
        card=AcceleratorState(
            backend="cuda-linux", total_bytes=TOTAL,
            free_bytes=budget + VLLM_CUDA_CONTEXT_BYTES,
            compute_apps=(), detail="fixture",
        ),
        desktop_allowance_bytes=TOTAL - budget,
        reclaimable_bytes=0,
    )
    assert plan is not None and not plan.fits
    assert plan.affordable_context == 10_192
    fitted = plan_vllm_memory(
        model_id="qwen3.8-27b-4bit",
        spec=spec(memory=terms),
        context=plan.affordable_context,
        card=AcceleratorState(
            backend="cuda-linux", total_bytes=TOTAL,
            free_bytes=budget + VLLM_CUDA_CONTEXT_BYTES,
            compute_apps=(), detail="fixture",
        ),
        desktop_allowance_bytes=TOTAL - budget,
        reclaimable_bytes=0,
    )
    assert fitted is not None and fitted.fits
    blocks = terms.kv_blocks
    assert blocks is not None
    assert fitted.pool_bytes - blocks.bytes >= blocks.request_blocks(10_192) * blocks.bytes


def test_the_budget_leaves_vllms_own_cuda_context_free_on_an_8_gib_card():
    """Victoria's RTX 3070, 2026-10-09: the probe saw 7.83 GiB free and the allowance left
    7.0 GiB, so the budget was 7.0 GiB (utilization 0.875); vLLM, holding its CUDA
    context, saw 6.95 GiB free and refused to start."""
    from crucible.vram import VLLM_CUDA_CONTEXT_BYTES

    total = 8 * GIB
    eight = AcceleratorState(
        backend="cuda-linux", total_bytes=total, free_bytes=int(7.83 * GIB),
        compute_apps=(), detail="fixture",
    )
    plan = plan_vllm_memory(
        model_id="qwen3.5-4b-8bit",
        spec=spec(memory=MemoryTerms(
            weights_bytes=5_120_000_000, overhead_bytes=1_600_000_000,
            kv_bytes_per_token=None, kv_blocks=FOUR_B_BLOCKS, basis="computed", measured_at_context=8192,
        ), args=("--max-num-seqs", "4")),
        context=8192,
        card=eight,
        desktop_allowance_bytes=GIB,
        reclaimable_bytes=0,
    )
    assert plan is not None and plan.fits
    assert plan.budget_bytes == int(7.83 * GIB) - VLLM_CUDA_CONTEXT_BYTES
    assert plan.budget_bytes <= int(6.95 * GIB), "what vLLM saw free there"
