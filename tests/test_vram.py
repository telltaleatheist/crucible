"""Sizing a vLLM KV pool against the card, rather than a fraction of it.

docs/MEASUREMENTS.md, 2026-09-18. `qwen3.5-9b` refused to load on its own
manifest's `--gpu-memory-utilization 0.84` with `Available KV cache memory:
-0.19 GiB`, and the identical argv loaded cleanly the next night at `+1.94 GiB`
with the desktop the same size. The term that moved was `total_consumed`, a
WHOLE-CARD delta across the profiling window, so the longer the load the more of
somebody else's allocation is charged to our pool.

These tests hold the three things that make the fix a fix rather than a bigger
number. That the budget is the smaller of the measurement and the allowance and
never a third invented term; that a block nobody has taken apart is LEFT ALONE
rather than given made-up terms; and that the pool reaches the command line as
BYTES, after the manifest's own flag so it overrides rather than decorates it.

**None of them is a regression test for the -0.19 GiB, and one of them says so
at length.** Crucible's arithmetic said yes that night; vLLM's said no, from a
fraction of the card's total and a profiling window that charged it somebody
else's allocation. No unit test reaches into that. What proves the fix is a
load, and the load is recorded in docs/MEASUREMENTS.md.
"""

from __future__ import annotations

import pytest

from crucible.accelerator import AcceleratorState
from crucible.manifests import BackendSpec, MemoryTerms, load_manifest
from crucible.vram import (
    KvPlan,
    engine_budget_bytes,
    max_num_seqs,
    plan_vllm_memory,
)

MIB = 1024**2
GIB = 1024**3

#: Owen's PC. `nvidia-smi --query-gpu=memory.total` on the 3090 Ti.
TOTAL = 24_564 * MIB
#: `desktop_allowance_bytes` from config.toml, the declared host fact.
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


#: The 9B's calibrated cuda-linux terms, 2026-09-18. Quoted rather than loaded so
#: that a change to the manifest fails the last test in this file — which is the
#: one that names the manifest — rather than silently changing every case here.
NINE_B = MemoryTerms(
    weights_bytes=18_038_862_643,
    overhead_bytes=1_476_395_008,
    kv_bytes_per_token=40_337,
    basis="measured",
    measured_at_context=16384,
)


# ----------------------------------------------------------- the budget


def test_the_measurement_wins_when_the_desktop_is_over_its_allowance():
    # 2026-09-17, the night of the failure: 3_330 MiB of desktop on a 24_564 MiB
    # card, so 21_234 MiB free against an allowance that would have said 21_492.
    free = 21_234 * MIB
    assert engine_budget_bytes(TOTAL, ALLOWANCE, free) == free
    assert free < TOTAL - ALLOWANCE, "this case is only interesting while free is the smaller"


def test_the_allowance_holds_the_room_open_when_the_desktop_is_under_it():
    # A quiet machine: 1_192 MiB of desktop, which is what both 2026-09-12
    # measurements started from. The desktop is free to grow back to its
    # allowance, so the budget must NOT be the whole 23_372 MiB that is free.
    free = 23_372 * MIB
    assert engine_budget_bytes(TOTAL, ALLOWANCE, free) == TOTAL - ALLOWANCE


def test_there_is_no_third_term():
    # capability.py's ruling 3: the allowance IS the margin, and a second reserve
    # would be "a number nobody has measured, invented to feel safe". So the
    # answer is exactly one of the two inputs, never a shaved version of either.
    for free_mib in (0, 5_000, 21_234, 21_492, 23_372, 24_564):
        budget = engine_budget_bytes(TOTAL, ALLOWANCE, free_mib * MIB)
        assert budget in (free_mib * MIB, TOTAL - ALLOWANCE)


def test_an_allowance_larger_than_the_card_is_zero_and_not_negative():
    assert engine_budget_bytes(4 * GIB, 8 * GIB, 4 * GIB) == 0


# ------------------------------------------------------ what is left alone


def test_a_block_with_no_terms_is_left_exactly_as_its_manifest_states():
    # dots-ocr's cuda-linux estimate IS a budget — 0.5 x the card, so page
    # reading can share the machine — rather than a sum of terms. There is
    # nothing to take apart and its fraction is a DECISION.
    plan = plan_vllm_memory(
        model_id="dots-ocr",
        spec=spec(memory=None),
        context=32768,
        card=card(21_300),
        desktop_allowance_bytes=ALLOWANCE,
    )
    assert plan is None


def test_a_backend_that_is_not_vllm_is_left_alone_even_with_terms():
    # llama-windows HAS terms (declared ones) and llama-server has no such flag.
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=spec(engine="llama-server", memory=NINE_B),
        context=16384,
        card=card(21_300),
        desktop_allowance_bytes=ALLOWANCE,
    )
    assert plan is None


# ------------------------------------------------------------ the ceiling


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
    # A huge card: the budget would afford far more KV than 16 requests of 4096
    # tokens can ever use, and reserving past that holds bytes nothing can read.
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
    )
    assert plan is not None
    assert plan.pool_bytes == NINE_B.kv_bytes_per_token * 4096 * 16


def test_a_block_that_states_no_concurrency_may_have_the_whole_budget():
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=spec(memory=NINE_B, args=("--dtype", "bfloat16")),
        context=16384,
        card=card(21_300),
        desktop_allowance_bytes=ALLOWANCE,
    )
    assert plan is not None
    assert plan.concurrency is None
    assert plan.pool_bytes == plan.budget_bytes - NINE_B.fixed_bytes


# ------------------------------------------- the night of the failure, again


def test_the_card_that_refused_on_2026_09_17_is_sized_and_not_refused():
    """The card as it stood that night, sized rather than fractioned.

    WHAT THIS DOES NOT PROVE, said plainly because the first draft of this
    docstring claimed it did: it is not a regression test for the failure. The
    old arithmetic would also have said yes here — `total − allowance` came to
    21_492 MiB and the old terms left 2.1 GiB for KV. **Crucible's arithmetic was
    never what refused.** vLLM's was, from a fraction of the card's total and a
    profiling window that charged it the desktop's growth, and no unit test can
    reach that.

    What this DOES hold is that the new path does not introduce a refusal of its
    own on the very card that matters: 21_234 MiB free is a real reading from a
    real night, and it must produce a workable pool rather than an
    `insufficient_kv_cache`. The proof that the load itself succeeds is a load,
    and it is recorded in docs/MEASUREMENTS.md.
    """
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=spec(memory=NINE_B),
        context=16384,
        card=card(21_234),
        desktop_allowance_bytes=ALLOWANCE,
    )
    assert plan is not None
    assert plan.fits, plan.sentence()
    assert plan.pool_bytes > 0
    # More than one full-context request, which is the floor vLLM itself refuses
    # below, and the thing -0.19 GiB was 0.19 GiB short of.
    assert plan.affordable_context > 16384


def test_the_pool_is_the_budget_less_weights_and_overhead():
    free_mib = 21_234
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=spec(memory=NINE_B),
        context=16384,
        card=card(free_mib),
        desktop_allowance_bytes=ALLOWANCE,
    )
    assert plan is not None
    assert plan.budget_bytes == free_mib * MIB
    assert plan.pool_bytes == free_mib * MIB - NINE_B.fixed_bytes


def test_a_card_too_full_refuses_by_name_with_every_term_in_the_sentence():
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=spec(memory=NINE_B),
        context=16384,
        card=card(19_000),
        desktop_allowance_bytes=ALLOWANCE,
    )
    assert plan is not None
    assert not plan.fits
    said = plan.sentence()
    for term in ("qwen3.5-9b", "16384", "40_337", "measured", "free", "budget"):
        assert term in said, f"the refusal does not name {term!r}: {said}"


# ------------------------------------------------------------- the flags


def test_both_flags_go_on_and_the_pool_is_stated_in_bytes():
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=spec(memory=NINE_B),
        context=16384,
        card=card(21_234),
        desktop_allowance_bytes=ALLOWANCE,
    )
    assert plan is not None
    flags = plan.flags()
    assert flags[0] == "--kv-cache-memory-bytes"
    assert int(flags[1]) == plan.pool_bytes
    # The utilisation survives as a GATE: `request_memory()` runs in init_device
    # whatever the pool says, and raises if free < total x util.
    assert flags[2] == "--gpu-memory-utilization"
    assert 0.0 < float(flags[3]) <= 1.0


def test_the_plan_flags_come_after_the_manifest_so_they_win():
    """argparse takes the LAST spelling of a flag, and the manifest states one.

    `qwen3.5-9b`'s block carries `--gpu-memory-utilization 0.84`. If the plan's
    flags went on first the manifest's constant would win and this whole module
    would be decorative.
    """
    from crucible.residency import Residency

    manifest = load_manifest("qwen3.5-9b")
    block = manifest.backends["cuda-linux"]
    plan = plan_vllm_memory(
        model_id="qwen3.5-9b",
        spec=block,
        context=16384,
        card=card(21_234),
        desktop_allowance_bytes=ALLOWANCE,
    )
    assert plan is not None
    args = Residency._engine_args(manifest, block, __import__("pathlib").Path("/w"), plan)
    assert "--kv-cache-memory-bytes" in args
    last = len(args) - 1 - args[::-1].index("--gpu-memory-utilization")
    assert args[last + 1] == f"{plan.budget_bytes / plan.total_bytes:.4f}"
    assert args.count("--gpu-memory-utilization") == 2, (
        "the manifest's own value should still be visible on the line, "
        "overridden rather than edited out"
    )


def test_none_leaves_the_manifest_line_untouched():
    from pathlib import Path

    from crucible.residency import Residency

    manifest = load_manifest("qwen3.5-9b")
    block = manifest.backends["cuda-linux"]
    args = Residency._engine_args(manifest, block, Path("/w"), None)
    assert "--kv-cache-memory-bytes" not in args
    assert args.count("--gpu-memory-utilization") == 1


# ------------------------------------------------- the manifest it came from


def test_the_calibrated_terms_in_this_file_are_the_manifests_own():
    """If the manifest is re-calibrated, this file must be re-read, not drift.

    Every case above uses `NINE_B` rather than the manifest, so that a new
    measurement changes ONE assertion here instead of silently changing what a
    dozen tests are about.
    """
    assert load_manifest("qwen3.5-9b").backends["cuda-linux"].memory == NINE_B


def test_the_slope_is_measured_and_not_computed():
    # The whole point of 2026-09-18. A computed slope on this engine has been
    # 23-24% light on both models anyone has checked.
    terms = load_manifest("qwen3.5-9b").backends["cuda-linux"].memory
    assert terms is not None
    assert terms.basis == "measured"
    assert terms.kv_bytes_per_token > 32_768
