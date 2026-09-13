"""The accelerator guard's refusals (PHASE2-LLM.md section 4).

Every probe is monkeypatched, so these assert on the guard's rules rather than on
whatever happens to be on the machine running the suite. `accelerator_busy` in
particular can only be proved this way: it requires somebody else's job to be on
the card, and Crucible's whole point is that it never puts one there.
"""

from __future__ import annotations

import pytest

from crucible import accelerator
from crucible.accelerator import GIB, ComputeApp, ProbeError, guard, read_state
from crucible.errors import ApiError

GIB_MIB = 1024

CARD_TOTAL = 24 * GIB


def fake_cuda(
    monkeypatch: pytest.MonkeyPatch,
    *,
    apps: list[ComputeApp],
    free_bytes: int,
    total_bytes: int = CARD_TOTAL,
) -> None:
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: list(apps))
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (free_bytes, total_bytes))


def fake_mac(
    monkeypatch: pytest.MonkeyPatch, *, available: int, total: int = 64 * GIB
) -> None:
    monkeypatch.setattr(
        accelerator, "probe_unified_memory", lambda: (available, total)
    )


# --------------------------------------------------------------- it proceeds


def test_an_idle_card_with_room_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda(monkeypatch, apps=[], free_bytes=22 * GIB)
    state = guard(
        "cuda-linux",
        model_id="qwen3.5-9b",
        need_bytes=20 * GIB,
        desktop_allowance_bytes=3 * GIB,
    )
    assert state.free_bytes == 22 * GIB
    assert state.total_bytes == CARD_TOTAL


def test_a_small_process_is_not_somebody_s_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under the 1 GiB floor is a compositor, not a job (section 4)."""
    fake_cuda(
        monkeypatch,
        apps=[ComputeApp(pid=99, name="Xorg", used_bytes=400 * 1024 ** 2)],
        free_bytes=22 * GIB,
    )
    guard(
        "cuda-linux",
        model_id="qwen3.5-9b",
        need_bytes=20 * GIB,
        desktop_allowance_bytes=3 * GIB,
    )


def test_crucible_s_own_engine_is_not_foreign(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda(
        monkeypatch,
        apps=[ComputeApp(pid=4242, name="python", used_bytes=20 * GIB)],
        free_bytes=3 * GIB,
    )
    guard(
        "cuda-linux",
        model_id="qwen3.5-9b",
        need_bytes=20 * GIB,
        owned_pids=frozenset({4242}),
        desktop_allowance_bytes=0,
        # Reloading after unloading our own 20 GiB engine: that memory is coming
        # back, so the guard counts it as free.
        reclaimable_bytes=20 * GIB,
    )


# ------------------------------------------------------------ accelerator_busy


def test_a_foreign_process_over_a_gib_is_accelerator_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda(
        monkeypatch,
        apps=[ComputeApp(pid=12769, name="sgl-omni", used_bytes=17 * GIB)],
        free_bytes=7 * GIB,
    )
    with pytest.raises(ApiError) as caught:
        guard(
            "cuda-linux",
            model_id="qwen3.5-9b",
            need_bytes=2 * GIB,
            desktop_allowance_bytes=3 * GIB,
        )
    error = caught.value
    assert error.status_code == 409
    assert error.code == "accelerator_busy"
    # It names the process, as section 4 requires.
    assert "pid 12769" in error.message
    assert "sgl-omni" in error.message
    assert "17.0 GiB" in error.message
    assert "never evicts" in error.message
    assert error.details["processes"][0]["pid"] == 12769


def test_a_process_whose_memory_the_driver_hides_is_still_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"[Insufficient Permissions]" is not zero and is not parsed as zero."""
    fake_cuda(
        monkeypatch,
        apps=[ComputeApp(pid=1460, name="[Insufficient Permissions]", used_bytes=None)],
        free_bytes=22 * GIB,
    )
    with pytest.raises(ApiError) as caught:
        guard(
            "cuda-linux",
            model_id="qwen3.5-9b",
            need_bytes=2 * GIB,
            desktop_allowance_bytes=3 * GIB,
        )
    assert caught.value.code == "accelerator_busy"
    assert "memory not reported" in caught.value.message


def test_vram_nobody_admits_to_is_accelerator_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WSL2 case, measured on Owen's PC 2026-09-12.

    The driver shim inside WSL2 answers --query-compute-apps with an empty list
    even while a process in that same VM holds 17 GiB. memory.free is accurate,
    so the unattributed figure is what catches it.
    """
    fake_cuda(monkeypatch, apps=[], free_bytes=7 * GIB)
    with pytest.raises(ApiError) as caught:
        guard(
            "cuda-linux",
            model_id="qwen3.5-9b",
            need_bytes=2 * GIB,
            desktop_allowance_bytes=3 * GIB,
        )
    assert caught.value.code == "accelerator_busy"
    assert "will not name" in caught.value.message
    assert caught.value.details["unattributed_bytes"] == (24 - 7 - 3) * GIB


def test_the_desktop_allowance_is_not_treated_as_a_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2.3 GiB of Windows desktop under a 3 GiB allowance is not somebody's job."""
    fake_cuda(monkeypatch, apps=[], free_bytes=CARD_TOTAL - int(2.3 * GIB))
    guard(
        "cuda-linux",
        model_id="qwen3.5-9b",
        need_bytes=20 * GIB,
        desktop_allowance_bytes=3 * GIB,
    )


# --------------------------------------------------------- insufficient_memory


def test_the_27b_on_a_24_gib_card_is_insufficient_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda(monkeypatch, apps=[], free_bytes=22 * GIB)
    with pytest.raises(ApiError) as caught:
        guard(
            "cuda-linux",
            model_id="qwen3.8-27b",
            need_bytes=56_368_313_144,
            desktop_allowance_bytes=3 * GIB,
        )
    error = caught.value
    assert error.status_code == 409
    assert error.code == "insufficient_memory"
    # It names both numbers, as section 4 requires.
    assert "needs 52.5 GiB" in error.message
    assert "22.0 GiB free" in error.message
    assert error.details["needed_bytes"] == 56_368_313_144
    assert error.details["free_bytes"] == 22 * GIB


def test_unified_memory_is_checked_the_same_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mac(monkeypatch, available=30 * GIB)
    guard("mlx-darwin", model_id="qwen3.5-9b", need_bytes=19 * GIB)
    with pytest.raises(ApiError) as caught:
        guard("mlx-darwin", model_id="qwen3.8-27b", need_bytes=55_518_912_853)
    assert caught.value.code == "insufficient_memory"
    assert "30.0 GiB free" in caught.value.message


def test_used_unified_memory_is_not_treated_as_a_squatting_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a Mac, 34 GiB "in use" is Chrome and Xcode, not somebody's GPU job.

    The unattributed-VRAM rule exists for a discrete card under WSL2; applying it
    to unified memory would refuse every load on a machine anyone actually uses.
    """
    fake_mac(monkeypatch, available=30 * GIB, total=64 * GIB)
    state = read_state("mlx-darwin", 0)
    assert state.used_bytes == 34 * GIB
    guard("mlx-darwin", model_id="qwen3.5-9b", need_bytes=19 * GIB)


def test_a_probe_that_cannot_answer_never_means_the_card_is_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode() -> list[ComputeApp]:
        raise ProbeError("nvidia-smi did not answer the process list within 30s")

    monkeypatch.setattr(accelerator, "probe_compute_apps", explode)
    with pytest.raises(ApiError) as caught:
        guard("cuda-linux", model_id="qwen3.5-9b", need_bytes=1)
    assert caught.value.code == "accelerator_unreadable"
    assert "did not answer" in caught.value.message


def test_an_unknown_backend_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ApiError) as caught:
        guard("rocm-linux", model_id="qwen3.5-9b", need_bytes=1)
    assert caught.value.code == "accelerator_unreadable"
    assert "not a Crucible backend" in caught.value.message


# ------------------------------------------------------------------- parsing


def test_compute_apps_parses_na_as_unknown_not_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        "1460, [Insufficient Permissions], [N/A]",
        "9001, /usr/bin/python3, 17000",
    ]
    monkeypatch.setattr(accelerator, "_nvidia_smi", lambda query, what: rows)
    apps = accelerator.probe_compute_apps()
    assert apps[0].used_bytes is None
    assert apps[1].used_bytes == 17000 * 1024 * 1024


def test_read_state_reports_what_it_saw(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda(
        monkeypatch,
        apps=[ComputeApp(pid=1, name="a", used_bytes=GIB)],
        free_bytes=20 * GIB,
    )
    state = read_state("cuda-linux", 3 * GIB)
    assert state.used_bytes == 4 * GIB
    assert "20.0 GiB free of 24.0 GiB" in state.detail
    assert state.to_dict()["compute_apps"][0]["pid"] == 1
