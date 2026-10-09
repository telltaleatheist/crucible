from __future__ import annotations

import os
from pathlib import Path

import pytest

from crucible import accelerator
from crucible.accelerator import ComputeApp, ProbeError, guard, read_state
from crucible.errors import ApiError
from crucible.memorybudget import GIB

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


def fake_windows(
    monkeypatch: pytest.MonkeyPatch,
    *,
    apps: list[ComputeApp],
    free_bytes: int,
    total_bytes: int = CARD_TOTAL,
) -> None:
    monkeypatch.setattr(
        accelerator, "nvidia_smi_path", lambda: "C:/Windows/System32/nvidia-smi.exe"
    )
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: list(apps))
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (free_bytes, total_bytes))


WINDOWS_DESKTOP = [
    ComputeApp(pid=1460, name="[Insufficient Permissions]", used_bytes=None),
    ComputeApp(
        pid=6028,
        name=(
            "C:\\WINDOWS\\SystemApps\\MicrosoftWindows.Client.CBS_cw5n1h2txyewy"
            "\\CrossDeviceResume.exe"
        ),
        used_bytes=None,
    ),
    ComputeApp(pid=11208, name="C:\\WINDOWS\\explorer.exe", used_bytes=None),
    ComputeApp(
        pid=2200,
        name="C:\\WINDOWS\\System32\\dwm.exe",
        used_bytes=1_800 * 1024 ** 2,
    ),
    ComputeApp(
        pid=7744,
        name="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        used_bytes=2 * GIB,
    ),
]

DOTS_NEEDS = 5_900_000_000


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
        reclaimable_bytes=20 * GIB,
    )


NARRATOR_PROCESSES = {
    3022: (3022, 3022),
    3053: (3053, 3022),
    4001: (4001, 4001),
}


def test_the_serving_child_of_our_own_narrator_is_not_foreign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda(
        monkeypatch,
        apps=[ComputeApp(pid=3053, name="sglang::scheduler", used_bytes=18 * GIB)],
        free_bytes=6 * GIB,
    )
    monkeypatch.setattr(
        accelerator, "probe_process_table", lambda: dict(NARRATOR_PROCESSES)
    )
    guard(
        "cuda-linux",
        model_id="zeroshot",
        need_bytes=18 * GIB,
        owned_pids=frozenset({3022}),
        desktop_allowance_bytes=0,
        reclaimable_bytes=18 * GIB,
    )


def test_a_trainer_in_its_own_session_is_still_somebody_else_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda(
        monkeypatch,
        apps=[ComputeApp(pid=4001, name="python train_lora.py", used_bytes=13 * GIB)],
        free_bytes=11 * GIB,
    )
    monkeypatch.setattr(
        accelerator, "probe_process_table", lambda: dict(NARRATOR_PROCESSES)
    )
    with pytest.raises(ApiError) as caught:
        guard(
            "cuda-linux",
            model_id="zeroshot",
            need_bytes=18 * GIB,
            owned_pids=frozenset({3022}),
            desktop_allowance_bytes=0,
        )
    assert caught.value.code == "accelerator_busy"
    assert caught.value.details["processes"][0]["pid"] == 4001


def test_a_pid_that_leads_nothing_claims_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda(
        monkeypatch,
        apps=[ComputeApp(pid=5200, name="python train_lora.py", used_bytes=13 * GIB)],
        free_bytes=11 * GIB,
    )
    monkeypatch.setattr(
        accelerator,
        "probe_process_table",
        lambda: {5000: (5000, 5000), 5100: (5000, 5000), 5200: (5000, 5000)},
    )
    with pytest.raises(ApiError) as caught:
        guard(
            "cuda-linux",
            model_id="zeroshot",
            need_bytes=18 * GIB,
            owned_pids=frozenset({5100}),
            desktop_allowance_bytes=0,
        )
    assert caught.value.code == "accelerator_busy"


def test_the_process_table_is_not_read_when_every_app_is_already_ours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda(monkeypatch, apps=[], free_bytes=22 * GIB)

    def refuse() -> dict[int, tuple[int, int]]:
        raise AssertionError("the guard read /proc with nothing to attribute")

    monkeypatch.setattr(accelerator, "probe_process_table", refuse)
    guard(
        "cuda-linux",
        model_id="qwen3.5-9b",
        need_bytes=20 * GIB,
        desktop_allowance_bytes=3 * GIB,
    )


def test_a_proc_that_will_not_parse_is_unreadable_and_never_all_foreign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda(
        monkeypatch,
        apps=[ComputeApp(pid=3053, name="sglang::scheduler", used_bytes=18 * GIB)],
        free_bytes=6 * GIB,
    )

    def broken() -> dict[int, tuple[int, int]]:
        raise ProbeError("could not parse /proc/3053/stat")

    monkeypatch.setattr(accelerator, "probe_process_table", broken)
    with pytest.raises(ApiError) as caught:
        guard(
            "cuda-linux",
            model_id="zeroshot",
            need_bytes=18 * GIB,
            owned_pids=frozenset({3022}),
            desktop_allowance_bytes=0,
        )
    assert caught.value.code == "accelerator_unreadable"


def test_the_process_table_reads_this_very_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not Path("/proc").is_dir():
        pytest.skip("no /proc on this host")
    table = accelerator.probe_process_table()
    mine = os.getpid()
    assert mine in table
    assert table[mine] == (os.getpgid(mine), os.getsid(mine))


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
    assert "pid 12769" in error.message
    assert "sgl-omni" in error.message
    assert "17.0 GiB" in error.message
    assert "never evicts" in error.message
    assert error.details["processes"][0]["pid"] == 12769


def test_a_process_whose_memory_the_driver_hides_is_still_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    fake_cuda(monkeypatch, apps=[], free_bytes=CARD_TOTAL - int(2.3 * GIB))
    guard(
        "cuda-linux",
        model_id="qwen3.5-9b",
        need_bytes=20 * GIB,
        desktop_allowance_bytes=3 * GIB,
    )


def test_the_27b_on_a_card_without_room_is_insufficient_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda(monkeypatch, apps=[], free_bytes=16 * GIB, total_bytes=16 * GIB)
    with pytest.raises(ApiError) as caught:
        guard(
            "cuda-linux",
            model_id="qwen3.8-27b-4bit",
            need_bytes=21_633_171_456,
            desktop_allowance_bytes=3 * GIB,
        )
    error = caught.value
    assert error.status_code == 409
    assert error.code == "insufficient_memory"
    assert "needs 20.1 GiB" in error.message
    assert "16.0 GiB free" in error.message
    assert error.details["needed_bytes"] == 21_633_171_456
    assert error.details["free_bytes"] == 16 * GIB


def test_unified_memory_is_sized_not_sampled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mac(monkeypatch, available=int(37.5 * GIB))
    allowance = 16 * GIB
    state = guard(
        "mlx-darwin",
        model_id="qwen3.8-27b-8bit",
        need_bytes=47_320_162_000,
        desktop_allowance_bytes=allowance,
    )
    assert state.free_bytes == int(37.5 * GIB), "the sample is still reported"


def test_unified_memory_still_refuses_what_the_pool_cannot_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mac(monkeypatch, available=63 * GIB)
    with pytest.raises(ApiError) as caught:
        guard(
            "mlx-darwin",
            model_id="qwen3.8-27b-bf16",
            need_bytes=55_518_912_853,
            desktop_allowance_bytes=16 * GIB,
        )
    assert caught.value.code == "insufficient_memory"
    assert "gives a model 48.0 GiB" in caught.value.message
    assert caught.value.details["room_bytes"] == 48 * GIB


def test_the_guard_and_the_walk_answer_the_mac_the_same_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.memorybudget import available_bytes

    total, allowance = 64 * GIB, 16 * GIB
    fake_mac(monkeypatch, available=1 * GIB, total=total)
    budget = available_bytes(total, allowance)
    guard("mlx-darwin", model_id="m", need_bytes=budget, desktop_allowance_bytes=allowance)
    with pytest.raises(ApiError):
        guard("mlx-darwin", model_id="m", need_bytes=budget + 1,
              desktop_allowance_bytes=allowance)


def test_used_unified_memory_is_not_treated_as_a_squatting_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_unattributed_never_goes_below_zero() -> None:
    state = accelerator.AcceleratorState(
        backend=accelerator.CUDA_LINUX,
        total_bytes=24 * GIB,
        free_bytes=24 * GIB - 1_711_276_032,
        compute_apps=(),
        detail="idle card, desktop only",
    )
    assert state.used_bytes == 1_711_276_032
    assert accelerator.unattributed_bytes(state, 3 * GIB) == 0


def test_an_unmeasured_card_reserve_scales_with_the_card_and_the_mac_reserve_scales() -> None:
    """Owen 2026-10-09 (docs/VERB-SIZING.md 1b): plan close to the card's limit. An
    unmeasured card keeps an eighth of itself, between 1 GiB and 3 GiB: the 24 GiB PC
    keeps the 3 GiB it always had, and an 8 GiB card keeps 1 GiB, not 3."""
    from crucible.config import (
        DEFAULT_DESKTOP_ALLOWANCE_BYTES,
        default_desktop_allowance_bytes,
    )

    expected = {6 * GIB: 1 * GIB, 8 * GIB: 1 * GIB, 12 * GIB: GIB * 3 // 2,
                16 * GIB: 2 * GIB, 24 * GIB: 3 * GIB, 80 * GIB: 3 * GIB}
    for backend in ("cuda-linux", "llama-windows"):
        for total, reserve in expected.items():
            assert default_desktop_allowance_bytes(backend, total, "nvidia") == reserve
    assert DEFAULT_DESKTOP_ALLOWANCE_BYTES == 3 * GIB

    for total in (8 * GIB, 64 * GIB):
        assert (
            default_desktop_allowance_bytes("llama-windows", total, "cpu")
            == DEFAULT_DESKTOP_ALLOWANCE_BYTES
        ), "a CPU build's pool is system memory, shared with the OS: the flat 3 GiB"

    small = default_desktop_allowance_bytes("mlx-darwin", 16 * GIB, "apple")
    large = default_desktop_allowance_bytes("mlx-darwin", 192 * GIB, "apple")
    assert large == 12 * small


def test_a_windows_desktop_is_not_a_holder_and_the_load_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_windows(monkeypatch, apps=WINDOWS_DESKTOP, free_bytes=20 * GIB)
    state = guard(
        "llama-windows",
        model_id="dots-ocr",
        need_bytes=DOTS_NEEDS,
        desktop_allowance_bytes=3 * GIB,
    )
    assert state.backend == "llama-windows"
    assert state.free_bytes == 20 * GIB
    assert [app.pid for app in state.compute_apps] == [1460, 6028, 11208, 2200, 7744]


def test_no_room_on_windows_is_refused_by_name_with_both_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_windows(monkeypatch, apps=WINDOWS_DESKTOP, free_bytes=3 * GIB)
    with pytest.raises(ApiError) as caught:
        guard(
            "llama-windows",
            model_id="dots-ocr",
            need_bytes=DOTS_NEEDS,
            desktop_allowance_bytes=3 * GIB,
        )
    error = caught.value
    assert error.status_code == 409
    assert error.code == "insufficient_memory"
    assert "needs 5.5 GiB" in error.message
    assert "3.0 GiB free of 24.0 GiB" in error.message
    assert error.details["needed_bytes"] == DOTS_NEEDS
    assert error.details["free_bytes"] == 3 * GIB
    assert [entry["pid"] for entry in error.details["processes"]] == [
        1460,
        6028,
        11208,
        2200,
        7744,
    ]
    assert "explorer" not in error.message
    assert "Insufficient Permissions" not in error.message
    assert "held by" not in error.message


def test_a_stray_llama_server_on_windows_is_accelerator_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orphan = ComputeApp(
        pid=31337,
        name="C:\\Users\\tellt\\AppData\\Local\\Crucible\\engine\\llama-server.exe",
        used_bytes=6 * GIB,
    )
    fake_windows(monkeypatch, apps=[*WINDOWS_DESKTOP, orphan], free_bytes=20 * GIB)
    with pytest.raises(ApiError) as caught:
        guard(
            "llama-windows",
            model_id="dots-ocr",
            need_bytes=DOTS_NEEDS,
            desktop_allowance_bytes=3 * GIB,
        )
    error = caught.value
    assert error.status_code == 409
    assert error.code == "accelerator_busy"
    assert "pid 31337" in error.message
    assert "llama-server" in error.message
    assert "left behind by an earlier run" in error.message
    assert "never evicts" in error.message
    assert [entry["pid"] for entry in error.details["processes"]] == [31337]


def test_crucibles_own_llama_server_child_is_not_a_stray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mine = ComputeApp(
        pid=4242,
        name="C:\\Users\\tellt\\AppData\\Local\\Crucible\\engine\\llama-server.exe",
        used_bytes=9 * GIB,
    )
    fake_windows(monkeypatch, apps=[*WINDOWS_DESKTOP, mine], free_bytes=3 * GIB)
    guard(
        "llama-windows",
        model_id="dots-ocr",
        need_bytes=DOTS_NEEDS,
        owned_pids=frozenset({4242}),
        desktop_allowance_bytes=3 * GIB,
        reclaimable_bytes=9 * GIB,
    )


def test_the_same_desktop_under_cuda_linux_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda(monkeypatch, apps=WINDOWS_DESKTOP, free_bytes=20 * GIB)
    with pytest.raises(ApiError) as caught:
        guard(
            "cuda-linux",
            model_id="dots-ocr",
            need_bytes=DOTS_NEEDS,
            desktop_allowance_bytes=3 * GIB,
        )
    assert caught.value.code == "accelerator_busy"
    assert "pid 1460" in caught.value.message


def test_a_llama_server_is_found_by_image_name_never_by_pid() -> None:
    assert accelerator.is_llama_server(
        "C:\\Users\\tellt\\AppData\\Local\\Crucible\\engine\\llama-server.exe"
    )
    assert accelerator.is_llama_server("C:\\Engine\\LLAMA-SERVER.EXE")
    assert accelerator.is_llama_server("/opt/llama.cpp/build/bin/llama-server")
    assert accelerator.is_llama_server("llama-server")
    assert not accelerator.is_llama_server("C:\\WINDOWS\\System32\\dwm.exe")
    assert not accelerator.is_llama_server("[Insufficient Permissions]")
    assert not accelerator.is_llama_server("C:\\Engine\\llama-server-bench.exe")
    assert not accelerator.is_llama_server("C:\\Engine\\llama-cli.exe")


def test_read_windows_state_survives_rows_the_driver_will_not_fill_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        "1460, [Insufficient Permissions], [N/A]",
        "6028, C:\\WINDOWS\\SystemApps\\MicrosoftWindows.Client.CBS_cw5n1h2txyewy"
        "\\CrossDeviceResume.exe, [N/A]",
        "11208, C:\\WINDOWS\\explorer.exe, [N/A]",
    ]
    monkeypatch.setattr(
        accelerator, "nvidia_smi_path", lambda: "C:/Windows/System32/nvidia-smi.exe"
    )
    monkeypatch.setattr(accelerator, "_nvidia_smi", lambda query, what: rows)
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (20 * GIB, CARD_TOTAL))
    state = accelerator.read_windows_state(3 * GIB)
    assert state.backend == "llama-windows"
    assert [app.used_bytes for app in state.compute_apps] == [None, None, None]
    assert state.compute_apps[1].name.endswith("CrossDeviceResume.exe")
    assert "3 compute app(s)" in state.detail


def test_the_mac_reserve_is_a_share_and_leaves_macos_a_quarter() -> None:
    from crucible.config import default_desktop_allowance_bytes

    total = 64 * 1000 ** 3
    reserve = default_desktop_allowance_bytes("mlx-darwin", total, "apple")
    available = total - reserve
    assert reserve == total // 4, "the reserve stopped being a quarter"

    eightbit_27b = 41_688_522_448
    fourbit_27b = 33_873_484_870
    assert eightbit_27b < available, (
        "the 8-bit must fit — it is what the Mac is meant to translate on, and "
        "the reason the bf16 was dropped rather than kept as an option"
    )
    assert fourbit_27b < available, "the 4-bit must still fit — it is the fallback"
    assert eightbit_27b > fourbit_27b
