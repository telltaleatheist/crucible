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


def fake_windows(
    monkeypatch: pytest.MonkeyPatch,
    *,
    apps: list[ComputeApp],
    free_bytes: int,
    total_bytes: int = CARD_TOTAL,
) -> None:
    """A Windows host WITH an NVIDIA driver: `read_windows_state` takes the card arm."""
    monkeypatch.setattr(
        accelerator, "nvidia_smi_path", lambda: "C:/Windows/System32/nvidia-smi.exe"
    )
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: list(apps))
    monkeypatch.setattr(accelerator, "probe_vram", lambda: (free_bytes, total_bytes))


#: What the Phase 15 button's T7 actually read off Owen's desktop, 2026-09-14 —
#: the compositor, the shell, a system app, and one row the driver would only
#: call `[Insufficient Permissions]` with no memory figure at all.
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

#: `dots-ocr`'s declared estimate on `llama-windows`: the GGUF pair plus
#: Foundry's overhead, ~5.9 GB (PHASE15-HOST.md 3.10 fact 2).
DOTS_NEEDS = 5_900_000_000


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


def test_unattributed_never_goes_below_zero() -> None:
    """An idle card with a generous desktop allowance is not owed VRAM.

    Measured on Owen's PC, 2026-09-13: the 3090 Ti held 1.71 GB with no compute
    app listed, against the configured 3.0 GiB desktop allowance, and the probe
    published `unattributed_bytes: -1509949440`. The guard never saw it because
    it only asks whether the figure clears a floor — but `GET /v1/accelerator`
    publishes it, and a client sizing a load against a negative number is reading
    headroom that does not exist.
    """
    state = accelerator.AcceleratorState(
        backend=accelerator.CUDA_LINUX,
        total_bytes=24 * GIB,
        free_bytes=24 * GIB - 1_711_276_032,
        compute_apps=(),
        detail="idle card, desktop only",
    )
    assert state.used_bytes == 1_711_276_032
    assert accelerator.unattributed_bytes(state, 3 * GIB) == 0


# ------------------------------------- the host reserve, which is per backend


def test_the_cuda_reserve_is_flat_and_the_mac_reserve_scales() -> None:
    """A discrete card and a unified pool are not one question.

    On `cuda-linux` the desktop's appetite does not grow with the card, so the
    reserve is a fixed byte count. On `mlx-darwin` the reserve has to cover the
    whole operating system out of the same pool the model allocates from, so it
    is a share — 3 GiB is defensible on a 16 GB Mac mini and absurd on a 192 GB
    Studio.
    """
    from crucible.config import (
        DEFAULT_DESKTOP_ALLOWANCE_BYTES,
        default_desktop_allowance_bytes,
    )

    for total in (12 * GIB, 24 * GIB, 80 * GIB):
        assert (
            default_desktop_allowance_bytes("cuda-linux", total)
            == DEFAULT_DESKTOP_ALLOWANCE_BYTES
        )

    small = default_desktop_allowance_bytes("mlx-darwin", 16 * GIB)
    large = default_desktop_allowance_bytes("mlx-darwin", 192 * GIB)
    assert large == 12 * small


# ------------------------- `llama-windows`: the card is SHARED by design


def test_a_windows_desktop_is_not_a_holder_and_the_load_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The T7 failure, 2026-09-14, and the rule that replaces it.

    The staged Windows server refused `load-model dots-ocr` with
    `accelerator_busy` naming dwm, explorer, SearchHost, CrossDeviceResume and
    an `[Insufficient Permissions]` row. Every one of those is Windows drawing
    a desktop. On this backend the card is shared by design, so the guard asks
    whether there is ROOM, not whether the card is untouched.
    """
    fake_windows(monkeypatch, apps=WINDOWS_DESKTOP, free_bytes=20 * GIB)
    state = guard(
        "llama-windows",
        model_id="dots-ocr",
        need_bytes=DOTS_NEEDS,
        desktop_allowance_bytes=3 * GIB,
    )
    assert state.backend == "llama-windows"
    assert state.free_bytes == 20 * GIB
    # And the neighbours are still SEEN — reported, never the reason.
    assert [app.pid for app in state.compute_apps] == [1460, 6028, 11208, 2200, 7744]


def test_no_room_on_windows_is_refused_by_name_with_both_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shortfall keeps the name it has always had: `insufficient_memory`.

    `capability.py` already points at this module for "is there room right
    now" and names that refusal; a second spelling of one refusal would be a
    fact with two owners.
    """
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
    # The desktop is REPORTED in the details...
    assert [entry["pid"] for entry in error.details["processes"]] == [
        1460,
        6028,
        11208,
        2200,
        7744,
    ]
    # ...and is nowhere in the reason.
    assert "explorer" not in error.message
    assert "Insufficient Permissions" not in error.message
    assert "held by" not in error.message


def test_a_stray_llama_server_on_windows_is_accelerator_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One image name still holds the card: our own engine, outliving its run.

    There is room (20 GiB free against 5.9 GB needed), so this cannot be the
    memory check catching it — it is the image, and only the image.
    """
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
    # ONLY the llama-server. The desktop is not a holder even in this refusal.
    assert [entry["pid"] for entry in error.details["processes"]] == [31337]


def test_crucibles_own_llama_server_child_is_not_a_stray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`owned_pids` still comes first: a resident engine of ours is ours."""
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
        # Unloading our 9 GiB resident to make room for this one.
        reclaimable_bytes=9 * GIB,
    )


def test_the_same_desktop_under_cuda_linux_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other two backends' guards are untouched, and this is the pin.

    Inside WSL2 a foreign compute app on the card is a trainer or another
    engine, and `accelerator_busy` is the right answer. The `llama-windows`
    rule is a rule about a Windows DESKTOP, not a softening of the guard.
    """
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
    """The pid of a crashed run is not knowable; the image is."""
    assert accelerator.is_llama_server(
        "C:\\Users\\tellt\\AppData\\Local\\Crucible\\engine\\llama-server.exe"
    )
    assert accelerator.is_llama_server("C:\\Engine\\LLAMA-SERVER.EXE")
    assert accelerator.is_llama_server("/opt/llama.cpp/build/bin/llama-server")
    assert accelerator.is_llama_server("llama-server")
    assert not accelerator.is_llama_server("C:\\WINDOWS\\System32\\dwm.exe")
    assert not accelerator.is_llama_server("[Insufficient Permissions]")
    # Not a prefix match: a different binary is a different binary.
    assert not accelerator.is_llama_server("C:\\Engine\\llama-server-bench.exe")
    assert not accelerator.is_llama_server("C:\\Engine\\llama-cli.exe")


def test_read_windows_state_survives_rows_the_driver_will_not_fill_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly the CSV the Windows driver printed under T7, parsed end to end."""
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


def test_the_mac_reserve_selects_the_4bit_27b_owen_already_runs() -> None:
    """The rule is checked against a known-good answer, not just written.

    Owen has translated with a 4-bit 27B on the 64 GB Studio for months. Under a
    flat 3 GiB reserve a best-first walk selects the **bf16** 27B instead and
    leaves macOS 8.5 GB — PHASE9-CAPABILITY.md section 1.1 is the record of that
    disagreement, and of the finding that the rule was wrong rather than the
    operator. This is the test that keeps it that way.
    """
    from crucible.config import default_desktop_allowance_bytes

    total = 64 * 1000 ** 3  # 64 GB as Apple counts it
    available = total - default_desktop_allowance_bytes("mlx-darwin", total)
    bf16_27b = 55.5 * 1000 ** 3
    fourbit_27b = 33.9 * 1000 ** 3
    assert bf16_27b > available, "bf16 must NOT fit — it leaves macOS nothing"
    assert fourbit_27b < available, "the 4-bit must fit — it is what he runs"
