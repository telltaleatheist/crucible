from __future__ import annotations

import sys
from pathlib import Path

import pytest

from crucible import accelerator, ladder, procgroup

from .conftest import end_process_tree

DEAF = (
    "import signal, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "if hasattr(signal, 'SIGBREAK'):\n"
    "    signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
    "time.sleep(60)\n"
)


@pytest.fixture(autouse=True)
def quick_clocks(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_card(*args: object) -> list[str]:
        raise accelerator.ProbeError("no card in this test")

    monkeypatch.setattr(accelerator, "_nvidia_smi", no_card)
    monkeypatch.setattr(ladder, "SMOKE_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(procgroup, "STOP_TIMEOUT_SECONDS", 2.0)


def test_a_smoke_past_its_clock_is_asked_to_stop_not_killed() -> None:
    result, why, _ = ladder._run_script(
        Path(sys.executable), "import time\ntime.sleep(60)\n"
    )
    assert result is None
    assert why == "did not finish within 1 s"


def test_a_smoke_deaf_to_sigterm_is_left_running_and_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[int] = []
    real_popen = ladder.subprocess.Popen

    def watched(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process.pid)
        return process

    monkeypatch.setattr(ladder.subprocess, "Popen", watched)
    try:
        result, why, _ = ladder._run_script(Path(sys.executable), DEAF)
        assert result is None
        assert f"`kill {spawned[0]}` (never -9)" in why
        assert "does not SIGKILL" in why
    finally:
        monkeypatch.undo()
        for pid in list(spawned):
            end_process_tree(pid)


# The watch's attribution. Under WSL2 the guest's nvidia-smi lists each CUDA process
# in the VM by its guest pid, named "[Not Found]" with no memory (measured on the PC
# 2026-10-11: a trainer listed as "1208643, [Not Found], [N/A]" was /proc/1208643 in
# the guest), so wsl_row is what every row looks like there.
ENGINE_SESSION = {
    3022: (3022, 3022),
    3053: (3022, 3022),
    4001: (4001, 4001),
}


def fake_card(
    monkeypatch: pytest.MonkeyPatch,
    apps: list[accelerator.ComputeApp],
    table: dict[int, tuple[int, int]] | None = None,
) -> None:
    monkeypatch.setattr(accelerator, "_nvidia_smi", lambda *args: ["6000, 40"])
    monkeypatch.setattr(accelerator, "probe_compute_apps", lambda: list(apps))
    session = ENGINE_SESSION if table is None else table
    monkeypatch.setattr(accelerator, "probe_process_table", lambda: dict(session))


def wsl_row(pid: int) -> accelerator.ComputeApp:
    return accelerator.ComputeApp(pid=pid, name="[Not Found]", used_bytes=None)


def test_the_engine_core_child_of_the_rung_s_own_vllm_is_not_foreign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_card(monkeypatch, [wsl_row(3053)])
    watch = ladder.Watch(lambda: frozenset({3022}))
    watch.sample()
    assert watch.foreign == {}
    assert watch.samples == 1


def test_the_engine_core_is_still_the_rung_s_after_its_api_server_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = [frozenset({3022})]
    fake_card(monkeypatch, [wsl_row(3053)])
    watch = ladder.Watch(lambda: owned[0])
    watch.sample()
    owned[0] = frozenset()
    monkeypatch.setattr(
        accelerator, "probe_process_table", lambda: {3053: (3022, 3022), 4001: (4001, 4001)}
    )
    watch.sample()
    assert watch.foreign == {}
    assert watch.samples == 2


def test_a_leader_never_seen_alive_claims_no_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_card(monkeypatch, [wsl_row(3053)], {3053: (3022, 3022)})
    watch = ladder.Watch(lambda: frozenset({3022}))
    watch.sample()
    assert list(watch.foreign) == [3053]


def test_a_wsl_row_from_another_session_is_foreign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_card(monkeypatch, [wsl_row(3053), wsl_row(4001)])
    watch = ladder.Watch(lambda: frozenset({3022}))
    watch.sample()
    assert watch.foreign == {4001: "pid 4001 ([Not Found], memory not reported)"}


def test_a_foreign_process_with_its_memory_reported_is_foreign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = accelerator.ComputeApp(
        pid=4001, name="python train_lora.py", used_bytes=13 * accelerator.GIB
    )
    fake_card(monkeypatch, [trainer])
    watch = ladder.Watch(lambda: frozenset({3022}))
    watch.sample()
    assert list(watch.foreign) == [4001]


def test_a_process_table_that_will_not_parse_is_an_unread_sample_not_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_card(monkeypatch, [wsl_row(3053)])

    def broken() -> dict[int, tuple[int, int]]:
        raise accelerator.ProbeError("could not parse /proc/3053/stat")

    monkeypatch.setattr(accelerator, "probe_process_table", broken)
    watch = ladder.Watch(lambda: frozenset({3022}))
    with watch:
        for _ in range(100):
            if watch.unread:
                break
            ladder.time.sleep(0.05)
    assert watch.foreign == {}
    assert watch.samples == 0
    assert list(watch.summary()["unread_samples"]) == ["could not parse /proc/3053/stat"]


SPAWNS_A_CORE = (
    "import multiprocessing, time\n"
    "def core():\n"
    "    time.sleep(60)\n"
    "if __name__ == '__main__':\n"
    "    child = multiprocessing.get_context('spawn').Process(target=core)\n"
    "    child.start()\n"
    "    print(child.pid, flush=True)\n"
    "    time.sleep(60)\n"
)


def test_a_spawned_child_of_a_real_leader_is_the_rung_s_and_a_stranger_is_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """vLLM 0.28's CoreEngineProcManager starts EngineCore with
    get_mp_context().Process and no setsid; this is that shape, on the real /proc."""
    if not Path("/proc").is_dir():
        pytest.skip("no /proc on this host")
    script = tmp_path / "leader.py"
    script.write_text(SPAWNS_A_CORE, encoding="utf-8")
    leader = ladder.subprocess.Popen(
        [sys.executable, str(script)],
        stdout=ladder.subprocess.PIPE,
        text=True,
        **procgroup.own_group(),
    )
    try:
        core = int(leader.stdout.readline())
        monkeypatch.setattr(accelerator, "_nvidia_smi", lambda *args: ["6000, 40"])
        monkeypatch.setattr(
            accelerator, "probe_compute_apps", lambda: [wsl_row(core), wsl_row(1)]
        )
        watch = ladder.Watch(lambda: frozenset({leader.pid}))
        watch.sample()
        assert core != leader.pid
        assert list(watch.foreign) == [1]
    finally:
        end_process_tree(leader.pid)
        leader.wait(timeout=30)
