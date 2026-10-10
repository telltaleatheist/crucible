from __future__ import annotations

from pathlib import Path

from crucible.workerexit import OomCount, how_it_ended, read_oom_count

EVENTS = "low 0\nhigh 0\nmax 0\noom 1\noom_kill {n}\noom_group_kill 0\n"


def _host(tmp_path: Path, *, cgroup: str | None, kills: int | None, vmstat: int | None) -> dict:
    proc_cgroup = tmp_path / "cgroup"
    if cgroup is not None:
        proc_cgroup.write_text(f"0::{cgroup}\n", encoding="utf-8")
    root = tmp_path / "sys-fs-cgroup"
    if cgroup is not None and kills is not None:
        events = root / cgroup.lstrip("/") / "memory.events"
        events.parent.mkdir(parents=True)
        events.write_text(EVENTS.format(n=kills), encoding="utf-8")
    vm = tmp_path / "vmstat"
    if vmstat is not None:
        vm.write_text(f"pgfault 12\noom_kill {vmstat}\n", encoding="utf-8")
    return {"proc_cgroup": proc_cgroup, "cgroup_root": root, "proc_vmstat": vm}


def test_the_count_is_read_from_the_servers_own_cgroup(tmp_path: Path) -> None:
    host = _host(tmp_path, cgroup="/system.slice/crucible.service", kills=2, vmstat=9)
    found = read_oom_count(**host)
    assert found is not None
    assert found.kills == 2
    assert found.source.endswith("system.slice/crucible.service/memory.events")


def test_without_a_cgroup_count_the_hosts_is_read(tmp_path: Path) -> None:
    host = _host(tmp_path, cgroup="/user.slice", kills=None, vmstat=9)
    found = read_oom_count(**host)
    assert found == OomCount(9, str(host["proc_vmstat"]))


def test_with_neither_there_is_no_count(tmp_path: Path) -> None:
    assert read_oom_count(**_host(tmp_path, cgroup=None, kills=None, vmstat=None)) is None


SOURCE = "/sys/fs/cgroup/system.slice/crucible.service/memory.events"


def test_a_sigkill_while_the_count_moved_is_the_out_of_memory_killer() -> None:
    ending = how_it_ended(-9, OomCount(0, SOURCE), OomCount(1, SOURCE))
    assert ending.phrase == "was killed by the out-of-memory killer (SIGKILL, signal 9)"
    assert "went from 0 to 1" in ending.why and SOURCE in ending.why


def test_a_sigkill_while_the_count_stayed_is_someone_else() -> None:
    ending = how_it_ended(-9, OomCount(4, SOURCE), OomCount(4, SOURCE))
    assert ending.phrase == "was killed (SIGKILL, signal 9)"
    assert "not the out-of-memory killer" in ending.why and "stayed at 4" in ending.why


def test_a_sigkill_with_no_count_asks_rather_than_says() -> None:
    ending = how_it_ended(-9, None, None)
    assert ending.phrase == "was killed (SIGKILL, signal 9; out of memory?)"
    assert "no out-of-memory count" in ending.why


def test_other_signals_and_exit_codes_are_named() -> None:
    assert how_it_ended(-11, None, None).phrase == "was killed by a signal (SIGSEGV, signal 11)"
    assert how_it_ended(1, None, None).phrase == "exited 1"
    assert how_it_ended(1, None, None).sentence() == ""
