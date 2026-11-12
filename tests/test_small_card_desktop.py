"""Victoria's 8 GiB laptop, 2026-11-04: qwen3.5-4b-8bit refused with "This card affords 0
tokens". Her desktop held 1.63 GiB against a 1.00 GiB allowance measured while it sat idle,
so the card's FREE memory set the engine's budget, and the refusal named the allowance as
the cause. These hold the refusal to the term that binds, and the doctor to reporting a
desktop seen above its reserve."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from crucible.accelerator import AcceleratorState
from crucible.cli.doctor import Host, check_desktop_reserve
from crucible.manifests import load_manifest
from crucible.vram import plan_vllm_memory

GIB = 1024**3
TOTAL = 8 * GIB
ALLOWANCE = 1 * GIB
# Her card when the load was refused: "the card has 6.37 GiB free of 8.00 GiB".
FREE = int(6.37 * GIB)


def _plan(free: int, allowance: int = ALLOWANCE):
    spec = load_manifest("qwen3.5-4b-8bit").spec("cuda-linux")
    card = AcceleratorState(
        backend="cuda-linux",
        total_bytes=TOTAL,
        free_bytes=free,
        compute_apps=(),
        detail="fixture",
    )
    plan = plan_vllm_memory(
        model_id="qwen3.5-4b-8bit",
        spec=spec,
        context=8192,
        card=card,
        desktop_allowance_bytes=allowance,
        reclaimable_bytes=0,
    )
    assert plan is not None
    return plan


def test_victorias_refusal_names_the_desktop_over_its_allowance_not_the_allowance():
    plan = _plan(FREE)
    assert not plan.fits
    assert plan.free_binds
    words = plan.sentence()
    assert "the desktop and other processes hold 1.63 GiB" in words
    assert "0.63 GiB more than the desktop's 1.00 GiB allowance" in words
    assert "vLLM's own CUDA context takes 0.94 GiB" in words
    assert "the engine's budget is 5.43 GiB" in words or "the engine's budget is 5.44 GiB" in words
    assert "This card affords 0 tokens" in words
    # What it would take, so the person can act: 5.51 weights and overhead, 0.94 context,
    # 0.37 for one 8192-token request: 16 blocks of 528 tokens, 6 state blocks and the
    # null block, 16.5 MiB each.
    assert "qwen3.5-4b-8bit needs 6.82 GiB free at 8192 tokens" in words
    assert "0.45 GiB more than the card has" in words


def test_the_need_it_names_is_the_need_it_holds_a_load_to():
    short = _plan(FREE)
    enough = _plan(short.free_needed_bytes)
    assert enough.fits
    assert not _plan(short.free_needed_bytes - 1024**2).fits


def test_where_the_allowance_binds_the_sentence_says_so():
    # The desktop at its allowance and nothing else on the card: free is total less the
    # allowance, and the budget is free less vLLM's context, which still binds. A large
    # allowance on a card with plenty free is where the allowance binds.
    plan = _plan(int(7.9 * GIB), allowance=int(2.6 * GIB))
    assert not plan.free_binds
    assert not plan.fits
    words = plan.sentence()
    assert "the card is 8.00 GiB and the desktop's allowance is 2.60 GiB" in words
    assert "needs" not in words.split("This card affords")[1]


def _host(seen: int | None, *, basis: str = "measured", allowance: int = ALLOWANCE) -> Host:
    facts = {} if seen is None else {"desktop_bytes_max": seen}
    return Host(
        home=Path("/nowhere"),
        backend=SimpleNamespace(kind="cuda-linux"),
        backend_refusal=None,
        config=SimpleNamespace(
            desktop_allowance_bytes=allowance,
            desktop_allowance_basis=basis,
            desktop_allowance_note=(
                "desktop held 0.00-0.00 GiB of 8.0 GiB over 5s on 2026-10-09"
            ),
        ),
        config_refusal=None,
        ladder={"rungs": {"card": {"measured_at": "2026-10-24T18:44:42+00:00", "facts": facts}}},
    )


def test_doctor_reports_a_desktop_seen_above_a_measured_reserve():
    # Her ladder card rung, 2026-10-24: desktop_bytes_max 2,035,286,016.
    section = check_desktop_reserve(_host(2_035_286_016))
    assert [f.code for f in section.findings] == ["desktop_over_reserve"]
    finding = section.findings[0]
    assert "up to 1.90 GiB on 2026-10-24" in finding.message
    assert "0.90 GiB more than its 1.00 GiB reserve (measured" in finding.message
    assert "--measure-desktop" in finding.fix


def test_doctor_is_quiet_when_the_desktop_stays_within_its_reserve():
    section = check_desktop_reserve(_host(900 * 1024**2))
    assert section.findings == () and section.notes == ()


def test_doctor_still_notes_an_unmeasured_reserve_larger_than_needed():
    section = check_desktop_reserve(_host(200 * 1024**2, basis="default", allowance=3 * GIB))
    assert section.findings == ()
    assert section.notes and section.notes[0].startswith("desktop_reserve_unmeasured")
