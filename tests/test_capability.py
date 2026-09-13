"""Install-time capability selection (PHASE9-CAPABILITY.md).

Section 1.1 of that document is the record of a selection rule that was written,
checked against a decision Owen had already made, and found to disagree with him —
and of the finding that **the disagreement was the rule's, not his**. So the first
two tests here are not about the code: they are the two answers Owen has been
running for months, asserted through the real selector. Everything after them is
the machinery that has to keep producing those two answers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible import capability, cli
from crucible.capability import (
    CLASSES,
    BY_NAME,
    available_bytes,
    classes_for_job_type,
    decide,
    decide_all,
    job_type_enabled,
)
from crucible.config import (
    CapabilityRecord,
    CapabilityRow,
    config_path,
    default_desktop_allowance_bytes,
    load_config,
    write_config,
)
from crucible.errors import ApiError, ConfigError
from crucible.jobs import disabled_error

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

GIB = 1024 ** 3
CUDA_RESERVE = 3 * GIB

#: Owen's 3090 Ti, exactly as `nvidia-smi --query-gpu=memory.total` reports it.
THREE_NINETY = FAKE_BACKEND.gpu.vram_bytes
#: The 64 GiB Mac Studio, as `sysctl hw.memsize` reports it.
STUDIO = FAKE_MAC_BACKEND.gpu.vram_bytes
MAC_RESERVE = default_desktop_allowance_bytes("mlx-darwin", STUDIO)

#: A card too small for anything a 9B-class model needs. Owen's example in the
#: ruling: *"if i run crucible on a 6 gb gpu…"*.
SIX_GIG = 6 * GIB


def _decide(name: str, backend: str, total: int, reserve: int) -> Any:
    return decide(
        BY_NAME[name], backend, total_bytes=total, desktop_allowance_bytes=reserve
    )


# ------------------------------------------------ the two known-good answers


def test_the_3090ti_translates_which_is_what_owen_already_does() -> None:
    """*"translate works just fine on the pc right now."* — Owen, 2026-09-13.

    The arithmetic is tight: 20.1 GiB of model against 21.0 GiB of budget. It is
    the tightest fit in the build, which makes it the check worth keeping — any
    invented safety margin bolted onto the fit test disables translation on a card
    that has been translating for months, and that is the phase doc's section 1.1
    failure happening a second time on the other backend.
    """
    verdict = _decide("translate", "cuda-linux", THREE_NINETY, CUDA_RESERVE)
    assert verdict.enabled is True
    assert verdict.selected == "qwen3.8-27b-4bit"
    assert verdict.shortfall_bytes == 0


def test_the_mac_selects_the_4bit_27b_and_refuses_the_bf16() -> None:
    """Owen translates on a 4-bit 27B on the 64 GB Studio, and has for months.

    A flat 3 GiB reserve would let a best-first walk take the **bf16** 27B at
    55.5 GB and leave macOS 8.5 GB out of the one pool everything allocates from.
    `default_desktop_allowance_bytes` reserves 25% there instead, and this asserts
    the consequence rather than the constant: the walk sees 48 GiB, bf16 does not
    fit, and the 4-bit does.
    """
    verdict = _decide("translate", "mlx-darwin", STUDIO, MAC_RESERVE)
    assert verdict.enabled is True
    assert verdict.selected == "qwen3.8-27b-4bit"
    ids = [c.id for c in verdict.candidates]
    assert ids[0] == "qwen3.8-27b", "the walk must SEE the bf16 and refuse it"
    assert verdict.fit_count == 1


# -------------------------------------------------------------- the ordering


def test_best_precision_first_not_smallest_that_fits() -> None:
    """A 4-bit translation is a worse translation. Given room, take the bf16.

    The order is read off `memory_bytes_estimate` descending rather than off a
    `precision` field, because the manifests already declare the size and a second
    field saying the same thing in other units is a second owner of one fact
    (ARCHITECTURE.md R1).
    """
    verdict = _decide("translate", "cuda-linux", 200 * GIB, CUDA_RESERVE)
    assert verdict.selected == "qwen3.8-27b", (
        "with room for both, the rule must take the better one, not the smaller"
    )
    assert verdict.fit_count == 2


def test_the_selected_id_does_not_depend_on_catalog_order() -> None:
    """Every voice declares the same 19 GB; every RVC model the same 2.5 GiB.

    Without a second sort key the "selected" id would be whichever the dict
    happened to yield first, and two runs on one host could record different
    answers to the same question. The tie-break is the id.
    """
    for name in ("tts", "rvc"):
        first = _decide(name, "cuda-linux", THREE_NINETY, CUDA_RESERVE)
        again = _decide(name, "cuda-linux", THREE_NINETY, CUDA_RESERVE)
        sizes = {c.memory_bytes_estimate for c in first.candidates}
        assert len(sizes) == 1, f"{name} is the tie case this test is about"
        assert first.selected == again.selected
        assert first.selected == min(c.id for c in first.candidates)


# ------------------------------------------------------- the binary rulings


def test_higgs_is_binary_and_a_six_gig_card_loses_tts_entirely() -> None:
    """*"if higgs doesnt fit in a card that crucible is installed on, it's
    disabled on that gpu."* — Owen, 2026-09-13.

    No reduced mode, and the refusal has to SAY there is no reduced mode, or the
    next person spends an afternoon looking for the quantized build.
    """
    verdict = _decide("tts", "cuda-linux", SIX_GIG, CUDA_RESERVE)
    assert verdict.enabled is False
    assert verdict.selected == ""
    assert "not quantized" in verdict.reason
    assert job_type_enabled("tts", decide_all(
        "cuda-linux", total_bytes=SIX_GIG, desktop_allowance_bytes=CUDA_RESERVE
    )) is False


def test_translate_is_binary_per_server_and_says_so_when_it_is_off() -> None:
    """*"translation is binary per server… if 27b doesnt fit on the card then it
    cant translate."* — Owen, 2026-09-13."""
    verdict = _decide("translate", "cuda-linux", SIX_GIG, CUDA_RESERVE)
    assert verdict.enabled is False
    assert "cannot translate" in verdict.reason


def test_a_six_gig_card_keeps_llm_only_if_something_behind_it_fits() -> None:
    """The phase doc's section 3 calls a 6 GB box "an `llm` + `rvc` + `asr`
    server". Section 1.1 says the opposite and is right: there is no 4-bit 9B in
    this build, so clean is gone; there is no 27B that small, so translate is
    gone; dots-ocr is 12 GB, so pages is gone. All three classes behind
    `enable_llm` fail, and the flag goes with them.
    """
    decisions = decide_all(
        "cuda-linux", total_bytes=SIX_GIG, desktop_allowance_bytes=CUDA_RESERVE
    )
    by_name = {d.capability: d for d in decisions}
    assert by_name["clean"].enabled is False
    assert by_name["translate"].enabled is False
    assert by_name["pages"].enabled is False
    assert job_type_enabled("llm", decisions) is False
    # And the two the doc got right, which is what makes the disagreement above
    # a finding rather than a quibble.
    assert job_type_enabled("asr", decisions) is True
    assert job_type_enabled("rvc", decisions) is True


def test_llm_survives_when_one_of_its_three_classes_survives() -> None:
    """A host that cleans and cannot translate is an `llm` host, and the per-class
    rows are where it says which half it has. `enable_llm` alone cannot."""
    decisions = decide_all(
        "cuda-linux", total_bytes=24 * GIB, desktop_allowance_bytes=8 * GIB
    )
    by_name = {d.capability: d for d in decisions}
    assert by_name["pages"].enabled is True
    assert by_name["translate"].enabled is False
    assert job_type_enabled("llm", decisions) is True


# ---------------------------------------------- refusing nothing silently


def test_a_disabled_class_records_the_number_that_disabled_it() -> None:
    """Section 2 step 4. The shortfall is a NUMBER on the row, not only a phrase
    inside the sentence: a log line is never load-bearing (ARCHITECTURE.md R4).

    And it is the shortfall of the SMALLEST candidate, because that is the one
    that says how much bigger a card would have to be. `asr` on a 4 GiB card is
    the case that can tell: six whisper models, none of them fitting, and
    `large-v3` is 2.8 GiB further out of reach than `tiny`. Reporting the largest
    would tell an operator to buy three times the card they need.
    """
    tiny_card = 4 * GIB
    verdict = _decide("asr", "cuda-linux", tiny_card, CUDA_RESERVE)
    assert verdict.enabled is False
    assert len(verdict.candidates) > 1, "the point of this test is several"
    sizes = [c.memory_bytes_estimate for c in verdict.candidates]
    assert verdict.shortfall_bytes == min(sizes) - (tiny_card - CUDA_RESERVE)
    assert verdict.shortfall_bytes > 0
    assert verdict.shortfall_bytes < max(sizes) - (tiny_card - CUDA_RESERVE)


def test_asr_and_align_have_no_mac_candidates_and_say_which_it_is() -> None:
    """"nothing this build can serve here" and "it does not fit" are different
    facts about a host, and a reader who cannot tell them apart goes looking for a
    bigger Mac to fix a missing manifest."""
    for name in ("asr", "align"):
        verdict = _decide(name, "mlx-darwin", STUDIO, MAC_RESERVE)
        assert verdict.enabled is False
        assert verdict.candidates == ()
        assert verdict.shortfall_bytes == 0
        assert "ships none with a mlx-darwin block" in verdict.reason


def test_echo_needs_no_accelerator_and_is_never_disabled_by_a_card() -> None:
    verdict = _decide("echo", "cuda-linux", 1, CUDA_RESERVE)
    assert verdict.enabled is True
    assert verdict.candidates == ()


def test_the_budget_never_goes_negative() -> None:
    """An allowance larger than the pool is a misconfiguration, and a NEGATIVE
    budget would report it as a number instead of as nonsense — the same clamp
    `accelerator.unattributed_bytes` carries for the mirror-image case."""
    assert available_bytes(2 * GIB, 3 * GIB) == 0


def test_every_job_type_has_a_capability_class() -> None:
    """The check `crucible/jobs/__init__.py` makes at import, asserted where a
    reader will look for it. A job type nothing decides about is a job type whose
    refusal has no number to name."""
    from crucible.jobs import ALL_JOB_TYPES

    covered = {entry.job_type for entry in CLASSES}
    assert set(ALL_JOB_TYPES.values()) <= covered
    for job_type in set(ALL_JOB_TYPES.values()):
        assert classes_for_job_type(job_type), job_type


def test_an_unknown_backend_is_refused_rather_than_priced() -> None:
    with pytest.raises(ValueError, match="not a Crucible backend"):
        _decide("clean", "cuda-windows", THREE_NINETY, CUDA_RESERVE)


# ------------------------------------------------------- the config record


def _write(home: Path, **overrides: Any) -> None:
    base: dict[str, Any] = {
        "name": "crucible@test",
        "host": "127.0.0.1",
        "port": 7100,
        "token": "token-token-token",
        "backend_kind": "cuda-linux",
        "enable_echo": True,
        "enable_llm": False,
        "enable_asr": False,
        "enable_tts": False,
        "enable_align": False,
        "enable_rvc": False,
        "desktop_allowance_bytes": CUDA_RESERVE,
    }
    base.update(overrides)
    write_config(home, **base)


def test_the_capability_record_round_trips_through_config_toml(home: Path) -> None:
    decisions = decide_all(
        "cuda-linux",
        total_bytes=THREE_NINETY,
        desktop_allowance_bytes=CUDA_RESERVE,
    )
    written = capability.record(
        "cuda-linux",
        total_bytes=THREE_NINETY,
        desktop_allowance_bytes=CUDA_RESERVE,
        decisions=decisions,
    )
    _write(home, capability=written)
    read_back = load_config(home).capability
    assert read_back == written
    assert read_back.row("translate").selected == "qwen3.8-27b-4bit"


def test_a_config_with_no_capability_table_reads_as_NOT_DECIDED(home: Path) -> None:
    """Absent must stay None and must never become an empty record.

    An empty record reads as "the card was probed and nothing fit", which is a
    different and false statement about the host — and it is the statement the
    refusal would then make to a client.
    """
    _write(home)
    config = load_config(home)
    assert config.capability is None
    assert config.flags_absent == (), "the phase-2 compat path is untouched"


def test_a_capability_table_with_an_unknown_key_is_refused(home: Path) -> None:
    _write(home)
    path = config_path(home)
    path.write_text(
        path.read_text(encoding="utf-8")
        + '\n[capability]\nbackend_kind = "cuda-linux"\ntotal_bytes = 1\n'
        "desktop_allowance_bytes = 1\nclasses = []\nmeasured = true\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(home)


def test_a_capability_table_missing_its_classes_is_refused(home: Path) -> None:
    _write(home)
    path = config_path(home)
    path.write_text(
        path.read_text(encoding="utf-8")
        + '\n[capability]\nbackend_kind = "cuda-linux"\ntotal_bytes = 1\n'
        "desktop_allowance_bytes = 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="capability.classes"):
        load_config(home)


def test_one_class_recorded_twice_is_refused(home: Path) -> None:
    _write(home)
    path = config_path(home)
    row = (
        "[[capability.classes]]\ncapability = \"tts\"\nenabled = false\n"
        "selected = \"\"\nreason = \"x\"\nshortfall_bytes = 1\n"
    )
    path.write_text(
        path.read_text(encoding="utf-8")
        + '\n[capability]\nbackend_kind = "cuda-linux"\ntotal_bytes = 1\n'
        "desktop_allowance_bytes = 1\n" + row + row,
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="recorded twice"):
        load_config(home)


# ------------------------------------------------------------ the refusal


def _config_with(home: Path, rows: tuple[CapabilityRow, ...], **flags: Any) -> Any:
    _write(
        home,
        capability=CapabilityRecord(
            backend_kind="cuda-linux",
            total_bytes=SIX_GIG,
            desktop_allowance_bytes=CUDA_RESERVE,
            rows=rows,
        ),
        **flags,
    )
    return load_config(home)


def test_the_refusal_names_the_number_and_never_says_flip_the_flag(
    home: Path,
) -> None:
    """PHASE9-CAPABILITY.md section 2.1. On a card that cannot hold Higgs, *"set
    [jobs] enable_tts = true"* is not merely unhelpful — it is the instruction
    that produces an OOM. A refusal that recommends a fix which cannot work is
    worse than one that just says no.
    """
    verdict = _decide("tts", "cuda-linux", SIX_GIG, CUDA_RESERVE)
    config = _config_with(home, (verdict.row(),))
    error = disabled_error("tts", config)
    assert isinstance(error, ApiError)
    assert error.status_code == 400
    assert error.code == "job_type_disabled"
    assert "not quantized" in error.message
    assert "short by" in error.message
    assert "would not change any of those numbers" in error.message
    assert "= true" not in error.message
    assert error.details["shortfall_bytes"]["tts"] == verdict.shortfall_bytes


def test_the_refusal_says_so_when_nothing_has_decided_anything_here(
    home: Path,
) -> None:
    """The honest answer for a config `crucible init` wrote and nothing probed.
    It must not invent a reason, and it must not tell anybody the flag is safe."""
    _write(home)
    error = disabled_error("tts", load_config(home))
    assert error.details["capability_recorded"] is False
    assert "no capability selection has been recorded" in error.message
    assert "crucible capability" in error.message


def test_the_refusal_offers_install_when_the_card_can_hold_it(home: Path) -> None:
    """The one case with an action that works is the one case that gets an
    action: recorded as fitting, flag off, env not built."""
    verdict = _decide("tts", "cuda-linux", THREE_NINETY, CUDA_RESERVE)
    assert verdict.enabled is True
    config = _config_with(home, (verdict.row(),))
    error = disabled_error("tts", config)
    assert "crucible install tts" in error.message
    assert error.details["fits"] == ["tts"]


def test_the_llm_refusal_reads_every_class_behind_the_flag(home: Path) -> None:
    """`/v1/models` refuses by CAPABILITY name (`llm`) and `POST /v1/jobs` by job
    type (`load-model`); both have to arrive at one sentence, which names all
    three classes rather than whichever one happened to be checked."""
    decisions = decide_all(
        "cuda-linux", total_bytes=SIX_GIG, desktop_allowance_bytes=CUDA_RESERVE
    )
    rows = tuple(d.row() for d in decisions if d.job_type == "llm")
    config = _config_with(home, rows)
    by_capability = disabled_error("llm", config)
    by_job_type = disabled_error("load-model", config)
    for name in ("clean", "translate", "pages"):
        assert name in by_capability.message
    assert by_capability.message.split(" is disabled", 1)[1] == (
        by_job_type.message.split(" is disabled", 1)[1]
    )


def test_the_refusal_refuses_a_name_that_is_neither() -> None:
    with pytest.raises(KeyError, match="neither a job type nor a capability"):
        disabled_error("wildly-made-up", None)


# ------------------------------------------------------------------- the CLI


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "detect_backend", lambda: FAKE_BACKEND)


@pytest.fixture
def tiny_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same host with a 6 GiB card in it, so the falling edge can be tested."""
    from crucible.backend import Backend, Gpu

    monkeypatch.setattr(
        cli,
        "detect_backend",
        lambda: Backend(
            kind="cuda-linux",
            platform="linux",
            arch="x86_64",
            gpu=Gpu(vendor="nvidia", name="NVIDIA T1000", vram_bytes=SIX_GIG),
            detail="test double",
        ),
    )


def test_capability_is_a_dry_run_unless_asked(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-echo"]) == 0
    capsys.readouterr()
    assert cli.main(["capability"]) == 0
    assert "dry run" in capsys.readouterr().out
    assert load_config(home).capability is None


def test_capability_write_records_the_verdict(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-echo"]) == 0
    capsys.readouterr()
    assert cli.main(["capability", "--write", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["written"] is True
    assert payload["job_types"]["tts"] is True
    record = load_config(home).capability
    assert record.total_bytes == THREE_NINETY
    assert record.row("tts").enabled is True


def test_capability_write_turns_a_type_OFF_but_never_ON(
    home: Path, tiny_card: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """A flag means "this server offers this type", which needs the card to fit
    AND the env to exist. Only `crucible install` knows the second, so this door
    is allowed to travel in one direction: off.
    """
    assert cli.main(["init", "--enable-tts", "--enable-rvc"]) == 0
    capsys.readouterr()
    assert cli.main(["capability", "--write"]) == 0
    config = load_config(home)
    assert config.enable_tts is False, "6 GiB cannot hold Higgs"
    assert config.enable_rvc is True, "2.5 GiB fits and was already on"
    assert config.enable_asr is False, "asr fits, but its env was never built"
    assert config.capability.row("asr").enabled is True
    assert config.token == load_config(home).token


def test_install_writes_the_flag_and_the_reason_even_when_it_disables(
    home: Path, tiny_card: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """R6: partial work survives failure. The selection step writes the record and
    the flag and THEN refuses, because the record is the only durable answer to
    "why is tts off on this box" — discarding it to tidy the exit code leaves an
    operator with a refusal and nothing to read.
    """
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    config = load_config(home)
    assert cli._capability_step(config, cli.detect_backend(), "tts") == 1
    out = capsys.readouterr()
    assert "DISABLED" in out.err
    after = load_config(home)
    assert after.enable_tts is False
    assert after.capability.row("tts").shortfall_bytes > 0


def test_install_turns_the_flag_on_when_the_card_holds_it(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    config = load_config(home)
    assert cli._capability_step(config, cli.detect_backend(), "tts") == 0
    after = load_config(home)
    assert after.enable_tts is True
    assert after.capability.row("tts").selected != ""
    assert after.token == config.token, "a capability decision never re-mints"


def test_doctor_reports_a_flag_the_numbers_contradict(
    home: Path, tiny_card: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """`enable_tts = true` with nothing behind it that fits is the dangerous
    direction, and the only one doctor calls a PROBLEM."""
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["capability", "--write"]) == 0
    capsys.readouterr()
    # Put the flag back on behind capability's back, the way a hand edit would.
    path = config_path(home)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "enable_tts = false", "enable_tts = true", 1
        ),
        encoding="utf-8",
    )
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    contradicted = [
        p for p in report["problems"] if p.startswith("capability_contradicted")
    ]
    assert len(contradicted) == 1
    assert "enable_tts" in contradicted[0]


def test_doctor_notices_the_card_was_swapped(
    home: Path, viable: None, tiny_card: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale record is caught by comparing NUMBERS, not by writing down a date:
    when the decision was made says nothing about whether it is still right."""
    assert cli.main(["init", "--enable-echo"]) == 0
    assert cli.main(["capability", "--write"]) == 0
    capsys.readouterr()
    # `viable` ran first and `tiny_card` second, so the record above was decided
    # on the 6 GiB card; put the 3090 Ti back and the record is stale.
    import crucible.cli as cli_module

    cli_module.detect_backend = lambda: FAKE_BACKEND  # type: ignore[assignment]
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert any(p.startswith("capability_stale") for p in report["problems"])
    assert report["capability"]["stale"] is True


def test_doctor_calls_an_unbuilt_but_fitting_type_a_note_not_a_problem(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ordinary state of a fresh host: the card can hold `tts`, the env has
    not been built, `enable_tts` is off. That is not a fault."""
    assert cli.main(["init", "--enable-echo"]) == 0
    assert cli.main(["capability", "--write"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is True
    assert "tts" in report["capability"]["could_enable"]
