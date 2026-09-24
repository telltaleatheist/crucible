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

from crucible import capability, cli, voices
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


def _decide(
    name: str,
    backend: str,
    total: int,
    reserve: int,
    vendor: str = "nvidia",
    chosen: str | None = None,
) -> Any:
    """One class, decided. `vendor` defaults to the card these tests are about.

    It is a keyword with a default HERE and a required argument in
    `capability.decide` on purpose: production has a `Backend` in hand and
    must state it, while every test below asks about a machine with a card
    and saying so eighteen times would bury the one line that differs.
    """
    return decide(
        BY_NAME[name],
        backend,
        total_bytes=total,
        desktop_allowance_bytes=reserve,
        gpu_vendor=vendor,
        chosen=chosen,
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
    # And the 4-bit is chosen because the 8-bit is not OFFERED here, not
    # because it was walked and refused: Owen, 2026-09-23, *"we shouldnt have
    # an 8 bit 27b on here. waste of space, wont fit in the gpu"*.
    assert "qwen3.8-27b-8bit" not in [c.id for c in verdict.candidates]


def test_the_mac_selects_the_8bit_27b_over_the_4bit() -> None:
    """The 64 GB Studio translates on the 8-bit, and that is the point of it.

    ── What changed on 2026-09-17, and why this test reads the other way now ──

    It was `test_the_mac_selects_the_4bit_27b_and_refuses_the_bf16`, and the
    thing it asserted was a REFUSAL: the walk saw the bf16 27B at 55.5 GB,
    refused it against the 25% reserve, and fell to the 4-bit. Owen removed the
    bf16 — *"I don't think the Mac can fit 27b 16 bit. It fits 8 bit at most.
    We should remove 16 bit and put 8 bit on the Mac. 4 bit for pc"* — and
    `qwen3.8-27b-8bit` took its place at 47_320_162_000.

    That number is the whole difference: 44.07 GiB against the 48.0 GiB this
    reserve leaves, so the largest candidate now FITS and best-first takes it.
    (It is 41_688_522_448 = 38.83 GiB since 2026-09-23, when the KV it counted
    twice came out of its overhead; it fits by more, for the same reason.)
    The Mac stops running a 4-bit translation it never had to.

    ── What is still being asserted ──────────────────────────────────────────

    The same rule, reaching a different answer because the catalog changed
    rather than because the rule did: a 25% share of a 64 GiB machine, a
    best-first walk over `memory_bytes_estimate`, and the consequence rather
    than the constant. The 4-bit is still a candidate and still fits; it is
    simply no longer the best one that does.
    """
    verdict = _decide("translate", "mlx-darwin", STUDIO, MAC_RESERVE)
    assert verdict.enabled is True
    assert verdict.selected == "qwen3.8-27b-8bit", (
        "the Mac must take the 8-bit: it is the best candidate that fits, and "
        "fitting it is the reason the bf16 was replaced rather than dropped"
    )
    ids = [c.id for c in verdict.candidates]
    assert ids[0] == "qwen3.8-27b-8bit", "best-first must still walk largest first"
    # THREE, and the count is the incidental half: the 8-bit, the 4-bit and the
    # 9B all fit this machine. It was two while the largest candidate could not.
    assert verdict.fit_count == 3


# -------------------------------------------------------------- the ordering


def test_best_precision_first_not_smallest_that_fits() -> None:
    """A 4-bit translation is a worse translation. Given room, take the 8-bit.

    The order is read off `memory_bytes_estimate` descending rather than off a
    `precision` field, because the manifests already declare the size and a second
    field saying the same thing in other units is a second owner of one fact
    (ARCHITECTURE.md R1).
    """
    # ON THE MAC since 2026-09-23: the 8-bit 27B has no cuda-linux block any
    # more (Owen: *"we shouldnt have an 8 bit 27b on here"*), so the one
    # backend where an 8-bit and a 4-bit of the same model are both offered is
    # mlx-darwin. A machine far larger than the Studio, so every candidate fits
    # and only the ORDER can decide.
    verdict = _decide("translate", "mlx-darwin", 200 * GIB, MAC_RESERVE)
    assert verdict.selected == "qwen3.8-27b-8bit", (
        "with room for both, the rule must take the better one, not the smaller"
    )
    # THREE SINCE 2026-09-16: the 9B joined translate's candidates. The claim
    # being made here is best-first, and it is sharper now than it was — with
    # three models fitting, a walk that took the smallest would land on the 9B
    # rather than merely on the 4-bit.
    assert verdict.fit_count == 3
    # And on cuda-linux, however large the card, best-first reaches the 4-bit
    # — the best 27B that backend is offered — never the 9B.
    pc = _decide("translate", "cuda-linux", 200 * GIB, CUDA_RESERVE)
    assert pc.selected == "qwen3.8-27b-4bit"
    assert pc.fit_count == 2
    assert "qwen3.8-27b-8bit" not in [c.id for c in pc.candidates]


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
        "cuda-linux",
        total_bytes=SIX_GIG,
        desktop_allowance_bytes=CUDA_RESERVE,
        gpu_vendor="nvidia",
        chosen={},
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
        "cuda-linux",
        total_bytes=SIX_GIG,
        desktop_allowance_bytes=CUDA_RESERVE,
        gpu_vendor="nvidia",
        chosen={},
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
        "cuda-linux",
        total_bytes=24 * GIB,
        desktop_allowance_bytes=8 * GIB,
        gpu_vendor="nvidia",
        chosen={},
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


def test_asr_and_align_are_both_enabled_on_the_studio() -> None:
    """Both stopped being the Mac's missing manifests on 2026-09-14.

    They got there differently and the selections say so: `align` runs the SAME
    engine on `mps`, so the Mac's candidate is the id the card already used;
    `asr` runs a SECOND engine, so its candidates are ids the card has never
    heard of and best-first picks the largest mlx conversion.
    """
    align = _decide("align", "mlx-darwin", STUDIO, MAC_RESERVE)
    assert align.enabled is True
    assert align.selected == "qwen3-aligner"

    asr = _decide("asr", "mlx-darwin", STUDIO, MAC_RESERVE)
    assert asr.enabled is True
    assert asr.selected == "mlx-whisper-large-v3"
    assert all(c.id.startswith("mlx-whisper-") for c in asr.candidates)
    # And the card's six are not among them, which is the id rule showing up
    # where a client would notice it.
    assert not any(c.id.startswith("faster-whisper-") for c in asr.candidates)


def _pages_with_no_block_on_any_backend() -> Any:
    """`pages` as it stood on the Mac until 2026-09-21: the class exists, the
    catalog has the model, and no manifest carries a block for this backend.

    SYNTHETIC since dots-ocr grew its mlx-darwin block, because no shipped
    class reads that way on any shipped backend any more — and the sentence
    still has to exist for the day one does."""
    import dataclasses

    from crucible.capability import CLASSES

    pages = next(entry for entry in CLASSES if entry.name == "pages")
    return dataclasses.replace(pages, candidates=lambda backend_kind: ())


def test_a_class_with_nothing_on_this_backend_still_says_which_it_is() -> None:
    """"nothing this build can serve here" and "it does not fit" are different
    facts about a host, and a reader who cannot tell them apart goes looking
    for a bigger Mac to fix a missing manifest. `pages` read that way on the
    Mac until 2026-09-21 (docs/PHASE15-HOST.md 7c); the state is kept here
    synthetically."""
    from crucible.capability import decide

    verdict = decide(
        _pages_with_no_block_on_any_backend(), "mlx-darwin",
        total_bytes=STUDIO, desktop_allowance_bytes=MAC_RESERVE,
        gpu_vendor="apple", chosen=None,
    )
    assert verdict.enabled is False
    assert verdict.candidates == ()
    assert verdict.shortfall_bytes == 0
    assert "ships none with a mlx-darwin block" in verdict.reason


def test_pages_on_the_mac_is_enabled_since_the_block_landed() -> None:
    """The other side of the test above, and the whole point of 2026-09-21:
    a 64 GiB Mac reads pages with dots-ocr, and says so in both voices."""
    verdict = _decide("pages", "mlx-darwin", STUDIO, MAC_RESERVE, vendor="apple")
    assert verdict.enabled is True
    assert verdict.selected == "dots-ocr"
    assert verdict.summary == "can read pages, using dots-ocr"


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
        gpu_vendor="nvidia",
        chosen={},
    )
    written = capability.record(
        "cuda-linux",
        total_bytes=THREE_NINETY,
        desktop_allowance_bytes=CUDA_RESERVE,
        decisions=decisions,
        routes={},
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
        "cuda-linux",
        total_bytes=SIX_GIG,
        desktop_allowance_bytes=CUDA_RESERVE,
        gpu_vendor="nvidia",
        chosen={},
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


# ------------------------------------------------------------- the route

def test_the_capability_route_answers_every_class_and_its_reason(
    make_client, auth
) -> None:
    """The read Foundry and BookForge decide what to ask for from.

    PHASE 9 made the act-to-model mapping a per-host fact, so a client that was
    handed a model id by configuration would be carrying a model this server may
    have refused.
    """
    total, allowance = 26 * 1024 ** 3, 3 * 1024 ** 3
    decided = capability.record(
        "cuda-linux",
        total_bytes=total,
        desktop_allowance_bytes=allowance,
        decisions=capability.decide_all(
            "cuda-linux",
            total_bytes=total,
            desktop_allowance_bytes=allowance,
            gpu_vendor="nvidia",
            chosen={},
        ),
        routes={},
    )
    with make_client(capability=decided) as instance:
        body = instance.get("/v1/capability", headers=auth)
    assert body.status_code == 200, body.text
    record = body.json()
    assert record["backend_kind"]
    assert record["total_bytes"] > 0
    names = {row["capability"] for row in record["classes"]}
    # translate / simplify / analysis are SEPARATE, and share a selected model.
    # Owen, 2026-09-13: a job must never be named as a different job.
    # `denoise` is its own class despite sharing the rvc ENV, because a class is
    # about what the card can hold and a 913 MB separator and a 2.5 GiB urvc
    # stack are different arithmetic. `generate` (2026-09-23) is the generic
    # chat-shaped act, separate for the same naming reason.
    assert names == {
        "align", "analysis", "asr", "clean", "decide", "denoise", "echo",
        "generate", "pages", "rvc", "simplify", "translate", "tts",
    }
    picked = {row["capability"]: row["selected"] for row in record["classes"]}
    assert picked["translate"] == picked["simplify"] == picked["analysis"], picked
    for row in record["classes"]:
        # Every row answers with a reason whichever way it went — a class turned
        # off records the number that turned it off.
        assert row["reason"], row["capability"]
        assert isinstance(row["enabled"], bool)
        if row["enabled"]:
            assert row["selected"] or row["capability"] == "echo"


def test_the_route_says_which_job_type_each_class_feeds_and_what_builds_it(
    make_client, auth
) -> None:
    """`job_types`, the operator page's Job types section in one read.

    PHASE13-OPERATOR.md section 3.2a. The page must not carry a list of job
    types, a list of narrator engines, or the knowledge that `denoise` has no
    installer of its own — all three are this BUILD's tables, and a copy in a
    page is the copy nobody updates (R1, and section 4's "never a hard-coded
    list").
    """
    total, allowance = 26 * 1024 ** 3, 3 * 1024 ** 3
    decided = capability.record(
        "cuda-linux",
        total_bytes=total,
        desktop_allowance_bytes=allowance,
        decisions=capability.decide_all(
            "cuda-linux",
            total_bytes=total,
            desktop_allowance_bytes=allowance,
            gpu_vendor="nvidia",
            chosen={},
        ),
        routes={},
    )
    with make_client(capability=decided) as instance:
        record = instance.get("/v1/capability", headers=auth).json()

    rows = {row["job_type"]: row for row in record["job_types"]}
    # Every class the same read reports belongs to exactly one of these rows,
    # and no row names a class that is not there — that is what makes the two
    # halves of the section one read rather than two that can disagree.
    classed = [name for row in record["job_types"] for name in row["classes"]]
    assert sorted(classed) == sorted(row["capability"] for row in record["classes"])
    assert len(classed) == len(set(classed))

    # `decide` since 2026-09-23 (PHASE22-DECIDE.md section 2.9), and
    # `generate` the same day, in CLASSES order.
    assert rows["llm"]["classes"] == [
        "clean", "translate", "simplify", "analysis", "generate", "decide", "pages"
    ]
    # Almost always itself. `denoise` shares `rvc`'s env, so a page that offered
    # it an Install button of its own would draw a control the task door refuses
    # `unknown_job_type` — and `echo` is compiled in, which is not the same as
    # "installed".
    assert rows["llm"]["installer"] == "llm"
    assert rows["denoise"]["installer"] == "rvc"
    assert rows["echo"]["installer"] is None

    # The whole of what `narrator_engine` may be, from the table the task door
    # validates against, and empty for every type the field means nothing for.
    assert rows["tts"]["narrator_engines"] == sorted(voices.NARRATOR_ENGINE_SAMPLING)
    assert all(
        row["narrator_engines"] == []
        for name, row in rows.items()
        if name != "tts"
    )

    # OFFERED is not answered here: `/v1/setup`'s `job_types` owns it.
    assert all("offered" not in row and "installed" not in row for row in rows.values())


def test_a_server_that_has_decided_nothing_says_so_rather_than_answering_empty(
    client, auth
) -> None:
    """Absent is its own answer. Empty rows would read as "probed, nothing fit",
    which is the opposite piece of news."""
    body = client.get("/v1/capability", headers=auth)
    assert body.status_code == 503, body.text
    assert body.json()["error"]["code"] == "capability_undecided"


def test_every_decision_carries_a_summary_a_person_can_read() -> None:
    """`reason` diagnoses; `summary` says what somebody can do (2026-09-20).

    A `pages` refusal reached a user as *"…cannot pages: disabled: reading page
    images (the VLM door) needs page readers, and this build ships none with a
    mlx-darwin block"*. Owen: *"it looks like an error."* It was a correct
    sentence written for the wrong reader — and stripping it would have cost the
    operator the only line naming which backend's block is missing. So the
    decision carries both, and neither reader gives way.

    EVERY branch, not only the one that was reported. A decision that reached a
    person with no summary would be the same defect back, on whichever class
    nobody happened to test.
    """
    from crucible.capability import CLASSES, decide

    internals = ("disabled:", "backend", "block", "the VLM door", "_")
    for entry in CLASSES:
        for backend, vendor, total in (
            ("mlx-darwin", "apple", 64 * 1024**3),
            ("cuda-linux", "nvidia", 24 * 1024**3),
            ("cuda-linux", "nvidia", 6 * 1024**3),
        ):
            verdict = decide(
                entry, backend, total_bytes=total,
                desktop_allowance_bytes=3 * 1024**3,
                gpu_vendor=vendor, chosen=None,
            )
            where = f"{entry.name} on {backend} at {total // 1024**3} GiB"
            assert verdict.summary, f"{where} has no summary"
            # SUBJECTLESS, so the caller supplies the server's name: the walk
            # does not know what this server is called and the caller does.
            assert verdict.summary.startswith(("can ", "cannot ")), where
            lowered = verdict.summary.lower()
            for token in internals:
                assert token not in lowered, f"{where} leaks {token!r}: {verdict.summary}"
            # And the operator's half keeps everything.
            assert verdict.reason, where


def test_the_reported_pages_case_reads_both_ways() -> None:
    """The exact refusal Owen saw, and the sentence that replaces it. The
    state that produced it (no mlx-darwin block on dots-ocr) is synthetic
    since 2026-09-21 — `_pages_with_no_block_on_any_backend` above."""
    from crucible.capability import decide

    pages = _pages_with_no_block_on_any_backend()
    verdict = decide(
        pages, "mlx-darwin", total_bytes=64 * 1024**3,
        desktop_allowance_bytes=3 * 1024**3, gpu_vendor="apple", chosen=None,
    )
    assert verdict.enabled is False
    # The user's half: a subject away from a whole sentence.
    assert verdict.summary.startswith("cannot read pages")
    assert "mlx-darwin" not in verdict.summary
    assert "VLM" not in verdict.summary
    # The operator's half, unchanged — this is what `crucible doctor` prints.
    assert "ships none with a mlx-darwin block" in verdict.reason
    assert "(the VLM door)" in verdict.reason


def test_the_summary_reaches_the_wire_and_not_only_the_decision() -> None:
    """THE GAP THAT SHIPPED ONCE. `summary` was added to `Decision.to_dict()`
    and announced as live — and `GET /v1/capability` serves `CapabilityRow`,
    a different projection, which dropped it. The live Mac answered
    `summary: None` on the very class the field was built for.

    So this asserts the whole path, not the object nearest the code that
    produces it: decision -> row -> the dict a client parses.
    """
    from crucible.capability import decide

    pages = _pages_with_no_block_on_any_backend()
    verdict = decide(
        pages, "mlx-darwin", total_bytes=64 * 1024**3,
        desktop_allowance_bytes=3 * 1024**3, gpu_vendor="apple", chosen=None,
    )
    served = verdict.row().to_dict()
    assert served["summary"] == verdict.summary
    assert served["summary"].startswith("cannot read pages")
    # And the operator's half is still beside it, unchanged.
    assert "ships none with a mlx-darwin block" in served["reason"]


def test_a_routed_class_summarises_where_the_work_goes() -> None:
    """A routed class is not a refusal and must not read like one."""
    from crucible.capability import CLASSES, decide, routed_row

    entry = next(c for c in CLASSES if c.name == "translate")
    verdict = decide(
        entry, "cuda-linux", total_bytes=6 * 1024**3,
        desktop_allowance_bytes=3 * 1024**3, gpu_vendor="nvidia", chosen=None,
    )
    row = routed_row(verdict.row(), "anthropic/claude-sonnet-4")
    assert row.enabled is True
    assert row.summary == "sends this work to anthropic"
    assert "cannot" not in row.summary


# ------------------------------------------ `generate`, and a client-sized fit
#
# Owen, 2026-09-23: *"if the only difference is the context limit then make it
# one class and give it the ability to set the context limit"*, and *"context
# limit can be set to 8k tokens by default, and it can request higher …
# requesting higher than that throws an error back to the app thats making the
# call"*.


def test_generate_is_one_routable_client_sized_class_on_the_9b_floor() -> None:
    entry = BY_NAME["generate"]
    assert entry.job_type == "llm"
    assert entry.routable is True
    assert entry.client_sized is True
    assert entry.min_params_b == capability.NINE_B_FLOOR == 9
    assert entry.work is not None
    assert (entry.work.tokens, entry.work.concurrency) == (8192, 1)
    assert entry.work.tokens == capability.GENERATE_DEFAULT_TOKENS
    # ContentStudio, its first measured user, is quoted as an example of asking
    # for more, never as the definition.
    assert "40960" in entry.work.source
    assert "ContentStudio" not in entry.purpose + entry.plainly
    # It sits after `analysis` and before `decide`, and it is the ONLY class a
    # client may size: every other class's context is a ruling about its act.
    names = [c.name for c in CLASSES]
    assert names.index("analysis") + 1 == names.index("generate")
    assert names.index("generate") + 1 == names.index("decide")
    assert [c.name for c in CLASSES if c.client_sized] == ["generate"]
    assert "generate" in capability.ROUTABLE_CLASSES
    # The 4B, 2B and 0.8B are below its floor on every backend.
    for kind in ("cuda-linux", "mlx-darwin", "llama-windows"):
        ids = {c.id for c in entry.candidates(kind)}
        assert ids and not ids & {"qwen3.5-4b", "qwen3.5-2b", "qwen3.5-0.8b"}, (kind, ids)


def test_a_ceiling_is_the_smaller_of_what_is_served_and_what_memory_affords() -> None:
    """Both halves are facts Crucible already owns; nothing here is typed.

    On the 3090 Ti the served half binds at one in flight — each block's
    `max_context` (65536 for the 9B, 32768 for the 27B-4bit, computed to sit
    just under what the card affords) — while the memory half is what the card
    could hold; the 27B's memory half shrinks as concurrency grows until it
    binds instead.
    """
    budget = available_bytes(THREE_NINETY, CUDA_RESERVE)
    one = {
        c.model: c
        for c in capability.context_ceilings(
            BY_NAME["generate"], "cuda-linux", available_bytes=budget, concurrency=1
        )
    }
    nine = one["qwen3.5-9b"]
    assert nine.served_context == 65536
    assert nine.memory_context == 74_887
    assert (nine.tokens, nine.bound_by) == (65536, "served")
    big = one["qwen3.8-27b-4bit"]
    assert (big.tokens, big.bound_by, big.memory_context) == (32768, "served", 33_945)

    two = {
        c.model: c
        for c in capability.context_ceilings(
            BY_NAME["generate"], "cuda-linux", available_bytes=budget, concurrency=2
        )
    }
    big = two["qwen3.8-27b-4bit"]
    assert big.bound_by == "memory"
    assert big.tokens == big.memory_context < big.served_context


def _generate_record(backend_kind: str, total: int, allowance: int, vendor: str):
    return capability.record(
        backend_kind,
        total_bytes=total,
        desktop_allowance_bytes=allowance,
        decisions=capability.decide_all(
            backend_kind,
            total_bytes=total,
            desktop_allowance_bytes=allowance,
            gpu_vendor=vendor,
            chosen={},
        ),
        routes={},
    )


def _rows(body) -> dict[str, dict]:
    return {row["capability"]: row for row in body.json()["classes"]}


def test_every_row_echoes_its_work_and_generate_its_ceilings(make_client, auth) -> None:
    decided = _generate_record("cuda-linux", THREE_NINETY, CUDA_RESERVE, "nvidia")
    with make_client(capability=decided) as instance:
        body = instance.get("/v1/capability", headers=auth)
    assert body.status_code == 200, body.text
    rows = _rows(body)
    assert rows["generate"]["work"]["tokens"] == 8192
    assert rows["generate"]["work"]["concurrency"] == 1
    assert rows["generate"]["work"]["from"] == "default"
    assert rows["translate"]["work"]["from"] == "default"
    # Not token-shaped: no work to state, and never an invented one.
    assert rows["tts"]["work"] is None
    ceilings = {c["model"]: c for c in rows["generate"]["context_ceilings"]}
    assert ceilings["qwen3.5-9b"]["tokens"] == 65536
    assert ceilings["qwen3.5-9b"]["served_context"] == 65536
    assert ceilings["qwen3.8-27b-4bit"]["tokens"] == 32768
    # Only a client-sized row has ceilings; every other row says null.
    assert rows["translate"]["context_ceilings"] is None
    assert rows["tts"]["context_ceilings"] is None


def test_the_fit_follows_the_clients_stated_context(make_client, auth) -> None:
    """A size the 27B serves picks the 27B; one only the 9B serves picks the 9B.

    At 32768 tokens x 2 in flight the 27B-4bit's KV no longer fits the 3090 Ti
    beside its weights, while the 9B's does — so the same class on the same card
    selects a different model because the CLIENT said how it will use it.
    """
    decided = _generate_record("cuda-linux", THREE_NINETY, CUDA_RESERVE, "nvidia")
    with make_client(capability=decided) as instance:
        small = instance.get(
            "/v1/capability?class=generate&context_tokens=4096", headers=auth
        )
        wide = instance.get(
            "/v1/capability?class=generate&context_tokens=32768&concurrency=2",
            headers=auth,
        )
        after = instance.get("/v1/capability", headers=auth)
    assert small.status_code == 200, small.text
    row = _rows(small)["generate"]
    assert row["enabled"] is True and row["selected"] == "qwen3.8-27b-4bit"
    assert row["work"]["from"] == "request"
    assert (row["work"]["tokens"], row["work"]["concurrency"]) == (4096, 1)

    assert wide.status_code == 200, wide.text
    row = _rows(wide)["generate"]
    assert row["enabled"] is True and row["selected"] == "qwen3.5-9b", row
    assert (row["work"]["tokens"], row["work"]["concurrency"]) == (32768, 2)
    # The ceilings are published at the concurrency asked about.
    assert {c["concurrency"] for c in row["context_ceilings"]} == {2}

    # A request re-decides for its caller alone; the record is untouched.
    row = _rows(after)["generate"]
    assert row["work"]["from"] == "default" and row["work"]["tokens"] == 8192


def test_a_context_above_the_ceiling_is_refused_by_name_and_never_clamped(
    make_client, auth
) -> None:
    decided = _generate_record("cuda-linux", THREE_NINETY, CUDA_RESERVE, "nvidia")
    with make_client(capability=decided) as instance:
        body = instance.get(
            "/v1/capability?class=generate&context_tokens=70000", headers=auth
        )
    assert body.status_code == 400, body.text
    error = body.json()["error"]
    assert error["code"] == "context_over_limit"
    details = error["details"]
    assert details["requested"] == {"tokens": 70000, "concurrency": 1}
    # The highest ceiling among the candidates, the model it belongs to, and
    # where each half came from: the 9B's cuda-linux max_context.
    assert details["ceiling"]["tokens"] == 65536
    assert details["ceiling"]["model"] == "qwen3.5-9b"
    assert details["ceiling"]["bound_by"] == "served"
    assert details["ceiling"]["served_context_source"]
    assert "70000" in error["message"] and "65536" in error["message"]


def test_a_mac_serves_what_the_card_cannot(make_client, auth) -> None:
    """The ceiling is PER HOST: the Studio takes 40960 on the 8-bit 27B, which
    the 3090 Ti is not offered at all, and both 27Bs there are capped at 131072."""
    decided = _generate_record("mlx-darwin", STUDIO, MAC_RESERVE, "apple")
    with make_client(capability=decided, backend=FAKE_MAC_BACKEND) as instance:
        body = instance.get(
            "/v1/capability?class=generate&context_tokens=40960", headers=auth
        )
    assert body.status_code == 200, body.text
    row = _rows(body)["generate"]
    assert row["enabled"] is True and row["selected"] == "qwen3.8-27b-8bit", row
    ceilings = {c["model"]: c for c in row["context_ceilings"]}
    assert ceilings["qwen3.8-27b-4bit"]["tokens"] == 131072
    assert ceilings["qwen3.8-27b-8bit"]["tokens"] == 131072


def test_a_chosen_model_is_the_one_whose_ceiling_governs() -> None:
    """An app's own choice is what will run, so its ceiling is the limit.

    On the 3090 Ti 40960 is inside the 9B's 65536 and past the 27B-4bit's
    32768: unchosen, the 9B's ceiling governs and it passes; with the 27B
    chosen, the 27B's does and it is refused.
    """
    entry = BY_NAME["generate"]
    budget = available_bytes(THREE_NINETY, CUDA_RESERVE)
    work = capability.WorkingContext(tokens=40960, concurrency=1, source="test")
    capability.check_ceiling(
        entry, "cuda-linux", available_bytes=budget, work=work, chosen=None
    )
    with pytest.raises(ApiError) as caught:
        capability.check_ceiling(
            entry,
            "cuda-linux",
            available_bytes=budget,
            work=work,
            chosen="qwen3.8-27b-4bit",
        )
    assert caught.value.code == "context_over_limit"
    assert caught.value.details["ceiling"]["model"] == "qwen3.8-27b-4bit"
    assert caught.value.details["ceiling"]["tokens"] == 32768


def test_a_host_that_cannot_hold_the_weights_is_not_a_length_refusal() -> None:
    """`MemoryTerms.max_context`: zero is its own answer, never "too long"."""
    work = capability.WorkingContext(tokens=4096, concurrency=1, source="test")
    capability.check_ceiling(
        BY_NAME["generate"],
        "cuda-linux",
        available_bytes=available_bytes(SIX_GIG, CUDA_RESERVE),
        work=work,
        chosen=None,
    )


@pytest.mark.parametrize(
    ("query", "code"),
    [
        ("class=translate&context_tokens=4096", "capability_not_client_sized"),
        ("class=tts&concurrency=2", "capability_not_client_sized"),
        ("context_tokens=4096", "capability_class_required"),
        ("class=generat&context_tokens=4096", "unknown_capability"),
        ("class=generate&context_tokens=0", "invalid_working_context"),
        ("class=generate&context_tokens=-5", "invalid_working_context"),
        ("class=generate&context_tokens=8k", "invalid_working_context"),
        ("class=generate&concurrency=0", "invalid_working_context"),
        ("class=generate&concurrency=1.5", "invalid_working_context"),
    ],
)
def test_a_bad_size_is_refused_by_name(make_client, auth, query, code) -> None:
    decided = _generate_record("cuda-linux", THREE_NINETY, CUDA_RESERVE, "nvidia")
    with make_client(capability=decided) as instance:
        body = instance.get(f"/v1/capability?{query}", headers=auth)
    assert body.status_code == 400, body.text
    assert body.json()["error"]["code"] == code


def test_a_record_that_predates_generate_says_so_rather_than_sizing_nothing(
    make_client, auth
) -> None:
    full = _generate_record("cuda-linux", THREE_NINETY, CUDA_RESERVE, "nvidia")
    older = CapabilityRecord(
        backend_kind=full.backend_kind,
        total_bytes=full.total_bytes,
        desktop_allowance_bytes=full.desktop_allowance_bytes,
        rows=tuple(row for row in full.rows if row.capability != "generate"),
    )
    with make_client(capability=older) as instance:
        body = instance.get(
            "/v1/capability?class=generate&context_tokens=4096", headers=auth
        )
    assert body.status_code == 503, body.text
    assert body.json()["error"]["code"] == "capability_undecided"


def test_the_8bit_mac_ceiling_is_131072_now_that_kv_is_counted_once() -> None:
    """It came out near 64k while the 8-bit's overhead still held the 4-bit
    run's 98_220 tokens of KV and the terms added KV on top. Counted once, the
    Studio's memory affords 162_603 tokens at one in flight and the manifest's
    max_context (131072) binds."""
    budget = available_bytes(STUDIO, MAC_RESERVE)
    ceilings = {
        c.model: c
        for c in capability.context_ceilings(
            BY_NAME["generate"], "mlx-darwin", available_bytes=budget, concurrency=1
        )
    }
    eight = ceilings["qwen3.8-27b-8bit"]
    assert (eight.tokens, eight.bound_by) == (131072, "served")
    assert eight.memory_context == 162_603
    # The 4-bit has no memory terms on this backend: its max_context is all
    # there is, and the row says so with a null memory half.
    four = ceilings["qwen3.8-27b-4bit"]
    assert (four.tokens, four.memory_context) == (131072, None)


def test_the_pc_ceilings_are_the_computed_maxima() -> None:
    budget = available_bytes(THREE_NINETY, CUDA_RESERVE)
    ceilings = {
        c.model: c.tokens
        for c in capability.context_ceilings(
            BY_NAME["generate"], "cuda-linux", available_bytes=budget, concurrency=1
        )
    }
    assert ceilings["qwen3.8-27b-4bit"] == 32768
    assert ceilings["qwen3.5-9b"] == 65536
    # No row for the 8-bit 27B: it has no cuda-linux block (Owen, 2026-09-23).
    assert "qwen3.8-27b-8bit" not in ceilings


def test_a_load_is_held_to_the_same_ceiling_capability_publishes() -> None:
    """`check_load_context` is `Candidate.context_ceiling` at one in flight:
    the load door and `GET /v1/capability` cannot disagree."""
    from crucible.manifests import load_manifest

    budget = available_bytes(THREE_NINETY, CUDA_RESERVE)
    big = load_manifest("qwen3.8-27b-4bit")
    published = {
        c.model: c
        for c in capability.context_ceilings(
            BY_NAME["generate"], "cuda-linux", available_bytes=budget, concurrency=1
        )
    }["qwen3.8-27b-4bit"]
    held = capability.check_load_context(
        big, "cuda-linux", available_bytes=budget, context=32768
    )
    assert held == published
    with pytest.raises(ApiError) as caught:
        capability.check_load_context(
            big, "cuda-linux", available_bytes=budget, context=32769
        )
    assert caught.value.code == "context_over_limit"
    assert caught.value.details["ceiling"] == published.to_dict()
    # A host that cannot hold the weights is not a length refusal: the guard
    # refuses it by its bytes instead.
    capability.check_load_context(
        big, "cuda-linux", available_bytes=available_bytes(SIX_GIG, CUDA_RESERVE),
        context=32769,
    )
