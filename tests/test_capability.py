from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible import (
    capabilityclasses,
    classnames,
    cli,
    contextceiling,
    fit,
    narratorengines,
    verdict,
)
from crucible.capabilityclasses import BY_NAME, CLASSES, classes_for_job_type
from crucible.capabilityrecord import CapabilityRecord, CapabilityRow
from crucible.config import config_path, default_desktop_allowance_bytes, load_config, write_config
from crucible.errors import ApiError, ConfigError
from crucible.jobs import disabled_error
from crucible.memorybudget import available_bytes
from crucible.verdict import decide, decide_all, job_type_enabled

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND

GIB = 1024 ** 3
CUDA_RESERVE = 3 * GIB

THREE_NINETY = FAKE_BACKEND.gpu.vram_bytes
STUDIO = FAKE_MAC_BACKEND.gpu.vram_bytes
MAC_RESERVE = default_desktop_allowance_bytes("mlx-darwin", STUDIO)

SIX_GIG = 6 * GIB


def _decide(
    name: str,
    backend: str,
    total: int,
    reserve: int,
    vendor: str = "nvidia",
    chosen: str | None = None,
) -> Any:
    return decide(
        BY_NAME[name],
        backend,
        total_bytes=total,
        desktop_allowance_bytes=reserve,
        gpu_vendor=vendor,
        chosen=chosen,
        audio_low_vram=False,
    )


def test_the_3090ti_translates_which_is_what_owen_already_does() -> None:
    verdict = _decide("translate", "cuda-linux", THREE_NINETY, CUDA_RESERVE)
    assert verdict.enabled is True
    assert verdict.selected == "qwen3.8-27b-4bit"
    assert verdict.shortfall_bytes == 0
    assert "qwen3.8-27b-8bit" not in [c.id for c in verdict.candidates]


def test_the_mac_selects_the_8bit_27b_over_the_4bit() -> None:
    verdict = _decide("translate", "mlx-darwin", STUDIO, MAC_RESERVE)
    assert verdict.enabled is True
    assert verdict.selected == "qwen3.8-27b-8bit", (
        "the Mac must take the 8-bit: it is the best candidate that fits, and "
        "fitting it is the reason the bf16 was replaced rather than dropped"
    )
    ids = [c.id for c in verdict.candidates]
    assert ids[0] == "qwen3.8-27b-8bit", "best-first must still walk largest first"
    assert verdict.fit_count == 3


def test_best_precision_first_not_smallest_that_fits() -> None:
    verdict = _decide("translate", "mlx-darwin", 200 * GIB, MAC_RESERVE)
    assert verdict.selected == "qwen3.8-27b-8bit", (
        "with room for both, the rule must take the better one, not the smaller"
    )
    assert verdict.fit_count == 3
    pc = _decide("translate", "cuda-linux", 200 * GIB, CUDA_RESERVE)
    assert pc.selected == "qwen3.8-27b-4bit"
    assert pc.fit_count == 2
    assert "qwen3.8-27b-8bit" not in [c.id for c in pc.candidates]


def test_the_selected_id_does_not_depend_on_catalog_order() -> None:
    for name in ("tts", "rvc"):
        first = _decide(name, "cuda-linux", THREE_NINETY, CUDA_RESERVE)
        again = _decide(name, "cuda-linux", THREE_NINETY, CUDA_RESERVE)
        sizes = {c.memory_bytes_estimate for c in first.candidates}
        assert len(sizes) == 1, f"{name} is the tie case this test is about"
        assert first.selected == again.selected
        assert first.selected == min(c.id for c in first.candidates)


def test_higgs_is_binary_and_a_six_gig_card_loses_tts_entirely() -> None:
    verdict = _decide("tts", "cuda-linux", SIX_GIG, CUDA_RESERVE)
    assert verdict.enabled is False
    assert verdict.selected == ""
    assert "nothing under 4-bit is ever offered" in verdict.reason
    assert job_type_enabled("tts", decide_all(
        "cuda-linux",
        total_bytes=SIX_GIG,
        desktop_allowance_bytes=CUDA_RESERVE,
        gpu_vendor="nvidia",
        chosen={},
        audio_low_vram=False,
    )) is False


def test_translate_is_binary_per_server_and_says_so_when_it_is_off() -> None:
    verdict = _decide("translate", "cuda-linux", SIX_GIG, CUDA_RESERVE)
    assert verdict.enabled is False
    assert "cannot translate" in verdict.reason


def test_a_six_gig_card_keeps_llm_only_if_something_behind_it_fits() -> None:
    decisions = decide_all(
        "cuda-linux",
        total_bytes=SIX_GIG,
        desktop_allowance_bytes=CUDA_RESERVE,
        gpu_vendor="nvidia",
        chosen={},
        audio_low_vram=False,
    )
    by_name = {d.capability: d for d in decisions}
    assert by_name["clean"].enabled is False
    assert by_name["translate"].enabled is False
    assert by_name["pages"].enabled is False
    assert job_type_enabled("llm", decisions) is False
    assert job_type_enabled("asr", decisions) is True
    assert job_type_enabled("rvc", decisions) is True


def test_llm_survives_when_one_of_its_three_classes_survives() -> None:
    decisions = decide_all(
        "cuda-linux",
        total_bytes=24 * GIB,
        desktop_allowance_bytes=8 * GIB,
        gpu_vendor="nvidia",
        chosen={},
        audio_low_vram=False,
    )
    by_name = {d.capability: d for d in decisions}
    assert by_name["pages"].enabled is True
    assert by_name["translate"].enabled is False
    assert job_type_enabled("llm", decisions) is True


def test_a_disabled_class_records_the_number_that_disabled_it() -> None:
    tiny_card = 4 * GIB
    verdict = _decide("asr", "cuda-linux", tiny_card, CUDA_RESERVE)
    assert verdict.enabled is False
    assert len(verdict.candidates) > 1, "the point of this test is several"
    sizes = [c.memory_bytes_estimate for c in verdict.candidates]
    assert verdict.shortfall_bytes == min(sizes) - (tiny_card - CUDA_RESERVE)
    assert verdict.shortfall_bytes > 0
    assert verdict.shortfall_bytes < max(sizes) - (tiny_card - CUDA_RESERVE)


def test_asr_and_align_are_both_enabled_on_the_studio() -> None:
    align = _decide("align", "mlx-darwin", STUDIO, MAC_RESERVE)
    assert align.enabled is True
    assert align.selected == "qwen3-aligner"

    asr = _decide("asr", "mlx-darwin", STUDIO, MAC_RESERVE)
    assert asr.enabled is True
    assert asr.selected == "qwen3-asr-1.7b"
    assert [c.id for c in asr.candidates] == [
        "qwen3-asr-1.7b",
        "qwen3-asr-1.7b-mlx",
        "qwen3-asr-0.6b",
        "qwen3-asr-0.6b-mlx",
        "whisper-large-v3-turbo",
        "whisper-tiny",
    ]


def _pages_with_no_block_on_any_backend() -> Any:
    import dataclasses

    from crucible.capabilityclasses import CLASSES

    pages = next(entry for entry in CLASSES if entry.name == "pages")
    return dataclasses.replace(pages, candidates=lambda backend_kind: ())


def test_a_class_with_nothing_on_this_backend_still_says_which_it_is() -> None:
    from crucible.verdict import decide

    verdict = decide(
        _pages_with_no_block_on_any_backend(), "mlx-darwin",
        total_bytes=STUDIO, desktop_allowance_bytes=MAC_RESERVE,
        gpu_vendor="apple", chosen=None,
        audio_low_vram=False,
    )
    assert verdict.enabled is False
    assert verdict.candidates == ()
    assert verdict.shortfall_bytes == 0
    assert "ships none with a mlx-darwin block" in verdict.reason


def test_pages_on_the_mac_is_enabled_since_the_block_landed() -> None:
    verdict = _decide("pages", "mlx-darwin", STUDIO, MAC_RESERVE, vendor="apple")
    assert verdict.enabled is True
    assert verdict.selected == "dots-ocr"
    assert verdict.summary == "can read pages, using dots-ocr"


def test_echo_needs_no_accelerator_and_is_never_disabled_by_a_card() -> None:
    verdict = _decide("echo", "cuda-linux", 1, CUDA_RESERVE)
    assert verdict.enabled is True
    assert verdict.candidates == ()


def test_the_budget_never_goes_negative() -> None:
    assert available_bytes(2 * GIB, 3 * GIB) == 0


def test_every_job_type_has_a_capability_class() -> None:
    from crucible.jobs import ALL_JOB_TYPES

    covered = {entry.job_type for entry in CLASSES}
    assert set(ALL_JOB_TYPES.values()) <= covered
    for job_type in set(ALL_JOB_TYPES.values()):
        assert classes_for_job_type(job_type), job_type


def test_an_unknown_backend_is_refused_rather_than_priced() -> None:
    with pytest.raises(ValueError, match="not a Crucible backend"):
        _decide("clean", "cuda-windows", THREE_NINETY, CUDA_RESERVE)


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
        "enable_denoise": False,
        "desktop_allowance_bytes": CUDA_RESERVE,
        "retention_days": 7,
        "desktop_allowance_basis": "stated",
        "desktop_allowance_note": "",
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
        audio_low_vram=False,
    )
    written = verdict.record(
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
    _write(home)
    config = load_config(home)
    assert config.capability is None


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
        "selected = \"\"\nreason = \"x\"\nsummary = \"x\"\nshortfall_bytes = 1\n"
    )
    path.write_text(
        path.read_text(encoding="utf-8")
        + '\n[capability]\nbackend_kind = "cuda-linux"\ntotal_bytes = 1\n'
        "desktop_allowance_bytes = 1\n" + row + row,
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="recorded twice"):
        load_config(home)


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
    verdict = _decide("tts", "cuda-linux", SIX_GIG, CUDA_RESERVE)
    config = _config_with(home, (verdict.row(),))
    error = disabled_error("tts", config)
    assert isinstance(error, ApiError)
    assert error.status_code == 400
    assert error.code == "job_type_disabled"
    assert "nothing under 4-bit is ever offered" in error.message
    assert "short by" in error.message
    assert "would not change any of those numbers" in error.message
    assert "= true" not in error.message
    assert error.details["shortfall_bytes"]["tts"] == verdict.shortfall_bytes


def test_the_refusal_says_so_when_nothing_has_decided_anything_here(
    home: Path,
) -> None:
    _write(home)
    error = disabled_error("tts", load_config(home))
    assert error.details["capability_recorded"] is False
    assert "no capability selection has been recorded" in error.message
    assert "crucible capability" in error.message


def test_the_refusal_offers_install_when_the_card_can_hold_it(home: Path) -> None:
    verdict = _decide("tts", "cuda-linux", THREE_NINETY, CUDA_RESERVE)
    assert verdict.enabled is True
    config = _config_with(home, (verdict.row(),))
    error = disabled_error("tts", config)
    assert "POST /v1/tasks" in error.message
    assert error.details["install"] == {"type": "install", "job_type": "tts"}
    assert error.details["reason"] == "not_installed"
    assert error.details["fits"] == ["tts"]


def test_the_llm_refusal_reads_every_class_behind_the_flag(home: Path) -> None:
    decisions = decide_all(
        "cuda-linux",
        total_bytes=SIX_GIG,
        desktop_allowance_bytes=CUDA_RESERVE,
        gpu_vendor="nvidia",
        chosen={},
        audio_low_vram=False,
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


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)


@pytest.fixture
def tiny_card(monkeypatch: pytest.MonkeyPatch) -> None:
    from crucible.backend import Backend, Gpu

    monkeypatch.setattr(
        cli.common,
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
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    config = load_config(home)
    assert cli.capability._capability_step(config, cli.common.detect_backend(), "tts") == 1
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
    assert cli.capability._capability_step(config, cli.common.detect_backend(), "tts") == 0
    after = load_config(home)
    assert after.enable_tts is True
    assert after.capability.row("tts").selected != ""
    assert after.token == config.token, "a capability decision never re-mints"


def test_doctor_reports_a_flag_the_numbers_contradict(
    home: Path, tiny_card: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-tts"]) == 0
    capsys.readouterr()
    assert cli.main(["capability", "--write"]) == 0
    capsys.readouterr()
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
    assert cli.main(["init", "--enable-echo"]) == 0
    assert cli.main(["capability", "--write"]) == 0
    capsys.readouterr()
    import crucible.cli.common as cli_module

    cli_module.detect_backend = lambda: FAKE_BACKEND
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert any(p.startswith("capability_stale") for p in report["problems"])
    assert report["capability"]["stale"] is True


def test_doctor_calls_an_unbuilt_but_fitting_type_a_note_not_a_problem(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-echo"]) == 0
    assert cli.main(["capability", "--write"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is True
    assert "tts" in report["capability"]["could_enable"]


def test_the_capability_route_answers_every_class_and_its_reason(
    make_client, auth
) -> None:
    total, allowance = 26 * 1024 ** 3, 3 * 1024 ** 3
    decided = verdict.record(
        "cuda-linux",
        total_bytes=total,
        desktop_allowance_bytes=allowance,
        decisions=verdict.decide_all(
            "cuda-linux",
            total_bytes=total,
            desktop_allowance_bytes=allowance,
            gpu_vendor="nvidia",
            chosen={},
            audio_low_vram=False,
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
    assert names == {
        "align", "analysis", "asr", "clean", "cutout", "decide", "denoise", "echo",
        "generate", "image", "music", "pages", "rvc", "select", "sfx", "simplify",
        "song", "translate", "tts", "video",
    }
    picked = {row["capability"]: row["selected"] for row in record["classes"]}
    assert picked["translate"] == picked["simplify"] == picked["analysis"], picked
    for row in record["classes"]:
        assert row["reason"], row["capability"]
        assert isinstance(row["enabled"], bool)
        if row["enabled"]:
            assert row["selected"] or row["capability"] == "echo"


def test_the_route_says_which_job_type_each_class_feeds_and_what_builds_it(
    make_client, auth
) -> None:
    total, allowance = 26 * 1024 ** 3, 3 * 1024 ** 3
    decided = verdict.record(
        "cuda-linux",
        total_bytes=total,
        desktop_allowance_bytes=allowance,
        decisions=verdict.decide_all(
            "cuda-linux",
            total_bytes=total,
            desktop_allowance_bytes=allowance,
            gpu_vendor="nvidia",
            chosen={},
            audio_low_vram=False,
        ),
        routes={},
    )
    with make_client(capability=decided) as instance:
        record = instance.get("/v1/capability", headers=auth).json()

    rows = {row["job_type"]: row for row in record["job_types"]}
    classed = [name for row in record["job_types"] for name in row["classes"]]
    assert sorted(classed) == sorted(row["capability"] for row in record["classes"])
    assert len(classed) == len(set(classed))

    assert rows["llm"]["classes"] == [
        "clean", "translate", "simplify", "analysis", "generate", "decide", "pages"
    ]
    assert rows["llm"]["installer"] == "llm"
    assert rows["denoise"]["installer"] == "rvc"
    assert rows["echo"]["installer"] is None

    assert rows["tts"]["narrator_engines"] == sorted(narratorengines.NARRATOR_ENGINE_SAMPLING)
    assert all(
        row["narrator_engines"] == []
        for name, row in rows.items()
        if name != "tts"
    )

    assert all("offered" not in row and "installed" not in row for row in rows.values())


def test_a_server_that_has_decided_nothing_says_so_rather_than_answering_empty(
    client, auth
) -> None:
    body = client.get("/v1/capability", headers=auth)
    assert body.status_code == 503, body.text
    assert body.json()["error"]["code"] == "capability_undecided"


def test_every_decision_carries_a_summary_a_person_can_read() -> None:
    from crucible.capabilityclasses import CLASSES
    from crucible.verdict import decide

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
                audio_low_vram=False,
            )
            where = f"{entry.name} on {backend} at {total // 1024**3} GiB"
            assert verdict.summary, f"{where} has no summary"
            assert verdict.summary.startswith(("can ", "cannot ")), where
            lowered = verdict.summary.lower()
            for token in internals:
                assert token not in lowered, f"{where} leaks {token!r}: {verdict.summary}"
            assert verdict.reason, where


def test_the_reported_pages_case_reads_both_ways() -> None:
    from crucible.verdict import decide

    pages = _pages_with_no_block_on_any_backend()
    verdict = decide(
        pages, "mlx-darwin", total_bytes=64 * 1024**3,
        desktop_allowance_bytes=3 * 1024**3, gpu_vendor="apple", chosen=None,
        audio_low_vram=False,
    )
    assert verdict.enabled is False
    assert verdict.summary.startswith("cannot read pages")
    assert "mlx-darwin" not in verdict.summary
    assert "VLM" not in verdict.summary
    assert "ships none with a mlx-darwin block" in verdict.reason
    assert "(the VLM door)" in verdict.reason


def test_the_summary_reaches_the_wire_and_not_only_the_decision() -> None:
    from crucible.verdict import decide

    pages = _pages_with_no_block_on_any_backend()
    verdict = decide(
        pages, "mlx-darwin", total_bytes=64 * 1024**3,
        desktop_allowance_bytes=3 * 1024**3, gpu_vendor="apple", chosen=None,
        audio_low_vram=False,
    )
    served = verdict.row().to_dict()
    assert served["summary"] == verdict.summary
    assert served["summary"].startswith("cannot read pages")
    assert "ships none with a mlx-darwin block" in served["reason"]


def test_a_routed_class_summarises_where_the_work_goes() -> None:
    from crucible.capabilityclasses import CLASSES
    from crucible.verdict import decide, routed_row

    entry = next(c for c in CLASSES if c.name == "translate")
    verdict = decide(
        entry, "cuda-linux", total_bytes=6 * 1024**3,
        desktop_allowance_bytes=3 * 1024**3, gpu_vendor="nvidia", chosen=None,
        audio_low_vram=False,
    )
    row = routed_row(verdict.row(), "anthropic/claude-sonnet-4")
    assert row.enabled is True
    assert row.summary == "sends this work to anthropic"
    assert "cannot" not in row.summary


def test_generate_is_one_routable_client_sized_class_on_the_9b_floor() -> None:
    entry = BY_NAME["generate"]
    assert entry.job_type == "llm"
    assert entry.routable is True
    assert entry.client_sized is True
    assert entry.min_params_b == capabilityclasses.NINE_B_FLOOR == 9
    assert entry.work is not None
    assert (entry.work.tokens, entry.work.concurrency) == (8192, 1)
    assert entry.work.tokens == capabilityclasses.GENERATE_DEFAULT_TOKENS
    assert "40960" in entry.work.source
    assert "ContentStudio" not in entry.purpose + entry.plainly
    names = [c.name for c in CLASSES]
    assert names.index("analysis") + 1 == names.index("generate")
    assert names.index("generate") + 1 == names.index("decide")
    assert [c.name for c in CLASSES if c.client_sized] == ["generate"]
    assert "generate" in classnames.ROUTABLE_CLASSES
    for kind in ("cuda-linux", "mlx-darwin", "llama-windows"):
        ids = {c.id for c in entry.candidates(kind)}
        assert ids and not ids & {"qwen3.5-4b", "qwen3.5-2b", "qwen3.5-0.8b"}, (kind, ids)


def test_a_ceiling_is_the_smaller_of_what_is_served_and_what_memory_affords() -> None:
    budget = available_bytes(THREE_NINETY, CUDA_RESERVE)
    one = {
        c.model: c
        for c in contextceiling.context_ceilings(
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
        for c in contextceiling.context_ceilings(
            BY_NAME["generate"], "cuda-linux", available_bytes=budget, concurrency=2
        )
    }
    big = two["qwen3.8-27b-4bit"]
    assert big.bound_by == "memory"
    assert big.tokens == big.memory_context < big.served_context


def _generate_record(backend_kind: str, total: int, allowance: int, vendor: str):
    return verdict.record(
        backend_kind,
        total_bytes=total,
        desktop_allowance_bytes=allowance,
        decisions=verdict.decide_all(
            backend_kind,
            total_bytes=total,
            desktop_allowance_bytes=allowance,
            gpu_vendor=vendor,
            chosen={},
            audio_low_vram=False,
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
    assert rows["tts"]["work"] is None
    ceilings = {c["model"]: c for c in rows["generate"]["context_ceilings"]}
    assert ceilings["qwen3.5-9b"]["tokens"] == 65536
    assert ceilings["qwen3.5-9b"]["served_context"] == 65536
    assert ceilings["qwen3.8-27b-4bit"]["tokens"] == 32768
    assert rows["translate"]["context_ceilings"] is None
    assert rows["tts"]["context_ceilings"] is None


def test_the_fit_follows_the_clients_stated_context(make_client, auth) -> None:
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
    assert {c["concurrency"] for c in row["context_ceilings"]} == {2}

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
    assert details["ceiling"]["tokens"] == 65536
    assert details["ceiling"]["model"] == "qwen3.5-9b"
    assert details["ceiling"]["bound_by"] == "served"
    assert details["ceiling"]["served_context_source"]
    assert "70000" in error["message"] and "65536" in error["message"]


def test_a_mac_serves_what_the_card_cannot(make_client, auth) -> None:
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
    entry = BY_NAME["generate"]
    budget = available_bytes(THREE_NINETY, CUDA_RESERVE)
    work = fit.WorkingContext(tokens=40960, concurrency=1, source="test")
    contextceiling.check_ceiling(
        entry, "cuda-linux", available_bytes=budget, work=work, chosen=None
    )
    with pytest.raises(ApiError) as caught:
        contextceiling.check_ceiling(
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
    work = fit.WorkingContext(tokens=4096, concurrency=1, source="test")
    contextceiling.check_ceiling(
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
    budget = available_bytes(STUDIO, MAC_RESERVE)
    ceilings = {
        c.model: c
        for c in contextceiling.context_ceilings(
            BY_NAME["generate"], "mlx-darwin", available_bytes=budget, concurrency=1
        )
    }
    eight = ceilings["qwen3.8-27b-8bit"]
    assert (eight.tokens, eight.bound_by) == (131072, "served")
    assert eight.memory_context == 162_603
    four = ceilings["qwen3.8-27b-4bit"]
    assert (four.tokens, four.memory_context) == (131072, None)


def test_the_pc_ceilings_are_the_computed_maxima() -> None:
    budget = available_bytes(THREE_NINETY, CUDA_RESERVE)
    ceilings = {
        c.model: c.tokens
        for c in contextceiling.context_ceilings(
            BY_NAME["generate"], "cuda-linux", available_bytes=budget, concurrency=1
        )
    }
    assert ceilings["qwen3.8-27b-4bit"] == 32768
    assert ceilings["qwen3.5-9b"] == 65536
    assert "qwen3.8-27b-8bit" not in ceilings


def test_a_load_is_held_to_the_same_ceiling_capability_publishes() -> None:
    from crucible.manifests import load_manifest

    budget = available_bytes(THREE_NINETY, CUDA_RESERVE)
    big = load_manifest("qwen3.8-27b-4bit")
    published = {
        c.model: c
        for c in contextceiling.context_ceilings(
            BY_NAME["generate"], "cuda-linux", available_bytes=budget, concurrency=1
        )
    }["qwen3.8-27b-4bit"]
    held = contextceiling.check_load_context(
        big, "cuda-linux", available_bytes=budget, context=32768
    )
    assert held == published
    with pytest.raises(ApiError) as caught:
        contextceiling.check_load_context(
            big, "cuda-linux", available_bytes=budget, context=32769
        )
    assert caught.value.code == "context_over_limit"
    assert caught.value.details["ceiling"] == published.to_dict()
    contextceiling.check_load_context(
        big, "cuda-linux", available_bytes=available_bytes(SIX_GIG, CUDA_RESERVE),
        context=32769,
    )
