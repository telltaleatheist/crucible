from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible import cli, weights
from crucible.config import load_config, write_config
from crucible.jobs import workerio

from .conftest import FAKE_BACKEND, configure_box

REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


def _snapshot(root: Path) -> Path:
    snapshot = root / "models--Qwen--Qwen-Image-2.1" / "snapshots" / REVISION
    (snapshot / "transformer").mkdir(parents=True)
    (snapshot / "transformer" / "shard.safetensors").write_bytes(b"w" * 4096)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    return snapshot


def _spec(files: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(
        backend="mlx-darwin", hf_repo="Qwen/Qwen-Image-2.1", revision=REVISION, files=files
    )


def test_the_hub_cache_at_the_pinned_revision_is_linked_not_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    snapshot = _snapshot(tmp_path / "hub")
    target = tmp_path / "store"
    lines: list[str] = []
    linked = weights.adopt_hub_cache(_spec(), target, lines.append)
    assert linked == 4096 + 2
    placed = target / "transformer" / "shard.safetensors"
    assert placed.read_bytes() == b"w" * 4096
    assert placed.stat().st_ino == (snapshot / "transformer" / "shard.safetensors").stat().st_ino
    assert "no second copy on disk" in lines[0]


def test_another_revision_in_the_cache_is_not_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    _snapshot(tmp_path / "hub")
    spec = _spec()
    spec.revision = "0" * 40
    assert weights.adopt_hub_cache(spec, tmp_path / "store") == 0
    assert not (tmp_path / "store").exists()


def test_only_the_files_a_block_names_are_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    _snapshot(tmp_path / "hub")
    assert weights.adopt_hub_cache(_spec(("model_index.json",)), tmp_path / "store") == 2
    assert not (tmp_path / "store" / "transformer").exists()


def test_a_config_written_before_images_existed_reads_image_as_off(home: Path) -> None:
    configure_box(home)
    path = home / "config.toml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("enable_image = false\n", ""), encoding="utf-8")
    assert "enable_image" not in path.read_text(encoding="utf-8")
    assert load_config(home).enable_image is False


def test_a_rewriter_that_does_not_name_the_image_flag_keeps_it(home: Path) -> None:
    configure_box(home, enable_image=True)
    config = load_config(home)
    write_config(
        home,
        name=config.name,
        host=config.host,
        port=config.port,
        token=config.token,
        backend_kind=config.backend_kind,
        enable_echo=config.enable_echo,
        enable_llm=config.enable_llm,
        enable_asr=config.enable_asr,
        enable_tts=config.enable_tts,
        enable_align=config.enable_align,
        enable_rvc=config.enable_rvc,
        enable_denoise=config.enable_denoise,
        desktop_allowance_bytes=config.desktop_allowance_bytes,
        retention_days=config.retention_days,
        desktop_allowance_basis=config.desktop_allowance_basis,
        desktop_allowance_note=config.desktop_allowance_note,
    )
    assert load_config(home).enable_image is True


def test_an_interrupt_reaches_a_handler_that_is_still_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = io.StringIO()
    monkeypatch.setattr(workerio, "_RESULTS", results)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO('{"op": "work"}\n{"op": "cancel", "request_id": "r1"}\n'),
    )
    asked = threading.Event()

    def work(request: dict) -> None:
        workerio.send("done", stopped=asked.wait(5.0))

    code = workerio.serve("test", {"work": work}, {"cancel": lambda request: asked.set()})
    assert code == 0
    assert [json.loads(line) for line in results.getvalue().splitlines()] == [
        {"type": "done", "stopped": True}
    ]


def test_without_interrupts_a_cancel_line_is_an_unknown_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = io.StringIO()
    monkeypatch.setattr(workerio, "_RESULTS", results)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"op": "cancel"}\n'))
    assert workerio.serve("test", {"work": lambda request: None}) == 1
    assert "this worker takes ['work']" in results.getvalue()


def test_init_turns_image_on_and_doctor_names_its_install(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    assert cli.main(["init", "--enable-image"]) == 0
    assert load_config(home).enable_image is True
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    envs = {entry["job_type"]: entry for entry in report["worker_envs"]}
    assert sorted(envs) == ["image"]
    assert "crucible install image" in envs["image"]["detail"]
    types = {entry["name"]: entry for entry in report["job_types"]}
    assert types["image"]["enabled"] is True
