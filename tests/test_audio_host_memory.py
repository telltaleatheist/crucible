"""What an audio model keeps in host memory, weighed against this machine's (Victoria's
3070 laptop, 2026-10-10: a 15.8 GB WSL guest, YuE2 OOM-killed on track 12 of an album
while every card figure read correct). crucible/hostmemory.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible import cli, hostmemory
from crucible.audiomodels import AudioManifestError, load_audio_manifest, parse_audio_manifest
from crucible.memorybudget import GIB

from .conftest import FAKE_BACKEND, FAKE_MAC_BACKEND
from .test_audio_api import SONG


def test_yue2_declares_what_it_keeps_in_host_memory_on_cuda() -> None:
    spec = load_audio_manifest(SONG).spec(FAKE_BACKEND.kind)
    assert spec.host_memory_bytes_estimate is not None
    assert spec.host_memory_note.strip()
    assert spec.to_dict()["host_memory_bytes_estimate"] == spec.host_memory_bytes_estimate


def test_memtotal_is_read_from_meminfo_in_bytes(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16223232 kB\nMemFree:  1 kB\n", encoding="ascii")
    assert hostmemory.memory_total_bytes(str(meminfo)) == 16223232 * 1024


def test_a_meminfo_without_memtotal_is_refused_by_name(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemFree:  1 kB\n", encoding="ascii")
    with pytest.raises(ValueError, match="has no MemTotal"):
        hostmemory.memory_total_bytes(str(meminfo))


def test_a_machine_with_less_host_memory_than_yue2_keeps_is_named() -> None:
    manifest = load_audio_manifest(SONG)
    need = manifest.spec(FAKE_BACKEND.kind).host_memory_bytes_estimate
    (short,) = hostmemory.short_of_host([manifest], FAKE_BACKEND.kind, need - 1)
    assert (short.id, short.host_bytes, short.total_bytes) == (SONG, need, need - 1)
    assert short.words.startswith(f"{SONG} keeps ")
    assert "this machine has" in short.words
    assert hostmemory.short_of_host([manifest], FAKE_BACKEND.kind, need) == ()


def test_only_cuda_linux_weighs_host_memory() -> None:
    assert hostmemory.weighed_here(FAKE_BACKEND.kind)
    assert not hostmemory.weighed_here(FAKE_MAC_BACKEND.kind)


def _without_note(text: str, replacement: str | None) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("host_memory_note"):
            if replacement is not None:
                lines.append(replacement)
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: _without_note(t, None), "go together"),
        (lambda t: _without_note(t, 'host_memory_note = "  "'), "host_memory_note is empty"),
        (lambda t: t.replace("host_memory_bytes_estimate = ", "host_memory_bytes_estimate = -"),
         "must be positive"),
    ],
)
def test_a_bad_host_memory_figure_is_refused(change: Any, message: str) -> None:
    path = load_audio_manifest(SONG).path
    text = path.read_text(encoding="utf-8")
    changed = change(text)
    assert changed != text
    with pytest.raises(AudioManifestError, match=message):
        parse_audio_manifest(changed, path, SONG)


def _doctor(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], total: int) -> dict:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)
    monkeypatch.setattr(hostmemory, "memory_total_bytes", lambda: total)
    assert cli.main(["init", "--enable-audio"]) == 0
    capsys.readouterr()
    cli.main(["doctor", "--json"])
    return json.loads(capsys.readouterr().out)


def test_doctor_names_the_model_and_both_figures_when_the_host_is_short(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _doctor(monkeypatch, capsys, 6 * GIB)
    (short,) = report["audio_host_memory"]["short"]
    assert (short["id"], short["total_bytes"]) == (SONG, 6 * GIB)
    found = [p for p in report["problems"] if p.startswith("audio_host_memory")]
    assert len(found) == 1 and SONG in found[0] and "6.0 GiB in all" in found[0]


def test_doctor_is_quiet_when_the_host_holds_it(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _doctor(monkeypatch, capsys, 64 * GIB)
    assert report["audio_host_memory"] == {"total_bytes": 64 * GIB, "short": []}
    assert not [p for p in report["problems"] if p.startswith("audio_host_memory")]
