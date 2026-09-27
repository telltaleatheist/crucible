from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from crucible import VERSION, cli
from crucible.backend import Backend, Gpu
from crucible.cli import doctor

from .conftest import FAKE_BACKEND


@pytest.fixture
def viable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.common, "detect_backend", lambda: FAKE_BACKEND)


def test_a_finding_without_a_fix_cannot_be_made() -> None:
    with pytest.raises(ValueError, match="names no fix"):
        doctor.Finding("x", "something is wrong", " ")


def test_a_finding_names_its_fix_once() -> None:
    assert doctor.Finding.run("a", "broken", "crucible install a").line == (
        "a: broken. Run `crucible install a`"
    )
    assert doctor.Finding.run("a", "run `crucible install a`", "crucible install a").line == (
        "a: run `crucible install a`"
    )


def test_the_report_keeps_its_key_order_and_refuses_a_key_it_does_not_know(
    tmp_path: Path,
) -> None:
    report = doctor.assemble(tmp_path, [
        doctor.Section("one", {"ffmpeg": {"x": 1}}, notes=("n",)),
        doctor.Section("two", {}, (doctor.Finding("p", "q", "crucible doctor"),)),
    ])
    assert list(report) == [key for key, _ in doctor.REPORT_DEFAULTS]
    assert report["ffmpeg"] == {"x": 1}
    assert (report["notes"], report["problems"], report["healthy"]) == (["n"], ["p: q"], False)
    with pytest.raises(KeyError, match="unknown keys"):
        doctor.assemble(tmp_path, [doctor.Section("bad", {"surprise": 1})])


def _synthetic_report(home: Path) -> dict[str, Any]:
    report = doctor.assemble(home, [])
    report.update({
        "backend": FAKE_BACKEND.to_dict(),
        "path": {"shell": "/bin", "service": None, "mechanism": "systemd",
                 "definition": "/u/crucible.service", "agree": None},
        "config": {"path": "/c.toml", "mode": "0o600", "name": "n", "host": "h",
                   "port": 1, "desktop_reserve": "kept 3.0 GiB", "desktop_allowance_note": None},
        "llm_env": {"installed": False, "detail": "no venv"},
        "worker_envs": [{"job_type": "asr", "installed": True, "detail": "ok"}],
        "ffmpeg": {"source": "missing", "pinned_version": None, "tools_bin": "/t",
                   "platform": "p", "path": None},
        "job_types": [{"name": "echo", "enabled": False, "ready": False, "detail": "off"}],
        "notes": ["a note"],
        "problems": ["llm_env: no venv. Run `crucible install llm`"],
    })
    return report


def test_the_text_renderer_is_one_pass_over_the_report(tmp_path: Path) -> None:
    out, err = doctor.render_text(_synthetic_report(tmp_path))
    assert out == [
        f"crucible {VERSION} (api 1)",
        f"home:    {tmp_path}",
        "backend: cuda-linux on linux/x86_64",
        "gpu:     nvidia NVIDIA GeForce RTX 3090 Ti (24.0 GiB) — test double",
        "PATH (this shell):   /bin",
        "PATH (the service):  none recorded — no systemd definition at /u/crucible.service",
        "config:  /c.toml (mode 0o600)",
        "serves:  n on h:1",
        "reserve: kept 3.0 GiB",
        "llm env: NOT READY — no venv",
        "asr env: ready — ok",
        "capability: NOT DECIDED — nothing has probed this host's card against the "
        "models; run `crucible capability`",
        "ffmpeg:  NONE — no pinned build for p, and none on PATH",
        "job echo: off — off",
        "note:    a note",
    ]
    assert err == ["PROBLEM: llm_env: no venv. Run `crucible install llm`"]


def test_every_finding_on_a_fresh_install_names_its_fix_in_its_line(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-llm", "--enable-tts", "--enable-asr"]) == 0
    capsys.readouterr()
    host = doctor.survey(home)
    findings = [finding for check in doctor.CHECKS for finding in check(host).findings]
    assert findings
    for finding in findings:
        assert f"`{finding.fix}`" in finding.line, finding.line

    assert cli.main(["doctor", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert list(report) == [key for key, _ in doctor.REPORT_DEFAULTS]
    assert report["problems"] == [finding.line for finding in findings]


def test_the_text_and_json_carry_the_same_problems(
    home: Path, viable: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["init", "--enable-asr"]) == 0
    capsys.readouterr()
    assert cli.main(["doctor", "--json"]) == 1
    problems = json.loads(capsys.readouterr().out)["problems"]
    assert cli.main(["doctor"]) == 1
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [f"PROBLEM: {problem}" for problem in problems]
    assert captured.out.splitlines()[-1] == "unhealthy"


def test_env_patch_checks_the_job_type_it_was_given(monkeypatch: pytest.MonkeyPatch) -> None:
    windows = Backend(
        kind="llama-windows", platform="win32", arch="x86_64",
        gpu=Gpu(vendor="nvidia", name="card", vram_bytes=1), detail="test double",
    )
    checked: list[str] = []
    config = argparse.Namespace(home=Path("/nowhere"))
    monkeypatch.setattr(cli.common, "here", lambda: (config, windows))
    monkeypatch.setattr(doctor.envpatches, "patches_for", lambda job_type: ())
    monkeypatch.setattr(
        doctor.envpatches, "check",
        lambda job_type, directory, pins: checked.append(job_type) or [],
    )
    assert doctor.cmd_env_patch(argparse.Namespace(job_type="llm")) == 0
    assert checked == ["llm"]
