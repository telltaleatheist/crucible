from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from crucible import engines as engines_module
from crucible.engines.narrator import NarratorEngine
from crucible.narratorvoices import VoicesDocument

FAKE_NARRATOR = Path(__file__).resolve().parent / "fake_narrator.py"

FAKE_TOTAL_BYTES = 64 * 1024 * 1024 * 1024


class FakeNarratorEngine(NarratorEngine):

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        return [
            sys.executable,
            str(FAKE_NARRATOR),
            "--engine",
            self.narrator_engine,
        ]


def install(monkeypatch: pytest.MonkeyPatch) -> list[FakeNarratorEngine]:
    built: list[FakeNarratorEngine] = []

    def build(
        narrator_engine: str,
        python: Path,
        log_path: Path,
        *,
        serving_stack: str | None,
        max_num_seqs: int | None,
        mem_fraction: float | None,
        context_length: int | None,
        stall_guard: str | None,
        voices: VoicesDocument | None,
        mlx_total_bytes: int | None,
    ) -> FakeNarratorEngine:
        engine = FakeNarratorEngine(
            narrator_engine=narrator_engine,
            python=python,
            log_path=log_path,
            serving_stack=None,
            max_num_seqs=None,
            mem_fraction=None,
            context_length=None,
            stall_guard=stall_guard,
            voices=voices,
            mlx_total_bytes=FAKE_TOTAL_BYTES,
        )
        built.append(engine)
        return engine

    monkeypatch.setattr(engines_module, "build_voice_engine", build)
    return built


def steer(monkeypatch: pytest.MonkeyPatch, **variables: Any) -> None:
    for name, value in variables.items():
        monkeypatch.setenv(f"CRUCIBLE_FAKE_{name.upper()}", str(value))


__all__: list[str] = ["FAKE_NARRATOR", "FakeNarratorEngine", "install", "steer"]
