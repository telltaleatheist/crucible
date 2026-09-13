"""`NarratorEngine` with one line changed: the argv.

`tests/fake_engine.py` replaces vLLM with an HTTP server because the only things
Crucible needs from vLLM are a process and a route. narrator is not like that —
the protocol *is* the thing being built — so this does the opposite: it is the
**real** `crucible/engines/narrator.py`, with its real pipes, its real reader
thread, its real correlation and its real `stop()`, pointed at
`tests/fake_narrator.py` instead of `python -m narrator.serve`.

That is the whole substitution. `command()` is overridden and nothing else is, so
a test that goes green here has exercised every line of the engine that a render
on the PC will run, minus the part that needs 8.5 GB of weights and a card.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from crucible import residency as residency_module
from crucible.engines.narrator import NarratorEngine

FAKE_NARRATOR = Path(__file__).resolve().parent / "fake_narrator.py"


class FakeNarratorEngine(NarratorEngine):
    """The real engine, started on the fake worker."""

    def command(
        self, model_dir: Path, served_name: str, port: int, args: list[str]
    ) -> list[str]:
        # `sys.executable` rather than `self._python`: the env fixtures stamp a
        # non-executable placeholder there, because what they are testing is that
        # `env_status` accepts a stamped env, not that it can run one.
        return [
            sys.executable,
            str(FAKE_NARRATOR),
            "--engine",
            self.narrator_engine,
        ]


def install(monkeypatch: pytest.MonkeyPatch) -> list[FakeNarratorEngine]:
    """Point the residency's voice loads at the fake worker.

    Returns the list the built engines are appended to, so a test can reach the
    engine the server is holding — its log path, its pids — without going through
    the API for it.
    """
    built: list[FakeNarratorEngine] = []

    def build(narrator_engine: str, python: Path, log_path: Path) -> FakeNarratorEngine:
        engine = FakeNarratorEngine(
            narrator_engine=narrator_engine, python=python, log_path=log_path
        )
        built.append(engine)
        return engine

    monkeypatch.setattr(residency_module, "build_voice_engine", build)
    return built


def steer(monkeypatch: pytest.MonkeyPatch, **variables: Any) -> None:
    """Set the fake worker's `CRUCIBLE_FAKE_*` switches for one test.

    They reach the subprocess through `os.environ`, which is what
    `SubprocessEngine.start()` hands it, so a `monkeypatch.setenv` here is read
    by the spawned worker.
    """
    for name, value in variables.items():
        monkeypatch.setenv(f"CRUCIBLE_FAKE_{name.upper()}", str(value))


__all__: list[str] = ["FAKE_NARRATOR", "FakeNarratorEngine", "install", "steer"]
